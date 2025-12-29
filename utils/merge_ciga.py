import os
import sys
import add_pypath
import argparse
import torch
import numpy as np
import logging
from tqdm import tqdm

from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.citygs_partitioning_utils import CityGSPartitioning, PartitionCoordinates

parser = argparse.ArgumentParser()
parser.add_argument("path", help="Path to the model output directory")
args = parser.parse_args()

checkpoint_dir = os.path.join(args.path, "blocks")
if not os.path.isdir(checkpoint_dir):
    raise FileNotFoundError(f"blocks directory not found: {checkpoint_dir}")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

# merge.log는 outputs/<NAME>/merge.log에 저장되도록 유지
merge_log_path = os.path.join(os.path.dirname(checkpoint_dir), "merge.log")
file_handler = logging.FileHandler(merge_log_path)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.handlers.clear()
logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

# --------------------------
# find max iteration
# --------------------------
logger.info("Searching checkpoint files...")

max_iteration = -1
checkpoint_files = []
skipped_blocks = []
non_ckpt_blocks = []

for entry in sorted(os.listdir(checkpoint_dir)):
    block_path = os.path.join(checkpoint_dir, entry)

    if not os.path.isdir(block_path):
        continue

    try:
        loadable_file = GaussianModelLoader.search_load_file(block_path)
    except AssertionError:
        skipped_blocks.append(block_path)
        logger.warning(f"[skip] empty or no loadable file in block: {block_path}")
        continue

    if not loadable_file:
        skipped_blocks.append(block_path)
        logger.warning(f"[skip] empty or no loadable file in block: {block_path}")
        continue

    if not loadable_file.endswith(".ckpt"):
        non_ckpt_blocks.append((block_path, loadable_file))
        logger.warning(f"[skip] loadable file is not .ckpt (got: {loadable_file}) in block: {block_path}")
        continue

    try:
        step = int(loadable_file[loadable_file.index("step=") + 5:loadable_file.index(".ckpt")])
    except Exception:
        logger.warning(f"[skip] cannot parse step from ckpt path: {loadable_file}")
        continue

    if step > max_iteration:
        max_iteration = step
        checkpoint_files = []

    if step == max_iteration:
        checkpoint_files.append(loadable_file)

checkpoint_files = sorted(checkpoint_files)

if len(checkpoint_files) == 0:
    msg = [
        "No mergeable checkpoint (.ckpt) files were found under:",
        f"  {checkpoint_dir}",
        "",
        f"Skipped empty blocks: {len(skipped_blocks)}",
    ]
    if skipped_blocks:
        msg.append("  - " + "\n  - ".join(skipped_blocks[:20]) + (" ..." if len(skipped_blocks) > 20 else ""))
    if non_ckpt_blocks:
        msg.append(f"Blocks with non-ckpt loadable files: {len(non_ckpt_blocks)} (e.g., ply)")
        for b, f in non_ckpt_blocks[:10]:
            msg.append(f"  - {b} -> {f}")
    raise RuntimeError("\n".join(msg))

logger.info(f"Using max_iteration(step)={max_iteration}")
logger.info("Checkpoint files to merge:")
for f in checkpoint_files:
    logger.info(f"  {f}")

# --------------------------
# Merge
# --------------------------
import torch
from internal.models.gaussian import Gaussian  # noqa: F401

is_new_model = True
param_list_key_by_name = {}
extra_param_list_key_by_name = {}
optimizer_state_exp_avg_list_key_by_index = {}
optimizer_state_exp_avg_sq_list_key_by_index = {}
density_controller_state_list_key_by_name = {}
number_of_gaussians = []

ckpt = torch.load(checkpoint_files[0], map_location="cpu")
dataparser_config = ckpt["datamodule_hyper_parameters"]["parser"]

partitions = torch.load(os.path.join(os.path.dirname(dataparser_config.image_list), "partitions.pt"))
partition_coordinates = PartitionCoordinates(
    id=partitions['partition_coordinates']['id'],
    xy=partitions['partition_coordinates']['xy'],
)
partition_bounding_boxes = partition_coordinates.get_bounding_boxes(
    partitions['scene_config']['partition_size'],
    enlarge=0.
)

del ckpt, dataparser_config

for ckpt_path in tqdm(checkpoint_files, desc="Loading checkpoints"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    dataparser_config = ckpt["datamodule_hyper_parameters"]["parser"]

    xyz_gs = ckpt['state_dict']['gaussian_model.gaussians.means'] @ partitions['extra_data']['rotation_transform'][:3, :3].T

    if partitions['scene_config']['contract']:
        xyz_gs = CityGSPartitioning.contract_to_unisphere(
            xyz_gs[:, :2],
            partitions['scene_config']['aabb'],
            ord=torch.inf
        )

    mask_preserved = CityGSPartitioning.is_in_bounding_boxes(
        bounding_boxes=partition_bounding_boxes,
        coordinates=xyz_gs[:, :2],
    )[dataparser_config.block_id]

    property_names = []
    gaussian_property_dict_key_prefix = "gaussian_model.gaussians."
    density_controller_state_dict_key_prefix = "density_controller."

    # extract gaussian properties + density controller state
    for key, value in ckpt["state_dict"].items():
        if key.startswith(gaussian_property_dict_key_prefix):
            param_list_key_by_name.setdefault(key, []).append(value[mask_preserved])
            property_names.append(key[len(gaussian_property_dict_key_prefix):])
        elif key.startswith(density_controller_state_dict_key_prefix):
            param_list_key_by_name.setdefault(key, []).append(value)

    # extract optimizer states (assume meet gaussian optimizers first)
    for optimizer_idx, optimizer in enumerate(ckpt["optimizer_states"]):
        for param_group_idx, param_group in enumerate(optimizer["param_groups"]):
            if param_group["name"] not in property_names:
                continue

            property_names.remove(param_group["name"])
            state = optimizer["state"][param_group_idx]

            optimizer_state_exp_avg_list_key_by_index.setdefault(optimizer_idx, {}).setdefault(param_group_idx, []).append(
                state["exp_avg"]
            )
            optimizer_state_exp_avg_sq_list_key_by_index.setdefault(optimizer_idx, {}).setdefault(param_group_idx, []).append(
                state["exp_avg_sq"]
            )

        if len(property_names) == 0:
            break

    number_of_gaussians.append(mask_preserved.sum().item())

logger.info("Merging Gaussians and density controller states...")

ckpt["datamodule_hyper_parameters"]["parser"] = torch.load(
    ckpt['hyper_parameters']['initialize_from'],
    map_location="cpu"
)["datamodule_hyper_parameters"]["parser"]

for k in param_list_key_by_name:
    ckpt["state_dict"][k] = torch.concat(param_list_key_by_name[k], dim=0)

if is_new_model is True:
    logger.info("Merging optimizers...")
    for optimizer_idx in optimizer_state_exp_avg_list_key_by_index.keys():
        for param_group_idx in optimizer_state_exp_avg_list_key_by_index[optimizer_idx].keys():
            ckpt["optimizer_states"][optimizer_idx]["state"][param_group_idx]["exp_avg"] = torch.concat(
                optimizer_state_exp_avg_list_key_by_index[optimizer_idx][param_group_idx],
                dim=0,
            )
            ckpt["optimizer_states"][optimizer_idx]["state"][param_group_idx]["exp_avg_sq"] = torch.concat(
                optimizer_state_exp_avg_sq_list_key_by_index[optimizer_idx][param_group_idx],
                dim=0,
            )
else:
    for k in extra_param_list_key_by_name:
        ckpt["gaussian_model_extra_state_dict"][k] = torch.concat(extra_param_list_key_by_name[k], dim=0)
    logger.info("Merging optimizers...")
    for i in optimizer_state_exp_avg_list_key_by_index.keys():
        ckpt["optimizer_states"][0]["state"][i]["exp_avg"] = torch.concat(optimizer_state_exp_avg_list_key_by_index[i], dim=0)
        ckpt["optimizer_states"][0]["state"][i]["exp_avg_sq"] = torch.concat(optimizer_state_exp_avg_sq_list_key_by_index[i], dim=0)

logger.info("number_of_gaussians=sum({})={}".format(number_of_gaussians, sum(number_of_gaussians)))

out_ckpt_dir = os.path.join(os.path.dirname(checkpoint_dir), "checkpoints")
os.makedirs(out_ckpt_dir, exist_ok=True)

output_path = os.path.join(
    out_ckpt_dir,
    ckpt["hyper_parameters"]["initialize_from"].split('/')[-1]
)

logger.info("Saving...")
torch.save(ckpt, output_path)
logger.info(f"Saved to '{output_path}'")
logger.info(f"Skipped empty blocks: {len(skipped_blocks)}")
