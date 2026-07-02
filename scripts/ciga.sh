#!/bin/bash

set -e

# ============================================================
# CIGA / View-Adaptive Gaussian Splatting Full Pipeline
# ============================================================
#
# Stage 1. Global coarse optimization
#   - Train the full scene globally.
#   - Learn Gaussian parameters and GateMLP jointly.
#
# Stage 2. Local block-wise fine-tuning
#   - Partition the scene into blocks.
#   - Fine-tune local Gaussian parameters for each block.
#   - GateMLP can be frozen during this stage to preserve global modulation behavior.
#
# Merge + checkpoint fixing
#   - Merge locally optimized blocks back into a single global checkpoint.
#   - Reset density-controller buffers and remove optimizer/loop states.
#   - Gaussian and GateMLP parameters are preserved.
#
# Stage 3. Final joint refinement
#   - Refine the merged Gaussian representation together with the GateMLP.
#   - This aligns the learned view-adaptive SH modulation with the final merged scene.
#
# ============================================================


# -----------------------------
# Environment
# -----------------------------
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ciga

cd /workspace/Ciga/git/tmp/capstone_ciga


# ============================================================
# Stage 1: Global coarse optimization
# ============================================================
# Output example:
#   outputs/v2coarse2/checkpoints/epoch=49-step=30000.ckpt

python main.py fit \
  --config configs/ciga_yaml/coarse.yaml \
  -n v2coarse2


# ============================================================
# Stage 2-A: Scene partitioning
# ============================================================
# This step prepares block metadata / partitions for local fine-tuning.
# Replace the script/config name if your repository uses a different file.

python tools/partition_scene.py \
  --config configs/ciga_yaml/partition.yaml \
  --input_ckpt outputs/v2coarse2/checkpoints/epoch=49-step=30000.ckpt \
  --output_dir outputs/v2partition


# ============================================================
# Stage 2-B: Local block-wise fine-tuning
# ============================================================
# Fine-tune Gaussian parameters block by block.
# GateMLP freeze should be enabled in the renderer if using the Stage 2 setting.
#
# Output example:
#   outputs/v2block_final/checkpoints/epoch=49-step=30000.ckpt

python main.py fit \
  --config configs/ciga_yaml/train.yaml \
  --ckpt_path outputs/v2coarse2/checkpoints/epoch=49-step=30000.ckpt \
  -n v2block_final


# ============================================================
# Merge local blocks
# ============================================================
# Merge block-wise optimized Gaussians back into a single global representation.
# Replace script/config names if your repo uses different names.

python tools/merge_blocks.py \
  --config configs/ciga_yaml/merge_citygs_ckpt.yaml \
  --input_dir outputs/v2block_final \
  --output_ckpt outputs/v2block_final/checkpoints/epoch=49-step=30000.ckpt


# ============================================================
# Fix merged checkpoint
# ============================================================
# This script preserves:
#   - gaussian_model.gaussians.*
#   - renderer.mlp_model.*
#
# It resets/removes:
#   - density_controller.max_radii2D
#   - density_controller.xyz_gradient_accum
#   - density_controller.denom
#   - optimizer_states
#   - loops
#
# Output:
#   outputs/v2block_final/checkpoints/merge_fixed_v2.ckpt

python after_merge.py


# ============================================================
# Stage 3: Final joint refinement after merging
# ============================================================
# This stage is best interpreted as final joint refinement:
#   - merged Gaussian representation
#   - GateMLP
#
# For strict MLP-only refinement, enable Gaussian freeze in the renderer code.

python main.py fit \
  --config configs/ciga_yaml/after_merge.yaml \
  --ckpt_path outputs/v2block_final/checkpoints/merge_fixed_v2.ckpt \
  -n rescale


# ============================================================
# Evaluation / Test
# ============================================================
# Use --model.save_val_output true to save rendered images.

python main.py test \
  --config configs/ciga_yaml/test.yaml \
  --ckpt_path outputs/rescale/checkpoints/epoch=29-step=20000.ckpt \
  -n checkresult \
  --data.parser.depth_rescaling false \
  --model.save_val_metrics true \
  --model.save_val_output true \
  --model.max_save_val_output 100