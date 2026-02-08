#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import logging
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional


def run(cmd: List[str]) -> None:
    """Run a command and raise with readable message on failure."""
    logging.info("RUN: %s", " ".join(map(str, cmd)))
    p = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed (exit={p.returncode}): {' '.join(cmd)}")


def find_model_dir(sparse_root: Path) -> Path:
    """
    Return a COLMAP model directory path.
    - If sparse_root/0 exists, use it.
    - Else if sparse_root contains a single numeric subdir, use the smallest (e.g., 0, 1, 2).
    - Else use sparse_root itself.
    """
    cand0 = sparse_root / "0"
    if cand0.is_dir():
        return cand0

    numeric_subdirs = []
    if sparse_root.is_dir():
        for p in sparse_root.iterdir():
            if p.is_dir() and p.name.isdigit():
                numeric_subdirs.append((int(p.name), p))
    if numeric_subdirs:
        numeric_subdirs.sort(key=lambda x: x[0])
        return numeric_subdirs[0][1]

    return sparse_root


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = ArgumentParser("CIGA COLMAP/GLOMAP converter")

    parser.add_argument("--source_path", "-s", required=True, type=str,
                        help="Scene root. Expects images under <source_path>/input/")
    parser.add_argument("--camera", default="OPENCV", type=str,
                        help="COLMAP camera model, e.g., OPENCV / PINHOLE / SIMPLE_PINHOLE")

    parser.add_argument("--skip_matching", action="store_true",
                        help="Skip feature extraction/matching/mapping and only undistort/resize.")
    parser.add_argument("--no_gpu", action="store_true",
                        help="Disable GPU usage for COLMAP feature extraction/matching (if supported).")
    parser.add_argument("--gpu_index", default=0, type=int,
                        help="GPU index for COLMAP feature extraction/matching.")
    parser.add_argument("--add_match", action="store_true",
                        help="Enable affine/dsp extraction + guided matching to improve robustness.")

    parser.add_argument("--colmap_executable", default="", type=str,
                        help="Optional path to colmap binary (default: colmap in PATH).")

    # ✅ NEW: use glomap for mapper
    parser.add_argument("--glomap", type=int, default=0,
                        help="If 1, use glomap mapper instead of colmap mapper.")
    parser.add_argument("--glomap_executable", default="", type=str,
                        help="Optional path to glomap binary (default: glomap in PATH).")
    parser.add_argument("--glomap_extra_args", default="", type=str,
                        help="Extra args appended to `glomap mapper ...` (raw string).")

    # optional resizing
    parser.add_argument("--resize", action="store_true",
                        help="Create images_2/images_4/images_8 via ImageMagick.")
    parser.add_argument("--magick_executable", default="", type=str,
                        help="Optional path to ImageMagick 'magick' executable (default: magick in PATH).")

    args = parser.parse_args()

    scene = Path(os.path.expanduser(args.source_path)).resolve()
    input_dir = scene / "input"
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input dir not found: {input_dir}  (expected <source_path>/input/)")

    colmap_bin = str(Path(args.colmap_executable).resolve()) if args.colmap_executable else "colmap"
    glomap_bin = str(Path(args.glomap_executable).resolve()) if args.glomap_executable else "glomap"
    magick_bin = str(Path(args.magick_executable).resolve()) if args.magick_executable else "magick"

    use_gpu = 0 if args.no_gpu else 1
    add_match = 1 if args.add_match else 0

    distorted_dir = scene / "distorted"
    db_path = distorted_dir / "database.db"
    sparse_out = distorted_dir / "sparse"

    if not args.skip_matching:
        sparse_out.mkdir(parents=True, exist_ok=True)

        # ---------------------------
        # 1) Feature extraction (COLMAP 3.14: FeatureExtraction.use_gpu / gpu_index)
        # ---------------------------
        feat_cmd = [
            colmap_bin, "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(input_dir),
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", str(args.camera),
            "--FeatureExtraction.use_gpu", str(use_gpu),
            "--FeatureExtraction.gpu_index", str(args.gpu_index),
            "--SiftExtraction.domain_size_pooling", str(add_match),
            "--SiftExtraction.estimate_affine_shape", str(add_match),
        ]
        run(feat_cmd)

        # ---------------------------
        # 2) Exhaustive matching (COLMAP 3.14: FeatureMatching.use_gpu / gpu_index / guided_matching)
        # ---------------------------
        match_cmd = [
            colmap_bin, "exhaustive_matcher",
            "--database_path", str(db_path),
            "--FeatureMatching.use_gpu", str(use_gpu),
            "--FeatureMatching.gpu_index", str(args.gpu_index),
            "--FeatureMatching.guided_matching", str(add_match),
        ]
        run(match_cmd)

        # ---------------------------
        # 3) Mapping / Bundle adjustment (COLMAP mapper OR GLOMAP mapper)
        # ---------------------------
        if int(args.glomap) == 1:
            # glomap mapper --database_path ... --image_path ... --output_path ...
            mapper_cmd = [
                glomap_bin, "mapper",
                "--database_path", str(db_path),
                "--image_path", str(input_dir),
                "--output_path", str(sparse_out),
            ]
            # Append raw extra args (space-split)
            extra = args.glomap_extra_args.strip()
            if extra:
                mapper_cmd += extra.split()
            run(mapper_cmd)
        else:
            # Original COLMAP mapper + smaller BA tolerance for speed
            mapper_cmd = [
                colmap_bin, "mapper",
                "--database_path", str(db_path),
                "--image_path", str(input_dir),
                "--output_path", str(sparse_out),
                "--Mapper.ba_global_function_tolerance", "0.000001",
            ]
            run(mapper_cmd)

    # ---------------------------
    # 4) Image undistortion
    #   - input_path must be a COLMAP model folder (cameras/images/points3D)
    #   - glomap may export to sparse/0 or directly sparse/
    # ---------------------------
    model_in = find_model_dir(sparse_out)
    undist_cmd = [
        colmap_bin, "image_undistorter",
        "--image_path", str(input_dir),
        "--input_path", str(model_in),
        "--output_path", str(scene),
        "--output_type", "COLMAP",
    ]
    run(undist_cmd)

    # After undistorter, COLMAP writes model files into <scene>/sparse (flat).
    # Keep compatibility with downstream code by moving them into <scene>/sparse/0
    sparse_dir = scene / "sparse"
    if sparse_dir.is_dir():
        (sparse_dir / "0").mkdir(parents=True, exist_ok=True)
        for f in list(sparse_dir.iterdir()):
            if f.name == "0":
                continue
            shutil.move(str(f), str(sparse_dir / "0" / f.name))

    # ---------------------------
    # 5) Optional resize (images -> images_2/images_4/images_8)
    # ---------------------------
    if args.resize:
        images_dir = scene / "images"
        if not images_dir.is_dir():
            raise FileNotFoundError(f"images dir not found after undistort: {images_dir}")

        out2 = scene / "images_2"
        out4 = scene / "images_4"
        out8 = scene / "images_8"
        out2.mkdir(exist_ok=True)
        out4.mkdir(exist_ok=True)
        out8.mkdir(exist_ok=True)

        for img in images_dir.iterdir():
            if not img.is_file():
                continue

            d2 = out2 / img.name
            d4 = out4 / img.name
            d8 = out8 / img.name

            shutil.copy2(str(img), str(d2))
            run([magick_bin, "mogrify", "-resize", "50%", str(d2)])

            shutil.copy2(str(img), str(d4))
            run([magick_bin, "mogrify", "-resize", "25%", str(d4)])

            shutil.copy2(str(img), str(d8))
            run([magick_bin, "mogrify", "-resize", "12.5%", str(d8)])

    print("Done.")


if __name__ == "__main__":
    main()
