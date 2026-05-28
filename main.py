"""
main.py
Entry point for KITTI stereo visual odometry.

Usage
-----
    python main.py                         # uses defaults in config.py
    python main.py --sequence 05           # different KITTI sequence
    python main.py --max_frames 500        # quick debug run
    python main.py --no_eval               # skip GT evaluation
"""

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from config import Config
from calibration import load_kitti_calib
from depth import DepthEstimator
from features import FeatureMatcher
from odometry import StereoVO
from evaluation import load_gt_poses, align_trajectories, compute_ate
from visualization import plot_trajectory


# ------------------------------------------------------------------ CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KITTI stereo visual odometry")
    p.add_argument("--data_root",   default=None)
    p.add_argument("--sequence",    default=None)
    p.add_argument("--max_frames",  type=int, default=None)
    p.add_argument("--no_eval",     action="store_true",
                   help="Skip ground-truth evaluation")
    p.add_argument("--no_plot",     action="store_true",
                   help="Skip trajectory plot")
    return p.parse_args()


# ------------------------------------------------------------------ helpers
def load_image_paths(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG images found in {directory}")
    return paths


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Could not read {path}")
    return img


# ------------------------------------------------------------------ main
def main() -> None:
    args = parse_args()

    # Build config, override from CLI if provided
    cfg = Config()
    if args.data_root:
        cfg.data_root = args.data_root
        cfg.__post_init__()
    if args.sequence:
        cfg.sequence = args.sequence
        cfg.__post_init__()
    if args.max_frames is not None:
        cfg.max_frames = args.max_frames

    print(f"\nKITTI stereo VO — sequence {cfg.sequence}")
    print(f"  Data root : {cfg.data_root}")
    print(f"  Calib     : {cfg.calib_path}\n")

    # --- calibration ---
    K, baseline = load_kitti_calib(cfg.calib_path)
    print(f"Camera K:\n{K}")
    print(f"Baseline: {baseline:.4f} m\n")

    # --- image lists ---
    left_paths  = load_image_paths(cfg.left_dir)
    right_paths = load_image_paths(cfg.right_dir)

    if len(left_paths) != len(right_paths):
        raise RuntimeError("Left/right image counts do not match")

    if cfg.max_frames is not None:
        left_paths  = left_paths[:cfg.max_frames]
        right_paths = right_paths[:cfg.max_frames]

    print(f"Processing {len(left_paths)} frames…\n")

    # --- build pipeline ---
    depth_est = DepthEstimator(
        K, baseline,
        num_disparities=cfg.num_disparities,
        block_size=cfg.block_size,
    )
    feat_matcher = FeatureMatcher(
        n_features=cfg.n_features,
        max_matches=cfg.max_matches,
    )
    vo = StereoVO(
        K, depth_est, feat_matcher,
        min_points=cfg.pnp_min_points,
    )

    # --- main loop ---
    for left_path, right_path in tqdm(
        zip(left_paths, right_paths),
        total=len(left_paths),
        desc="VO",
    ):
        left  = read_gray(left_path)
        right = read_gray(right_path)
        vo.process_frame(left, right)

    trajectory = vo.get_trajectory()
    print(f"\nTrajectory shape: {trajectory.shape}")

    # --- evaluation ---
    if not args.no_eval and cfg.pose_path.exists():
        print(f"\nLoading ground truth from {cfg.pose_path}")
        gt_poses     = load_gt_poses(cfg.pose_path)
        gt_positions = gt_poses[:, :3, 3]

        N = min(len(gt_positions), len(trajectory))
        gt_positions = gt_positions[:N]
        est_positions = trajectory[:N]

        est_aligned, _, _ = align_trajectories(gt_positions, est_positions)
        metrics = compute_ate(gt_positions, est_aligned)

        print(f"\n{'='*40}")
        print(f"  ATE RMSE : {metrics['rmse']:.3f} m")
        print(f"  ATE mean : {metrics['mean']:.3f} m")
        print(f"  ATE max  : {metrics['max']:.3f} m")
        print(f"{'='*40}\n")

        if not args.no_plot:
            save_path = Path("results") / f"seq{cfg.sequence}_trajectory.png"
            plot_trajectory(
                gt_positions, est_aligned, metrics,
                sequence=cfg.sequence,
                save_path=save_path,
            )
    else:
        if not cfg.pose_path.exists():
            print("No ground-truth poses found — skipping evaluation.")

        if not args.no_plot:
            # Plot estimate only
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 6))
            plt.plot(trajectory[:, 0], trajectory[:, 2], color="#7F77DD")
            plt.xlabel("X [m]"); plt.ylabel("Z [m]")
            plt.title(f"Estimated trajectory — seq {cfg.sequence}")
            plt.axis("equal"); plt.grid(True)
            plt.tight_layout()
            save_path = Path("results") / f"seq{cfg.sequence}_trajectory.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.show()


if __name__ == "__main__":
    main()
