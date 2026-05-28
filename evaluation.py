"""
evaluation.py
Ground-truth loading and trajectory evaluation metrics.

Functions
---------
load_gt_poses()        parse KITTI poses/*.txt into (N, 4, 4) array
align_trajectories()   rigid Umeyama alignment of estimated to GT
compute_ate()          Absolute Trajectory Error (RMSE, mean, max)
"""

from pathlib import Path
import numpy as np


def load_gt_poses(pose_path: Path) -> np.ndarray:
    """
    Load KITTI ground-truth poses from a poses/*.txt file.

    Each line encodes a 3×4 pose matrix (12 values, row-major).
    Returns them as (N, 4, 4) homogeneous matrices.

    Parameters
    ----------
    pose_path : Path

    Returns
    -------
    poses : np.ndarray, shape (N, 4, 4), float64
    """
    poses = []
    with open(pose_path, "r") as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) != 12:
                continue
            T = np.eye(4, dtype=np.float64)
            T[:3, :] = np.array(vals, dtype=np.float64).reshape(3, 4)
            poses.append(T)

    if not poses:
        raise RuntimeError(f"No valid poses found in {pose_path}")

    return np.stack(poses, axis=0)   # (N, 4, 4)


def align_trajectories(
    gt: np.ndarray, est: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rigid (rotation + translation, no scale) Umeyama alignment.

    Finds (R, t) that minimises ||gt - (R @ est.T).T - t||.
    Removes gauge freedom so ATE reflects real drift, not
    a global offset between the two coordinate frames.

    Parameters
    ----------
    gt  : np.ndarray, shape (N, 3)   ground-truth positions
    est : np.ndarray, shape (N, 3)   estimated positions

    Returns
    -------
    est_aligned : np.ndarray, shape (N, 3)
    R           : np.ndarray, shape (3, 3)   rotation applied
    t           : np.ndarray, shape (3,)     translation applied
    """
    gt  = np.asarray(gt,  dtype=np.float64)
    est = np.asarray(est, dtype=np.float64)

    mu_gt  = gt.mean(axis=0)
    mu_est = est.mean(axis=0)

    gt_c  = gt  - mu_gt
    est_c = est - mu_est

    W = est_c.T @ gt_c / gt.shape[0]
    U, _, Vt = np.linalg.svd(W)

    R = Vt.T @ U.T

    # Fix improper rotation (det = -1 means a reflection, not a rotation)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = mu_gt - R @ mu_est
    est_aligned = (R @ est.T).T + t

    return est_aligned, R, t


def compute_ate(
    gt: np.ndarray, est_aligned: np.ndarray
) -> dict[str, float]:
    """
    Compute Absolute Trajectory Error between aligned estimate and GT.

    Parameters
    ----------
    gt          : np.ndarray, shape (N, 3)
    est_aligned : np.ndarray, shape (N, 3)

    Returns
    -------
    metrics : dict with keys 'rmse', 'mean', 'max', 'min' (all in metres)
    """
    errors = np.linalg.norm(est_aligned - gt, axis=1)
    return {
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mean": float(np.mean(errors)),
        "max":  float(np.max(errors)),
        "min":  float(np.min(errors)),
    }
