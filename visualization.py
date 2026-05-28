"""
visualization.py
Plotting utilities for trajectory comparison and ATE results.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_trajectory(
    gt_positions: np.ndarray,
    est_aligned: np.ndarray,
    metrics: dict[str, float],
    sequence: str,
    save_path: Path | None = None,
) -> None:
    """
    Two-panel figure:
      Left  — X-Z top-down trajectory (GT vs estimated)
      Right — per-frame ATE error over time

    Parameters
    ----------
    gt_positions : np.ndarray, shape (N, 3)
    est_aligned  : np.ndarray, shape (N, 3)
    metrics      : dict from evaluation.compute_ate()
    sequence     : str, e.g. "00"
    save_path    : if given, save the figure to this path as PNG
    """
    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.4, 1], wspace=0.35)

    # ---- left panel: X-Z trajectory ----
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(gt_positions[:, 0],  gt_positions[:, 2],
             color="#1D9E75", linewidth=1.8, label="Ground truth")
    ax1.plot(est_aligned[:, 0], est_aligned[:, 2],
             color="#7F77DD", linewidth=1.8, linestyle="--", label="Estimated")
    ax1.scatter(gt_positions[0, 0], gt_positions[0, 2],
                color="#639922", s=80, zorder=5, label="Start")
    ax1.scatter(gt_positions[-1, 0], gt_positions[-1, 2],
                color="#E24B4A", s=80, zorder=5, label="End")
    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Z [m]")
    ax1.set_title(f"Trajectory — KITTI seq {sequence}", fontsize=12)
    ax1.axis("equal")
    ax1.grid(True, linewidth=0.4, alpha=0.6)
    ax1.legend(fontsize=9)

    # ---- right panel: per-frame ATE ----
    errors = np.linalg.norm(est_aligned - gt_positions, axis=1)
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(errors, color="#7F77DD", linewidth=1.2)
    ax2.axhline(metrics["rmse"], color="#E24B4A", linewidth=1,
                linestyle="--", label=f"RMSE {metrics['rmse']:.2f} m")
    ax2.axhline(metrics["mean"], color="#EF9F27", linewidth=1,
                linestyle=":",  label=f"Mean {metrics['mean']:.2f} m")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("ATE [m]")
    ax2.set_title("Per-frame ATE", fontsize=12)
    ax2.grid(True, linewidth=0.4, alpha=0.6)
    ax2.legend(fontsize=9)

    # ---- overall title ----
    fig.suptitle(
        f"Stereo VO · seq {sequence} · "
        f"ATE RMSE {metrics['rmse']:.3f} m  "
        f"mean {metrics['mean']:.3f} m  "
        f"max {metrics['max']:.3f} m",
        fontsize=10,
        y=1.01,
    )

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved → {save_path}")

    plt.show()
