"""
calibration.py
Loads KITTI stereo calibration from calib.txt.
Returns the 3x3 intrinsic matrix K and the stereo baseline in metres.
"""

from pathlib import Path
import numpy as np


def load_kitti_calib(calib_path: Path) -> tuple[np.ndarray, float]:
    """
    Parse a KITTI calib.txt file and return (K, baseline).

    Parameters
    ----------
    calib_path : Path
        Path to calib.txt for a KITTI sequence.

    Returns
    -------
    K : np.ndarray, shape (3, 3), dtype float64
        Camera intrinsic matrix for the left camera (P2).
    baseline : float
        Stereo baseline in metres (always positive).
    """
    P2, P3 = None, None

    with open(calib_path, "r") as f:
        for line in f:
            if line.startswith("P2:"):
                vals = line.split()[1:]
                P2 = np.array(vals, dtype=np.float64).reshape(3, 4)
            elif line.startswith("P3:"):
                vals = line.split()[1:]
                P3 = np.array(vals, dtype=np.float64).reshape(3, 4)

    if P2 is None or P3 is None:
        raise RuntimeError(f"Could not find P2/P3 in {calib_path}")

    K = P2[:, :3]

    # P3[0,3] = -fx * baseline  (negative in KITTI convention)
    # P2[0,3] = 0 for the rectified left camera
    baseline = (P2[0, 3] - P3[0, 3]) / K[0, 0]

    return K, float(baseline)
