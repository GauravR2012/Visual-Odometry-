"""
config.py
All dataset paths and tunable parameters in one place.
Change DATA_ROOT and SEQUENCE to point at your KITTI download.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ------------------------------------------------------------------ paths
    data_root: str = "/kaggle/input/kitti-odometry"
    sequence: str = "00"
    max_frames: int = None          # set e.g. 500 to do a quick debug run

    # ------------------------------------------------------------------ ORB
    n_features: int = 3000          # keypoints per frame
    max_matches: int = 500          # top-N matches kept after BFMatcher sort

    # ------------------------------------------------------------------ StereoBM
    num_disparities: int = 112      # must be divisible by 16
    block_size: int = 15            # must be odd

    # ------------------------------------------------------------------ PnP
    pnp_min_points: int = 6         # minimum 3D-2D pairs for solvePnPRansac

    # ------------------------------------------------------------------ derived (auto-filled, do not edit)
    seq_path: Path = field(init=False)
    left_dir: Path = field(init=False)
    right_dir: Path = field(init=False)
    calib_path: Path = field(init=False)
    pose_path: Path = field(init=False)

    def __post_init__(self):
        root = Path(self.data_root)
        self.seq_path   = root / "sequences" / self.sequence
        self.left_dir   = self.seq_path / "image_2"
        self.right_dir  = self.seq_path / "image_3"
        self.calib_path = self.seq_path / "calib.txt"
        self.pose_path  = root / "poses" / f"{self.sequence}.txt"
