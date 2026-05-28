"""
odometry.py
Frame-by-frame stereo visual odometry using PnP.

StereoVO processes one stereo pair at a time:
  1. Compute depth from stereo pair
  2. Detect ORB features in left image
  3. Match against previous frame's features
  4. Build 3D (prev frame) ↔ 2D (curr frame) correspondences
  5. Estimate relative pose via solvePnPRansac
  6. Accumulate global pose
"""

import cv2
import numpy as np

from depth import DepthEstimator
from features import FeatureMatcher


class StereoVO:
    """
    Incremental stereo visual odometry.

    Parameters
    ----------
    K : np.ndarray, shape (3, 3)
        Camera intrinsic matrix.
    depth_estimator : DepthEstimator
    feature_matcher : FeatureMatcher
    min_points : int
        Minimum 3D-2D correspondences required to attempt PnP.
    """

    def __init__(
        self,
        K: np.ndarray,
        depth_estimator: DepthEstimator,
        feature_matcher: FeatureMatcher,
        min_points: int = 6,
    ) -> None:
        self.K = K
        self.depth_est = depth_estimator
        self.feat = feature_matcher
        self.min_points = min_points

        # Global pose: world_T_cam (4x4, starts at identity = origin)
        self._pose = np.eye(4, dtype=np.float64)
        self._trajectory: list[np.ndarray] = []

        # State carried from previous frame
        self._prev_kp = None
        self._prev_des = None
        self._prev_pts3d: np.ndarray | None = None

        self._frame_idx = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(
        self, left: np.ndarray, right: np.ndarray
    ) -> np.ndarray:
        """
        Process one stereo pair and return the updated camera pose.

        Parameters
        ----------
        left, right : np.ndarray, uint8 grayscale
            Rectified stereo images.

        Returns
        -------
        pose : np.ndarray, shape (4, 4)
            Current world_T_cam pose matrix.
        """
        depth = self.depth_est.compute(left, right)
        kp, des = self.feat.detect(left)

        if self._frame_idx == 0:
            self._init_first_frame(kp, des, depth)
            return self._pose.copy()

        pose = self._estimate_pose(kp, des, depth)
        self._trajectory.append(pose[:3, 3].copy())

        # Refresh landmarks for next frame
        self._prev_kp = kp
        self._prev_des = des
        self._prev_pts3d = self.depth_est.backproject_keypoints(kp, depth)
        self._frame_idx += 1

        return pose

    def get_trajectory(self) -> np.ndarray:
        """
        Return all accumulated camera positions as an (N, 3) array.
        """
        if not self._trajectory:
            return np.zeros((0, 3), dtype=np.float64)
        return np.array(self._trajectory, dtype=np.float64)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_first_frame(
        self, kp: list, des: np.ndarray, depth: np.ndarray
    ) -> None:
        """Store first-frame landmarks; no motion to estimate yet."""
        self._prev_kp = kp
        self._prev_des = des
        self._prev_pts3d = self.depth_est.backproject_keypoints(kp, depth)
        self._trajectory.append(self._pose[:3, 3].copy())
        self._frame_idx = 1

    def _estimate_pose(
        self, kp: list, des: np.ndarray, depth: np.ndarray
    ) -> np.ndarray:
        """
        Match current frame against previous, run PnP, update global pose.
        Falls back to repeating the previous pose if PnP fails.
        """
        matches = self.feat.match(self._prev_des, des)

        if len(matches) < self.min_points:
            return self._pose.copy()

        # Build 3D (from prev frame) ↔ 2D (in curr frame) correspondences
        pts3d = np.array(
            [self._prev_pts3d[m.queryIdx] for m in matches], dtype=np.float32
        )
        pts2d = np.array(
            [kp[m.trainIdx].pt for m in matches], dtype=np.float32
        )

        if len(pts3d) < self.min_points:
            return self._pose.copy()

        ok, rvec, tvec, _ = cv2.solvePnPRansac(
            pts3d,
            pts2d,
            self.K,
            None,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not ok:
            return self._pose.copy()

        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()

        # world_T_cam = world_T_cam @ inv(cam_N_T_cam_N+1)
        self._pose = self._pose @ np.linalg.inv(T)
        return self._pose.copy()
