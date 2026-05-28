"""
depth.py
Stereo depth estimation and 3D backprojection.

DepthEstimator wraps OpenCV's StereoBM and provides:
  - compute()             disparity → metric depth map
  - backproject()         single (u, v) pixel → 3D point
  - backproject_keypoints()  batch lift of all keypoints to 3D
"""

import cv2
import numpy as np


class DepthEstimator:
    """
    Computes dense depth from a stereo pair and backprojects 2D pixels to 3D.

    Parameters
    ----------
    K : np.ndarray, shape (3, 3)
        Camera intrinsic matrix.
    baseline : float
        Stereo baseline in metres.
    num_disparities : int
        StereoBM parameter — must be divisible by 16.
    block_size : int
        StereoBM parameter — must be odd.
    """

    def __init__(
        self,
        K: np.ndarray,
        baseline: float,
        num_disparities: int = 112,
        block_size: int = 15,
    ) -> None:
        self.K = K
        self.baseline = baseline
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]
        self._stereo = cv2.StereoBM_create(
            numDisparities=num_disparities,
            blockSize=block_size,
        )

    def compute(
        self, left_gray: np.ndarray, right_gray: np.ndarray
    ) -> np.ndarray:
        """
        Compute a metric depth map from a stereo pair.

        Parameters
        ----------
        left_gray, right_gray : np.ndarray, uint8
            Rectified grayscale images.

        Returns
        -------
        depth : np.ndarray, float32, same shape as input
            Per-pixel depth in metres. Invalid pixels are set to 0.1 m.
        """
        disp = self._stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        disp[disp <= 0.0] = 0.1
        depth = (self.fx * self.baseline) / disp
        return depth

    def backproject(self, u: float, v: float, depth: np.ndarray) -> np.ndarray:
        """
        Lift a single 2D pixel to a 3D point in the camera frame.

        Parameters
        ----------
        u, v : float
            Pixel coordinates (column, row).
        depth : np.ndarray
            Dense depth map from compute().

        Returns
        -------
        point3d : np.ndarray, shape (3,), float32
            [X, Y, Z] in the camera frame (metres).
        """
        z = depth[int(v), int(u)]
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return np.array([x, y, z], dtype=np.float32)

    def backproject_keypoints(
        self, keypoints: list, depth: np.ndarray
    ) -> np.ndarray:
        """
        Batch-lift a list of cv2.KeyPoint objects to 3D.

        Parameters
        ----------
        keypoints : list of cv2.KeyPoint
        depth : np.ndarray

        Returns
        -------
        pts3d : np.ndarray, shape (N, 3), float32
        """
        h, w = depth.shape
        pts3d = []
        for kp in keypoints:
            u, v = kp.pt
            # Clamp to valid pixel range
            u = np.clip(u, 0, w - 1)
            v = np.clip(v, 0, h - 1)
            pts3d.append(self.backproject(u, v, depth))
        return np.array(pts3d, dtype=np.float32)
