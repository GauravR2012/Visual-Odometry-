"""
features.py
ORB feature detection and brute-force descriptor matching.

FeatureMatcher wraps cv2.ORB and cv2.BFMatcher and exposes:
  - detect()    extract keypoints + descriptors from a grayscale frame
  - match()     match two descriptor sets, return sorted top-N matches
"""

import cv2
import numpy as np


class FeatureMatcher:
    """
    Detects ORB keypoints and matches descriptors between frames.

    Parameters
    ----------
    n_features : int
        Maximum number of ORB keypoints to detect per frame.
    max_matches : int
        Maximum number of matches to keep (sorted by Hamming distance).
    """

    def __init__(self, n_features: int = 3000, max_matches: int = 500) -> None:
        self.max_matches = max_matches
        self._orb = cv2.ORB_create(nfeatures=n_features)
        # crossCheck=True: accept a match only if it's mutual (A→B and B→A)
        # This eliminates most false positives without a ratio test
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def detect(
        self, gray: np.ndarray
    ) -> tuple[list, np.ndarray | None]:
        """
        Detect ORB keypoints and compute descriptors.

        Parameters
        ----------
        gray : np.ndarray, uint8
            Grayscale image.

        Returns
        -------
        keypoints : list of cv2.KeyPoint
        descriptors : np.ndarray, shape (N, 32), uint8 — or None if no features
        """
        keypoints, descriptors = self._orb.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match(
        self,
        des1: np.ndarray,
        des2: np.ndarray,
    ) -> list:
        """
        Match two descriptor arrays and return the best matches.

        Parameters
        ----------
        des1, des2 : np.ndarray
            Descriptor arrays from detect(). If either is None or empty,
            returns an empty list.

        Returns
        -------
        matches : list of cv2.DMatch
            Top matches sorted by ascending Hamming distance,
            capped at self.max_matches.
        """
        if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
            return []

        matches = self._bf.match(des1, des2)
        matches = sorted(matches, key=lambda m: m.distance)
        return matches[: self.max_matches]
