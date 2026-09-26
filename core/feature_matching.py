"""
Feature matching module for lunar image registration.
Implements FLANN and Brute-Force matchers with Lowe's ratio test outlier rejection.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import cv2
import numpy as np

class FeatureMatchingError(Exception):
    """Raised when matching fails due to missing descriptors or insufficient pairs."""
    pass

class BaseFeatureMatcher(ABC):
    """
    Abstract Base Class for feature matchers.
    Extensible for deep-learning matchers like LightGlue / SuperGlue.
    """

    @abstractmethod
    def match(self, keypoints_ref: List[cv2.KeyPoint], descriptors_ref: Optional[np.ndarray],
              keypoints_src: List[cv2.KeyPoint], descriptors_src: Optional[np.ndarray],
              norm_type: int = cv2.NORM_L2) -> Tuple[List[cv2.DMatch], np.ndarray, np.ndarray]:
        """
        Match descriptors between reference and source images.

        Returns:
            Tuple containing:
            - List of good cv2.DMatch objects passing ratio test
            - np.ndarray of matched reference (x,y) points [N x 2]
            - np.ndarray of matched source (x,y) points [N x 2]
        """
        pass


class FLANNMatcher(BaseFeatureMatcher):
    """
    FLANN (Fast Library for Approximate Nearest Neighbors) Matcher.
    Uses KDTree for L2 descriptors (SIFT) and LSH Index for Binary descriptors (ORB/AKAZE).
    """

    def __init__(self, ratio_threshold: float = 0.75):
        self.ratio_threshold = ratio_threshold

    def match(self, keypoints_ref: List[cv2.KeyPoint], descriptors_ref: Optional[np.ndarray],
              keypoints_src: List[cv2.KeyPoint], descriptors_src: Optional[np.ndarray],
              norm_type: int = cv2.NORM_L2) -> Tuple[List[cv2.DMatch], np.ndarray, np.ndarray]:

        if descriptors_ref is None or descriptors_src is None:
            return [], np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        if len(descriptors_ref) < 2 or len(descriptors_src) < 2:
            return [], np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        try:
            if norm_type == cv2.NORM_L2:
                # FLANN KDTree index for SIFT (floating point descriptors)
                INDEX_KDTREE = 1
                index_params = dict(algorithm=INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
            else:
                # FLANN LSH index for binary descriptors (ORB, AKAZE)
                INDEX_LSH = 6
                index_params = dict(algorithm=INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
                search_params = dict(checks=50)

            flann = cv2.FlannBasedMatcher(index_params, search_params)

            # Convert binary descriptors to uint8 if needed
            if norm_type != cv2.NORM_L2:
                if descriptors_ref.dtype != np.uint8:
                    descriptors_ref = descriptors_ref.astype(np.uint8)
                if descriptors_src.dtype != np.uint8:
                    descriptors_src = descriptors_src.astype(np.uint8)
            else:
                if descriptors_ref.dtype != np.float32:
                    descriptors_ref = descriptors_ref.astype(np.float32)
                if descriptors_src.dtype != np.float32:
                    descriptors_src = descriptors_src.astype(np.float32)

            knn_matches = flann.knnMatch(descriptors_ref, descriptors_src, k=2)

        except Exception as e:
            # Fallback to Brute-Force if FLANN fails on edge cases
            bf = cv2.BFMatcher(norm_type)
            knn_matches = bf.knnMatch(descriptors_ref, descriptors_src, k=2)

        good_matches: List[cv2.DMatch] = []
        for m_n in knn_matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)

        pts_ref = np.float32([keypoints_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 2) if good_matches else np.empty((0, 2), dtype=np.float32)
        pts_src = np.float32([keypoints_src[m.trainIdx].pt for m in good_matches]).reshape(-1, 2) if good_matches else np.empty((0, 2), dtype=np.float32)

        return good_matches, pts_ref, pts_src


class BFMatcher(BaseFeatureMatcher):
    """
    Brute-Force Matcher with Lowe's Ratio Test.
    """

    def __init__(self, ratio_threshold: float = 0.75):
        self.ratio_threshold = ratio_threshold

    def match(self, keypoints_ref: List[cv2.KeyPoint], descriptors_ref: Optional[np.ndarray],
              keypoints_src: List[cv2.KeyPoint], descriptors_src: Optional[np.ndarray],
              norm_type: int = cv2.NORM_L2) -> Tuple[List[cv2.DMatch], np.ndarray, np.ndarray]:

        if descriptors_ref is None or descriptors_src is None:
            return [], np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        if len(descriptors_ref) < 2 or len(descriptors_src) < 2:
            return [], np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        bf = cv2.BFMatcher(norm_type)
        knn_matches = bf.knnMatch(descriptors_ref, descriptors_src, k=2)

        good_matches: List[cv2.DMatch] = []
        for m_n in knn_matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)

        pts_ref = np.float32([keypoints_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 2) if good_matches else np.empty((0, 2), dtype=np.float32)
        pts_src = np.float32([keypoints_src[m.trainIdx].pt for m in good_matches]).reshape(-1, 2) if good_matches else np.empty((0, 2), dtype=np.float32)

        return good_matches, pts_ref, pts_src
