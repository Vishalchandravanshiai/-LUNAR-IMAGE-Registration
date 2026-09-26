"""
Feature detection module for optical lunar image correspondence.
Defines extensible interfaces for keypoint detection and descriptor extraction.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Dict, Any
import cv2
import numpy as np

class FeatureDetectionError(Exception):
    """Raised when feature detection or descriptor computation fails."""
    pass

class BaseFeatureDetector(ABC):
    """
    Abstract Base Class for local feature detectors.
    Designed so deep learning models (SuperPoint, DISK, ALIKE) can be plugged in seamlessly.
    """

    @abstractmethod
    def detect_and_compute(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """
        Detect keypoints and extract descriptors from an 8-bit grayscale image.

        Args:
            image (np.ndarray): Single-channel 8-bit grayscale image.

        Returns:
            Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]: Keypoints list and descriptor array.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name of the detector."""
        pass

    @property
    @abstractmethod
    def norm_type(self) -> int:
        """Returns OpenCV distance norm enum (e.g., cv2.NORM_L2 or cv2.NORM_HAMMING)."""
        pass


class SIFTFeatureDetector(BaseFeatureDetector):
    """
    SIFT (Scale-Invariant Feature Transform) Detector baseline.
    Invertible to scale and rotation; widely recognized classical baseline.
    """

    def __init__(self, nfeatures: int = 0, contrastThreshold: float = 0.04,
                 edgeThreshold: float = 10.0, sigma: float = 1.6):
        self.nfeatures = nfeatures
        self.contrastThreshold = contrastThreshold
        self.edgeThreshold = edgeThreshold
        self.sigma = sigma

        try:
            self._sift = cv2.SIFT_create(
                nfeatures=self.nfeatures,
                contrastThreshold=self.contrastThreshold,
                edgeThreshold=self.edgeThreshold,
                sigma=self.sigma
            )
        except AttributeError as e:
            raise FeatureDetectionError("OpenCV SIFT module not available in this OpenCV build.") from e

    @property
    def name(self) -> str:
        return "SIFT"

    @property
    def norm_type(self) -> int:
        return cv2.NORM_L2

    def detect_and_compute(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        if image is None or image.size == 0:
            raise FeatureDetectionError("Invalid image input for SIFT feature detection.")

        keypoints, descriptors = self._sift.detectAndCompute(image, None)
        if keypoints is None:
            keypoints = []
        else:
            keypoints = list(keypoints)
        return keypoints, descriptors


class ORBFeatureDetector(BaseFeatureDetector):
    """
    ORB (Oriented FAST and Rotated BRIEF) Detector baseline.
    Fast binary descriptor baseline.
    """

    def __init__(self, nfeatures: int = 2000, scaleFactor: float = 1.2, nlevels: int = 8):
        self.nfeatures = nfeatures
        self.scaleFactor = scaleFactor
        self.nlevels = nlevels
        self._orb = cv2.ORB_create(
            nfeatures=self.nfeatures,
            scaleFactor=self.scaleFactor,
            nlevels=self.nlevels
        )

    @property
    def name(self) -> str:
        return "ORB"

    @property
    def norm_type(self) -> int:
        return cv2.NORM_HAMMING

    def detect_and_compute(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        if image is None or image.size == 0:
            raise FeatureDetectionError("Invalid image input for ORB feature detection.")

        keypoints, descriptors = self._orb.detectAndCompute(image, None)
        if keypoints is None:
            keypoints = []
        else:
            keypoints = list(keypoints)
        return keypoints, descriptors


class AKAZEFeatureDetector(BaseFeatureDetector):
    """
    AKAZE (Accelerated KAZE) Feature Detector.
    Uses non-linear scale spaces for sharp edge retention.
    """

    def __init__(self, threshold: float = 0.001):
        self.threshold = threshold
        if hasattr(cv2, 'AKAZE_create'):
            self._akaze = cv2.AKAZE_create(threshold=self.threshold)
        elif hasattr(cv2, 'AKAZE'):
            self._akaze = cv2.AKAZE.create(threshold=self.threshold)
        else:
            # Fallback for OpenCV builds where AKAZE is absent
            self._akaze = None

    @property
    def name(self) -> str:
        return "AKAZE"

    @property
    def norm_type(self) -> int:
        return cv2.NORM_HAMMING

    def detect_and_compute(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        if image is None or image.size == 0:
            raise FeatureDetectionError("Invalid image input for AKAZE feature detection.")

        if self._akaze is None:
            raise FeatureDetectionError("AKAZE feature detector is not available in current OpenCV build.")

        keypoints, descriptors = self._akaze.detectAndCompute(image, None)
        if keypoints is None:
            keypoints = []
        else:
            keypoints = list(keypoints)
        return keypoints, descriptors
