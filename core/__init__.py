"""
Lunar Image Registration Core Package.
Provides modules for preprocessing, feature detection, feature matching,
geometric verification, image warping/registration, and reprojection metrics.
"""

from .preprocessing import ImagePreprocessor
from .feature_detection import SIFTFeatureDetector, ORBFeatureDetector, AKAZEFeatureDetector, BaseFeatureDetector
from .feature_matching import FLANNMatcher, BFMatcher, BaseFeatureMatcher
from .geometric_verification import RANSACVerifier, BaseGeometricVerifier
from .registration import ImageRegistrar
from .metrics import RegistrationMetrics, compute_registration_metrics

__all__ = [
    "ImagePreprocessor",
    "BaseFeatureDetector",
    "SIFTFeatureDetector",
    "ORBFeatureDetector",
    "AKAZEFeatureDetector",
    "BaseFeatureMatcher",
    "FLANNMatcher",
    "BFMatcher",
    "BaseGeometricVerifier",
    "RANSACVerifier",
    "ImageRegistrar",
    "RegistrationMetrics",
    "compute_registration_metrics",
]
