"""
Unit tests for core/feature_detection.py.
"""

import unittest
import numpy as np
import cv2

from core.feature_detection import SIFTFeatureDetector, ORBFeatureDetector, AKAZEFeatureDetector, FeatureDetectionError
from image_fixtures import textured_test_image

class TestFeatureDetection(unittest.TestCase):

    def setUp(self):
        # Create textured lunar surface with abundant keypoints
        lunar_bgr = textured_test_image(width=400, height=400, seed=42)
        self.gray_img = cv2.cvtColor(lunar_bgr, cv2.COLOR_BGR2GRAY)

    def test_sift_detector(self):
        detector = SIFTFeatureDetector()
        kp, desc = detector.detect_and_compute(self.gray_img)
        self.assertIsInstance(kp, list)
        self.assertGreater(len(kp), 10, "SIFT should detect >10 keypoints on crater texture.")
        self.assertIsNotNone(desc)
        self.assertEqual(desc.shape[0], len(kp))
        self.assertEqual(desc.shape[1], 128)  # SIFT 128D descriptor

    def test_orb_detector(self):
        detector = ORBFeatureDetector(nfeatures=500)
        kp, desc = detector.detect_and_compute(self.gray_img)
        self.assertIsInstance(kp, list)
        self.assertGreater(len(kp), 10, "ORB should detect >10 keypoints on crater texture.")
        self.assertIsNotNone(desc)
        self.assertEqual(desc.shape[0], len(kp))
        self.assertEqual(desc.shape[1], 32)   # ORB 32-byte binary descriptor

    def test_akaze_detector(self):
        try:
            detector = AKAZEFeatureDetector()
            kp, desc = detector.detect_and_compute(self.gray_img)
            self.assertIsInstance(kp, list)
            self.assertIsNotNone(desc)
        except FeatureDetectionError:
            pass  # AKAZE not compiled in this OpenCV build

    def test_empty_image_error(self):
        detector = SIFTFeatureDetector()
        with self.assertRaises(FeatureDetectionError):
            detector.detect_and_compute(None)

if __name__ == "__main__":
    unittest.main()
