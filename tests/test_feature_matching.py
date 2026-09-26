"""
Unit tests for core/feature_matching.py.
"""

import unittest
import numpy as np
import cv2

from core.feature_detection import SIFTFeatureDetector
from core.feature_matching import FLANNMatcher, BFMatcher
from image_fixtures import textured_test_image

class TestFeatureMatching(unittest.TestCase):

    def setUp(self):
        # Generate ref and slightly shifted source image
        img1 = textured_test_image(width=400, height=400, seed=1)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        # Shift gray1 by (10, 5) pixels
        M = np.float32([[1, 0, 10], [0, 1, 5]])
        gray2 = cv2.warpAffine(gray1, M, (400, 400))

        detector = SIFTFeatureDetector()
        self.kp1, self.desc1 = detector.detect_and_compute(gray1)
        self.kp2, self.desc2 = detector.detect_and_compute(gray2)

    def test_flann_matcher(self):
        matcher = FLANNMatcher(ratio_threshold=0.75)
        matches, pts_ref, pts_src = matcher.match(self.kp1, self.desc1, self.kp2, self.desc2, norm_type=cv2.NORM_L2)

        self.assertGreater(len(matches), 5, "FLANN should find good matches on shifted image.")
        self.assertEqual(len(pts_ref), len(matches))
        self.assertEqual(len(pts_src), len(matches))

    def test_bf_matcher(self):
        matcher = BFMatcher(ratio_threshold=0.75)
        matches, pts_ref, pts_src = matcher.match(self.kp1, self.desc1, self.kp2, self.desc2, norm_type=cv2.NORM_L2)

        self.assertGreater(len(matches), 5, "BFMatcher should find good matches on shifted image.")
        self.assertEqual(len(pts_ref), len(matches))
        self.assertEqual(len(pts_src), len(matches))

    def test_matching_with_empty_descriptors(self):
        matcher = FLANNMatcher()
        matches, pts_ref, pts_src = matcher.match(self.kp1, None, self.kp2, self.desc2)
        self.assertEqual(len(matches), 0)
        self.assertEqual(len(pts_ref), 0)

if __name__ == "__main__":
    unittest.main()
