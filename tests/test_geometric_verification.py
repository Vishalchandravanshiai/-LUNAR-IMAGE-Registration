"""
Unit tests for core/geometric_verification.py.
"""

import unittest
import numpy as np
import cv2

from core.geometric_verification import RANSACVerifier

class TestGeometricVerification(unittest.TestCase):

    def setUp(self):
        self.verifier_homography = RANSACVerifier(transform_type='homography', ransac_reproj_threshold=3.0)
        self.verifier_affine = RANSACVerifier(transform_type='affine', ransac_reproj_threshold=3.0)

    def test_verify_known_homography(self):
        # Generate 20 synthetic points
        np.random.seed(42)
        pts_src = np.random.uniform(50, 350, (20, 2)).astype(np.float32)

        # Ground truth homography: rotation + translation + small perspective
        H_true = np.array([
            [0.98, -0.17, 25.0],
            [0.17,  0.98, -15.0],
            [0.0001, -0.0002, 1.0]
        ], dtype=np.float32)

        # Transform src points to ref points
        ones = np.ones((20, 1), dtype=np.float32)
        pts_src_homo = np.hstack([pts_src, ones])
        transformed = (H_true @ pts_src_homo.T).T
        pts_ref = transformed[:, :2] / transformed[:, 2:3]

        # Add small Gaussian noise to ref points
        noise = np.random.normal(0, 0.2, pts_ref.shape).astype(np.float32)
        pts_ref_noisy = pts_ref + noise

        H_est, mask, is_valid, msg = self.verifier_homography.verify(pts_ref_noisy, pts_src)

        self.assertTrue(is_valid, f"RANSAC homography estimation failed: {msg}")
        self.assertIsNotNone(H_est)
        self.assertEqual(H_est.shape, (3, 3))
        self.assertGreaterEqual(np.sum(mask), 15)

    def test_verify_insufficient_points(self):
        pts_src = np.array([[10, 10], [20, 20]], dtype=np.float32)
        pts_ref = np.array([[15, 12], [25, 22]], dtype=np.float32)

        H, mask, is_valid, msg = self.verifier_homography.verify(pts_ref, pts_src)
        self.assertFalse(is_valid)
        self.assertIn("Insufficient matches", msg)

    def test_verify_degenerate_homography(self):
        # All points collinear (on a single straight line)
        pts_src = np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]], dtype=np.float32)
        pts_ref = pts_src.copy()

        H, mask, is_valid, msg = self.verifier_homography.verify(pts_ref, pts_src)
        self.assertFalse(is_valid)

if __name__ == "__main__":
    unittest.main()
