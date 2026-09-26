"""
Unit tests for core/metrics.py.
"""

import unittest
import numpy as np

from core.metrics import compute_reprojection_errors, compute_registration_metrics, RegistrationMetrics

class TestMetrics(unittest.TestCase):

    def test_compute_reprojection_errors_exact(self):
        # Shift translation matrix: tx=10, ty=5
        H = np.array([
            [1.0, 0.0, 10.0],
            [0.0, 1.0, 5.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        pts_src = np.array([[0, 0], [100, 100]], dtype=np.float64)
        # Perfect reference points: src + (10, 5)
        pts_ref_perfect = np.array([[10, 5], [110, 105]], dtype=np.float64)

        errors = compute_reprojection_errors(H, pts_src, pts_ref_perfect)
        self.assertEqual(len(errors), 2)
        np.testing.assert_allclose(errors, [0.0, 0.0], atol=1e-5)

    def test_compute_reprojection_errors_with_offset(self):
        H = np.eye(3, dtype=np.float64)
        pts_src = np.array([[0, 0], [0, 0]], dtype=np.float64)
        # Ref points offset by 3 and 4 -> distance = 5.0
        pts_ref = np.array([[3, 4], [-3, -4]], dtype=np.float64)

        errors = compute_reprojection_errors(H, pts_src, pts_ref)
        np.testing.assert_allclose(errors, [5.0, 5.0], atol=1e-5)

    def test_compute_registration_metrics(self):
        H = np.eye(3, dtype=np.float64)
        pts_src = np.array([[0, 0], [10, 10], [20, 20]], dtype=np.float64)
        pts_ref = np.array([[0, 0], [10, 10], [20, 20]], dtype=np.float64)
        mask = np.array([[1], [1], [0]], dtype=np.uint8)

        metrics = compute_registration_metrics(
            pts_ref=pts_ref,
            pts_src=pts_src,
            inlier_mask=mask,
            H=H,
            total_keypoints_ref=50,
            total_keypoints_src=50,
            total_matches=4,
            execution_time_ms=12.5,
            success=True,
            status_message="Success"
        )

        self.assertTrue(metrics.success)
        self.assertEqual(metrics.inlier_matches, 2)
        self.assertEqual(metrics.total_matches, 4)
        self.assertAlmostEqual(metrics.inlier_ratio, 0.5)
        self.assertAlmostEqual(metrics.rmse_pixels, 0.0, places=4)

if __name__ == "__main__":
    unittest.main()
