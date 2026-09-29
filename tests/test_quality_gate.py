import unittest
import numpy as np

from core.quality_gate import MIN_INLIERS, assess_registration


class QualityGateTests(unittest.TestCase):
    def test_six_points_are_rejected(self):
        src = np.arange(12, dtype=np.float32).reshape(6, 2)
        ref = src + 3
        result = assess_registration(ref, src, np.ones(6), (100, 100), "affine")
        self.assertEqual(result["confidence"], "REJECTED")

    def test_well_spread_points_pass_with_low_holdout_error(self):
        rng = np.random.default_rng(7)
        src = rng.uniform(10, 990, (200, 2))
        ref = src @ np.array([[1.001, 0.002], [-0.002, 0.999]]).T + [2, -3]
        ref += rng.normal(0, 0.08, ref.shape)
        result = assess_registration(ref, src, np.ones(200), (1024, 1024), "affine")
        self.assertIn(result["confidence"], ("HIGH", "MEDIUM"))
        self.assertLess(result["holdout_rmse"], 2)
        self.assertGreater(result["coverage"], 0.5)

    def test_clustered_points_have_low_coverage(self):
        rng = np.random.default_rng(3)
        ref = rng.uniform(10, 100, (60, 2))
        src = ref + [3, 2]
        result = assess_registration(ref, src, np.ones(60), (1024, 1024), "affine")
        self.assertLess(result["coverage"], 0.25)

    def test_minimum_is_at_least_ten(self):
        self.assertGreaterEqual(MIN_INLIERS, 10)


if __name__ == "__main__":
    unittest.main()
