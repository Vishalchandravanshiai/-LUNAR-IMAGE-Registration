import unittest
import cv2
import numpy as np

from core.bridge_registration import plan_chain, sun_separation_deg, register_via_bridge
from core.metrics import RegistrationMetrics
from core.feature_detection import SIFTFeatureDetector
from core.feature_matching import FLANNMatcher
from core.geometric_verification import RANSACVerifier
from core.preprocessing import ImagePreprocessor
from core.registration import ImageRegistrar
from tests.synthetic_terrain import generate_crater_terrain, hillshade


class BridgePlanningTests(unittest.TestCase):
    def test_sun_vector_separation_and_ordered_chain(self):
        self.assertAlmostEqual(sun_separation_deg((0, 45), (30, 45)), 21.1, delta=0.3)
        candidates = [(f"i{i}", (az, 30)) for i, az in enumerate((285, 255, 225, 195, 165, 135))]
        chain = plan_chain((315, 45), (105, 22), candidates, 30)
        self.assertIsNotNone(chain)
        self.assertEqual([p for p, _ in chain[1:-1]], [f"i{i}" for i in range(6)])
        for a, b in zip(chain, chain[1:]):
            self.assertLessEqual(sun_separation_deg(a[1], b[1]), 30)

    def test_composes_source_to_reference_transforms(self):
        class FakeRegistrar:
            def __init__(self):
                self.transforms = {
                    ("ref", "mid"): np.array([[1, 0, 10], [0, 1, 0], [0, 0, 1.]]),
                    ("mid", "src"): np.array([[1, 0, 0], [0, 1, 20], [0, 0, 1.]]),
                }
            def register(self, ref, src):
                H = self.transforms.get((ref, src))
                if ref == "ref" and src == "src":
                    m = RegistrationMetrics(success=False, confidence="REJECTED")
                    return {"metrics": m, "H": None}
                return {"metrics": RegistrationMetrics(success=True, confidence="HIGH", inlier_matches=60), "H": H,
                        "registered_source": np.zeros((10, 10), np.uint8)}
        result = register_via_bridge("ref", "src", (0, 30), (60, 30), [("mid", (30, 30))], FakeRegistrar(), 35)
        self.assertEqual(result["mode"], "bridge", result["metrics"].status_message)
        np.testing.assert_allclose(result["H"], [[1, 0, 10], [0, 1, 20], [0, 0, 1]])
        self.assertEqual(result["metrics"].confidence, "HIGH")

    def test_synthetic_crater_chain_recovers_large_sun_gap_and_affine(self):
        cv2.setRNGSeed(17)
        height = generate_crater_terrain(size=1024, n_craters=220, seed=42)
        ref_sun = (315, 45)
        ref = hillshade(height, *ref_sun)
        intermediate_suns = ((285, 41), (255, 37), (225, 33), (195, 30), (165, 27), (135, 24))
        candidates = [(hillshade(height, *sun), sun) for sun in intermediate_suns]
        src_sun = (105, 22)
        base = hillshade(height, *src_sun)
        affine = cv2.getRotationMatrix2D((511.5, 511.5), 12, .75)
        affine[:, 2] += [40, -25]
        truth = np.vstack((affine, [0, 0, 1])).astype(float)
        source = cv2.warpPerspective(base, np.linalg.inv(truth), (1024, 1024))
        registrar = ImageRegistrar(SIFTFeatureDetector(), FLANNMatcher(.85), RANSACVerifier("affine", 5.0, 5000),
                                   ImagePreprocessor(use_clahe=True, use_blur=False))
        direct = registrar.register(ref, source)
        self.assertNotIn(direct["metrics"].confidence, ("HIGH", "MEDIUM"))
        cv2.setRNGSeed(17)
        result = register_via_bridge(ref, source, ref_sun, src_sun, candidates, registrar, 30)
        self.assertEqual(result["mode"], "bridge", result["metrics"].status_message)
        self.assertTrue(result["metrics"].success, result["metrics"].status_message)
        grid = np.array([[x, y] for x in range(80, 945, 80) for y in range(80, 945, 80)], dtype=np.float32)
        true_pts = cv2.perspectiveTransform(grid.reshape(-1, 1, 2), truth).reshape(-1, 2)
        est_pts = cv2.perspectiveTransform(grid.reshape(-1, 1, 2), result["H"]).reshape(-1, 2)
        valid = ((true_pts[:, 0] >= 0) & (true_pts[:, 0] < 1024) & (true_pts[:, 1] >= 0) & (true_pts[:, 1] < 1024))
        median_error = np.median(np.linalg.norm(true_pts[valid] - est_pts[valid], axis=1))
        self.assertLess(median_error, 1.5)


if __name__ == "__main__":
    unittest.main()
