import csv
import os
import tempfile
import unittest
import cv2
import numpy as np

from core.export import write_match_points_csv
from core.feature_detection import SIFTFeatureDetector, GridFeatureDetector
from core.quality_gate import spatial_coverage


class ExportAndGridDetectionTests(unittest.TestCase):
    def test_csv_has_expected_match_rows(self):
        refs = np.array([[1, 2], [3, 4]], dtype=float)
        srcs = np.array([[5, 6], [7, 8]], dtype=float)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "points.csv")
            write_match_points_csv(refs, srcs, np.array([[1], [0]], dtype=np.uint8), path)
            with open(path, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(list(rows[0]), ["ref_x", "ref_y", "src_x", "src_y", "is_inlier"])
        self.assertEqual(rows, [
            {"ref_x": "1.0", "ref_y": "2.0", "src_x": "5.0", "src_y": "6.0", "is_inlier": "True"},
            {"ref_x": "3.0", "ref_y": "4.0", "src_x": "7.0", "src_y": "8.0", "is_inlier": "False"},
        ])

    def test_grid_detection_spreads_features_beyond_dense_patch(self):
        rng = np.random.default_rng(9)
        image = np.zeros((512, 512), dtype=np.uint8)
        # A highly textured center dominates a globally capped detector.
        image[160:352, 160:352] = cv2.GaussianBlur(rng.integers(0, 256, (192, 192), dtype=np.uint8), (3, 3), .4)
        # Small repeatable corner patterns provide sparse features in the other grid cells.
        for row in range(4):
            for col in range(4):
                if 1 <= row <= 2 and 1 <= col <= 2:
                    continue
                x, y = col * 128 + 45, row * 128 + 45
                cv2.rectangle(image, (x, y), (x + 22, y + 22), 255, -1)
                cv2.line(image, (x, y + 11), (x + 22, y + 11), 0, 2)
                cv2.line(image, (x + 11, y), (x + 11, y + 22), 0, 2)
        standard = SIFTFeatureDetector(nfeatures=40)
        grid = GridFeatureDetector(SIFTFeatureDetector(nfeatures=80), features_per_tile=20)
        kp_standard, _ = standard.detect_and_compute(image)
        kp_grid, _ = grid.detect_and_compute(image)
        standard_cov = spatial_coverage(np.array([k.pt for k in kp_standard]), image.shape)
        grid_cov = spatial_coverage(np.array([k.pt for k in kp_grid]), image.shape)
        self.assertLess(standard_cov, 0.75) if standard_cov < 0.75 else self.assertLess(standard_cov, grid_cov)
        self.assertGreaterEqual(grid_cov, 0.75)


if __name__ == "__main__":
    unittest.main()
