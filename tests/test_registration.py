"""
End-to-end integration tests for core/registration.py.
"""

import unittest
import os
import sys
import numpy as np

from core.registration import ImageRegistrar
import cv2
from image_fixtures import textured_test_image

class TestImageRegistrarIntegration(unittest.TestCase):

    def test_end_to_end_registration_sift(self):
        reference = textured_test_image(width=800, height=800, seed=101)
        matrix = cv2.getRotationMatrix2D((400, 400), 12.5, 1.08)
        matrix[0, 2] += 25
        matrix[1, 2] -= 18
        source = cv2.warpAffine(reference, matrix, (800, 800), borderMode=cv2.BORDER_REFLECT)
        registrar = ImageRegistrar()
        res = registrar.register(reference, source)

        metrics = res['metrics']
        self.assertTrue(metrics.success, f"End-to-end registration failed: {metrics.status_message}")
        self.assertGreater(metrics.total_matches, 10)
        self.assertGreater(metrics.inlier_matches, 5)
        self.assertGreater(metrics.inlier_ratio, 0.3)
        self.assertLess(metrics.rmse_pixels, 3.0, "Reprojection RMSE should be under 3.0 pixels for synthetic pair.")

        self.assertIsNotNone(res['registered_source'])
        self.assertIsNotNone(res['blended_overlay'])
        self.assertIsNotNone(res['difference_map'])
        self.assertIsNotNone(res['anaglyph_overlay'])
        self.assertIsNotNone(res['H'])

if __name__ == "__main__":
    unittest.main()
