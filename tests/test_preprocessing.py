"""
Unit tests for core/preprocessing.py.
"""

import unittest
import numpy as np
import cv2
import tempfile
import os

from core.preprocessing import ImagePreprocessor, PreprocessingError

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.preprocessor = ImagePreprocessor(use_clahe=True)
        # Create a synthetic test image
        self.test_img_bgr = np.uint8(np.random.randint(0, 255, (100, 100, 3)))

        # Save temp image
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(self.temp_file.name, self.test_img_bgr)
        self.temp_path = self.temp_file.name
        self.temp_file.close()

    def tearDown(self):
        if os.path.exists(self.temp_path):
            os.remove(self.temp_path)

    def test_load_image_from_path(self):
        img = self.preprocessor.load_image(self.temp_path)
        self.assertIsNotNone(img)
        self.assertEqual(img.shape[:2], (100, 100))

    def test_load_image_invalid_path(self):
        with self.assertRaises(PreprocessingError):
            self.preprocessor.load_image("non_existent_file_path_12345.png")

    def test_load_image_none(self):
        with self.assertRaises(PreprocessingError):
            self.preprocessor.load_image(None)

    def test_to_grayscale_3channel(self):
        gray = self.preprocessor.to_grayscale(self.test_img_bgr)
        self.assertEqual(len(gray.shape), 2)
        self.assertEqual(gray.shape, (100, 100))
        self.assertEqual(gray.dtype, np.uint8)

    def test_apply_clahe(self):
        gray = self.preprocessor.to_grayscale(self.test_img_bgr)
        clahe_img = self.preprocessor.apply_clahe(gray)
        self.assertEqual(clahe_img.shape, gray.shape)
        self.assertEqual(clahe_img.dtype, np.uint8)

    def test_preprocess_full_pipeline(self):
        raw, proc = self.preprocessor.preprocess(self.temp_path)
        self.assertEqual(raw.shape[:2], (100, 100))
        self.assertEqual(proc.shape, (100, 100))
        self.assertEqual(proc.dtype, np.uint8)

if __name__ == "__main__":
    unittest.main()
