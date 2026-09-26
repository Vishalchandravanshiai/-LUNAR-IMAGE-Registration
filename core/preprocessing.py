"""
Preprocessing module for lunar optical images.
Handles image loading, color conversion, CLAHE illumination enhancement,
noise filtering, and validation.
"""

from typing import Union, Tuple, Optional
import os
import cv2
import numpy as np
from PIL import Image, ImageOps

class PreprocessingError(Exception):
    """Raised when image preprocessing or validation fails."""
    pass

class ImagePreprocessor:
    """
    Handles robust loading and preprocessing of lunar images for feature extraction.

    Supports:
    - Loading from filepath, bytes, PIL Image, or NumPy array
    - Grayscale conversion (handling 1, 3, or 4 channels)
    - CLAHE (Contrast Limited Adaptive Histogram Equalization) for sun-angle contrast
    - Gaussian blurring for sensor noise reduction
    """

    def __init__(self, use_clahe: bool = True, clahe_clip_limit: float = 2.0,
                 clahe_tile_grid_size: Tuple[int, int] = (8, 8),
                 use_blur: bool = False, blur_kernel_size: Tuple[int, int] = (3, 3)):
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.use_blur = use_blur
        self.blur_kernel_size = blur_kernel_size

    def load_image(self, source: Union[str, bytes, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Loads an image from various source formats into a standard BGR uint8 NumPy array.

        Args:
            source: Filepath string, bytes buffer, PIL Image, or NumPy array.

        Returns:
            np.ndarray: Loaded BGR or Grayscale uint8 NumPy array.

        Raises:
            PreprocessingError: If source is invalid or corrupt.
        """
        if source is None:
            raise PreprocessingError("Image source cannot be None.")

        try:
            if isinstance(source, str):
                if not os.path.exists(source):
                    raise PreprocessingError(f"File not found at path: '{source}'")
                img = cv2.imread(source, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise PreprocessingError(f"Failed to read image from path: '{source}'. File may be corrupted or unreadable format.")
            elif isinstance(source, bytes):
                nparr = np.frombuffer(source, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise PreprocessingError("Failed to decode image from bytes. File corrupted or unsupported format.")
            elif isinstance(source, Image.Image):
                source = source.convert('RGB')
                img = cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR)
            elif isinstance(source, np.ndarray):
                if source.size == 0:
                    raise PreprocessingError("Input NumPy array is empty.")
                img = source.copy()
            else:
                raise PreprocessingError(f"Unsupported image source type: {type(source)}")

            # Validate array non-emptiness and dimensions
            if img is None or img.size == 0:
                raise PreprocessingError("Image loaded with zero size.")

            if len(img.shape) < 2 or img.shape[0] == 0 or img.shape[1] == 0:
                raise PreprocessingError(f"Invalid image dimensions: {img.shape}")

            return img

        except Exception as e:
            if isinstance(e, PreprocessingError):
                raise e
            raise PreprocessingError(f"Unexpected error loading image: {str(e)}") from e

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Converts BGR / BGRA / RGB image to 8-bit single-channel grayscale.
        """
        if len(image.shape) == 2:
            gray = image.copy()
        elif len(image.shape) == 3:
            channels = image.shape[2]
            if channels == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif channels == 4:
                gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            elif channels == 1:
                gray = image[:, :, 0].copy()
            else:
                raise PreprocessingError(f"Unsupported channel count: {channels}")
        else:
            raise PreprocessingError(f"Unsupported image shape: {image.shape}")

        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return gray

    def apply_clahe(self, gray_img: np.ndarray) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
        Crucial for enhancing lunar surface features under variable sun angles.
        """
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid_size
        )
        return clahe.apply(gray_img)

    def preprocess(self, source: Union[str, bytes, np.ndarray, Image.Image]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads and executes full preprocessing pipeline.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (original_bgr_or_gray, preprocessed_gray)
        """
        raw_img = self.load_image(source)
        gray = self.to_grayscale(raw_img)

        proc = gray.copy()
        if self.use_clahe:
            proc = self.apply_clahe(proc)

        if self.use_blur:
            k = self.blur_kernel_size
            proc = cv2.GaussianBlur(proc, k, 0)

        return raw_img, proc
