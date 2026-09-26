"""Small in-memory image fixtures used only by the automated tests."""

import cv2
import numpy as np


def textured_test_image(width=400, height=400, seed=42):
    """Build repeatable feature-rich pixels for detector and registration tests."""
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, (height, width), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (5, 5), 0.8)
    for _ in range(45):
        x = int(rng.integers(20, width - 20))
        y = int(rng.integers(20, height - 20))
        radius = int(rng.integers(5, 22))
        color = int(rng.integers(0, 256))
        cv2.circle(image, (x, y), radius, color, 2)
        cv2.circle(image, (x, y), max(2, radius // 2), 255 - color, 1)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
