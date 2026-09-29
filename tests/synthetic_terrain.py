"""Deterministic cratered height fields and Lambertian hillshade rendering."""
import cv2
import numpy as np


def generate_crater_terrain(size=1024, n_craters=220, seed=42):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:size, :size].astype(np.float32)
    height = cv2.GaussianBlur(rng.normal(0, 0.025, (size, size)).astype(np.float32), (0, 0), size / 180)
    for _ in range(n_craters):
        cx, cy = rng.uniform(0, size, 2)
        radius = rng.uniform(size * .012, size * .065)
        r2 = ((x - cx) ** 2 + (y - cy) ** 2) / (radius * radius)
        bowl = -0.22 * np.exp(-r2 * 3.2)
        rim = 0.16 * np.exp(-((np.sqrt(r2) - 1.0) / .11) ** 2)
        height += (bowl + rim).astype(np.float32)
    height += 0.025 * np.sin(x / 29) * np.cos(y / 37)
    return height


def hillshade(height_field, azimuth, elevation):
    z = np.asarray(height_field, dtype=np.float32)
    # Exaggerate terrain height into visible photometric relief for compact synthetic tests.
    dy, dx = np.gradient(z * 5.0)
    # x points east, y points south; azimuth is clockwise from north.
    az, el = np.radians([azimuth, elevation])
    lx, ly, lz = np.sin(az) * np.cos(el), -np.cos(az) * np.cos(el), np.sin(el)
    nx, ny, nz = -dx, -dy, np.ones_like(z)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    shade = np.clip((nx * lx + ny * ly + nz * lz) / norm, 0, 1)
    shade = 0.2 + 0.8 * shade
    return np.uint8(np.clip(shade * 255, 0, 255))
