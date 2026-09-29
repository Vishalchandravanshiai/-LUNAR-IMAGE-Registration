"""Measure synthetic SIFT registration under controlled rotation and scale changes."""
import os
import sys
import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from core.feature_detection import SIFTFeatureDetector
from core.feature_matching import FLANNMatcher
from core.geometric_verification import RANSACVerifier
from core.preprocessing import ImagePreprocessor
from core.registration import ImageRegistrar
from tests.synthetic_terrain import generate_crater_terrain, hillshade


def main():
    size = 1024
    ref = hillshade(generate_crater_terrain(size=size), 315, 45)
    registrar = ImageRegistrar(SIFTFeatureDetector(), FLANNMatcher(.75),
                               RANSACVerifier("homography", 3.0),
                               ImagePreprocessor(use_clahe=True, use_blur=False))
    cases = [(angle, 1.0) for angle in (0, 15, 30, 45, 60, 90)]
    cases += [(0, scale) for scale in (1.0, .75, .5, .35, .25)]
    cases += [(15, .75), (30, .5), (45, .35), (60, .25)]
    yy, xx = np.mgrid[100:size-100:100, 100:size-100:100]
    points = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
    rows = []
    center = (size - 1) / 2
    for rotation, scale in cases:
        affine = cv2.getRotationMatrix2D((center, center), rotation, scale)
        affine[:, 2] += [0, 0]
        truth = np.vstack((affine, [0, 0, 1])).astype(float)
        source = cv2.warpPerspective(ref, np.linalg.inv(truth), (size, size), flags=cv2.INTER_LINEAR)
        result = registrar.register(ref, source)
        metrics, estimated = result["metrics"], result["H"]
        error = "n/a"
        if estimated is not None:
            projected_true = cv2.perspectiveTransform(points.reshape(-1, 1, 2), truth).reshape(-1, 2)
            projected_est = cv2.perspectiveTransform(points.reshape(-1, 1, 2), np.asarray(estimated)).reshape(-1, 2)
            valid = ((projected_true[:, 0] >= 0) & (projected_true[:, 0] < size) &
                     (projected_true[:, 1] >= 0) & (projected_true[:, 1] < size))
            error = float(np.median(np.linalg.norm(projected_est[valid] - projected_true[valid], axis=1))) if valid.any() else float("nan")
        rows.append((rotation, scale, metrics.inlier_matches, metrics.confidence or "REJECTED", error))
        print(f"rotation={rotation:>2} scale={scale:.2f}: {metrics.inlier_matches} inliers, {metrics.confidence or 'REJECTED'}, true median error={error}")
    output = ["# Synthetic viewpoint and scale sweep", "",
              "These measurements use a deterministic synthetic crater height field and fixed illumination (azimuth 315°, elevation 45°). They are not real Chandrayaan-2 validation.", "",
              "True error is the median displacement difference from the known source-to-reference transform over a reference point grid. Rejected estimates have no accepted transform and are reported as n/a.", "",
              "| Rotation (°) | Scale | Inliers | Confidence | Median true error (px) |", "|---:|---:|---:|:---|---:|"]
    for rotation, scale, count, confidence, error in rows:
        output.append(f"| {rotation} | {scale:.2f} | {count} | {confidence} | {error if isinstance(error, str) else f'{error:.3f}'} |")
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    path = os.path.join(ROOT, "docs", "viewpoint_scale_results.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(output) + "\n")
    print(f"Synthetic results saved to {path}")


if __name__ == "__main__":
    main()
