"""Evaluate basename-paired reference/source images with the quality-gated pipeline."""
import argparse
import csv
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from core.feature_detection import SIFTFeatureDetector
from core.feature_matching import FLANNMatcher
from core.geometric_verification import RANSACVerifier
from core.preprocessing import ImagePreprocessor
from core.registration import ImageRegistrar

EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.join(ROOT, "data", "real"))
    parser.add_argument("--output", help="CSV destination (default: <data-dir>/results.csv)")
    args = parser.parse_args()
    ref_dir, src_dir = os.path.join(args.data_dir, "reference"), os.path.join(args.data_dir, "source")
    refs = {os.path.splitext(f)[0]: os.path.join(ref_dir, f) for f in os.listdir(ref_dir) if os.path.splitext(f)[1].lower() in EXTENSIONS} if os.path.isdir(ref_dir) else {}
    srcs = {os.path.splitext(f)[0]: os.path.join(src_dir, f) for f in os.listdir(src_dir) if os.path.splitext(f)[1].lower() in EXTENSIONS} if os.path.isdir(src_dir) else {}
    pairs = sorted(set(refs) & set(srcs))
    if not pairs:
        print("no real data found, see data/README.md")
        return 0
    metadata_path = os.path.join(args.data_dir, "sun_angles.json")
    metadata = {}
    if os.path.isfile(metadata_path):
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
    registrar = ImageRegistrar(SIFTFeatureDetector(), FLANNMatcher(.75), RANSACVerifier("homography", 3.0),
                               ImagePreprocessor(use_clahe=True, use_blur=False))
    columns = ["pair", "success", "status", "reference_keypoints", "source_keypoints", "matches", "inliers", "inlier_ratio", "reprojection_rmse_px", "heldout_rmse_px", "match_coverage", "confidence", "runtime_ms", "reference_sun_az_el", "source_sun_az_el"]
    rows = []
    for key in pairs:
        result = registrar.register(refs[key], srcs[key])
        m = result["metrics"]
        angle = metadata.get(key, {})
        rows.append({"pair": key, "success": m.success, "status": m.status_message,
                     "reference_keypoints": m.total_keypoints_ref, "source_keypoints": m.total_keypoints_src,
                     "matches": m.total_matches, "inliers": m.inlier_matches, "inlier_ratio": m.inlier_ratio,
                     "reprojection_rmse_px": m.rmse_pixels, "heldout_rmse_px": m.holdout_rmse_pixels,
                     "match_coverage": m.match_coverage, "confidence": m.confidence,
                     "runtime_ms": m.execution_time_ms, "reference_sun_az_el": angle.get("reference", ""),
                     "source_sun_az_el": angle.get("source", "")})
    out = args.output or os.path.join(args.data_dir, "results.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Evaluated {len(rows)} pair(s); CSV saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
