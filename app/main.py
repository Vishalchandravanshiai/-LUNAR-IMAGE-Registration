"""
Main CLI entrypoint and application launcher for Lunar Image Registration MVP.
"""

import os
import sys
import argparse
import subprocess
import cv2


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.registration import ImageRegistrar
from core.feature_detection import SIFTFeatureDetector, ORBFeatureDetector, AKAZEFeatureDetector, GridFeatureDetector
from core.feature_matching import FLANNMatcher, BFMatcher
from core.geometric_verification import RANSACVerifier
from core.preprocessing import ImagePreprocessor
from core.bridge_registration import register_via_bridge
from core.export import write_match_points_csv
import json

def launch_ui():
    """Launches the Gradio web application."""
    ui_script = os.path.join(PROJECT_ROOT, "app.py")
    print(f"Launching Gradio Web UI: {ui_script}...")
    subprocess.run([sys.executable, ui_script], check=False)

def run_cli_registration(args):
    """Executes headless image registration from CLI arguments."""
    ref_path = args.ref
    src_path = args.src
    out_dir = args.out or os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    print(f"--- Chandrayaan-2 Lunar Image Registration MVP ---")
    print(f"Reference Image: {ref_path}")
    print(f"Source Image:    {src_path}")
    print(f"Detector:        {args.detector}")
    print(f"Matcher:         {args.matcher}")
    print(f"Model:           {args.model}")

    # Build components
    preproc = ImagePreprocessor(use_clahe=not args.no_clahe)

    if args.detector.upper() == "SIFT":
        det = SIFTFeatureDetector()
    elif args.detector.upper() == "ORB":
        det = ORBFeatureDetector()
    else:
        det = AKAZEFeatureDetector()
    if args.uniform_detection:
        det = GridFeatureDetector(det)

    if args.matcher.upper() == "FLANN":
        mat = FLANNMatcher(ratio_threshold=args.ratio)
    else:
        mat = BFMatcher(ratio_threshold=args.ratio)

    ver = RANSACVerifier(transform_type=args.model.lower(), ransac_reproj_threshold=args.ransac_thresh)

    registrar = ImageRegistrar(detector=det, matcher=mat, verifier=ver, preprocessor=preproc)

    # Run registration
    if args.bridge_images and args.sun_angles:
        with open(args.sun_angles, encoding="utf-8") as handle:
            sun_map = json.load(handle)
        candidates = [(path, sun_map[path]) for path in args.bridge_images]
        res = register_via_bridge(ref_path, src_path, sun_map[ref_path], sun_map[src_path], candidates, registrar)
        print(f"Registration mode: {res.get('mode', 'direct')}")
        if res.get('chain'):
            print("Sun-angle chain: " + " -> ".join(str(item[0] or 'reference') for item in res['chain']))
            for i, link in enumerate(res.get('links', []), 1):
                print(f"  Link {i}: {link['inliers']} inliers, {link['confidence']}")
    else:
        res = registrar.register(ref_path, src_path)
    metrics = res['metrics']
    if args.export_matches:
        write_match_points_csv(res.get('pts_ref', []), res.get('pts_src', []), res.get('inlier_mask'), args.export_matches)
        print(f"Match point CSV:    {args.export_matches}")

    print("\n--- REGISTRATION RESULTS ---")
    print(f"Status:             {metrics.status_message}")
    print(f"Success:            {metrics.success}")
    print(f"Total Keypoints (Ref): {metrics.total_keypoints_ref}")
    print(f"Total Keypoints (Src): {metrics.total_keypoints_src}")
    print(f"Total Matches:      {metrics.total_matches}")
    print(f"Inlier Matches:     {metrics.inlier_matches}")
    print(f"Inlier Ratio:       {metrics.inlier_ratio * 100:.2f}%")
    print(f"Reprojection RMSE:  {metrics.rmse_pixels:.4f} pixels")
    holdout = "n/a" if metrics.holdout_rmse_pixels is None else f"{metrics.holdout_rmse_pixels:.4f} pixels"
    print(f"Held-out RMSE:      {holdout}")
    print(f"Match Coverage:     {metrics.match_coverage * 100:.1f}%")
    print(f"Confidence:         {metrics.confidence or 'n/a'}")
    print(f"Processing Time:    {metrics.execution_time_ms:.2f} ms")

    if metrics.success and res['registered_source'] is not None:
        reg_out_path = os.path.join(out_dir, "registered_source.png")
        all_vis_path = os.path.join(out_dir, "matches_all.png")
        inlier_vis_path = os.path.join(out_dir, "matches_inliers.png")
        blend_path = os.path.join(out_dir, "blended_overlay.png")
        diff_path = os.path.join(out_dir, "difference_map.png")

        cv2.imwrite(reg_out_path, res['registered_source'])
        cv2.imwrite(all_vis_path, res['all_matches_vis'])
        cv2.imwrite(inlier_vis_path, res['inlier_matches_vis'])
        cv2.imwrite(blend_path, res['blended_overlay'])
        cv2.imwrite(diff_path, res['difference_map'])

        print(f"\nOutputs successfully saved to directory: {out_dir}")
        print(f"  - Registered Image:   {reg_out_path}")
        print(f"  - All Matches:        {all_vis_path}")
        print(f"  - Inlier Matches:     {inlier_vis_path}")
        print(f"  - Blended Overlay:    {blend_path}")
        print(f"  - Difference Heatmap: {diff_path}")
    else:
        print("\nRegistration failed to produce a valid warped image.")

def main():
    parser = argparse.ArgumentParser(description="Lunar Image Registration MVP (SIH 2026 Baseline)")
    parser.add_argument("--ui", action="store_true", help="Launch Streamlit Web UI")
    parser.add_argument("--ref", type=str, help="Path to Reference/Fixed image")
    parser.add_argument("--src", type=str, help="Path to Source/Moving image")
    parser.add_argument("--out", type=str, help="Path to output directory")
    parser.add_argument("--detector", type=str, default="SIFT", choices=["SIFT", "ORB", "AKAZE"], help="Feature detector")
    parser.add_argument("--matcher", type=str, default="FLANN", choices=["FLANN", "BFMatcher"], help="Feature matcher")
    parser.add_argument("--model", type=str, default="homography", choices=["homography", "affine"], help="Geometric model")
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio threshold")
    parser.add_argument("--ransac-thresh", type=float, default=3.0, help="RANSAC reprojection threshold in pixels")
    parser.add_argument("--no-clahe", action="store_true", help="Disable CLAHE preprocessing")
    parser.add_argument("--bridge-images", nargs="*", default=[], metavar="IMG", help="Intermediate images for sun-angle bridging")
    parser.add_argument("--sun-angles", type=str, help="JSON mapping image paths to [azimuth_deg, elevation_deg]")
    parser.add_argument("--export-matches", type=str, metavar="PATH.csv", help="Export matched points and inlier labels to CSV")
    parser.add_argument("--uniform-detection", action="store_true", help="Detect features separately in a 4x4 image grid")

    args = parser.parse_args()

    if args.ui or (args.ref is None and args.src is None):
        launch_ui()
    else:
        run_cli_registration(args)

if __name__ == "__main__":
    main()
