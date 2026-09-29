"""Gradio interface for the lunar image registration prototype."""

from functools import lru_cache
import os
import tempfile
import urllib.request

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from core.feature_detection import (
    AKAZEFeatureDetector,
    ORBFeatureDetector,
    SIFTFeatureDetector,
)
from core.feature_matching import BFMatcher, FLANNMatcher
from core.geometric_verification import RANSACVerifier
from core.preprocessing import ImagePreprocessor
from core.registration import ImageRegistrar
from core.export import write_match_points_csv
from core.bridge_registration import register_via_bridge

NASA_STEREO_PAIR_URL = (
    "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/"
    "LROLRC_2001/EXTRAS/ANAGLYPH/NAC_M1181613435_M1181606332/"
    "NAC_ANAGLYPH_M1181613435_M1181606332.TIF"
)


@lru_cache(maxsize=1)
def fetch_nasa_stereo_pair():
    """Fetch the public LROC anaglyph and return its two view channels."""
    request = urllib.request.Request(
        NASA_STEREO_PAIR_URL,
        headers={"User-Agent": "LunarImageRegistrationPrototype/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        encoded = np.frombuffer(response.read(), dtype=np.uint8)

    anaglyph = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if anaglyph is None:
        raise ValueError("NASA's stereo image could not be decoded.")

    reference = anaglyph[:, :, 1]
    source = anaglyph[:, :, 2]
    max_dimension = max(reference.shape)
    if max_dimension > 2200:
        scale = 2200 / max_dimension
        size = (round(reference.shape[1] * scale), round(reference.shape[0] * scale))
        reference = cv2.resize(reference, size, interpolation=cv2.INTER_AREA)
        source = cv2.resize(source, size, interpolation=cv2.INTER_AREA)

    return Image.fromarray(reference), Image.fromarray(source)


def load_test_pair():
    try:
        reference, source = fetch_nasa_stereo_pair()
        return reference.copy(), source.copy(), "NASA LROC multi-angle test pair loaded."
    except (OSError, TimeoutError, ValueError) as exc:
        return None, None, f"Could not fetch the NASA test pair: {exc}"


def _as_rgb(image):
    if image is None:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def run_registration(reference, source, detector_name, matcher_name, model_name,
                     ratio_threshold, ransac_threshold, use_clahe, use_blur,
                     bridge_files=None, ref_azimuth=315, ref_elevation=45,
                     src_azimuth=105, src_elevation=22, *intermediate_angles):
    if reference is None or source is None:
        return (
            "⚠️ Upload both images or load the NASA test pair first.",
            "", None, None, None, None, None, None, None, None,
        )

    try:
        detectors = {
            "SIFT": SIFTFeatureDetector,
            "ORB": ORBFeatureDetector,
            "AKAZE": AKAZEFeatureDetector,
        }
        detector = detectors[detector_name]()
        matcher = (
            FLANNMatcher(ratio_threshold=ratio_threshold)
            if matcher_name == "FLANN"
            else BFMatcher(ratio_threshold=ratio_threshold)
        )
        registrar = ImageRegistrar(
            detector=detector,
            matcher=matcher,
            verifier=RANSACVerifier(
                transform_type=model_name.lower(),
                ransac_reproj_threshold=ransac_threshold,
            ),
            preprocessor=ImagePreprocessor(use_clahe=use_clahe, use_blur=use_blur),
        )
        if bridge_files:
            files = bridge_files if isinstance(bridge_files, list) else [bridge_files]
            angles = list(intermediate_angles)
            candidates = []
            for index, path in enumerate(files[:6]):
                az = angles[index * 2] if len(angles) > index * 2 and angles[index * 2] is not None else None
                el = angles[index * 2 + 1] if len(angles) > index * 2 + 1 and angles[index * 2 + 1] is not None else None
                if az is not None and el is not None:
                    path = path.get("path") if isinstance(path, dict) else path
                    candidates.append((path, (float(az), float(el))))
            result = register_via_bridge(reference, source, (ref_azimuth, ref_elevation),
                                         (src_azimuth, src_elevation), candidates, registrar)
        else:
            result = registrar.register(reference, source)
        metrics = result["metrics"]
        status = ("✅ " if metrics.success and metrics.confidence not in ("LOW", "REJECTED") else "⚠️ ") + metrics.status_message
        rmse = f"{metrics.rmse_pixels:.3f} px" if metrics.success else "N/A"
        holdout = "n/a" if metrics.holdout_rmse_pixels is None else f"{metrics.holdout_rmse_pixels:.3f} px"
        summary = (
            f"| Metric | Result |\n|:--|--:|\n"
            f"| Total matches | {metrics.total_matches} |\n"
            f"| Inlier matches | {metrics.inlier_matches} |\n"
            f"| Inlier ratio | {metrics.inlier_ratio * 100:.1f}% |\n"
            f"| Reprojection RMSE | {rmse} |\n"
            f"| Held-out RMSE | {holdout} |\n"
            f"| Match coverage | {metrics.match_coverage * 100:.1f}% |\n"
            f"| Confidence | {metrics.confidence or 'n/a'} |\n"
            f"| Transform | {metrics.transformation_type} |\n"
            f"| Runtime | {metrics.execution_time_ms:.1f} ms |"
        )
        if result.get("chain"):
            summary += "\n\n**Sun-angle chain:** " + " → ".join(
                "Reference" if i == 0 else ("Source" if i == len(result['chain']) - 1 else os.path.basename(str(item[0])))
                for i, item in enumerate(result['chain'])
            )
            summary += "\n\n" + "\n".join(
                f"- Link {i}: {link['inliers']} inliers · {link['confidence']}" for i, link in enumerate(result.get('links', []), 1)
            )

        all_matches = result.get("all_matches_vis")
        inlier_matches = result.get("inlier_matches_vis")
        registered = result.get("registered_source")
        reference_result = result.get("reference_image")
        if registered is not None:
            blend = registrar.create_blended_overlay(reference_result, registered, alpha=0.5)
            anaglyph = result.get("anaglyph_overlay")
            state = {"reference": reference_result, "registered": registered}
        else:
            blend = None
            anaglyph = None
            state = None

        csv_path = tempfile.NamedTemporaryFile(prefix="lunar_matches_", suffix=".csv", delete=False).name
        write_match_points_csv(result.get("pts_ref", []), result.get("pts_src", []), result.get("inlier_mask"), csv_path)

        return (
            status, summary,
            _as_rgb(all_matches), _as_rgb(inlier_matches), _as_rgb(registered),
            _as_rgb(result.get("difference_map")), _as_rgb(blend),
            _as_rgb(anaglyph), state, csv_path,
        )
    except Exception as exc:
        return (
            f"❌ Registration error: {exc}",
            "", None, None, None, None, None, None, None, None,
        )


def update_blend(state, alpha):
    if not state:
        return None
    image = cv2.addWeighted(
        state["reference"], alpha,
        state["registered"], 1.0 - alpha, 0,
    )
    return _as_rgb(image)


with gr.Blocks(title="Lunar Image Registration") as demo:
    gr.Markdown(
        "# 🌕 Lunar Image Registration\n"
        "### Classical computer vision prototype for aligning lunar images\n"
        "SIH problem statement: multi-modal, sun-angle and scale-invariant image correspondence."
    )
    gr.Markdown(
        "Upload a fixed reference and a moving source image, or fetch the public NASA LROC "
        "multi-angle test pair. This is an experimental baseline, not a validated Chandrayaan-2 solution."
    )

    with gr.Row():
        reference_input = gr.Image(label="Reference / fixed image", type="pil")
        source_input = gr.Image(label="Source / moving image", type="pil")
    with gr.Row():
        fetch_button = gr.Button("🌑 Fetch NASA Multi-Angle Test Pair", variant="secondary")
        pair_status = gr.Markdown()
    fetch_button.click(
        load_test_pair,
        outputs=[reference_input, source_input, pair_status],
    )

    with gr.Accordion("Registration settings", open=True):
        with gr.Row():
            detector_input = gr.Dropdown(
                ["SIFT", "ORB", "AKAZE"], value="SIFT", label="Feature detector"
            )
            matcher_input = gr.Dropdown(
                ["FLANN", "BFMatcher"], value="FLANN", label="Feature matcher"
            )
            model_input = gr.Dropdown(
                ["Homography", "Affine"], value="Homography", label="Geometric model"
            )
        with gr.Row():
            ratio_input = gr.Slider(0.50, 0.95, value=0.75, step=0.05, label="Lowe ratio threshold")
            ransac_input = gr.Slider(0.5, 10.0, value=3.0, step=0.5, label="RANSAC threshold (px)")
        with gr.Row():
            clahe_input = gr.Checkbox(value=True, label="Enable CLAHE contrast enhancement")
            blur_input = gr.Checkbox(value=False, label="Enable Gaussian blur")

    with gr.Accordion("Large sun-angle difference (advanced)", open=False):
        bridge_files_input = gr.File(label="Intermediate images (in chain order)", file_count="multiple", type="filepath")
        gr.Markdown("Enter sun azimuth/elevation in degrees. Intermediate angle rows correspond to uploaded files in order.")
        with gr.Row():
            ref_az_input = gr.Number(value=315, label="Reference azimuth")
            ref_el_input = gr.Number(value=45, label="Reference elevation")
            src_az_input = gr.Number(value=105, label="Source azimuth")
            src_el_input = gr.Number(value=22, label="Source elevation")
        intermediate_angle_inputs = []
        for index in range(6):
            with gr.Row():
                intermediate_angle_inputs.extend([
                    gr.Number(label=f"Intermediate {index + 1} azimuth"),
                    gr.Number(label=f"Intermediate {index + 1} elevation"),
                ])

    register_button = gr.Button("⚡ Register Images", variant="primary")
    status_output = gr.Markdown()
    metrics_output = gr.Markdown()
    match_csv_output = gr.File(label="Download match points (CSV)")
    blend_state = gr.State()

    with gr.Tab("Matches"):
        with gr.Row():
            matches_output = gr.Image(label="All ratio-test matches", type="numpy")
            inliers_output = gr.Image(label="Geometrically verified inliers", type="numpy")
    with gr.Tab("Alignment"):
        with gr.Row():
            registered_output = gr.Image(label="Registered source", type="numpy")
            difference_output = gr.Image(label="Difference heatmap", type="numpy")
    with gr.Tab("Overlay inspection"):
        alpha_input = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Reference / source blend")
        with gr.Row():
            blend_output = gr.Image(label="Interactive alpha blend", type="numpy")
            anaglyph_output = gr.Image(label="Red-cyan anaglyph", type="numpy")

    registration_inputs = [
        reference_input, source_input, detector_input, matcher_input, model_input,
        ratio_input, ransac_input, clahe_input, blur_input,
        bridge_files_input, ref_az_input, ref_el_input, src_az_input, src_el_input,
        *intermediate_angle_inputs,
    ]
    registration_outputs = [
        status_output, metrics_output, matches_output, inliers_output,
        registered_output, difference_output, blend_output, anaglyph_output,
        blend_state,
        match_csv_output,
    ]
    register_button.click(run_registration, inputs=registration_inputs, outputs=registration_outputs)
    alpha_input.change(update_blend, inputs=[blend_state, alpha_input], outputs=blend_output)
