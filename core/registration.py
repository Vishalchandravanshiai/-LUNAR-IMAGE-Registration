"""
Image Registration Orchestrator and Visualizer.
Executes the end-to-end pipeline, warps source image, and creates comparison visualizations.
"""

from typing import Union, Tuple, Optional, Dict, Any
import time
import cv2
import numpy as np
from PIL import Image

from .preprocessing import ImagePreprocessor, PreprocessingError
from .feature_detection import (
    BaseFeatureDetector, SIFTFeatureDetector, ORBFeatureDetector, AKAZEFeatureDetector, FeatureDetectionError
)
from .feature_matching import (
    BaseFeatureMatcher, FLANNMatcher, BFMatcher, FeatureMatchingError
)
from .geometric_verification import (
    BaseGeometricVerifier, RANSACVerifier, VerificationError
)
from .metrics import RegistrationMetrics, compute_registration_metrics
from .quality_gate import assess_registration

class RegistrationError(Exception):
    """Raised when end-to-end image registration fails."""
    pass

class ImageRegistrar:
    """
    High-level orchestrator for multi-modal lunar image registration.
    Combines preprocessing, feature detection, matching, geometric verification,
    perspective warping, metric evaluation, and visual overlay generation.
    """

    def __init__(
        self,
        detector: Optional[BaseFeatureDetector] = None,
        matcher: Optional[BaseFeatureMatcher] = None,
        verifier: Optional[BaseGeometricVerifier] = None,
        preprocessor: Optional[ImagePreprocessor] = None
    ):
        self.preprocessor = preprocessor or ImagePreprocessor(use_clahe=True, use_blur=False)
        self.detector = detector or SIFTFeatureDetector()
        self.matcher = matcher or FLANNMatcher(ratio_threshold=0.75)
        self.verifier = verifier or RANSACVerifier(transform_type='homography', ransac_reproj_threshold=3.0)

    def warp_source_image(
        self,
        src_image: np.ndarray,
        ref_shape: Tuple[int, int],
        H: np.ndarray,
        transform_type: str = 'homography'
    ) -> np.ndarray:
        """
        Warps the source image into the coordinate system of the reference image.

        Args:
            src_image: Original or preprocessed source image (BGR or Grayscale).
            ref_shape: (height, width) of the reference image.
            H: 3x3 Homography matrix or 2x3/3x3 Affine matrix.
            transform_type: 'homography' or 'affine'.

        Returns:
            np.ndarray: Registered source image warped to match ref_shape.
        """
        ref_h, ref_w = ref_shape[:2]

        if transform_type == 'homography' or H.shape == (3, 3):
            registered = cv2.warpPerspective(
                src_image, H, (ref_w, ref_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        else:
            # 2x3 Affine
            M = H[:2, :] if H.shape == (3, 3) else H
            registered = cv2.warpAffine(
                src_image, M, (ref_w, ref_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

        return registered

    def draw_matches_visualization(
        self,
        ref_img: np.ndarray,
        keypoints_ref: list,
        src_img: np.ndarray,
        keypoints_src: list,
        matches: list,
        inlier_mask: Optional[np.ndarray] = None,
        inliers_only: bool = False
    ) -> np.ndarray:
        """
        Draws side-by-side feature match lines between reference and source images.
        """
        ref_vis = ref_img.copy()
        src_vis = src_img.copy()

        if len(ref_vis.shape) == 2:
            ref_vis = cv2.cvtColor(ref_vis, cv2.COLOR_GRAY2BGR)
        if len(src_vis.shape) == 2:
            src_vis = cv2.cvtColor(src_vis, cv2.COLOR_GRAY2BGR)

        if matches is None or len(matches) == 0:
            # Return concatenated images without match lines
            h1, w1 = ref_vis.shape[:2]
            h2, w2 = src_vis.shape[:2]
            max_h = max(h1, h2)
            canvas = np.zeros((max_h, w1 + w2, 3), dtype=np.uint8)
            canvas[:h1, :w1] = ref_vis
            canvas[:h2, w1:w1+w2] = src_vis
            return canvas

        if inliers_only and inlier_mask is not None:
            mask_list = inlier_mask.ravel().tolist()
            display_matches = [m for m, inlier in zip(matches, mask_list) if inlier]
            line_color = (0, 255, 0)  # Green for inliers
            match_mask = None
        else:
            display_matches = matches
            line_color = (255, 150, 0) # Cyan/Orange for raw matches
            match_mask = inlier_mask.ravel().tolist() if (inlier_mask is not None and not inliers_only) else None

        # Draw matches using cv2.drawMatches
        vis = cv2.drawMatches(
            ref_vis, keypoints_ref,
            src_vis, keypoints_src,
            display_matches, None,
            matchColor=line_color,
            singlePointColor=(0, 0, 255),
            matchesMask=match_mask,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        return vis

    def create_blended_overlay(self, ref_img: np.ndarray, registered_src: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Creates a weighted alpha blend between the reference image and registered source image.
        """
        ref_c = ref_img.copy()
        src_c = registered_src.copy()

        if len(ref_c.shape) == 2:
            ref_c = cv2.cvtColor(ref_c, cv2.COLOR_GRAY2BGR)
        if len(src_c.shape) == 2:
            src_c = cv2.cvtColor(src_c, cv2.COLOR_GRAY2BGR)

        # Resize if slight shape mismatch
        if ref_c.shape[:2] != src_c.shape[:2]:
            src_c = cv2.resize(src_c, (ref_c.shape[1], ref_c.shape[0]))

        blended = cv2.addWeighted(ref_c, alpha, src_c, 1.0 - alpha, 0)
        return blended

    def create_difference_map(self, ref_img: np.ndarray, registered_src: np.ndarray) -> np.ndarray:
        """
        Creates an absolute difference map between reference image and registered source image.
        Zero difference (perfect alignment) appears black.
        """
        ref_gray = self.preprocessor.to_grayscale(ref_img)
        src_gray = self.preprocessor.to_grayscale(registered_src)

        if ref_gray.shape != src_gray.shape:
            src_gray = cv2.resize(src_gray, (ref_gray.shape[1], ref_gray.shape[0]))

        diff = cv2.absdiff(ref_gray, src_gray)
        # Apply colormap (JET or HOT) for clear visual disparity inspection
        diff_color = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        return diff_color

    def create_false_color_anaglyph(self, ref_img: np.ndarray, registered_src: np.ndarray) -> np.ndarray:
        """
        Creates a false-color red-cyan anaglyph image.
        Red channel = Reference Image
        Cyan channel (Green + Blue) = Registered Source Image
        Perfectly aligned features show up in crisp grayscale; misaligned regions show red/cyan fringes!
        """
        ref_gray = self.preprocessor.to_grayscale(ref_img)
        src_gray = self.preprocessor.to_grayscale(registered_src)

        if ref_gray.shape != src_gray.shape:
            src_gray = cv2.resize(src_gray, (ref_gray.shape[1], ref_gray.shape[0]))

        h, w = ref_gray.shape
        anaglyph = np.zeros((h, w, 3), dtype=np.uint8)
        anaglyph[:, :, 2] = ref_gray  # Red channel = Ref
        anaglyph[:, :, 1] = src_gray  # Green channel = Src
        anaglyph[:, :, 0] = src_gray  # Blue channel = Src

        return anaglyph

    def register(
        self,
        ref_source: Union[str, bytes, np.ndarray, Image.Image],
        src_source: Union[str, bytes, np.ndarray, Image.Image]
    ) -> Dict[str, Any]:
        """
        Executes complete image registration pipeline.

        Returns:
            Dict containing:
            - 'metrics': RegistrationMetrics object
            - 'registered_source': np.ndarray (registered source BGR/gray)
            - 'all_matches_vis': np.ndarray visualization of all raw matches
            - 'inlier_matches_vis': np.ndarray visualization of RANSAC inliers only
            - 'blended_overlay': np.ndarray alpha-blended image
            - 'difference_map': np.ndarray false-color absolute difference image
            - 'anaglyph_overlay': np.ndarray red-cyan false-color anaglyph
            - 'reference_image': np.ndarray raw reference BGR
            - 'source_image': np.ndarray raw source BGR
            - 'H': np.ndarray estimated transformation matrix (or None)
        """
        start_time = time.time()

        # Step 1: Preprocess reference and source images
        try:
            raw_ref, proc_ref = self.preprocessor.preprocess(ref_source)
            raw_src, proc_src = self.preprocessor.preprocess(src_source)
        except PreprocessingError as pe:
            elapsed_ms = (time.time() - start_time) * 1000.0
            metrics = RegistrationMetrics(
                success=False,
                status_message=f"Preprocessing error: {str(pe)}",
                execution_time_ms=elapsed_ms
            )
            return {'metrics': metrics, 'registered_source': None, 'H': None}

        # Step 2 & 3: Detect local features and compute descriptors
        try:
            kp_ref, desc_ref = self.detector.detect_and_compute(proc_ref)
            kp_src, desc_src = self.detector.detect_and_compute(proc_src)
        except Exception as fe:
            elapsed_ms = (time.time() - start_time) * 1000.0
            metrics = RegistrationMetrics(
                total_keypoints_ref=0,
                total_keypoints_src=0,
                success=False,
                status_message=f"Feature detection error: {str(fe)}",
                execution_time_ms=elapsed_ms
            )
            return {'metrics': metrics, 'registered_source': None, 'H': None}

        num_kp_ref = len(kp_ref)
        num_kp_src = len(kp_src)

        if num_kp_ref == 0 or num_kp_src == 0:
            elapsed_ms = (time.time() - start_time) * 1000.0
            metrics = RegistrationMetrics(
                total_keypoints_ref=num_kp_ref,
                total_keypoints_src=num_kp_src,
                success=False,
                status_message="Registration failed — insufficient local features detected in one or both images.",
                execution_time_ms=elapsed_ms
            )
            return {'metrics': metrics, 'registered_source': None, 'H': None}

        # Step 4 & 5: Match corresponding features & apply Lowe ratio test
        try:
            matches, pts_ref, pts_src = self.matcher.match(
                kp_ref, desc_ref, kp_src, desc_src, norm_type=self.detector.norm_type
            )
        except Exception as me:
            elapsed_ms = (time.time() - start_time) * 1000.0
            metrics = RegistrationMetrics(
                total_keypoints_ref=num_kp_ref,
                total_keypoints_src=num_kp_src,
                success=False,
                status_message=f"Feature matching error: {str(me)}",
                execution_time_ms=elapsed_ms
            )
            return {'metrics': metrics, 'registered_source': None, 'H': None}

        total_matches = len(matches)

        if total_matches == 0:
            elapsed_ms = (time.time() - start_time) * 1000.0
            metrics = RegistrationMetrics(
                total_keypoints_ref=num_kp_ref,
                total_keypoints_src=num_kp_src,
                total_matches=0,
                success=False,
                status_message="Registration failed — insufficient feature matches passing ratio test.",
                execution_time_ms=elapsed_ms
            )
            all_vis = self.draw_matches_visualization(raw_ref, kp_ref, raw_src, kp_src, matches)
            return {
                'metrics': metrics,
                'registered_source': None,
                'all_matches_vis': all_vis,
                'inlier_matches_vis': all_vis,
                'H': None
            }

        # Step 6 & 7: Geometric verification with RANSAC & degeneracy rejection
        H, inlier_mask, is_valid, status_msg = self.verifier.verify(pts_ref, pts_src)

        elapsed_ms = (time.time() - start_time) * 1000.0

        metrics = compute_registration_metrics(
            pts_ref=pts_ref,
            pts_src=pts_src,
            inlier_mask=inlier_mask,
            H=H,
            total_keypoints_ref=num_kp_ref,
            total_keypoints_src=num_kp_src,
            total_matches=total_matches,
            execution_time_ms=elapsed_ms,
            detector_name=self.detector.name,
            matcher_name=getattr(self.matcher, 'name', 'FLANN'),
            transformation_type=getattr(self.verifier, 'transform_type', 'homography').capitalize(),
            success=is_valid,
            status_message=status_msg if is_valid else f"Registration failed — {status_msg}"
        )

        if is_valid and H is not None and inlier_mask is not None:
            quality = assess_registration(
                pts_ref, pts_src, inlier_mask, raw_ref.shape[:2],
                getattr(self.verifier, 'transform_type', 'homography')
            )
            metrics.holdout_rmse_pixels = quality['holdout_rmse']
            metrics.match_coverage = quality['coverage']
            metrics.confidence = quality['confidence']
            if quality['confidence'] == 'REJECTED':
                is_valid = False
                metrics.success = False
                metrics.status_message = f"Registration rejected by quality gate - {quality['reason']}"
            else:
                metrics.status_message = f"Registration successful (confidence: {quality['confidence']}; {quality['reason']})"

        # Generate visualizations
        all_vis = self.draw_matches_visualization(raw_ref, kp_ref, raw_src, kp_src, matches, inlier_mask=inlier_mask, inliers_only=False)
        inlier_vis = self.draw_matches_visualization(raw_ref, kp_ref, raw_src, kp_src, matches, inlier_mask=inlier_mask, inliers_only=True)

        if not is_valid or H is None:
            return {
                'metrics': metrics,
                'registered_source': None,
                'all_matches_vis': all_vis,
                'inlier_matches_vis': inlier_vis,
                'reference_image': raw_ref,
                'source_image': raw_src,
                'H': None,
                'pts_ref': pts_ref, 'pts_src': pts_src, 'inlier_mask': inlier_mask
            }

        # Step 8: Warp Source Image into Reference Image Coordinate Frame
        registered_src = self.warp_source_image(
            raw_src, raw_ref.shape[:2], H, transform_type=getattr(self.verifier, 'transform_type', 'homography')
        )

        # Generate alignment evaluation overlays
        blended = self.create_blended_overlay(raw_ref, registered_src, alpha=0.5)
        diff_map = self.create_difference_map(raw_ref, registered_src)
        anaglyph = self.create_false_color_anaglyph(raw_ref, registered_src)

        return {
            'metrics': metrics,
            'registered_source': registered_src,
            'all_matches_vis': all_vis,
            'inlier_matches_vis': inlier_vis,
            'blended_overlay': blended,
            'difference_map': diff_map,
            'anaglyph_overlay': anaglyph,
            'reference_image': raw_ref,
            'source_image': raw_src,
            'H': H,
            'pts_ref': pts_ref, 'pts_src': pts_src, 'inlier_mask': inlier_mask
        }
