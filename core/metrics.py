"""
Metrics calculation module for image registration assessment.
Calculates exact reprojection RMSE (Root Mean Square Error), inlier ratios, and runtime metrics.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import numpy as np

@dataclass
class RegistrationMetrics:
    """Dataclass holding performance and evaluation metrics for an image registration run."""
    total_keypoints_ref: int = 0
    total_keypoints_src: int = 0
    total_matches: int = 0
    inlier_matches: int = 0
    inlier_ratio: float = 0.0
    rmse_pixels: float = 0.0
    mae_pixels: float = 0.0
    max_reprojection_error_pixels: float = 0.0
    execution_time_ms: float = 0.0
    detector_name: str = "SIFT"
    matcher_name: str = "FLANN"
    transformation_type: str = "Homography"
    success: bool = False
    status_message: str = ""
    holdout_rmse_pixels: Optional[float] = None
    match_coverage: float = 0.0
    confidence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_reprojection_errors(H: np.ndarray, pts_src: np.ndarray, pts_ref: np.ndarray) -> np.ndarray:
    """
    Computes Euclidean reprojection error distance for each point pair:
    maps pts_src via H and compares to pts_ref.

    Args:
        H: 3x3 Homography matrix or 2x3/3x3 Affine matrix.
        pts_src: NumPy array of shape [N, 2] (source points x, y).
        pts_ref: NumPy array of shape [N, 2] (reference points x, y).

    Returns:
        np.ndarray: Array of Euclidean reprojection errors of shape [N].
    """
    if H is None or len(pts_src) == 0 or len(pts_ref) == 0:
        return np.empty((0,), dtype=np.float64)

    if H.shape == (2, 3):
        H_3x3 = np.eye(3, dtype=np.float64)
        H_3x3[:2, :] = H
    else:
        H_3x3 = H

    # Convert source points to homogeneous coordinates [N x 3]
    num_pts = len(pts_src)
    ones = np.ones((num_pts, 1), dtype=np.float64)
    src_homo = np.hstack([pts_src.astype(np.float64), ones])  # [N, 3]

    # Transform: (H * src^T)^T -> [N, 3]
    transformed_homo = (H_3x3 @ src_homo.T).T

    # Normalize by third coordinate w (z)
    w = transformed_homo[:, 2:3]
    # Avoid division by zero
    w_safe = np.where(np.abs(w) < 1e-12, 1e-12, w)
    pts_src_proj = transformed_homo[:, :2] / w_safe

    # Euclidean distance per point: sqrt((x_proj - x_ref)^2 + (y_proj - y_ref)^2)
    diffs = pts_src_proj - pts_ref.astype(np.float64)
    errors = np.sqrt(np.sum(diffs ** 2, axis=1))

    return errors


def compute_registration_metrics(
    pts_ref: np.ndarray,
    pts_src: np.ndarray,
    inlier_mask: Optional[np.ndarray],
    H: Optional[np.ndarray],
    total_keypoints_ref: int,
    total_keypoints_src: int,
    total_matches: int,
    execution_time_ms: float,
    detector_name: str = "SIFT",
    matcher_name: str = "FLANN",
    transformation_type: str = "Homography",
    success: bool = False,
    status_message: str = ""
) -> RegistrationMetrics:
    """
    Calculates complete metrics for a registration run.
    """
    metrics = RegistrationMetrics(
        total_keypoints_ref=total_keypoints_ref,
        total_keypoints_src=total_keypoints_src,
        total_matches=total_matches,
        execution_time_ms=execution_time_ms,
        detector_name=detector_name,
        matcher_name=matcher_name,
        transformation_type=transformation_type,
        success=success,
        status_message=status_message
    )

    if not success or H is None or inlier_mask is None or total_matches == 0:
        metrics.inlier_matches = 0
        metrics.inlier_ratio = 0.0
        metrics.rmse_pixels = 0.0
        metrics.mae_pixels = 0.0
        metrics.max_reprojection_error_pixels = 0.0
        return metrics

    mask_bool = inlier_mask.ravel().astype(bool)
    inlier_count = int(np.sum(mask_bool))
    metrics.inlier_matches = inlier_count
    metrics.inlier_ratio = float(inlier_count / total_matches) if total_matches > 0 else 0.0

    if inlier_count > 0 and len(pts_ref) > 0 and len(pts_src) > 0:
        inlier_pts_ref = pts_ref[mask_bool]
        inlier_pts_src = pts_src[mask_bool]

        errors = compute_reprojection_errors(H, inlier_pts_src, inlier_pts_ref)

        if len(errors) > 0:
            metrics.rmse_pixels = float(np.sqrt(np.mean(errors ** 2)))
            metrics.mae_pixels = float(np.mean(errors))
            metrics.max_reprojection_error_pixels = float(np.max(errors))

    return metrics
