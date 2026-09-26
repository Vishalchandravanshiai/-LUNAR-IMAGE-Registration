"""
Geometric verification and RANSAC outlier rejection module for lunar image correspondence.
Estimates Homography or Affine transformations and detects degenerate geometric models.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import cv2
import numpy as np

class VerificationError(Exception):
    """Raised when geometric verification fails or model is degenerate."""
    pass

class BaseGeometricVerifier(ABC):
    """
    Abstract Base Class for geometric verification.
    """

    @abstractmethod
    def verify(self, pts_ref: np.ndarray, pts_src: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool, str]:
        """
        Estimate transformation matrix from source points to reference points.

        Returns:
            Tuple of:
            - Transformation matrix (3x3 for homography, 2x3 or 3x3 for affine) or None
            - Inlier mask array [N x 1] (1 for inlier, 0 for outlier) or None
            - Boolean flag indicating success/validity
            - Status / error explanation message string
        """
        pass


class RANSACVerifier(BaseGeometricVerifier):
    """
    RANSAC-based geometric model estimation (Homography or Affine).
    Includes strict degeneracy checks to reject unstable or mirror-flipped transformations.
    """

    def __init__(self, transform_type: str = 'homography', ransac_reproj_threshold: float = 3.0, max_iters: int = 2000):
        self.transform_type = transform_type.lower()
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.max_iters = max_iters

    def _check_homography_validity(self, H: np.ndarray) -> Tuple[bool, str]:
        """
        Validates whether estimated 3x3 homography matrix is non-degenerate.
        """
        if H is None or H.shape != (3, 3):
            return False, "Matrix shape is not 3x3."

        if np.isnan(H).any() or np.isinf(H).any():
            return False, "Matrix contains NaN or Inf values."

        # Check normalization
        if abs(H[2, 2]) < 1e-7:
            return False, "Homography element H[2,2] is near zero."

        H_norm = H / H[2, 2]

        # Determinant of top 2x2 submatrix (affine/scale component)
        det = np.linalg.det(H_norm[:2, :2])
        if det <= 0.01 or det > 100.0:
            return False, f"Degenerate scale change or mirror flip (det={det:.4f})."

        # Singular values check to detect severe shear or collinearity
        s = np.linalg.svd(H_norm[:2, :2], compute_uv=False)
        if s[1] < 1e-5:
            return False, "Matrix singular values indicate severe collapse."

        condition_number = s[0] / s[1]
        if condition_number > 50.0:
            return False, f"Homography ill-conditioned (condition number={condition_number:.2f})."

        return True, "Valid Homography"

    def verify(self, pts_ref: np.ndarray, pts_src: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool, str]:
        """
        Performs geometric verification mapping source points -> reference points.
        Note: OpenCV findHomography takes (srcPoints, dstPoints, ...), so:
        srcPoints = pts_src (moving image)
        dstPoints = pts_ref (fixed image)
        """
        if pts_ref is None or pts_src is None or len(pts_ref) == 0 or len(pts_src) == 0:
            return None, None, False, "No feature correspondences provided."

        if len(pts_ref) != len(pts_src):
            return None, None, False, "Mismatched number of points between reference and source."

        num_pts = len(pts_ref)

        if self.transform_type == 'homography':
            if num_pts < 4:
                return None, None, False, f"Insufficient matches for homography estimation (need >= 4, got {num_pts})."

            try:
                # Use USAC_MAGSAC if available, else RANSAC
                ransac_method = getattr(cv2, 'USAC_MAGSAC', cv2.RANSAC)
                H, mask = cv2.findHomography(
                    pts_src, pts_ref,
                    method=ransac_method,
                    ransacReprojThreshold=self.ransac_reproj_threshold,
                    maxIters=self.max_iters
                )
            except Exception as e:
                return None, None, False, f"OpenCV findHomography failed: {str(e)}"

            if H is None or mask is None:
                return None, None, False, "RANSAC failed to find a valid Homography model."

            inlier_count = int(np.sum(mask))
            if inlier_count < 4:
                return None, None, False, f"RANSAC found insufficient inliers ({inlier_count} < 4)."

            is_valid, reason = self._check_homography_validity(H)
            if not is_valid:
                return None, mask, False, f"Degenerate homography rejected: {reason}"

            return H, mask, True, "Geometric verification successful."

        elif self.transform_type == 'affine':
            if num_pts < 3:
                return None, None, False, f"Insufficient matches for affine estimation (need >= 3, got {num_pts})."

            try:
                M, mask = cv2.estimateAffinePartial2D(
                    pts_src, pts_ref,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=self.ransac_reproj_threshold,
                    maxIters=self.max_iters
                )
            except Exception as e:
                return None, None, False, f"OpenCV estimateAffinePartial2D failed: {str(e)}"

            if M is None or mask is None:
                return None, None, False, "RANSAC failed to estimate an Affine transformation."

            inlier_count = int(np.sum(mask))
            if inlier_count < 3:
                return None, None, False, f"RANSAC found insufficient affine inliers ({inlier_count} < 3)."

            # Convert 2x3 Affine matrix to 3x3 for homogeneous coordinate consistency
            H = np.eye(3, dtype=np.float64)
            H[:2, :] = M

            return H, mask, True, "Affine geometric verification successful."

        else:
            return None, None, False, f"Unknown transform_type '{self.transform_type}'. Choose 'homography' or 'affine'."
