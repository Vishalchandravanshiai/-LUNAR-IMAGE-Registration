"""Cross-validation and spatial-support checks for estimated registrations."""
import cv2
import numpy as np

MIN_INLIERS = 15
GRID = 4
N_SPLITS = 10
TEST_FRACTION = 0.3


def spatial_coverage(pts_ref, ref_shape, grid=GRID):
    points = np.asarray(pts_ref, dtype=float).reshape(-1, 2)
    h, w = ref_shape[:2]
    if not len(points) or h <= 0 or w <= 0 or grid <= 0:
        return 0.0
    cells = set()
    for x, y in points:
        if 0 <= x < w and 0 <= y < h:
            cells.add((min(grid - 1, int(x * grid / w)), min(grid - 1, int(y * grid / h))))
    return len(cells) / float(grid * grid)


def _estimate(pts_ref, pts_src, transform_type):
    src = np.asarray(pts_src, dtype=np.float32).reshape(-1, 1, 2)
    ref = np.asarray(pts_ref, dtype=np.float32).reshape(-1, 1, 2)
    if transform_type.lower() == "affine":
        matrix, _ = cv2.estimateAffinePartial2D(src, ref, method=cv2.LMEDS)
        return matrix
    return cv2.findHomography(src, ref, method=0)[0]


def holdout_rmse(pts_ref, pts_src, transform_type="homography", n_splits=N_SPLITS, seed=42):
    ref = np.asarray(pts_ref, dtype=float).reshape(-1, 2)
    src = np.asarray(pts_src, dtype=float).reshape(-1, 2)
    n = len(ref)
    if n != len(src) or n < 4:
        return None
    rng = np.random.default_rng(seed)
    errors = []
    test_n = max(1, int(round(n * TEST_FRACTION)))
    for _ in range(max(1, n_splits)):
        order = rng.permutation(n)
        test_idx, train_idx = order[:test_n], order[test_n:]
        min_train = 3 if transform_type.lower() == "affine" else 4
        if len(train_idx) < min_train:
            continue
        try:
            H = _estimate(ref[train_idx], src[train_idx], transform_type)
            if H is None or not np.isfinite(H).all():
                continue
            if H.shape == (2, 3):
                H = np.vstack((H, [0, 0, 1]))
            projected = cv2.perspectiveTransform(src[test_idx].astype(np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)
            errors.extend(np.linalg.norm(projected - ref[test_idx], axis=1).tolist())
        except (cv2.error, ValueError, np.linalg.LinAlgError):
            continue
    return float(np.sqrt(np.mean(np.square(errors)))) if errors else None


def assess_registration(pts_ref, pts_src, inlier_mask, ref_shape, transform_type="homography"):
    mask = np.asarray(inlier_mask).reshape(-1).astype(bool)
    ref = np.asarray(pts_ref).reshape(-1, 2)
    src = np.asarray(pts_src).reshape(-1, 2)
    n = min(len(mask), len(ref), len(src))
    mask, ref, src = mask[:n], ref[:n], src[:n]
    ref, src = ref[mask], src[mask]
    count = len(ref)
    coverage = spatial_coverage(ref, ref_shape)
    rmse = holdout_rmse(ref, src, transform_type)
    if count < MIN_INLIERS:
        confidence, reason = "REJECTED", f"only {count} inliers (minimum {MIN_INLIERS})"
    elif rmse is None:
        confidence, reason = "REJECTED", "held-out error could not be computed"
    elif rmse > 5:
        confidence, reason = "REJECTED", f"held-out RMSE {rmse:.2f}px exceeds 5px"
    elif coverage < 0.25 or rmse > 2:
        confidence, reason = "LOW", f"coverage {coverage:.0%}; held-out RMSE {rmse:.2f}px"
    elif count < 50 or coverage < 0.5 or rmse > 1:
        confidence, reason = "MEDIUM", f"{count} inliers; coverage {coverage:.0%}; held-out RMSE {rmse:.2f}px"
    else:
        confidence, reason = "HIGH", f"{count} inliers; coverage {coverage:.0%}; held-out RMSE {rmse:.2f}px"
    return {"confidence": confidence, "reason": reason, "holdout_rmse": rmse, "coverage": coverage, "inliers": count}
