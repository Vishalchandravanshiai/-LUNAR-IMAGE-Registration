"""Export matched point coordinates and geometric inlier labels."""
import csv


def write_match_points_csv(pts_ref, pts_src, inlier_mask, path):
    refs = list(pts_ref)
    srcs = list(pts_src)
    mask = list(inlier_mask.reshape(-1)) if inlier_mask is not None else [False] * len(refs)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ref_x", "ref_y", "src_x", "src_y", "is_inlier"])
        for ref, src, inlier in zip(refs, srcs, mask):
            writer.writerow([float(ref[0]), float(ref[1]), float(src[0]), float(src[1]), bool(inlier)])
    return path
