"""Route difficult illumination changes through images with nearby sun angles."""
import heapq
import math
import numpy as np


def sun_separation_deg(sun_a, sun_b):
    def vector(sun):
        az, el = np.radians(sun)
        return np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])
    dot = float(np.clip(np.dot(vector(sun_a), vector(sun_b)), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def plan_chain(ref_sun, src_sun, candidates, max_step_deg=30):
    nodes = [(None, tuple(ref_sun))] + [(p, tuple(s)) for p, s in candidates] + [(None, tuple(src_sun))]
    start, goal = 0, len(nodes) - 1
    queue = [(0, start, [start])]
    best = {start: 0}
    while queue:
        cost, i, path = heapq.heappop(queue)
        if i == goal:
            return [(nodes[j][0], nodes[j][1]) for j in path]
        for j in range(1, len(nodes)):
            if j == i or j in path:
                continue
            gap = sun_separation_deg(nodes[i][1], nodes[j][1])
            if gap <= max_step_deg and cost + 1 < best.get(j, 10**9):
                best[j] = cost + 1
                heapq.heappush(queue, (cost + 1, j, path + [j]))
    return None


def _as_3x3(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape == (2, 3):
        matrix = np.vstack((matrix, [0.0, 0.0, 1.0]))
    if matrix.shape != (3, 3):
        raise ValueError("Registration transform must be 2x3 or 3x3")
    return matrix / matrix[2, 2] if abs(matrix[2, 2]) > 1e-12 else matrix


def register_via_bridge(reference, source, ref_sun, src_sun, candidates, registrar, max_step_deg=30):
    direct = registrar.register(reference, source)
    metrics = direct.get("metrics")
    if metrics and metrics.confidence in ("HIGH", "MEDIUM") and direct.get("H") is not None:
        direct["mode"] = "direct"
        return direct
    chain = plan_chain(ref_sun, src_sun, candidates, max_step_deg)
    gap = sun_separation_deg(ref_sun, src_sun)
    if chain is None:
        if metrics:
            metrics.success = False
            metrics.status_message = f"Direct registration not accepted; sun gap {gap:.1f}° needs intermediate images within {max_step_deg}°."
        direct.update({"H": None, "registered_source": None, "mode": "failed", "chain": None, "links": []})
        return direct
    images = [reference] + [p for p, _ in chain[1:-1]] + [source]
    composed = np.eye(3)
    links = []
    link_matrices = []
    link_ranks = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "REJECTED": 3}
    for idx, (left, right) in enumerate(zip(images, images[1:])):
        result = registrar.register(left, right)
        m = result.get("metrics")
        confidence = m.confidence if m else "REJECTED"
        if not m or confidence in ("REJECTED", "LOW") or result.get("H") is None:
            if metrics:
                metrics.success = False
                detail = m.status_message if m else "registration did not return metrics"
                metrics.status_message = f"Bridge link {idx + 1} failed quality gate ({confidence}): {detail}; chain discarded."
            direct.update({"H": None, "registered_source": None, "mode": "failed", "chain": chain, "links": links})
            return direct
        link_matrices.append(_as_3x3(result["H"]))
        links.append({"from": left, "to": right, "inliers": m.inlier_matches, "confidence": confidence,
                      "holdout_rmse": m.holdout_rmse_pixels})
    worst = max((x["confidence"] for x in links), key=lambda c: link_ranks[c])
    # Each matrix maps the next image into the current one: H_ref<-mid @ H_mid<-src.
    composed = np.eye(3)
    for H in link_matrices:
        composed = composed @ H
    composed = _as_3x3(composed)
    if metrics:
        metrics.success = True
        metrics.confidence = worst
        metrics.status_message = f"Bridge registration successful (confidence: {worst}; {len(links)} links; errors accumulate along chain)."
    try:
        raw_ref, _ = registrar.preprocessor.preprocess(reference)
        raw_src, _ = registrar.preprocessor.preprocess(source)
        warped = registrar.warp_source_image(raw_src, raw_ref.shape[:2], composed,
                                             getattr(registrar.verifier, "transform_type", "homography"))
        direct["registered_source"] = warped
        direct["reference_image"], direct["source_image"] = raw_ref, raw_src
        direct["blended_overlay"] = registrar.create_blended_overlay(raw_ref, warped)
        direct["difference_map"] = registrar.create_difference_map(raw_ref, warped)
        direct["anaglyph_overlay"] = registrar.create_false_color_anaglyph(raw_ref, warped)
    except Exception:
        direct["registered_source"] = None
    direct.update({"H": composed, "mode": "bridge", "chain": chain, "links": links})
    return direct
