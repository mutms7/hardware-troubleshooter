"""Board-edge registration of one camera frame.

Segment the board, fit a straight line to each outline edge in undistorted pixels,
intersect neighbouring lines for sub-pixel corners, intersect the corner rays with
the board's top-surface plane, and fit an in-plane rigid pose (x, y, theta). Every
cyclic corner correspondence is a hypothesis; the configured asymmetric feature
must select exactly one, or the frame is rejected as ambiguous.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import board_to_camera, kabsch_2d, line_intersection, polygon_area, rot2

_MIN_FEATURE_PIXELS = 4
_MORPH_KERNEL = np.ones((3, 3), np.uint8)


@dataclass
class Hypothesis:
    shift: int  # observed corner i corresponds to model corner (i + shift) % n
    theta: float
    t: np.ndarray
    residual_px: float
    feature_contrast: list = field(default_factory=list)  # None when not measurable


@dataclass
class FrameResult:
    ok: bool = False
    reasons: list = field(default_factory=list)
    pose: tuple | None = None  # (x, y, theta) of the board in the top-surface plane frame
    rotation: np.ndarray | None = None  # board -> camera
    translation: np.ndarray | None = None
    residual_px: float | None = None
    edge_points: int = 0
    symmetry_resolved: bool = False
    contour_px: np.ndarray | None = None
    hypotheses: list = field(default_factory=list)
    ippe_tilt_deg: float | None = None
    ippe_offset_m: float | None = None


def failed(*reasons: str) -> FrameResult:
    return FrameResult(reasons=list(reasons))


def register_frame(gray, camera, plane, board, board_is_brighter, thresholds) -> FrameResult:
    result = FrameResult()

    def fail(reason):
        result.reasons.append(reason)
        return result

    contour, board_mask = _segment(gray, board_is_brighter)
    if contour is None:
        return fail("no_board_segment")
    result.contour_px = contour
    h, w = gray.shape
    if (
        contour[:, 0].min() <= 0
        or contour[:, 1].min() <= 0
        or contour[:, 0].max() >= w - 1
        or contour[:, 1].max() >= h - 1
    ):
        return fail("board_at_image_border")

    ideal = camera.undistort_px(contour)
    uv = plane.rays_to_plane(camera.ideal_to_rays(ideal))
    if uv is None:
        return fail("fit_failed:outline doesn't intersect the board plane")
    if polygon_area(uv) < 0:
        ideal, uv = ideal[::-1], uv[::-1]
    start = int(np.argmax(np.linalg.norm(uv - uv.mean(axis=0), axis=1)))
    ideal = np.roll(ideal, -start, axis=0)

    n = len(board.edges)
    vertices = _polygon_vertices(ideal, n)
    if vertices is None:
        return fail(f"outline_mismatch:segmented outline doesn't have {n} edges")
    runs = []
    for i in range(n):
        a, b = vertices[i], vertices[(i + 1) % n]
        idx = np.arange(a, b if b > a else b + len(ideal)) % len(ideal)
        cut = max(1, math.ceil(board.outline.trim_fraction * len(idx)))
        run = ideal[idx[cut : len(idx) - cut]]
        if len(run) < 2:
            return fail("fit_failed:edge too short to fit")
        runs.append(run)
    result.edge_points = int(sum(len(run) for run in runs))

    lines = [
        cv2.fitLine(run.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel() for run in runs
    ]
    corners = []
    for i in range(n):
        (vx0, vy0, x0, y0), (vx1, vy1, x1, y1) = lines[i - 1], lines[i]
        corner = line_intersection((x0, y0), (vx0, vy0), (x1, y1), (vx1, vy1))
        if corner is None:
            return fail("fit_failed:neighbouring edges are parallel")
        corners.append(corner)
    corners = np.array(corners)
    observed = plane.rays_to_plane(camera.ideal_to_rays(corners))
    if observed is None:
        return fail("fit_failed:corner doesn't intersect the board plane")

    for shift in range(n):
        theta, t = kabsch_2d(np.roll(board.corners, -shift, axis=0), observed)
        residual = _edge_residual(runs, board, camera, plane, theta, t, shift)
        result.hypotheses.append(Hypothesis(shift, theta, t, residual))

    limit = thresholds.max_fit_residual_px
    candidates = result.hypotheses
    if limit is None:
        # Keep going for diagnostics, but the frame can't be accepted.
        result.reasons.append("unconfigured:rig.thresholds.max_fit_residual_px")
    else:
        candidates = [hyp for hyp in candidates if hyp.residual_px <= limit]
        if not candidates:
            best = min(hyp.residual_px for hyp in result.hypotheses)
            result.residual_px = best
            return fail(f"residual_above_limit:{best:.3f}px > {limit}px")

    chosen, reason = _resolve_symmetry(
        gray, board_mask, candidates, board, camera, plane, thresholds
    )
    if chosen is None:
        return fail(reason)
    result.symmetry_resolved = True
    result.pose = (float(chosen.t[0]), float(chosen.t[1]), float(chosen.theta))
    result.rotation, result.translation = plane.pose_in_camera(*result.pose)
    result.residual_px = chosen.residual_px
    _ippe_cross_check(result, board, camera, corners, chosen.shift)
    result.ok = not result.reasons
    return result


def _segment(gray, board_is_brighter):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if not board_is_brighter:
        mask = cv2.bitwise_not(mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)
    if mask.all() or not mask.any():
        return None, None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None
    contour = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
    return contour.reshape(-1, 2).astype(np.float64), filled


def _polygon_vertices(points: np.ndarray, n: int):
    """Contour indices of exactly n dominant vertices, by bisecting approxPolyDP's epsilon."""
    curve = points.astype(np.float32).reshape(-1, 1, 2)
    flat = curve.reshape(-1, 2)
    lo, hi = 0.0, float(cv2.arcLength(curve, True))
    for _ in range(60):
        eps = (lo + hi) / 2
        approx = cv2.approxPolyDP(curve, eps, True).reshape(-1, 2)
        if len(approx) == n:
            idx = [int(np.flatnonzero((flat == v).all(axis=1))[0]) for v in approx]
            return sorted(idx)
        if len(approx) > n:
            lo = eps
        else:
            hi = eps
    return None


def _edge_residual(runs, board, camera, plane, theta, t, shift) -> float:
    """RMS pixel distance from each run's edge points to its projected model edge."""
    rotation, translation = plane.pose_in_camera(float(t[0]), float(t[1]), theta)
    squared = []
    for j, run in enumerate(runs):
        p0, p1 = board.edges[(j + shift) % len(runs)]
        a, b = camera.project_ideal(board_to_camera(np.array([p0, p1]), rotation, translation))
        d = b - a
        cross = d[0] * (run[:, 1] - a[1]) - d[1] * (run[:, 0] - a[0])
        squared.append((cross / np.linalg.norm(d)) ** 2)
    return float(math.sqrt(np.concatenate(squared).mean()))


def _resolve_symmetry(gray, board_mask, candidates, board, camera, plane, thresholds):
    if not board.features:
        return None, "unconfigured:board.features"
    min_contrast = thresholds.feature_min_contrast
    if min_contrast is None:
        return None, "unconfigured:rig.thresholds.feature_min_contrast"
    h, w = gray.shape
    present, unknown = [], 0
    for hyp in candidates:
        rotation, translation = plane.pose_in_camera(float(hyp.t[0]), float(hyp.t[1]), hyp.theta)
        masks = []
        for feature in board.features:
            px = camera.project(board_to_camera(feature.outline, rotation, translation))
            inside = (px[:, 0] >= 0) & (px[:, 1] >= 0) & (px[:, 0] <= w - 1) & (px[:, 1] <= h - 1)
            mask = np.zeros((h, w), np.uint8)
            if inside.all():
                cv2.fillPoly(mask, [np.round(px).astype(np.int32)], 255)
            masks.append(mask)
        reference = board_mask.copy()
        for mask in masks:
            reference[mask > 0] = 0
        ref = float(np.median(gray[reference > 0])) if reference.any() else None
        hyp.feature_contrast = []
        for feature, mask in zip(board.features, masks):
            selected = mask > 0
            if ref is None or selected.sum() < _MIN_FEATURE_PIXELS:
                hyp.feature_contrast.append(None)
                continue
            delta = (float(gray[selected].mean()) - ref) / 255.0
            hyp.feature_contrast.append(-delta if feature.expect == "darker" else delta)
        if any(c is None for c in hyp.feature_contrast):
            unknown += 1
        elif all(c >= min_contrast for c in hyp.feature_contrast):
            present.append(hyp)
    if unknown:
        return None, f"symmetry_ambiguous:feature not measurable in {unknown} orientation(s)"
    if len(present) != 1:
        return None, (
            f"symmetry_ambiguous:feature matched {len(present)} of {len(candidates)} orientations"
        )
    return present[0], ""


def _ippe_cross_check(result, board, camera, corners, shift):
    """Unconstrained planar solve, reported as a diagnostic only."""
    if len(corners) < 4:
        return
    model = np.roll(board.corners, -shift, axis=0)
    obj = np.column_stack([model, np.zeros(len(model))]).astype(np.float64)
    try:
        ok, rvec, tvec = cv2.solvePnP(
            obj, corners.astype(np.float64), camera.k, None, flags=cv2.SOLVEPNP_IPPE
        )
    except cv2.error:
        return
    if not ok:
        return
    rotation = cv2.Rodrigues(rvec)[0]
    cos_tilt = float(np.clip(rotation[:, 2] @ result.rotation[:, 2], -1.0, 1.0))
    result.ippe_tilt_deg = math.degrees(math.acos(cos_tilt))
    result.ippe_offset_m = float(np.linalg.norm(tvec.ravel() - result.translation))


def board_corners_in_plane(board, pose) -> np.ndarray:
    x, y, theta = pose
    return board.corners @ rot2(theta).T + np.array([x, y])
