"""Confirm a live registration from the board edges that an arm doesn't cover.

Full registration needs the whole outline, so it can't run while an arm is over the
board. When the caller supplies the arm's image mask, sample the live revision's
predicted outline, skip samples near the mask, find each visible edge along its normal
and refit the in-plane pose to those points. The refit pose then goes through the
engine's usual move check, so a board that moves under an arm is still caught.

This only confirms or invalidates an existing revision. It never creates one.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .geometry import board_to_camera, rot2
from .registration import FrameResult, _edge_residual

_SAMPLES_PER_EDGE = 40
_MIN_EDGE_SAMPLES = 8  # visible samples an edge needs before it constrains the fit
_MIN_EDGE_ANGLE_RAD = math.radians(20.0)  # two constraining edges must be this far from parallel
_MIN_EDGE_GRADIENT = 10.0  # grey levels per pixel at the edge
_PROFILE_STEP_PX = 0.25
_NORMAL_PROBE_M = 0.0005
_SEARCH_MARGIN_PX = 4.0
_REFIT_ITERATIONS = 8


def search_radius_px(camera, plane, thresholds) -> float:
    """Normal search half-width: twice the move threshold in pixels, plus a margin.

    A board displaced by more than the move threshold is still found, so the move
    check (not a failed search) reports it."""
    focal = 0.5 * (camera.k[0, 0] + camera.k[1, 1])
    return 2.0 * thresholds.move_translation_m * focal / plane.offset + _SEARCH_MARGIN_PX


def board_polygon_px(board, camera, plane, pose) -> np.ndarray:
    """Raw image pixels of the board outline at an in-plane pose."""
    rotation, translation = plane.pose_in_camera(*pose)
    return camera.project(board_to_camera(board.outline.polyline, rotation, translation))


def mask_overlaps_board(mask, board, camera, plane, pose, margin_px: float) -> bool:
    filled = np.zeros(mask.shape, np.uint8)
    polygon = np.round(board_polygon_px(board, camera, plane, pose)).astype(np.int32)
    cv2.fillPoly(filled, [polygon.reshape(-1, 1, 2)], 255)
    size = 2 * math.ceil(margin_px) + 1
    filled = cv2.dilate(filled, np.ones((size, size), np.uint8))
    return bool(((filled > 0) & (mask > 0)).any())


def confirm_partial(
    gray, mask, camera, plane, board, pose, board_is_brighter, thresholds
) -> FrameResult:
    """Refit `pose` (x, y, theta) to the visible board edges. The result is ok only when
    enough edges were found to fix all three degrees of freedom and the fit is within
    max_fit_residual_px."""
    result = FrameResult(symmetry_resolved=True)

    def fail(reason):
        result.reasons.append(reason)
        return result

    if thresholds.min_visible_edge_fraction is None:
        return fail("unconfigured:rig.thresholds.min_visible_edge_fraction")
    if thresholds.max_fit_residual_px is None:
        return fail("unconfigured:rig.thresholds.max_fit_residual_px")

    radius = search_radius_px(camera, plane, thresholds)
    size = 2 * math.ceil(radius) + 1
    blocked = cv2.dilate((mask > 0).astype(np.uint8), np.ones((size, size), np.uint8)) > 0
    rotation, translation = plane.pose_in_camera(*pose)
    h, w = gray.shape
    image = gray.astype(np.float64)
    offsets = np.arange(-radius, radius + _PROFILE_STEP_PX / 2, _PROFILE_STEP_PX)

    points, runs, total = [], [], 0
    for p0, p1 in board.edges:
        direction = (p1 - p0) / np.linalg.norm(p1 - p0)
        outward = np.array([direction[1], -direction[0]])  # counter-clockwise outline
        trim = board.outline.trim_fraction
        fractions = np.linspace(trim, 1 - trim, _SAMPLES_PER_EDGE)
        model = p0 + fractions[:, None] * (p1 - p0)
        px = camera.project(board_to_camera(model, rotation, translation))
        probe = camera.project(
            board_to_camera(model + _NORMAL_PROBE_M * outward, rotation, translation)
        )
        normals = probe - px
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        total += len(model)

        found = []
        for centre, normal in zip(px, normals):
            line = centre + offsets[:, None] * normal
            if (
                line[:, 0].min() < 1
                or line[:, 1].min() < 1
                or line[:, 0].max() > w - 2
                or line[:, 1].max() > h - 2
                or blocked[round(centre[1]), round(centre[0])]
            ):
                continue
            edge = _find_edge(image, line, offsets, board_is_brighter)
            if edge is not None:
                found.append(centre + edge * normal)
        found = np.array(found).reshape(-1, 2)
        ideal = camera.undistort_px(found) if len(found) else found
        runs.append(ideal)
        plane_points = plane.rays_to_plane(camera.ideal_to_rays(ideal)) if len(found) else found
        if plane_points is None:
            return fail("fit_failed:edge point doesn't intersect the board plane")
        points.append(plane_points)

    result.edge_points = int(sum(len(run) for run in runs))
    result.visible_fraction = result.edge_points / total
    if result.visible_fraction < thresholds.min_visible_edge_fraction:
        return fail(f"unconfirmed:only {result.visible_fraction:.0%} of the board edge is visible")
    if not _constrained(board, points):
        return fail("unconfirmed:visible edges don't fix the board's position and rotation")

    x, y, theta = _refit(board, points, pose)
    residual = _edge_residual(runs, board, camera, plane, theta, np.array([x, y]), 0)
    result.residual_px = residual
    if residual > thresholds.max_fit_residual_px:
        return fail(f"residual_above_limit:{residual:.3f}px > {thresholds.max_fit_residual_px}px")
    result.pose = (float(x), float(y), float(theta))
    result.rotation, result.translation = plane.pose_in_camera(*result.pose)
    result.ok = True
    return result


def _find_edge(image, line, offsets, board_is_brighter) -> float | None:
    """Signed offset along the outward normal of the board's edge, or None."""
    profile = _bilinear(image, line)
    gradient = np.gradient(profile, _PROFILE_STEP_PX)
    signed = -gradient if board_is_brighter else gradient  # board side is at negative offsets
    i = int(np.argmax(signed))
    if signed[i] < _MIN_EDGE_GRADIENT or i in (0, len(signed) - 1):
        return None
    a, b, c = signed[i - 1], signed[i], signed[i + 1]
    denom = a - 2 * b + c
    shift = 0.5 * (a - c) / denom if denom != 0 else 0.0
    return float(offsets[i] + shift * _PROFILE_STEP_PX)


def _bilinear(image, points) -> np.ndarray:
    x, y = points[:, 0], points[:, 1]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    top = image[y0, x0] * (1 - fx) + image[y0, x0 + 1] * fx
    bottom = image[y0 + 1, x0] * (1 - fx) + image[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bottom * fy


def _constrained(board, points) -> bool:
    directions = [
        math.atan2(*(p1 - p0)[::-1])
        for (p0, p1), pts in zip(board.edges, points)
        if len(pts) >= _MIN_EDGE_SAMPLES
    ]
    for i, a in enumerate(directions):
        for b in directions[i + 1 :]:
            gap = abs((a - b + math.pi / 2) % math.pi - math.pi / 2)
            if gap >= _MIN_EDGE_ANGLE_RAD:
                return True
    return False


def _refit(board, points, pose):
    """Gauss-Newton on point-to-edge-line distances in the plane."""
    x, y, theta = pose
    for _ in range(_REFIT_ITERATIONS):
        rows, residuals = [], []
        rotation = rot2(theta)
        d_rotation = np.array(
            [[-math.sin(theta), -math.cos(theta)], [math.cos(theta), -math.sin(theta)]]
        )
        t = np.array([x, y])
        for (p0, p1), pts in zip(board.edges, points):
            if len(pts) == 0:
                continue
            direction = (p1 - p0) / np.linalg.norm(p1 - p0)
            normal_model = np.array([direction[1], -direction[0]])
            normal = rotation @ normal_model
            anchor = rotation @ p0 + t
            r = (pts - anchor) @ normal
            # d r / d theta, from both the rotated normal and the rotated anchor.
            d_theta = (pts - anchor) @ (d_rotation @ normal_model) - (d_rotation @ p0) @ normal
            rows.append(
                np.column_stack(
                    [-np.full(len(pts), normal[0]), -np.full(len(pts), normal[1]), d_theta]
                )
            )
            residuals.append(r)
        jacobian, r = np.vstack(rows), np.concatenate(residuals)
        step, *_ = np.linalg.lstsq(jacobian, -r, rcond=None)
        x, y, theta = x + step[0], y + step[1], theta + step[2]
        if np.abs(step).max() < 1e-12:
            break
    return x, y, theta
