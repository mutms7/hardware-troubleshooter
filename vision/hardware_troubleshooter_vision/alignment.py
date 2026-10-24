"""Wrist-camera alignment of a tool tip over a target pad.

At the hover pose, the wrist image is resampled onto the board plane as the predicted
pose says it should look (registration plus FK), and the copper around the target,
rendered from the pad geometry, is matched against it. The match offset is where the
board really is relative to the prediction. The tool tip's landing point comes from
the calibrated tip pixel, mapped through the same prediction, so most of the
prediction's error cancels: both are image measurements.

The error is target minus tip landing point, in the board plane. The motion adapter
moves the tool by it (correction_in_base) and the next frame checks again.

The result is evidence of where the tip will land. It isn't evidence of contact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import transforms
from .config import AlignmentConfig
from .pads import Pad

_EXCLUSION_PX = 3  # around the best match, before looking for the runner-up


@dataclass
class AlignmentResult:
    ok: bool
    target: str
    reasons: list[str] = field(default_factory=list)
    aligned: bool = False  # ok and the error is within converged_m
    error_board_m: np.ndarray | None = None  # target - tip landing point, board x/y
    error_m: float | None = None
    offset_board_m: np.ndarray | None = None  # seen board position - predicted
    score: float | None = None
    margin: float | None = None
    standoff_m: float | None = None
    tip_pixel: tuple[float, float] | None = None
    target_pixel: tuple[float, float] | None = None

    @property
    def failure_reason(self) -> str:
        return "; ".join(self.reasons)


def align(
    gray: np.ndarray,
    camera,
    camera_from_board: np.ndarray,
    pads: list[Pad],
    target: str,
    tip_pixel,
    tip_height_m: float,
    config: AlignmentConfig,
) -> AlignmentResult:
    """Measure target minus tip landing point.

    camera_from_board: predicted board -> wrist camera, from registration and measured
    joints at the image stamp. tip_pixel: the tool's calibrated TipPixel, or None.
    tip_height_m: the tool tip's height above the board top surface, from the same FK.
    """
    result = AlignmentResult(False, target)

    def fail(reason):
        result.reasons.append(reason)
        return result

    missing = config.missing()
    if missing:
        return fail("; ".join(f"unconfigured:{key}" for key in missing))
    pad = next((p for p in pads if p.name == target), None)
    if pad is None:
        return fail(f"unknown_target:{target}")
    if tip_pixel is None:
        return fail("tip_pixel_uncalibrated")
    result.standoff_m = float(tip_height_m)
    if abs(tip_height_m - tip_pixel.standoff_m) > config.max_standoff_error_m:
        return fail(
            f"standoff_mismatch:tip is {tip_height_m * 1000:.2f} mm above the board, "
            f"tip pixel was calibrated at {tip_pixel.standoff_m * 1000:.2f} mm"
        )

    res = config.resolution_m
    n_template = math.ceil(config.patch_radius_m / res)
    n_search = math.ceil(config.search_radius_m / res)
    half = n_template + n_search
    rectified = _rectify(gray, camera, camera_from_board, pad.center, half, res)
    if rectified is None:
        return fail("target_not_in_view:the search area isn't wholly in the image")
    template = _render(pads, pad.center, n_template, res, config.pads_are_brighter)
    if template.std() == 0:
        return fail("no_pad_contrast:nothing but copper or nothing but mask around the target")

    scores = cv2.matchTemplate(rectified, template, cv2.TM_CCOEFF_NORMED)
    _, best, _, (col, row) = cv2.minMaxLoc(scores)
    result.score = float(best)
    hidden = scores.copy()
    cv2.circle(
        hidden, (col, row), _EXCLUSION_PX + _min_pad_px(pads, pad.center, n_template, res), -1.0, -1
    )
    result.margin = float(best - hidden.max())
    if col in (0, scores.shape[1] - 1) or row in (0, scores.shape[0] - 1):
        return fail("match_at_search_edge:the board may be further off than search_radius_m")
    if best < config.min_match_score:
        return fail(f"match_weak:{best:.3f} < {config.min_match_score}")
    if result.margin < config.min_uniqueness_margin:
        return fail(f"match_ambiguous:margin {result.margin:.3f} < {config.min_uniqueness_margin}")

    dc = _subpixel(scores[row, col - 1 : col + 2])
    dr = _subpixel(scores[row - 1 : row + 2, col])
    offset = np.array([(col + dc - n_search) * res, -(row + dr - n_search) * res])
    result.offset_board_m = offset

    board_from_camera = transforms.invert(camera_from_board)
    ray = camera.ideal_to_rays(camera.undistort_px(np.array([tip_pixel.uv])))[0]
    origin, direction = board_from_camera[:3, 3], board_from_camera[:3, :3] @ ray
    if direction[2] >= 0:
        return fail("target_not_in_view:tip pixel ray doesn't reach the board")
    landing = origin + direction * (-origin[2] / direction[2])
    seen_target = pad.center + offset
    result.error_board_m = seen_target - landing[:2]
    result.error_m = float(np.linalg.norm(result.error_board_m))
    result.tip_pixel = tuple(float(v) for v in tip_pixel.uv)
    target_px = camera.project(transforms.apply(camera_from_board, [[*seen_target, 0.0]]))[0]
    result.target_pixel = (float(target_px[0]), float(target_px[1]))
    result.ok = True
    result.aligned = result.error_m <= config.converged_m
    return result


def correction_in_base(result: AlignmentResult, base_from_board: np.ndarray) -> np.ndarray:
    """Translation (arm base frame) that moves the tool tip onto the target."""
    if not result.ok:
        raise ValueError(f"no correction from a failed alignment: {result.failure_reason}")
    return base_from_board[:3, :3] @ np.array([*result.error_board_m, 0.0])


def _rectify(gray, camera, camera_from_board, center, half, res):
    """Board-plane resampling around center: column j is x, row i is -y (y up)."""
    steps = np.arange(-half, half + 1) * res
    xs, ys = np.meshgrid(center[0] + steps, center[1] - steps)
    points = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])
    camera_points = transforms.apply(camera_from_board, points)
    if np.any(camera_points[:, 2] <= 0):
        return None
    px = camera.project(camera_points).reshape(xs.shape + (2,)).astype(np.float32)
    h, w = gray.shape
    if (
        px[..., 0].min() < 0
        or px[..., 1].min() < 0
        or px[..., 0].max() > w - 1
        or px[..., 1].max() > h - 1
    ):
        return None
    image = gray if gray.dtype == np.float32 else gray.astype(np.float32)
    return cv2.remap(image, px[..., 0], px[..., 1], cv2.INTER_LINEAR)


def _render(pads, center, n, res, pads_are_brighter):
    """Expected copper around center at res, supersampled for sub-pixel pad edges."""
    supersample = 4
    size = (2 * n + 1) * supersample
    canvas = np.zeros((size, size), np.uint8)
    scale = supersample / res
    origin = np.array([center[0] - (n + 0.5) * res, center[1] + (n + 0.5) * res])
    reach = (n + 1) * res
    for pad in pads:
        if np.abs(pad.outline - center).min(axis=0).max() > reach:
            continue
        local = (pad.outline - origin) * [scale, -scale] - 0.5  # pixel centres
        cv2.fillPoly(
            canvas, [np.round(local * 16).astype(np.int32).reshape(-1, 1, 2)], 255, shift=4
        )
    small = cv2.resize(canvas, (2 * n + 1, 2 * n + 1), interpolation=cv2.INTER_AREA).astype(
        np.float32
    )
    return small if pads_are_brighter else 255.0 - small


def _min_pad_px(pads, center, n, res) -> int:
    nearby = [p.min_size for p in pads if np.linalg.norm(p.center - center) <= (n + 1) * res]
    return math.ceil(min(nearby) / 2 / res) if nearby else 0


def _subpixel(values) -> float:
    a, b, c = (float(v) for v in values)
    denom = a - 2 * b + c
    return 0.5 * (a - c) / denom if denom < 0 else 0.0
