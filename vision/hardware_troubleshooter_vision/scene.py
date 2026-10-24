"""Fixed-camera scene guard: anything in the workspace that wasn't there when it was empty.

Compare against a reference frame of the empty rig, after matching overall brightness.
The arms (their projected masks) and a band around the board are left out: the arms
are known, and the board is registration's to watch. What's left and changed, in a
region at least min_blob_area_px, is an obstruction: a hand, a cable, a dropped part.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import SceneConfig

_OPEN = np.ones((3, 3), np.uint8)


@dataclass
class SceneResult:
    clear: bool
    reasons: list[str] = field(default_factory=list)
    blobs: list[tuple[int, tuple[int, int, int, int]]] = field(default_factory=list)  # area, box


class SceneGuard:
    def __init__(self, config: SceneConfig):
        self.config = config
        self.reference: np.ndarray | None = None

    def set_reference(self, gray: np.ndarray) -> None:
        """Capture with no arm over the bench and nothing but the clamped board on it."""
        self.reference = _gray(gray).astype(np.float32)

    def check(self, gray, arm_mask=None, board_polygon_px=None) -> SceneResult:
        missing = self.config.missing()
        if missing:
            return SceneResult(False, [f"unconfigured:{key}" for key in missing])
        if self.reference is None:
            return SceneResult(False, ["no_reference:capture the empty rig first"])
        image = _gray(gray).astype(np.float32)
        if image.shape != self.reference.shape:
            return SceneResult(False, ["invalid:image size differs from the reference"])

        ignore = np.zeros(image.shape, np.uint8)
        if arm_mask is not None:
            ignore |= (arm_mask > 0).astype(np.uint8) * 255
        if board_polygon_px is not None:
            polygon = np.round(np.asarray(board_polygon_px)).astype(np.int32).reshape(-1, 1, 2)
            board = np.zeros_like(ignore)
            cv2.fillPoly(board, [polygon], 255)
            size = 2 * math.ceil(self.config.board_margin_px) + 1
            ignore |= cv2.dilate(board, np.ones((size, size), np.uint8))
        keep = ignore == 0
        if not keep.any():
            return SceneResult(False, ["scene_unobservable:everything is masked"])

        # Match brightness on what's watched, so a lighting change isn't an obstruction.
        gain = float(np.median(self.reference[keep])) / max(float(np.median(image[keep])), 1.0)
        changed = (np.abs(image * gain - self.reference) > self.config.diff_threshold) & keep
        changed = cv2.morphologyEx(changed.astype(np.uint8) * 255, cv2.MORPH_OPEN, _OPEN)
        count, _, stats, _ = cv2.connectedComponentsWithStats(changed)
        blobs = [
            (int(stats[i, cv2.CC_STAT_AREA]), tuple(int(v) for v in stats[i, :4]))
            for i in range(1, count)
            if stats[i, cv2.CC_STAT_AREA] >= self.config.min_blob_area_px
        ]
        if blobs:
            area, (x, y, w, h) = max(blobs)
            return SceneResult(False, [f"scene_obstructed:{area} px at x={x} y={y} {w}x{h}"], blobs)
        return SceneResult(True)


def _gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
