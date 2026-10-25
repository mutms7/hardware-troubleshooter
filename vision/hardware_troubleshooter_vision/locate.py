"""Where named pads appear in a camera image, from the PCB file and the current pose.

A pad's board-frame outline goes through camera <- board (registration for the fixed
camera, registration plus FK for a wrist camera) and the camera model. The pixel spread
combines the registration residual with a board-frame position uncertainty. This is a
prediction for finding the pad, not a measurement: it never establishes where the probe is,
and identity always comes from the pad name, never from the image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import transforms
from .pads import Pad


@dataclass
class Located:
    name: str
    reasons: list = field(default_factory=list)
    center_px: np.ndarray | None = None
    outline_px: np.ndarray | None = None
    sigma_px: float | None = None
    px_per_mm: float | None = None
    in_image: bool = False
    occluded_fraction: float | None = None

    @property
    def ok(self) -> bool:
        return not self.reasons

    def as_dict(self) -> dict:
        return {
            "pad_id": self.name,
            "ok": self.ok,
            "reasons": list(self.reasons),
            "center_px": None if self.center_px is None else [float(v) for v in self.center_px],
            "outline_px": None
            if self.outline_px is None
            else [[float(u), float(v)] for u, v in self.outline_px],
            "sigma_px": self.sigma_px,
            "px_per_mm": self.px_per_mm,
            "in_image": self.in_image,
            "occluded_fraction": self.occluded_fraction,
        }


def locate(
    pads: dict[str, Pad],
    names: list[str],
    camera,
    camera_from_board: np.ndarray,
    sigma_board_m: float,
    residual_px: float | None = None,
    mask: np.ndarray | None = None,
) -> list[Located]:
    """`mask` is nonzero where the view is blocked (the arms); None skips that check."""
    out = []
    for name in names:
        found = Located(name)
        out.append(found)
        pad = pads.get(name)
        if pad is None:
            found.reasons.append("unknown_pad")
            continue
        board = np.vstack(
            [
                pad.center,
                pad.center + [1e-3, 0.0],
                pad.center + [0.0, 1e-3],
                pad.outline,
            ]
        )
        points = transforms.apply(camera_from_board, np.column_stack([board, np.zeros(len(board))]))
        if np.any(points[:, 2] <= 0):
            found.reasons.append("behind_camera")
            continue
        pixels = camera.project(points)
        center, step_x, step_y = pixels[0], pixels[1], pixels[2]
        found.center_px = center
        found.outline_px = pixels[3:]
        found.px_per_mm = float(
            (np.linalg.norm(step_x - center) + np.linalg.norm(step_y - center)) / 2
        )
        found.sigma_px = float(
            math.hypot(residual_px or 0.0, sigma_board_m * 1000.0 * found.px_per_mm)
        )
        u, v = center
        found.in_image = 0 <= u < camera.width and 0 <= v < camera.height
        if not found.in_image:
            found.reasons.append("outside_image")
            continue
        if mask is not None:
            inside = [
                (round(pu), round(pv))
                for pu, pv in np.vstack([center[None], found.outline_px])
                if 0 <= pu < mask.shape[1] and 0 <= pv < mask.shape[0]
            ]
            blocked = sum(mask[pv, pu] > 0 for pu, pv in inside)
            found.occluded_fraction = blocked / len(inside)
            if mask[int(v), int(u)] > 0:
                found.reasons.append("occluded")
    return out


def draw(image: np.ndarray, located: list[Located]) -> np.ndarray:
    """Outline, centre cross, 2-sigma circle and name for each located pad. Debug only."""
    canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for item in located:
        if item.center_px is None:
            continue
        colour = (0, 220, 0) if item.ok else (0, 0, 255)
        outline = np.round(item.outline_px).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [outline], True, colour, 1, cv2.LINE_AA)
        center = tuple(round(c) for c in item.center_px)
        cv2.drawMarker(canvas, center, colour, cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)
        if item.sigma_px:
            cv2.circle(
                canvas, center, max(1, round(2 * item.sigma_px)), colour, 1, cv2.LINE_AA
            )
        cv2.putText(
            canvas,
            item.name,
            (center[0] + 6, center[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            colour,
            1,
            cv2.LINE_AA,
        )
    return canvas


def encode_png(image: np.ndarray) -> bytes:
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("PNG encoding failed")
    return data.tobytes()
