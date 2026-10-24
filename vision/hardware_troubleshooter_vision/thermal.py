"""AMG8833 8x8 thermal frames placed on the registered board.

A homography maps thermal cell coordinates (column, row; cell centres at integers) to
fixed camera ideal pixels at the board's top surface. Calibrate it once from four or
more warm points seen by both (thermal_homography). With the board registered, each
cell then has a board position, and the hottest region can be named by nearby pads.

At 8x8 over a whole board a cell is centimetres wide: this is a hint about which part
runs hot, never a measurement on a pad.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import transforms
from .config import ThermalConfig

GRID = 8


@dataclass(frozen=True)
class HotSpot:
    position_m: np.ndarray  # board frame
    celsius: float
    delta_c: float  # above the median of the cells on the board
    nearby_pads: tuple[str, ...]


def thermal_homography(cells, camera_ideal_px) -> np.ndarray:
    """Thermal (col, row) -> fixed camera ideal pixel, from >= 4 correspondences."""
    cells = np.asarray(cells, np.float64).reshape(-1, 2)
    pixels = np.asarray(camera_ideal_px, np.float64).reshape(-1, 2)
    if len(cells) < 4 or len(cells) != len(pixels):
        raise ValueError("need at least four matching points")
    homography, _ = cv2.findHomography(cells, pixels, 0)
    if homography is None:
        raise ValueError("points are degenerate (collinear?)")
    return homography


def cells_on_board(homography, camera, camera_from_board) -> np.ndarray:
    """(8, 8, 2) board-frame position of each cell centre; NaN where a ray misses."""
    rows, cols = np.mgrid[0:GRID, 0:GRID]
    cells = np.column_stack([cols.ravel(), rows.ravel(), np.ones(GRID * GRID)])
    ideal = cells @ homography.T
    ideal = ideal[:, :2] / ideal[:, 2:3]
    rays = camera.ideal_to_rays(ideal)
    board_from_camera = transforms.invert(camera_from_board)
    origin, directions = board_from_camera[:3, 3], rays @ board_from_camera[:3, :3].T
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -origin[2] / directions[:, 2]
    points = origin[:2] + directions[:, :2] * t[:, None]
    points[~(t > 0)] = np.nan
    return points.reshape(GRID, GRID, 2)


def hot_spot(
    celsius, positions, outline, pads, config: ThermalConfig, nearby=3
) -> tuple[HotSpot | None, str | None]:
    """The hottest on-board cell if it stands min_delta_c above the board's median."""
    missing = config.missing()
    if missing:
        return None, "; ".join(f"unconfigured:{key}" for key in missing)
    celsius = np.asarray(celsius, float).reshape(GRID, GRID)
    flat = positions.reshape(-1, 2)
    path = outline.astype(np.float32).reshape(-1, 1, 2)
    on_board = np.array(
        [
            np.all(np.isfinite(p)) and cv2.pointPolygonTest(path, tuple(map(float, p)), False) >= 0
            for p in flat
        ]
    )
    if not on_board.any():
        return None, "thermal_off_board:no thermal cell lands on the board"
    temps = celsius.ravel()
    median = float(np.median(temps[on_board]))
    best = int(np.argmax(np.where(on_board, temps, -np.inf)))
    delta = float(temps[best] - median)
    if delta < config.min_delta_c:
        return None, None  # nothing stands out; not a failure
    position = flat[best]
    names = tuple(
        p.name
        for p in sorted(pads, key=lambda p: float(np.linalg.norm(p.center - position)))[:nearby]
    )
    return HotSpot(position, float(temps[best]), delta, names), None
