"""Synthetic wrist-camera views of a board's copper. Test-only values throughout."""

import math

import cv2
import numpy as np
from hardware_troubleshooter_vision import transforms
from hardware_troubleshooter_vision.config import AlignmentConfig
from hardware_troubleshooter_vision.pads import Pad, circle_outline, rect_outline

TARGET = "U1.1"
TARGET_CENTER = np.array([0.010, 0.010])

ALIGNMENT = AlignmentConfig(
    pads_are_brighter=True,
    patch_radius_m=0.0025,
    resolution_m=0.0001,
    search_radius_m=0.0015,
    min_match_score=0.6,
    min_uniqueness_margin=0.1,
    max_standoff_error_m=0.003,
    converged_m=0.0002,
)


def cluster_pads() -> list[Pad]:
    """An irregular group of pads around the target, so only one offset matches."""
    c = TARGET_CENTER
    specs = [
        (TARGET, rect_outline(c, (0.0008, 0.0006), 0.0)),
        ("U1.2", rect_outline(c + [0.0013, 0.0002], (0.0006, 0.0010), 0.0)),
        ("R1.1", rect_outline(c + [-0.0010, 0.0014], (0.0012, 0.0005), 0.3)),
        ("C1.1", circle_outline(c + [-0.0012, -0.0011], 0.0004)),
        ("TP9", rect_outline(c + [0.0007, -0.0016], (0.0004, 0.0004), 0.0)),
    ]
    return [Pad(name, outline.mean(axis=0), outline) for name, outline in specs]


def row_pads(pitch=0.001, count=15) -> list[Pad]:
    """Identical pads on a regular pitch, longer than the template: ambiguous by design."""
    pads = []
    for i in range(count):
        center = TARGET_CENTER + [(i - count // 2) * pitch, 0.0]
        name = TARGET if i == count // 2 else f"J1.{i}"
        pads.append(Pad(name, center, rect_outline(center, (0.0005, 0.0008), 0.0)))
    return pads


def board_from_camera(x, y, height, tilt_rad=0.0) -> np.ndarray:
    """Camera looking down at the board: optical z = -board z, image y = -board y."""
    c, s = math.cos(tilt_rad), math.sin(tilt_rad)
    tilt = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return transforms.transform(np.diag([1.0, -1.0, -1.0]) @ tilt, [x, y, height])


def shifted(matrix, dx, dy, dtheta=0.0) -> np.ndarray:
    """board <- camera after the board itself moves by (dx, dy, dtheta) in its plane."""
    c, s = math.cos(dtheta), math.sin(dtheta)
    move = transforms.transform(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]), [dx, dy, 0])
    return transforms.invert(move) @ matrix


def render(camera, camera_from_board, pads, mask_gray=60, copper_gray=180, supersample=2):
    h, w = camera.height * supersample, camera.width * supersample
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    raw = np.column_stack(
        [(us.ravel() + 0.5) / supersample - 0.5, (vs.ravel() + 0.5) / supersample - 0.5]
    )
    rays = camera.ideal_to_rays(camera.undistort_px(raw))
    board_from_cam = transforms.invert(camera_from_board)
    origin, directions = board_from_cam[:3, 3], rays @ board_from_cam[:3, :3].T
    t = -origin[2] / directions[:, 2]
    points = origin[:2] + directions[:, :2] * t[:, None]

    res = 0.00001
    lo = TARGET_CENTER - 0.02
    raster = np.full((int(0.04 / res), int(0.04 / res)), mask_gray, np.uint8)
    for pad in pads:
        local = (pad.outline - lo) / res
        cv2.fillPoly(
            raster, [np.round(local * 16).astype(np.int32).reshape(-1, 1, 2)], copper_gray, shift=4
        )
    map_xy = ((points - lo) / res).astype(np.float32).reshape(h, w, 2)
    image = cv2.remap(
        raster, map_xy[..., 0], map_xy[..., 1], cv2.INTER_LINEAR, borderValue=mask_gray
    )
    return cv2.resize(image, (camera.width, camera.height), interpolation=cv2.INTER_AREA)


def project(camera, camera_from_board, board_xy) -> np.ndarray:
    return camera.project(transforms.apply(camera_from_board, [[*board_xy, 0.0]]))[0]
