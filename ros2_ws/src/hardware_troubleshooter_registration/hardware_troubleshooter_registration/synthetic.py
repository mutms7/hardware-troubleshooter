"""Synthetic board images rendered through a known camera model, for tests and playback.

Every number in this module describes an imaginary rig. None of it is a measurement
of real hardware, and none of it may be copied into a live board or rig config.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .geometry import board_to_camera

SYNTHETIC_CAMERA_INFO = {
    "image_width": 1280,
    "image_height": 960,
    "camera_name": "synthetic",
    "camera_matrix": {"rows": 3, "cols": 3, "data": [1100.0, 0, 640.0, 0, 1100.0, 480.0, 0, 0, 1]},
    "distortion_model": "plumb_bob",
    "distortion_coefficients": {"rows": 1, "cols": 5, "data": [-0.12, 0.05, 0.0005, -0.0003, 0.0]},
    "rectification_matrix": {"rows": 3, "cols": 3, "data": [1, 0, 0, 0, 1, 0, 0, 0, 1]},
    "projection_matrix": {
        "rows": 3,
        "cols": 4,
        "data": [1100.0, 0, 640.0, 0, 0, 1100.0, 480.0, 0, 0, 0, 1, 0],
    },
}

_TILT = math.radians(8.0)
SYNTHETIC_BOARD = {
    "board_id": "synthetic_board",
    "frame_id": "synthetic_board",
    "thickness_m": 0.0016,
    "outline_source": "measured",
    "outline_m": [[0.0, 0.0], [0.1, 0.0], [0.1, 0.06], [0.0, 0.06]],
    "features": [
        {
            "name": "mounting_hole",
            "shape": "circle",
            "center_m": [0.008, 0.008],
            "radius_m": 0.002,
            "expect": "darker",
        }
    ],
    "overlay_targets": [{"name": "TP_SYNTH", "position_m": [0.07, 0.04]}],
}
SYNTHETIC_RIG = {
    "base_frame": "synthetic_base",
    "bench_plane": {"normal": [0.0, math.sin(_TILT), math.cos(_TILT)], "distance_m": 0.45},
    "segmentation": {"board_is_brighter": True},
    "thresholds": {
        "max_fit_residual_px": 1.5,
        "move_translation_m": 0.002,
        "move_rotation_rad": 0.01,
        "tracking_loss_timeout_s": 1.0,
        "feature_min_contrast": 0.15,
    },
    "camera_extrinsic": {
        "camera_frame": "synthetic_camera_optical",
        "translation_m": [0.3, 0.0, 0.5],
        "rotation_xyzw": [1.0, 0.0, 0.0, 0.0],
    },
}


def render_frame(
    board,
    camera,
    plane,
    pose,
    *,
    board_gray=170,
    background_gray=60,
    show_features=True,
    occluders=(),
    occluder_gray=110,
    supersample=4,
    noise_sigma=0.0,
    seed=0,
) -> np.ndarray:
    """Grayscale image of the board top surface at an in-plane pose (x, y, theta)."""
    rotation, translation = plane.pose_in_camera(*pose)
    canvas = np.full(
        (camera.height * supersample, camera.width * supersample), background_gray, np.uint8
    )

    def fill(points_px, value):
        # Map low-res pixel coordinates to high-res pixel centres, then fill sub-pixel.
        hi = (np.asarray(points_px) + 0.5) * supersample - 0.5
        pts = np.round(hi * 16).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(canvas, [pts], int(value), lineType=cv2.LINE_8, shift=4)

    def project(points):
        return camera.project(board_to_camera(points, rotation, translation))

    fill(project(board.outline.polyline), board_gray)
    if show_features:
        for feature in board.features:
            value = background_gray if feature.expect == "darker" else min(255, board_gray + 70)
            fill(project(feature.outline), value)
    for polygon in occluders:
        fill(polygon, occluder_gray)
    image = cv2.resize(canvas, (camera.width, camera.height), interpolation=cv2.INTER_AREA)
    if noise_sigma > 0:
        noise = np.random.default_rng(seed).normal(0, noise_sigma, image.shape)
        image = np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return image
