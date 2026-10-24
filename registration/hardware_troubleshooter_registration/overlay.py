"""Debug overlay. A drawing aid only; it verifies nothing about the arm or contact."""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import board_to_camera

_GREEN, _RED, _YELLOW, _MAGENTA, _CYAN, _BLUE = (
    (0, 200, 0),
    (0, 0, 255),
    (0, 220, 220),
    (255, 0, 255),
    (255, 255, 0),
    (255, 80, 0),
)


def draw_overlay(image, result, snapshot, board, camera) -> np.ndarray:
    out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    if result is not None and result.contour_px is not None:
        cv2.polylines(out, [result.contour_px.astype(np.int32).reshape(-1, 1, 2)], True, _YELLOW, 1)

    pose_ok = (
        result is not None
        and result.rotation is not None
        and board is not None
        and camera is not None
    )
    if pose_ok:
        rotation, translation = result.rotation, result.translation

        def project(points):
            return camera.project(board_to_camera(points, rotation, translation))

        outline = project(board.outline.polyline)
        cv2.polylines(out, [_int(outline)], True, _GREEN, 2, cv2.LINE_AA)
        for corner in project(board.corners):
            cv2.circle(out, _pt(corner), 4, _RED, 1, cv2.LINE_AA)
        for feature in board.features:
            cv2.polylines(out, [_int(project(feature.outline))], True, _MAGENTA, 1, cv2.LINE_AA)
            label = project(feature.outline.mean(axis=0, keepdims=True))[0]
            _text(out, feature.name, label + (6, -6), _MAGENTA)
        centre = board.corners.mean(axis=0)
        span = float(np.ptp(board.corners[:, 0])) / 4
        start, tip = project(np.array([centre, centre + (span, 0.0)]))
        cv2.arrowedLine(out, _pt(start), _pt(tip), _BLUE, 2, cv2.LINE_AA, tipLength=0.2)
        _text(out, "+x", tip + (4, 4), _BLUE)
        for target in board.targets:
            p = project(target.position.reshape(1, 2))[0]
            cv2.drawMarker(out, _pt(p), _CYAN, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
            _text(out, target.name, p + (8, 14), _CYAN)

    lines = ["DEBUG OVERLAY - not verification"]
    if snapshot is not None:
        lines.append(f"{snapshot.status}  rev={snapshot.registration_id or '-'}")
        lines.append(
            f"calibration_valid={snapshot.calibration_valid}  tracking_valid={snapshot.tracking_valid}"
        )
    if result is not None:
        residual = "-" if result.residual_px is None else f"{result.residual_px:.3f}px"
        lines.append(
            f"frame ok={result.ok} residual={residual} edge_points={result.edge_points} "
            f"symmetry_resolved={result.symmetry_resolved}"
        )
    if snapshot is not None:
        lines += snapshot.reasons[:4]
    for i, line in enumerate(lines):
        _text(out, line, np.array([10.0, 22.0 + 20 * i]), _RED if i == 0 else (255, 255, 255))
    return out


def _int(points):
    return np.round(points).astype(np.int32).reshape(-1, 1, 2)


def _pt(point):
    return int(np.rint(point[0])), int(np.rint(point[1]))


def _text(image, text, origin, colour):
    org = _pt(origin)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
