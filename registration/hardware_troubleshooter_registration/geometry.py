"""Planar geometry helpers. Frames are right-handed, lengths are metres, angles radians."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def rot2(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def polygon_area(points: np.ndarray) -> float:
    """Signed area; positive for counter-clockwise vertices."""
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def line_intersection(p0, d0, p1, d1) -> np.ndarray | None:
    """Intersection of the 2D lines p0 + s*d0 and p1 + u*d1, or None when parallel."""
    p0, d0, p1, d1 = (np.asarray(v, dtype=float) for v in (p0, d0, p1, d1))
    cross = d0[0] * d1[1] - d0[1] * d1[0]
    if abs(cross) <= 1e-12 * np.linalg.norm(d0) * np.linalg.norm(d1):
        return None
    diff = p1 - p0
    s = (diff[0] * d1[1] - diff[1] * d1[0]) / cross
    return p0 + s * d0


def kabsch_2d(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray]:
    """Rigid (rotation + translation, no scale) least-squares fit dst ~ R(theta) src + t."""
    src_c, dst_c = src.mean(axis=0), dst.mean(axis=0)
    h = (src - src_c).T @ (dst - dst_c)
    theta = math.atan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
    return theta, dst_c - rot2(theta) @ src_c


def board_to_camera(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray):
    """Board-frame points on the top surface (z=0) expressed in the camera frame."""
    points = np.asarray(points, dtype=float)
    return np.column_stack([points, np.zeros(len(points))]) @ rotation.T + translation


def rotation_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Quaternion (x, y, z, w) of a proper rotation matrix."""
    m = rotation
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q = ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, s / 4)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = (s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s)
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = ((m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s)
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = ((m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4, (m[1, 0] - m[0, 1]) / s)
    norm = math.sqrt(sum(v * v for v in q))
    return tuple(v / norm for v in q)


@dataclass(frozen=True)
class PlaneFrame:
    """A plane parallel to the calibrated bench, with an in-plane 2D coordinate frame.

    The bench is n.X = d in the camera optical frame, with n pointing from the camera
    toward the bench. The board's top surface is the same plane lifted by the board
    thickness. The frame's z axis points back toward the camera (out of the board),
    its origin is the foot of the perpendicular from the camera, and its x axis is the
    camera x axis projected onto the plane.
    """

    origin: np.ndarray
    axes: np.ndarray  # columns: plane x, y, z expressed in the camera frame
    normal: np.ndarray
    offset: float

    @classmethod
    def from_bench(cls, normal, distance_m: float, height_m: float) -> PlaneFrame:
        n = np.asarray(normal, dtype=float)
        n = n / np.linalg.norm(n)
        offset = float(distance_m) - float(height_m)
        if offset <= 0:
            raise ValueError("board top surface isn't in front of the camera")
        z = -n
        x = np.array([1.0, 0.0, 0.0])
        x = x - x.dot(z) * z
        if np.linalg.norm(x) < 1e-6:
            x = np.array([0.0, 1.0, 0.0]) - z[1] * z
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        return cls(offset * n, np.column_stack([x, y, z]), n, offset)

    def rays_to_plane(self, rays: np.ndarray) -> np.ndarray | None:
        """Plane coordinates where camera rays hit the plane; None if any ray misses."""
        denom = rays @ self.normal
        if np.any(denom <= 1e-9):
            return None
        points = rays * (self.offset / denom)[:, None]
        return (points - self.origin) @ self.axes[:, :2]

    def pose_in_camera(self, x: float, y: float, theta: float):
        """Board-to-camera rotation and translation for an in-plane pose."""
        c, s = math.cos(theta), math.sin(theta)
        rotation = self.axes @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        translation = self.origin + x * self.axes[:, 0] + y * self.axes[:, 1]
        return rotation, translation
