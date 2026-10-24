"""4x4 rigid transforms. `a_from_b` maps points in frame b to frame a."""

from __future__ import annotations

import numpy as np
from hardware_troubleshooter_arm.kinematics import rpy_matrix, transform
from hardware_troubleshooter_registration.geometry import (
    quaternion_to_rotation,
    rotation_to_quaternion,
)

__all__ = [
    "apply",
    "from_pose",
    "from_quaternion",
    "invert",
    "rotation_angle",
    "to_quaternion",
    "transform",
]


def from_pose(pose) -> np.ndarray:
    """((x, y, z), (roll, pitch, yaw)), as the arm config stores URDF origins."""
    xyz, rpy = pose
    return transform(rpy_matrix(*rpy), xyz)


def from_quaternion(translation, quaternion_xyzw) -> np.ndarray:
    return transform(quaternion_to_rotation(quaternion_xyzw), translation)


def to_quaternion(matrix: np.ndarray) -> tuple[np.ndarray, tuple]:
    """(translation, quaternion xyzw), the form RegistrationEngine.set_extrinsic takes."""
    return matrix[:3, 3].copy(), rotation_to_quaternion(matrix[:3, :3])


def invert(matrix: np.ndarray) -> np.ndarray:
    rotation, translation = matrix[:3, :3], matrix[:3, 3]
    return transform(rotation.T, -rotation.T @ translation)


def apply(matrix: np.ndarray, points) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def rotation_angle(rotation: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0)))
