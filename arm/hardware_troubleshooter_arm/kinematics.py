"""Forward kinematics over a URDF tree, and SDK joint positions to URDF joint positions.

FK only places frames. It reads joint positions it is given; it doesn't know whether
they were measured, and it doesn't model compliance, backlash or zero error."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .config import ArmCalibration, ArmConfig, JointMap
from .urdf import TOOLHEAD_JOINT, prefixed
from .vendor import ARM_JOINTS, SDK_JOINT_COUNT


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + s * k + (1 - c) * (k @ k)


def transform(rotation: np.ndarray, translation) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = translation
    return out


@dataclass(frozen=True)
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    mimic: tuple[str, float, float] | None


class Kinematics:
    def __init__(self, joints: Sequence[Joint], root: str):
        self.joints = {joint.name: joint for joint in joints}
        self.root = root
        self._by_parent: dict[str, list[Joint]] = {}
        for joint in joints:
            self._by_parent.setdefault(joint.parent, []).append(joint)

    @classmethod
    def from_urdf(cls, text: str) -> Kinematics:
        robot = ET.fromstring(text)
        joints = []
        for element in robot.findall("joint"):
            origin = element.find("origin")
            xyz = [
                float(v)
                for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()
            ]
            rpy = [
                float(v)
                for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()
            ]
            axis_element = element.find("axis")
            axis = np.array(
                [
                    float(v)
                    for v in (
                        axis_element.get("xyz") if axis_element is not None else "1 0 0"
                    ).split()
                ]
            )
            mimic = element.find("mimic")
            joints.append(
                Joint(
                    element.get("name"),
                    element.get("type"),
                    element.find("parent").get("link"),
                    element.find("child").get("link"),
                    transform(rpy_matrix(*rpy), xyz),
                    axis / np.linalg.norm(axis),
                    (
                        mimic.get("joint"),
                        float(mimic.get("multiplier", 1)),
                        float(mimic.get("offset", 0)),
                    )
                    if mimic is not None
                    else None,
                )
            )
        children = {joint.child for joint in joints}
        roots = [
            link.get("name") for link in robot.findall("link") if link.get("name") not in children
        ]
        if len(roots) != 1:
            raise ValueError(f"URDF must have exactly one root link, found {roots}")
        return cls(joints, roots[0])

    @property
    def movable(self) -> list[str]:
        return [
            name
            for name, joint in self.joints.items()
            if joint.kind in ("revolute", "continuous", "prismatic") and joint.mimic is None
        ]

    def link_poses(self, positions: Mapping[str, float]) -> dict[str, np.ndarray]:
        """Every link's 4x4 pose in the root link frame."""
        missing = [name for name in self.movable if name not in positions]
        if missing:
            raise ValueError(f"missing joint positions: {missing}")
        bad = [name for name in self.movable if not math.isfinite(positions[name])]
        if bad:
            raise ValueError(f"non-finite joint positions: {bad}")
        poses = {self.root: np.eye(4)}
        stack = [self.root]
        while stack:
            parent = stack.pop()
            for joint in self._by_parent.get(parent, []):
                poses[joint.child] = poses[parent] @ joint.origin @ self._motion(joint, positions)
                stack.append(joint.child)
        return poses

    def _motion(self, joint: Joint, positions: Mapping[str, float]) -> np.ndarray:
        if joint.kind == "fixed":
            return np.eye(4)
        if joint.mimic:
            source, multiplier, offset = joint.mimic
            q = multiplier * positions[source] + offset
        else:
            q = positions[joint.name]
        if joint.kind == "prismatic":
            return transform(np.eye(3), joint.axis * q)
        if joint.kind in ("revolute", "continuous"):
            return transform(axis_angle_matrix(joint.axis, q), (0, 0, 0))
        raise ValueError(f"joint {joint.name}: unsupported type {joint.kind!r}")


def urdf_joint_positions(
    config: ArmConfig,
    sdk_positions: Sequence[float],
    calibration: ArmCalibration | None = None,
) -> dict[str, float]:
    """Map the SDK's seven joint positions (J1..J6, then motor 7) to this arm's URDF joints.

    The calibrated joint map wins over the nominal one. Raises ValueError when neither
    is set."""
    joint_map: tuple[JointMap, ...] | None = (
        calibration.joint_map if calibration and calibration.joint_map else config.joint_map
    )
    if joint_map is None:
        raise ValueError("arm.joint_map is unset and the calibration has none")
    if len(sdk_positions) != SDK_JOINT_COUNT:
        raise ValueError(
            f"expected {SDK_JOINT_COUNT} SDK joint positions, got {len(sdk_positions)}"
        )
    names = [prefixed(config, name) for name in (*ARM_JOINTS, TOOLHEAD_JOINT)]
    return {name: entry.to_urdf(q) for name, entry, q in zip(names, joint_map, sdk_positions)}
