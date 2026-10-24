"""The rig's frames: the fixed camera and each arm, all in the registration base frame.

`rig` is the frame RegistrationEngine reports in (rig.base_frame). Each arm's
base_link sits at a calibrated pose in it, and the fixed camera's extrinsic is
expressed in it. Every arm-relative quantity here is computed from measured joints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from hardware_troubleshooter_arm.config import (
    ArmCalibration,
    ArmConfig,
    load_arm_calibration,
    load_arm_config,
)
from hardware_troubleshooter_arm.kinematics import Kinematics, urdf_joint_positions
from hardware_troubleshooter_arm.urdf import CAMERA_LINK, build_urdf, prefixed, tool_link
from hardware_troubleshooter_registration.camera import CameraModel

from . import transforms
from .config import ArmEntry
from .occlusion import ArmGeometry


@dataclass
class ArmRuntime:
    name: str
    config: ArmConfig
    calibration: ArmCalibration | None
    kinematics: Kinematics
    geometry: ArmGeometry
    rig_from_base: np.ndarray
    wrist_camera: CameraModel

    @classmethod
    def load(cls, entry: ArmEntry) -> tuple[ArmRuntime | None, list[str]]:
        problems = [f"unconfigured:{key}" for key in entry.missing()]
        if problems:
            return None, problems
        config, problems = load_arm_config(str(entry.arm_config))
        calibration, calibration_problems = load_arm_calibration(
            str(entry.calibration) if entry.calibration else None
        )
        problems += calibration_problems
        if config.name is not None and config.name != entry.name:
            problems.append(f"invalid:arms.{entry.name} (arm config is named {config.name!r})")
        problems += [f"unconfigured:{key}" for key in config.missing_for_joint_map()]
        if problems:
            return None, problems
        # Mesh paths are written relative to the arm config's folder, and read back from it.
        urdf_dir = Path(entry.arm_config).parent
        urdf, problems = build_urdf(config, calibration, urdf_dir)
        if urdf is None:
            return None, problems
        camera, problem = _camera(entry.wrist_camera_info)
        if camera is None:
            return None, [f"arms.{entry.name}.wrist_camera_info:{problem}"]
        return cls(
            entry.name,
            config,
            calibration,
            Kinematics.from_urdf(urdf),
            ArmGeometry.from_urdf(urdf, urdf_dir),
            transforms.from_pose(entry.base_in_rig),
            camera,
        ), []

    def link_poses(self, sdk_positions) -> dict[str, np.ndarray]:
        """Arm base -> every link, from the SDK's seven measured positions."""
        q = urdf_joint_positions(self.config, list(sdk_positions), self.calibration)
        return self.kinematics.link_poses(q)

    def base_from_wrist_camera(self, poses) -> np.ndarray:
        return poses[prefixed(self.config, CAMERA_LINK)]

    def base_from_tool(self, poses, tool: str) -> np.ndarray:
        return poses[prefixed(self.config, tool_link(tool))]

    def toolhead_angle(self, sdk_positions) -> float:
        q = urdf_joint_positions(self.config, list(sdk_positions), self.calibration)
        return q[prefixed(self.config, "toolhead_joint")]


def fixed_camera_extrinsic(rig_yaml: Path) -> tuple[np.ndarray | None, str | None, str | None]:
    """(rig <- fixed camera, camera_frame, problem) from the registration rig YAML's
    camera_extrinsic block."""
    try:
        data = yaml.safe_load(Path(rig_yaml).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        return None, None, f"invalid:rig_config ({exc})"
    block = data.get("camera_extrinsic") or {}
    frame = block.get("camera_frame")
    translation, rotation = block.get("translation_m"), block.get("rotation_xyzw")
    if not frame or translation is None or rotation is None:
        return None, frame, "extrinsic_unavailable:rig camera_extrinsic is unset"
    try:
        translation = np.asarray(translation, dtype=float).reshape(3)
        rotation = np.asarray(rotation, dtype=float).reshape(4)
        if not (np.all(np.isfinite(translation)) and np.all(np.isfinite(rotation))):
            raise ValueError("not finite")
        if np.linalg.norm(rotation) == 0:
            raise ValueError("zero quaternion")
    except (TypeError, ValueError) as exc:
        return None, frame, f"invalid:rig.camera_extrinsic ({exc})"
    return transforms.from_quaternion(translation, rotation), frame, None


def _camera(path) -> tuple[CameraModel | None, str | None]:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        return None, f"camera_info_invalid:{exc}"
    return CameraModel.from_calibration_yaml(data)
