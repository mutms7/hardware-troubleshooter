"""Per-arm toolhead and calibration YAML loading. Unknown values stay None; nothing is
filled in with defaults."""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .vendor import SDK_JOINT_COUNT

NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
SDK7_KEYS = ("direction", "offset", "lo", "hi", "kp", "kd")

# ((x, y, z) metres, (roll, pitch, yaw) radians), as in a URDF <origin>.
Pose = tuple[tuple[float, float, float], tuple[float, float, float]]


@dataclass(frozen=True)
class Tool:
    name: str
    select_rad: float | None = None
    tip: Pose | None = None


@dataclass(frozen=True)
class JointMap:
    """URDF joint position = scale * SDK joint position + offset_rad."""

    scale: float
    offset_rad: float

    def to_urdf(self, sdk_position: float) -> float:
        return self.scale * sdk_position + self.offset_rad


@dataclass(frozen=True)
class SdkJoint7:
    """Replacement for the gripper entry (motor 7) in the Maker SDK profile."""

    direction: int | None = None
    offset: float | None = None
    lo: float | None = None
    hi: float | None = None
    kp: float | None = None
    kd: float | None = None


@dataclass(frozen=True)
class ArmConfig:
    name: str | None = None
    remove_link_007_meshes: tuple[str, ...] | None = None
    flange: Pose | None = None
    toolhead_mesh: Path | None = None
    toolhead_mesh_scale: tuple[float, float, float] | None = None
    toolhead_origin: Pose | None = None
    toolhead_axis: tuple[float, float, float] | None = None
    toolhead_lower_rad: float | None = None
    toolhead_upper_rad: float | None = None
    select_tolerance_rad: float | None = None
    tools: tuple[Tool, ...] = ()
    wrist_camera: Pose | None = None
    joint_map: tuple[JointMap, ...] | None = None
    sdk_joint7: SdkJoint7 = field(default_factory=SdkJoint7)

    def tool(self, name: str) -> Tool | None:
        return next((tool for tool in self.tools if tool.name == name), None)

    def missing_for_urdf(self) -> list[str]:
        missing = [
            f"arm.{key}"
            for key in (
                "name",
                "remove_link_007_meshes",
                "flange",
                "wrist_camera",
            )
            if getattr(self, key) is None
        ]
        toolhead = {
            "mesh": self.toolhead_mesh,
            "mesh_scale": self.toolhead_mesh_scale,
            "joint_origin": self.toolhead_origin,
            "axis": self.toolhead_axis,
            "lower_rad": self.toolhead_lower_rad,
            "upper_rad": self.toolhead_upper_rad,
        }
        missing += [f"arm.toolhead.{key}" for key, value in toolhead.items() if value is None]
        missing += self._missing_tools("tip")
        return missing

    def missing_for_tool_check(self) -> list[str]:
        missing = (
            [] if self.select_tolerance_rad is not None else ["arm.toolhead.select_tolerance_rad"]
        )
        return missing + self._missing_tools("select_rad")

    def missing_for_joint_map(self) -> list[str]:
        return [] if self.joint_map is not None else ["arm.joint_map"]

    def missing_for_sdk_profile(self) -> list[str]:
        return [
            f"arm.sdk_joint7.{key}" for key in SDK7_KEYS if getattr(self.sdk_joint7, key) is None
        ]

    def _missing_tools(self, key: str) -> list[str]:
        if not self.tools:
            return ["arm.toolhead.tools"]
        return [
            f"arm.toolhead.tools.{tool.name}.{key}"
            for tool in self.tools
            if getattr(tool, key) is None
        ]


@dataclass(frozen=True)
class TipPixel:
    """Where a tool's tip appears in the wrist camera image at a known stand-off."""

    uv: tuple[float, float]
    standoff_m: float


@dataclass(frozen=True)
class ArmCalibration:
    """Measured values that override the nominal CAD ones. Each part may be absent."""

    calibration_id: str
    arm: str
    joint_map: tuple[JointMap, ...] | None = None
    wrist_camera: Pose | None = None
    tool_tips: dict[str, Pose] = field(default_factory=dict)
    tip_pixels: dict[str, TipPixel] = field(default_factory=dict)


def load_arm_config(path: str | None) -> tuple[ArmConfig, list[str]]:
    if not path:
        return ArmConfig(), ["unconfigured:arm_config"]
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        return ArmConfig(), [f"invalid:arm_config ({exc})"]
    return arm_from_dict(data, Path(path).parent)


def arm_from_dict(data: dict, base_dir: Path) -> tuple[ArmConfig, list[str]]:
    problems = []
    name = data.get("name")
    if name is not None and not (isinstance(name, str) and NAME_PATTERN.match(name)):
        problems.append(f"invalid:arm.name ({name!r} must match {NAME_PATTERN.pattern})")
        name = None

    remove = data.get("remove_link_007_meshes")
    if remove is not None:
        if isinstance(remove, list) and all(isinstance(item, str) for item in remove):
            remove = tuple(remove)
        else:
            problems.append("invalid:arm.remove_link_007_meshes (must be a list of mesh filenames)")
            remove = None

    head = data.get("toolhead") or {}
    mesh = head.get("mesh")
    if mesh is not None:
        mesh = (base_dir / str(mesh)).resolve()
        if not mesh.is_file():
            problems.append(f"invalid:arm.toolhead.mesh ({mesh} not found)")
            mesh = None
    scale = _vector(head.get("mesh_scale"), "arm.toolhead.mesh_scale", problems, positive=True)
    axis = _vector(head.get("axis"), "arm.toolhead.axis", problems, unit=True)
    lower = _number(head.get("lower_rad"), "arm.toolhead.lower_rad", problems)
    upper = _number(head.get("upper_rad"), "arm.toolhead.upper_rad", problems)
    if lower is not None and upper is not None and not lower < upper:
        problems.append("invalid:arm.toolhead (lower_rad must be below upper_rad)")
        lower = upper = None
    tolerance = _number(
        head.get("select_tolerance_rad"),
        "arm.toolhead.select_tolerance_rad",
        problems,
        positive=True,
    )
    tools = _tools(head.get("tools"), problems)
    _check_tool_angles(tools, lower, upper, tolerance, problems)

    joint_map = _joint_map(data.get("joint_map"), "arm.joint_map", problems)
    return ArmConfig(
        name=name,
        remove_link_007_meshes=remove,
        flange=_pose(data.get("flange"), "arm.flange", problems),
        toolhead_mesh=mesh,
        toolhead_mesh_scale=scale,
        toolhead_origin=_pose(head.get("joint_origin"), "arm.toolhead.joint_origin", problems),
        toolhead_axis=axis,
        toolhead_lower_rad=lower,
        toolhead_upper_rad=upper,
        select_tolerance_rad=tolerance,
        tools=tools,
        wrist_camera=_pose(data.get("wrist_camera"), "arm.wrist_camera", problems),
        joint_map=joint_map,
        sdk_joint7=_sdk_joint7(data.get("sdk_joint7") or {}, problems),
    ), problems


def load_arm_calibration(path: str | None) -> tuple[ArmCalibration | None, list[str]]:
    if not path:
        return None, []
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        return None, [f"invalid:arm_calibration ({exc})"]
    return calibration_from_dict(data)


def calibration_from_dict(data: dict) -> tuple[ArmCalibration | None, list[str]]:
    problems = []
    calibration_id, arm = data.get("calibration_id"), data.get("arm")
    if not calibration_id:
        problems.append("unconfigured:calibration.calibration_id")
    if not arm:
        problems.append("unconfigured:calibration.arm")

    tool_tips, tip_pixels = {}, {}
    tools = data.get("tools") or {}
    if not isinstance(tools, dict):
        problems.append("invalid:calibration.tools (must map tool names to values)")
        tools = {}
    for name, spec in tools.items():
        spec = spec or {}
        tip = _pose(spec.get("tip"), f"calibration.tools.{name}.tip", problems)
        if tip is not None:
            tool_tips[str(name)] = tip
        pixel = spec.get("tip_pixel")
        if pixel is not None:
            uv = _vector(
                pixel.get("uv"), f"calibration.tools.{name}.tip_pixel.uv", problems, size=2
            )
            standoff = _number(
                pixel.get("standoff_m"),
                f"calibration.tools.{name}.tip_pixel.standoff_m",
                problems,
                positive=True,
            )
            if uv is None or standoff is None:
                problems.append(
                    f"invalid:calibration.tools.{name}.tip_pixel (needs uv and standoff_m)"
                )
            else:
                tip_pixels[str(name)] = TipPixel(uv, standoff)

    joint_map = _joint_map(data.get("joint_map"), "calibration.joint_map", problems)
    wrist_camera = _pose(data.get("wrist_camera"), "calibration.wrist_camera", problems)
    if problems:
        return None, problems
    return ArmCalibration(
        calibration_id=str(calibration_id),
        arm=str(arm),
        joint_map=joint_map,
        wrist_camera=wrist_camera,
        tool_tips=tool_tips,
        tip_pixels=tip_pixels,
    ), problems


def check_calibration(config: ArmConfig, calibration: ArmCalibration | None) -> list[str]:
    """Problems that stop a calibration from applying to this arm's configuration."""
    if calibration is None:
        return []
    problems = []
    if calibration.arm != config.name:
        problems.append(f"invalid:calibration.arm ({calibration.arm!r} is not {config.name!r})")
    known = {tool.name for tool in config.tools}
    for name in sorted((set(calibration.tool_tips) | set(calibration.tip_pixels)) - known):
        problems.append(f"invalid:calibration.tools.{name} (no such tool in the arm config)")
    return problems


def _tools(raw, problems: list) -> tuple[Tool, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        problems.append("invalid:arm.toolhead.tools (must be a list)")
        return ()
    tools, names = [], set()
    for i, spec in enumerate(raw):
        name = (spec or {}).get("name")
        if not (isinstance(name, str) and NAME_PATTERN.match(name)) or name in names:
            problems.append(
                f"invalid:arm.toolhead.tools[{i}].name ({name!r} missing, malformed or repeated)"
            )
            continue
        names.add(name)
        key = f"arm.toolhead.tools.{name}"
        tools.append(
            Tool(
                name,
                _number(spec.get("select_rad"), f"{key}.select_rad", problems),
                _pose(spec.get("tip"), f"{key}.tip", problems),
            )
        )
    return tuple(tools)


def _check_tool_angles(tools, lower, upper, tolerance, problems: list) -> None:
    angles = sorted((tool.select_rad, tool.name) for tool in tools if tool.select_rad is not None)
    if lower is not None and upper is not None:
        for angle, name in angles:
            if not lower <= angle <= upper:
                problems.append(
                    f"invalid:arm.toolhead.tools.{name}.select_rad (outside the joint limits)"
                )
    if tolerance is not None:
        for (a, first), (b, second) in itertools.pairwise(angles):
            if b - a <= 2 * tolerance:
                problems.append(
                    f"invalid:arm.toolhead.tools ({first} and {second} are within "
                    "2 x select_tolerance_rad)"
                )


def _joint_map(raw, key: str, problems: list) -> tuple[JointMap, ...] | None:
    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) != SDK_JOINT_COUNT:
        problems.append(f"invalid:{key} (needs one entry per SDK joint, {SDK_JOINT_COUNT} in all)")
        return None
    entries = []
    for i, spec in enumerate(raw):
        spec = spec or {}
        scale = _number(spec.get("scale"), f"{key}[{i}].scale", problems)
        offset = _number(spec.get("offset_rad"), f"{key}[{i}].offset_rad", problems)
        if scale is None or offset is None:
            # A partly filled map is unset as a whole; it never mixes guesses with values.
            if spec.get("scale") is not None or spec.get("offset_rad") is not None:
                problems.append(f"invalid:{key}[{i}] (needs both scale and offset_rad)")
            return None
        if scale == 0:
            problems.append(f"invalid:{key}[{i}].scale (must be non-zero)")
            return None
        entries.append(JointMap(scale, offset))
    return tuple(entries)


def _sdk_joint7(raw: dict, problems: list) -> SdkJoint7:
    values = {key: _number(raw.get(key), f"arm.sdk_joint7.{key}", problems) for key in SDK7_KEYS}
    direction = values["direction"]
    if direction is not None:
        if direction not in (1.0, -1.0):
            problems.append("invalid:arm.sdk_joint7.direction (must be 1 or -1)")
            values["direction"] = None
        else:
            values["direction"] = int(direction)
    return SdkJoint7(**values)


def _pose(raw, key: str, problems: list) -> Pose | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        problems.append(f"invalid:{key} (needs xyz_m and rpy_rad)")
        return None
    xyz = _vector(raw.get("xyz_m"), f"{key}.xyz_m", problems)
    rpy = _vector(raw.get("rpy_rad"), f"{key}.rpy_rad", problems)
    if xyz is None or rpy is None:
        if (xyz is None) != (rpy is None):
            problems.append(f"invalid:{key} (needs both xyz_m and rpy_rad)")
        return None
    return xyz, rpy


def _vector(raw, key: str, problems: list, size=3, unit=False, positive=False):
    if raw is None:
        return None
    try:
        values = tuple(float(v) for v in raw if not isinstance(v, bool))
    except (TypeError, ValueError):
        values = ()
    if len(values) != size or not all(math.isfinite(v) for v in values):
        problems.append(f"invalid:{key} (must be {size} finite numbers)")
        return None
    if positive and not all(v > 0 for v in values):
        problems.append(f"invalid:{key} (must be positive)")
        return None
    if unit:
        norm = math.sqrt(sum(v * v for v in values))
        if norm == 0:
            problems.append(f"invalid:{key} (must be non-zero)")
            return None
        values = tuple(v / norm for v in values)
    return values


def _number(value, key: str, problems: list, positive=False) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        problems.append(f"invalid:{key} (not a number)")
        return None
    if not math.isfinite(number) or isinstance(value, bool) or (positive and number <= 0):
        problems.append(f"invalid:{key} (must be a {'positive' if positive else 'finite'} number)")
        return None
    return number
