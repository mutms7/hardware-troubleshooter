"""Vision YAML loading. Unknown values stay None; nothing is filled in with defaults.

Each section is a frozen dataclass. A consumer asks its section what's missing and
refuses to run until it's empty, so an unset threshold can never act as a guess."""

from __future__ import annotations

import dataclasses
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def _missing(section, prefix: str) -> list[str]:
    return [
        f"{prefix}.{f.name}"
        for f in dataclasses.fields(section)
        if getattr(section, f.name) is None
    ]


@dataclass(frozen=True)
class CaptureConfig:
    """Stop-and-look: a wrist frame counts only when the arm was still at its stamp."""

    max_joint_skew_s: float | None = None  # |image stamp - nearest joint sample stamp|
    settle_velocity_rad_s: float | None = None  # every joint below this ...
    settle_duration_s: float | None = None  # ... for this long before the image stamp

    def missing(self) -> list[str]:
        return _missing(self, "capture")


@dataclass(frozen=True)
class OcclusionConfig:
    margin_px: float | None = None  # grows each projected arm part, for calibration error

    def missing(self) -> list[str]:
        return _missing(self, "occlusion")


@dataclass(frozen=True)
class AlignmentConfig:
    pads_are_brighter: bool | None = None  # copper brighter than the solder mask around it
    patch_radius_m: float | None = None  # board area around the target rendered for matching
    resolution_m: float | None = None  # board-plane size of one rectified pixel
    search_radius_m: float | None = None  # largest predicted-vs-seen offset searched
    min_match_score: float | None = None  # normalised correlation of the best match, 0-1
    min_uniqueness_margin: float | None = None  # best minus best elsewhere, 0-1
    max_standoff_error_m: float | None = None  # hover height vs the tip pixel's stand-off
    converged_m: float | None = None  # tip-to-target error that counts as aligned

    def missing(self) -> list[str]:
        return _missing(self, "alignment")


@dataclass(frozen=True)
class SceneConfig:
    diff_threshold: float | None = None  # grey-level change from the empty-rig reference
    min_blob_area_px: float | None = None  # smallest changed area reported as an obstruction
    board_margin_px: float | None = None  # board region excluded (registration watches it)

    def missing(self) -> list[str]:
        return _missing(self, "scene")


@dataclass(frozen=True)
class ContactConfig:
    """Torque touchdown: a sustained step from the pre-descent baseline."""

    joints: tuple[int, ...] | None = None  # 1-based SDK joints whose torque is watched
    torque_step_nm: float | None = None
    persistence_s: float | None = None
    baseline_s: float | None = None  # window before descent used as the baseline

    def missing(self) -> list[str]:
        return _missing(self, "contact")


@dataclass(frozen=True)
class ThermalConfig:
    # 3x3 homography from thermal-grid cell coordinates to fixed camera ideal pixels,
    # valid for the board's top-surface height.
    homography: np.ndarray | None = None
    min_delta_c: float | None = None  # hot spot must be this far above the board median

    def missing(self) -> list[str]:
        missing = [] if self.homography is not None else ["thermal.homography"]
        return missing + ([] if self.min_delta_c is not None else ["thermal.min_delta_c"])


@dataclass(frozen=True)
class PadSource:
    source: str | None = None  # kicad or measured
    kicad_pcb: Path | None = None
    measured: tuple = ()  # raw dicts, parsed by pads.pads_from_specs

    def missing(self) -> list[str]:
        if self.source is None:
            return ["pads.source"]
        if self.source == "kicad" and self.kicad_pcb is None:
            return ["pads.kicad_pcb"]
        if self.source == "measured" and not self.measured:
            return ["pads.measured"]
        return []


@dataclass(frozen=True)
class ArmEntry:
    name: str
    arm_config: Path | None = None  # hardware_troubleshooter_arm YAML
    calibration: Path | None = None  # optional; without it the arm is nominal only
    base_in_rig: tuple | None = None  # pose of the arm's base_link in the rig base frame
    wrist_camera_info: Path | None = None  # camera_calibration YAML

    def missing(self) -> list[str]:
        prefix = f"arms.{self.name}"
        return [
            f"{prefix}.{key}"
            for key in ("arm_config", "base_in_rig", "wrist_camera_info")
            if getattr(self, key) is None
        ]


@dataclass(frozen=True)
class VisionConfig:
    board_config: Path | None = None  # registration board YAML
    rig_config: Path | None = None  # registration rig YAML
    fixed_camera_info: Path | None = None
    arms: dict[str, ArmEntry] = field(default_factory=dict)
    pads: PadSource = field(default_factory=PadSource)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    occlusion: OcclusionConfig = field(default_factory=OcclusionConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    contact: ContactConfig = field(default_factory=ContactConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)

    def missing_for_registration(self) -> list[str]:
        return [
            key
            for key in ("board_config", "rig_config", "fixed_camera_info")
            if getattr(self, key) is None
        ]


def load_vision_config(path: str | None) -> tuple[VisionConfig, list[str]]:
    if not path:
        return VisionConfig(), ["unconfigured:vision_config"]
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        return VisionConfig(), [f"invalid:vision_config ({exc})"]
    return vision_from_dict(data, Path(path).parent)


def vision_from_dict(data: dict, base_dir: Path) -> tuple[VisionConfig, list[str]]:
    problems: list[str] = []

    def path(value, key):
        if value is None:
            return None
        resolved = (base_dir / str(value)).resolve()
        if not resolved.is_file():
            problems.append(f"invalid:{key} ({resolved} not found)")
            return None
        return resolved

    arms = {}
    for name, spec in (data.get("arms") or {}).items():
        if not (isinstance(name, str) and NAME_PATTERN.match(name)):
            problems.append(f"invalid:arms ({name!r} isn't a valid arm name)")
            continue
        spec = spec or {}
        arms[name] = ArmEntry(
            name,
            path(spec.get("arm_config"), f"arms.{name}.arm_config"),
            path(spec.get("calibration"), f"arms.{name}.calibration"),
            _pose(spec.get("base_in_rig"), f"arms.{name}.base_in_rig", problems),
            path(spec.get("wrist_camera_info"), f"arms.{name}.wrist_camera_info"),
        )

    raw_pads = data.get("pads") or {}
    source = raw_pads.get("source")
    if source not in (None, "kicad", "measured"):
        problems.append(f"invalid:pads.source ({source!r}; use kicad or measured)")
        source = None
    pads = PadSource(
        source,
        path(raw_pads.get("kicad_pcb"), "pads.kicad_pcb"),
        tuple(raw_pads.get("measured") or ()),
    )

    alignment_raw = data.get("alignment") or {}
    brighter = alignment_raw.get("pads_are_brighter")
    if brighter is not None and not isinstance(brighter, bool):
        problems.append("invalid:alignment.pads_are_brighter (must be true or false)")
        brighter = None
    alignment = _numbers(
        AlignmentConfig, alignment_raw, "alignment", problems, skip=("pads_are_brighter",)
    )
    alignment = dataclasses.replace(alignment, pads_are_brighter=brighter)
    for name in ("min_match_score", "min_uniqueness_margin"):
        if (getattr(alignment, name) or 0) > 1:
            problems.append(f"invalid:alignment.{name} (must be <= 1)")
            alignment = dataclasses.replace(alignment, **{name: None})

    contact_raw = data.get("contact") or {}
    joints = contact_raw.get("joints")
    if joints is not None:
        if (
            isinstance(joints, list)
            and joints
            and all(isinstance(j, int) and not isinstance(j, bool) and 1 <= j <= 7 for j in joints)
        ):
            joints = tuple(joints)
        else:
            problems.append("invalid:contact.joints (must list SDK joints 1-7)")
            joints = None
    contact = dataclasses.replace(
        _numbers(ContactConfig, contact_raw, "contact", problems, skip=("joints",)), joints=joints
    )

    thermal_raw = data.get("thermal") or {}
    homography = thermal_raw.get("homography")
    if homography is not None:
        try:
            homography = np.asarray(homography, dtype=float).reshape(3, 3)
            if not np.all(np.isfinite(homography)) or abs(np.linalg.det(homography)) < 1e-12:
                raise ValueError("must be finite and invertible")
        except (TypeError, ValueError) as exc:
            problems.append(f"invalid:thermal.homography ({exc})")
            homography = None
    thermal = ThermalConfig(
        homography, _positive(thermal_raw.get("min_delta_c"), "thermal.min_delta_c", problems)
    )

    return VisionConfig(
        board_config=path(data.get("board_config"), "board_config"),
        rig_config=path(data.get("rig_config"), "rig_config"),
        fixed_camera_info=path(data.get("fixed_camera_info"), "fixed_camera_info"),
        arms=arms,
        pads=pads,
        capture=_numbers(CaptureConfig, data.get("capture") or {}, "capture", problems),
        occlusion=_numbers(OcclusionConfig, data.get("occlusion") or {}, "occlusion", problems),
        alignment=alignment,
        scene=_numbers(SceneConfig, data.get("scene") or {}, "scene", problems),
        contact=contact,
        thermal=thermal,
    ), problems


def _numbers(cls, raw: dict, prefix: str, problems: list, skip=()):
    values = {
        f.name: _positive(raw.get(f.name), f"{prefix}.{f.name}", problems)
        for f in dataclasses.fields(cls)
        if f.name not in skip
    }
    return cls(**values)


def _pose(raw, key: str, problems: list):
    if raw is None or (
        isinstance(raw, dict) and raw.get("xyz_m") is None and raw.get("rpy_rad") is None
    ):
        return None
    try:
        xyz = tuple(float(v) for v in raw["xyz_m"])
        rpy = tuple(float(v) for v in raw["rpy_rad"])
        if len(xyz) != 3 or len(rpy) != 3 or not all(math.isfinite(v) for v in xyz + rpy):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        problems.append(f"invalid:{key} (needs xyz_m and rpy_rad, 3 finite numbers each)")
        return None
    return xyz, rpy


def _positive(value, key: str, problems: list) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        problems.append(f"invalid:{key} (not a number)")
        return None
    if not math.isfinite(number) or number <= 0 or isinstance(value, bool):
        problems.append(f"invalid:{key} (must be a positive number)")
        return None
    return number
