"""Board and rig YAML loading. Unknown values stay None; nothing is filled in with defaults."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .board_model import (
    BoardModel,
    Feature,
    Target,
    feature_outline,
    outline_from_kicad,
    outline_from_polygon,
)

THRESHOLD_NAMES = (
    "max_fit_residual_px",
    "move_translation_m",
    "move_rotation_rad",
    "tracking_loss_timeout_s",
    "feature_min_contrast",
)


@dataclass(frozen=True)
class Thresholds:
    max_fit_residual_px: float | None = None
    move_translation_m: float | None = None
    move_rotation_rad: float | None = None
    tracking_loss_timeout_s: float | None = None
    feature_min_contrast: float | None = None
    # Optional: only occluded confirmation needs it. Unset, an occluded frame never
    # confirms the registration.
    min_visible_edge_fraction: float | None = None


@dataclass(frozen=True)
class RigConfig:
    base_frame: str | None = None
    bench_normal: tuple | None = None
    bench_distance_m: float | None = None
    board_is_brighter: bool | None = None
    thresholds: Thresholds = field(default_factory=Thresholds)

    def missing_for_tracking(self) -> list[str]:
        missing = []
        if self.bench_normal is None or self.bench_distance_m is None:
            missing.append("rig.bench_plane")
        if self.board_is_brighter is None:
            missing.append("rig.segmentation.board_is_brighter")
        missing += [
            f"rig.thresholds.{name}"
            for name in THRESHOLD_NAMES
            if getattr(self.thresholds, name) is None
        ]
        return missing

    def missing_for_calibration(self) -> list[str]:
        missing = []
        if self.bench_normal is None or self.bench_distance_m is None:
            missing.append("rig.bench_plane")
        if not self.base_frame:
            missing.append("rig.base_frame")
        return missing


def load_board_config(path: str | None) -> tuple[BoardModel | None, list[str]]:
    if not path:
        return None, ["unconfigured:board_config"]
    try:
        data = yaml.safe_load(Path(path).read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        return None, [f"invalid:board_config ({exc})"]
    return board_from_dict(data, Path(path).parent)


def board_from_dict(data: dict, base_dir: Path) -> tuple[BoardModel | None, list[str]]:
    problems = []
    board_id = data.get("board_id")
    if not board_id:
        problems.append("unconfigured:board.board_id")
    thickness = _positive(data.get("thickness_m"), "board.thickness_m", problems)
    if data.get("thickness_m") is None:
        problems.append("unconfigured:board.thickness_m")

    outline = None
    source = data.get("outline_source")
    try:
        if source == "measured":
            if data.get("outline_m") is None:
                problems.append("unconfigured:board.outline_m")
            else:
                outline = outline_from_polygon(data["outline_m"])
        elif source == "kicad":
            if not data.get("kicad_pcb"):
                problems.append("unconfigured:board.kicad_pcb")
            else:
                outline = outline_from_kicad((base_dir / data["kicad_pcb"]).read_text())
        elif source is None:
            problems.append("unconfigured:board.outline_source")
        else:
            problems.append(f"invalid:board.outline_source ({source!r})")
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        problems.append(f"invalid:board.outline ({exc})")

    features = []
    for i, spec in enumerate(data.get("features") or []):
        try:
            if spec.get("expect") not in ("darker", "brighter"):
                raise ValueError("expect must be darker or brighter")
            features.append(
                Feature(spec.get("name") or f"feature_{i}", feature_outline(spec), spec["expect"])
            )
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            problems.append(f"invalid:board.features[{i}] ({exc})")
    if not data.get("features"):
        problems.append("unconfigured:board.features")

    targets = []
    for i, spec in enumerate(data.get("overlay_targets") or []):
        try:
            position = np.asarray(spec["position_m"], dtype=float).reshape(2)
            targets.append(Target(str(spec["name"]), position))
        except (ValueError, TypeError, KeyError) as exc:
            problems.append(f"invalid:board.overlay_targets[{i}] ({exc})")

    if problems:
        return None, problems
    frame_id = data.get("frame_id") or f"board_{board_id}"
    return BoardModel(
        str(board_id), frame_id, thickness, outline, tuple(features), tuple(targets)
    ), []


def load_rig_config(path: str | None) -> tuple[RigConfig, list[str]]:
    if not path:
        return RigConfig(), ["unconfigured:rig_config"]
    try:
        data = yaml.safe_load(Path(path).read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        return RigConfig(), [f"invalid:rig_config ({exc})"]
    return rig_from_dict(data)


def rig_from_dict(data: dict) -> tuple[RigConfig, list[str]]:
    problems = []
    plane = data.get("bench_plane") or {}
    normal = plane.get("normal")
    if normal is not None:
        try:
            normal = np.asarray(normal, dtype=float).reshape(3)
            if not np.all(np.isfinite(normal)) or np.linalg.norm(normal) == 0:
                raise ValueError("must be a finite non-zero 3-vector")
            normal = tuple(normal / np.linalg.norm(normal))
        except (ValueError, TypeError) as exc:
            problems.append(f"invalid:rig.bench_plane.normal ({exc})")
            normal = None
    distance = _positive(plane.get("distance_m"), "rig.bench_plane.distance_m", problems)

    brighter = (data.get("segmentation") or {}).get("board_is_brighter")
    if brighter is not None and not isinstance(brighter, bool):
        problems.append("invalid:rig.segmentation.board_is_brighter (must be true or false)")
        brighter = None

    raw = data.get("thresholds") or {}
    values = {
        name: _positive(raw.get(name), f"rig.thresholds.{name}", problems)
        for name in THRESHOLD_NAMES
    }
    values["min_visible_edge_fraction"] = _positive(
        raw.get("min_visible_edge_fraction"), "rig.thresholds.min_visible_edge_fraction", problems
    )
    for name in ("feature_min_contrast", "min_visible_edge_fraction"):
        if values[name] is not None and values[name] > 1:
            problems.append(f"invalid:rig.thresholds.{name} (must be <= 1)")
            values[name] = None

    base_frame = data.get("base_frame") or None
    return RigConfig(base_frame, normal, distance, brighter, Thresholds(**values)), problems


def _positive(value, name: str, problems: list) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        problems.append(f"invalid:{name} (not a number)")
        return None
    if not math.isfinite(number) or number <= 0 or isinstance(value, bool):
        problems.append(f"invalid:{name} (must be a positive number)")
        return None
    return number
