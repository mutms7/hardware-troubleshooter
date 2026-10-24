"""Tool selection checks on the rotating toolhead. Angles are URDF toolhead_joint
positions (map SDK motor 7 through kinematics.urdf_joint_positions first)."""

from __future__ import annotations

import math

from .config import ArmCalibration, ArmConfig, TipPixel


def check_tool(config: ArmConfig, tool: str, toolhead_rad: float) -> str | None:
    """None when `tool` is the selected tool at this toolhead angle, else a failure reason.

    Use the measured angle at the image or contact stamp, not the commanded one."""
    missing = config.missing_for_tool_check()
    if missing:
        return "; ".join(f"unconfigured:{key}" for key in missing)
    spec = config.tool(tool)
    if spec is None:
        return f"unknown_tool:{tool}"
    if not math.isfinite(toolhead_rad):
        return "toolhead_angle_invalid"
    error = toolhead_rad - spec.select_rad
    if abs(error) > config.select_tolerance_rad:
        return f"tool_not_selected:{tool} (toolhead off by {error:+.4f} rad)"
    return None


def active_tool(config: ArmConfig, toolhead_rad: float) -> str | None:
    """The tool selected at this angle, or None between tools or when unconfigured."""
    if config.missing_for_tool_check() or not math.isfinite(toolhead_rad):
        return None
    return next(
        (
            tool.name
            for tool in config.tools
            if abs(toolhead_rad - tool.select_rad) <= config.select_tolerance_rad
        ),
        None,
    )


def tip_pixel(calibration: ArmCalibration | None, tool: str) -> TipPixel | None:
    """Calibrated wrist camera pixel of the tool tip. There is no nominal fallback."""
    if calibration is None:
        return None
    return calibration.tip_pixels.get(tool)
