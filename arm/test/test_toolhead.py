import copy
import math

from arm_synthetic import SYNTHETIC_ARM
from hardware_troubleshooter_arm.toolhead import active_tool, check_tool, tip_pixel


def test_selected_tool_passes(arm):
    assert check_tool(arm, "probe", 0.015) is None
    assert check_tool(arm, "clip", 1.49) is None


def test_off_angle_fails(arm):
    reason = check_tool(arm, "probe", 0.05)
    assert reason == "tool_not_selected:probe (toolhead off by +0.0500 rad)"
    assert check_tool(arm, "clip", 0.0).startswith("tool_not_selected:clip")


def test_unknown_tool_and_bad_angle(arm):
    assert check_tool(arm, "drill", 0.0) == "unknown_tool:drill"
    assert check_tool(arm, "probe", math.nan) == "toolhead_angle_invalid"


def test_unconfigured_tolerance_fails_closed(make_arm):
    head = copy.deepcopy(SYNTHETIC_ARM["toolhead"])
    head["select_tolerance_rad"] = None
    config, _ = make_arm(toolhead=head)
    assert check_tool(config, "probe", 0.0) == "unconfigured:arm.toolhead.select_tolerance_rad"
    assert active_tool(config, 0.0) is None


def test_active_tool(arm):
    assert active_tool(arm, -0.01) == "probe"
    assert active_tool(arm, 1.51) == "clip"
    assert active_tool(arm, 0.75) is None
    assert active_tool(arm, math.inf) is None


def test_tip_pixel_comes_only_from_calibration(calibration):
    pixel = tip_pixel(calibration, "probe")
    assert pixel.uv == (322.5, 401.0) and pixel.standoff_m == 0.04
    assert tip_pixel(calibration, "clip") is None
    assert tip_pixel(None, "probe") is None
