import copy
from pathlib import Path

from arm_synthetic import SYNTHETIC_ARM, SYNTHETIC_CALIBRATION
from hardware_troubleshooter_arm.config import (
    calibration_from_dict,
    check_calibration,
    load_arm_calibration,
    load_arm_config,
)
from hardware_troubleshooter_arm.profile import build_sdk_profile
from hardware_troubleshooter_arm.urdf import build_urdf

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def test_example_config_loads_but_is_entirely_unset(tmp_path):
    config, problems = load_arm_config(str(CONFIG_DIR / "arm.example.yaml"))
    assert problems == []
    assert "arm.name" in config.missing_for_urdf()
    assert "arm.toolhead.mesh" in config.missing_for_urdf()
    assert config.missing_for_tool_check()
    assert config.missing_for_joint_map() == ["arm.joint_map"]
    assert len(config.missing_for_sdk_profile()) == 6

    urdf, problems = build_urdf(config, output_dir=tmp_path)
    assert urdf is None and problems and all(p.startswith("unconfigured:") for p in problems)
    profile, problems = build_sdk_profile(config)
    assert profile is None and problems


def test_example_calibration_is_unset():
    calibration, problems = load_arm_calibration(str(CONFIG_DIR / "calibration.example.yaml"))
    assert calibration is None
    assert "unconfigured:calibration.calibration_id" in problems


def test_synthetic_config_is_complete(arm):
    assert arm.missing_for_urdf() == []
    assert arm.missing_for_tool_check() == []
    assert arm.missing_for_joint_map() == []
    assert arm.missing_for_sdk_profile() == []
    assert arm.toolhead_mesh.is_file()


def test_invalid_values_are_reported(make_arm):
    head = copy.deepcopy(SYNTHETIC_ARM["toolhead"])
    head["axis"] = [0, 0, 0]
    head["tools"][1]["select_rad"] = 0.03  # within 2 x tolerance of the probe
    _, problems = make_arm(
        name="Left!",
        toolhead=head,
        joint_map=[{"scale": 1.0, "offset_rad": 0.0}] * 6,
        sdk_joint7={**SYNTHETIC_ARM["sdk_joint7"], "direction": 2},
    )
    text = "\n".join(problems)
    assert "invalid:arm.name" in text
    assert "invalid:arm.toolhead.axis" in text
    assert "probe and clip are within" in text
    assert "invalid:arm.joint_map" in text
    assert "invalid:arm.sdk_joint7.direction" in text


def test_tool_outside_joint_limits(make_arm):
    head = copy.deepcopy(SYNTHETIC_ARM["toolhead"])
    head["tools"][1]["select_rad"] = 3.5
    _, problems = make_arm(toolhead=head)
    assert any("clip.select_rad (outside the joint limits)" in p for p in problems)


def test_missing_mesh_file(make_arm):
    head = copy.deepcopy(SYNTHETIC_ARM["toolhead"])
    head["mesh"] = "absent.stl"
    config, problems = make_arm(toolhead=head)
    assert any(p.startswith("invalid:arm.toolhead.mesh") for p in problems)
    assert "arm.toolhead.mesh" in config.missing_for_urdf()


def test_partial_joint_map_is_unset_not_mixed(make_arm):
    joint_map = [{"scale": 1.0, "offset_rad": 0.0}] * 6 + [{"scale": 1.0, "offset_rad": None}]
    config, problems = make_arm(joint_map=joint_map)
    assert config.joint_map is None
    assert any("needs both scale and offset_rad" in p for p in problems)


def test_half_pose_is_invalid(make_arm):
    config, problems = make_arm(flange={"xyz_m": [0, 0, 0], "rpy_rad": None})
    assert config.flange is None
    assert "invalid:arm.flange (needs both xyz_m and rpy_rad)" in problems


def test_calibration_must_match_the_arm(arm):
    data = copy.deepcopy(SYNTHETIC_CALIBRATION)
    data["arm"] = "right"
    data["tools"]["drill"] = {"tip": {"xyz_m": [0, 0, 0], "rpy_rad": [0, 0, 0]}}
    calibration, problems = calibration_from_dict(data)
    assert not problems
    problems = check_calibration(arm, calibration)
    assert "invalid:calibration.arm ('right' is not 'left')" in problems
    assert any("calibration.tools.drill" in p for p in problems)


def test_tip_pixel_needs_uv_and_standoff():
    data = copy.deepcopy(SYNTHETIC_CALIBRATION)
    del data["tools"]["probe"]["tip_pixel"]["standoff_m"]
    calibration, problems = calibration_from_dict(data)
    assert calibration is None
    assert any("tip_pixel (needs uv and standoff_m)" in p for p in problems)
