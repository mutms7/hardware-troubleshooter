import copy
import xml.etree.ElementTree as ET
from pathlib import Path

from arm_synthetic import SYNTHETIC_ARM
from hardware_troubleshooter_arm.urdf import build_urdf
from hardware_troubleshooter_arm.vendor import VENDOR_URDF


def built(arm, tmp_path, calibration=None):
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    text, problems = build_urdf(arm, calibration, out)
    assert not problems, problems
    return ET.fromstring(text), text, out


def test_gripper_replaced_by_toolhead(arm, tmp_path):
    robot, _, _ = built(arm, tmp_path)
    links = {link.get("name") for link in robot.findall("link")}
    joints = {joint.get("name"): joint for joint in robot.findall("joint")}
    assert not {"left_gripper_left", "left_gripper_right", "left_grasp_center"} & links
    assert {"left_flange", "left_toolhead_link", "left_tool_probe", "left_tool_clip"} <= links
    assert "left_wrist_camera_optical" in links

    toolhead = joints["left_toolhead_joint"]
    assert toolhead.get("type") == "revolute"
    assert toolhead.find("parent").get("link") == "left_flange"
    assert toolhead.find("child").get("link") == "left_toolhead_link"
    limit = toolhead.find("limit")
    assert (float(limit.get("lower")), float(limit.get("upper"))) == (-3.0, 3.0)
    camera = joints["left_wrist_camera_joint"]
    assert camera.find("parent").get("link") == "left_link_007"
    assert joints["left_tool_probe_joint"].find("parent").get("link") == "left_toolhead_link"


def test_every_name_is_prefixed(arm, tmp_path):
    robot, _, _ = built(arm, tmp_path)
    assert robot.get("name") == "maker_arm_left"
    for element in robot.findall("link") + robot.findall("joint"):
        assert element.get("name").startswith("left_")
    for ref in robot.iter():
        if ref.tag in ("parent", "child"):
            assert ref.get("link").startswith("left_")


def test_only_listed_wrist_meshes_are_removed(arm, tmp_path):
    robot, _, _ = built(arm, tmp_path)
    vendor = ET.parse(VENDOR_URDF).getroot().find("link[@name='link_007']")
    wrist = robot.find("link[@name='left_link_007']")
    before = {Path(m.get("filename")).name for m in vendor.iter("mesh")}
    after = {Path(m.get("filename")).name for m in wrist.iter("mesh")}
    assert before - after == {"part_026_solid_026.stl", "part_027_solid_027.stl"}


def test_mesh_paths_resolve_from_the_output_dir(arm, tmp_path):
    robot, _, out = built(arm, tmp_path)
    meshes = list(robot.iter("mesh"))
    assert meshes
    for mesh in meshes:
        assert (out / mesh.get("filename")).resolve().is_file(), mesh.get("filename")


def test_unknown_wrist_mesh_is_rejected(make_arm, tmp_path):
    config, problems = make_arm(remove_link_007_meshes=["meshes/part_999.stl"])
    assert not problems
    urdf, problems = build_urdf(config, output_dir=tmp_path)
    assert urdf is None
    assert problems == ["invalid:arm.remove_link_007_meshes (meshes/part_999.stl not on link_007)"]


def test_empty_removal_keeps_the_wrist(make_arm, tmp_path):
    config, _ = make_arm(remove_link_007_meshes=[])
    text, problems = build_urdf(config, output_dir=tmp_path)
    assert not problems
    assert "part_026_solid_026.stl" in text


def test_calibration_overrides_nominal_values(arm, calibration, tmp_path):
    robot, text, _ = built(arm, tmp_path, calibration)
    joints = {joint.get("name"): joint for joint in robot.findall("joint")}
    assert joints["left_tool_probe_joint"].find("origin").get("xyz") == "0.0005 0 -0.0507"
    assert joints["left_tool_clip_joint"].find("origin").get("xyz") == "0 0 -0.04"
    assert joints["left_wrist_camera_joint"].find("origin").get("rpy") == "0 0.31 0"
    assert "Calibration: left-cal-1." in text
    assert "Calibrated: wrist_camera, tool_probe.tip." in text
    assert "Nominal CAD values: tool_clip.tip." in text


def test_uncalibrated_build_says_so(arm, tmp_path):
    _, text, _ = built(arm, tmp_path)
    assert "Calibration: none." in text
    assert "Nominal CAD values: wrist_camera, tool_probe.tip, tool_clip.tip." in text


def test_mismatched_calibration_blocks_the_build(make_arm, calibration, tmp_path):
    config, _ = make_arm(name="right")
    urdf, problems = build_urdf(config, calibration, tmp_path)
    assert urdf is None
    assert problems == ["invalid:calibration.arm ('left' is not 'right')"]


def test_missing_tool_tip_blocks_the_build(make_arm, tmp_path):
    head = copy.deepcopy(SYNTHETIC_ARM["toolhead"])
    del head["tools"][0]["tip"]
    config, _ = make_arm(toolhead=head)
    urdf, problems = build_urdf(config, output_dir=tmp_path)
    assert urdf is None
    assert problems == ["unconfigured:arm.toolhead.tools.probe.tip"]
