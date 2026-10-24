import math

import numpy as np
import pytest
from hardware_troubleshooter_arm.kinematics import Kinematics, rpy_matrix, urdf_joint_positions
from hardware_troubleshooter_arm.urdf import build_urdf
from hardware_troubleshooter_arm.vendor import ARM_JOINTS, VENDOR_URDF

ARM_POSE = [0.3, -0.8, 1.2, 0.4, -0.5, 0.7]


def built_kinematics(arm, tmp_path, calibration=None):
    text, problems = build_urdf(arm, calibration, tmp_path)
    assert not problems, problems
    return Kinematics.from_urdf(text)


def positions(arm_pose, toolhead):
    q = {f"left_{name}": value for name, value in zip(ARM_JOINTS, arm_pose)}
    q["left_toolhead_joint"] = toolhead
    return q


def test_rpy_is_fixed_axis_xyz():
    def rx(a):
        return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])

    def ry(a):
        return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])

    def rz(a):
        return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])

    assert np.allclose(rpy_matrix(0.1, 0.2, 0.3), rz(0.3) @ ry(0.2) @ rx(0.1))


def test_vendor_urdf_loads_with_the_gripper():
    kinematics = Kinematics.from_urdf(VENDOR_URDF.read_text(encoding="utf-8"))
    assert kinematics.root == "base_link"
    assert set(kinematics.movable) == set(ARM_JOINTS) | {"gripper_left_joint"}
    q = dict(zip(ARM_JOINTS, ARM_POSE), gripper_left_joint=0.02)
    poses = kinematics.link_poses(q)
    gap = poses["gripper_left"][:3, 3] - poses["gripper_right"][:3, 3]
    assert np.isclose(np.linalg.norm(gap), 0.04)


def test_arm_chain_is_unchanged_by_the_build(arm, tmp_path):
    vendor = Kinematics.from_urdf(VENDOR_URDF.read_text(encoding="utf-8"))
    built = built_kinematics(arm, tmp_path)
    expected = vendor.link_poses(dict(zip(ARM_JOINTS, ARM_POSE), gripper_left_joint=0.0))
    actual = built.link_poses(positions(ARM_POSE, 0.4))
    for link in ("link_002", "link_005", "link_007"):
        assert np.allclose(actual[f"left_{link}"], expected[link])


def test_toolhead_rotation_moves_tips_not_the_camera(arm, tmp_path):
    kinematics = built_kinematics(arm, tmp_path)
    a = kinematics.link_poses(positions(ARM_POSE, 0.0))
    b = kinematics.link_poses(positions(ARM_POSE, 1.5))
    assert np.allclose(a["left_wrist_camera_optical"], b["left_wrist_camera_optical"])
    assert not np.allclose(a["left_tool_probe"][:3, 3], b["left_tool_probe"][:3, 3])

    # The tip swings about the toolhead axis: its distance to the axis is unchanged.
    axis_point = a["left_toolhead_link"][:3, 3]
    axis = a["left_toolhead_link"][:3, :3] @ np.array([0.0, 1.0, 0.0])

    def radius(pose):
        offset = pose["left_tool_probe"][:3, 3] - axis_point
        return np.linalg.norm(offset - np.dot(offset, axis) * axis)

    assert np.isclose(radius(a), radius(b))
    assert np.isclose(radius(a), 0.05)


def test_wrist_roll_moves_the_camera(arm, tmp_path):
    kinematics = built_kinematics(arm, tmp_path)
    rolled = ARM_POSE[:5] + [ARM_POSE[5] + 0.5]
    a = kinematics.link_poses(positions(ARM_POSE, 0.0))
    b = kinematics.link_poses(positions(rolled, 0.0))
    assert not np.allclose(a["left_wrist_camera_optical"], b["left_wrist_camera_optical"])


def test_selected_tool_points_where_its_tip_frame_says(arm, tmp_path):
    kinematics = built_kinematics(arm, tmp_path)
    poses = kinematics.link_poses(positions(ARM_POSE, 1.5))
    toolhead = poses["left_toolhead_link"]
    tip = np.linalg.inv(toolhead) @ poses["left_tool_clip"]
    assert np.allclose(tip[:3, 3], [0.0, 0.0, -0.04])


def test_missing_or_non_finite_positions_are_rejected(arm, tmp_path):
    kinematics = built_kinematics(arm, tmp_path)
    q = positions(ARM_POSE, 0.0)
    del q["left_toolhead_joint"]
    with pytest.raises(ValueError, match="missing joint positions"):
        kinematics.link_poses(q)
    with pytest.raises(ValueError, match="non-finite"):
        kinematics.link_poses(positions(ARM_POSE, math.nan))


def test_sdk_positions_map_to_urdf_joints(make_arm):
    joint_map = [{"scale": -1.0, "offset_rad": 0.5}] + [{"scale": 1.0, "offset_rad": 0.0}] * 6
    config, problems = make_arm(joint_map=joint_map)
    assert not problems
    q = urdf_joint_positions(config, [1.0, 2, 3, 4, 5, 6, 7])
    assert q["left_link_002_joint"] == -0.5
    assert q["left_link_007_joint"] == 6
    assert q["left_toolhead_joint"] == 7


def test_calibrated_joint_map_wins(arm, calibration):
    from dataclasses import replace

    from hardware_troubleshooter_arm.config import JointMap

    calibrated = replace(calibration, joint_map=(JointMap(1.0, 0.1),) * 7)
    q = urdf_joint_positions(arm, [0.0] * 7, calibrated)
    assert all(value == 0.1 for value in q.values())


def test_unset_joint_map_raises(make_arm):
    config, _ = make_arm(joint_map=None)
    with pytest.raises(ValueError, match="joint_map is unset"):
        urdf_joint_positions(config, [0.0] * 7)
    with pytest.raises(ValueError, match="expected 7"):
        urdf_joint_positions(make_arm()[0], [0.0] * 6)
