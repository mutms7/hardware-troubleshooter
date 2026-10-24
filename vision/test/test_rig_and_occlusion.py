import copy
from pathlib import Path

import numpy as np
import pytest
import yaml
from conftest import ARM_POSE, overhead_camera, write_arm
from hardware_troubleshooter_vision import transforms
from hardware_troubleshooter_vision.config import load_vision_config, vision_from_dict
from hardware_troubleshooter_vision.occlusion import ArmGeometry, arm_mask, load_stl
from hardware_troubleshooter_vision.rig import ArmRuntime, fixed_camera_extrinsic

ROOT = Path(__file__).resolve().parents[2]


def test_example_config_loads_entirely_unset():
    config, problems = load_vision_config(str(ROOT / "vision/config/vision.example.yaml"))
    assert problems == []
    assert config.registration_source == "wrist"
    assert "homing.min_lateral_baseline_m" in config.missing_for_registration()
    assert "board_reference.image" in config.missing_for_registration()
    assert "fixed_camera_info" not in config.missing_for_registration()
    assert config.arms == {}
    for section in (config.capture, config.occlusion, config.alignment, config.scene):
        assert section.missing()
    assert config.contact.missing() and config.thermal.missing() and config.pads.missing()


def test_invalid_values_are_reported(tmp_path):
    _, problems = vision_from_dict(
        {
            "capture": {"max_joint_skew_s": -1},
            "alignment": {"pads_are_brighter": "yes", "min_match_score": 2},
            "contact": {"joints": [0, 9]},
            "thermal": {"homography": [[1, 0], [0, 1]]},
            "pads": {"source": "photo"},
            "arms": {"Left": {}, "right": {"base_in_rig": {"xyz_m": [0, 0], "rpy_rad": [0, 0, 0]}}},
            "board_config": "absent.yaml",
        },
        tmp_path,
    )
    text = "\n".join(problems)
    for expected in (
        "invalid:capture.max_joint_skew_s",
        "invalid:alignment.pads_are_brighter",
        "invalid:alignment.min_match_score",
        "invalid:contact.joints",
        "invalid:thermal.homography",
        "invalid:pads.source",
        "invalid:arms ('Left'",
        "invalid:arms.right.base_in_rig",
        "invalid:board_config",
    ):
        assert expected in text, expected


def test_arm_runtime_places_frames_from_measured_joints(arm):
    poses = arm.link_poses(ARM_POSE)
    camera = arm.base_from_wrist_camera(poses)
    rotated = arm.link_poses(ARM_POSE[:6] + (1.5,))
    assert np.allclose(camera, arm.base_from_wrist_camera(rotated))  # toolhead turn
    assert not np.allclose(arm.base_from_tool(poses, "probe"), arm.base_from_tool(rotated, "probe"))
    assert arm.toolhead_angle(ARM_POSE[:6] + (1.5,)) == pytest.approx(1.5)


def test_arm_runtime_rejects_a_name_mismatch(tmp_path):
    entry = write_arm(tmp_path / "x", name="left")
    entry = type(entry)(
        "right",
        *[
            getattr(entry, f)
            for f in ("arm_config", "calibration", "base_in_rig", "wrist_camera_info")
        ],
    )
    runtime, problems = ArmRuntime.load(entry)
    assert runtime is None
    assert "invalid:arms.right (arm config is named 'left')" in problems


def test_arm_runtime_needs_the_joint_map(tmp_path):
    runtime, problems = ArmRuntime.load(write_arm(tmp_path / "x", joint_map=None))
    assert runtime is None and problems == ["unconfigured:arm.joint_map"]


def test_fixed_camera_extrinsic(tmp_path):
    _, frame, problem = fixed_camera_extrinsic(ROOT / "registration/config/rig.example.yaml")
    assert problem.startswith("extrinsic_unavailable")
    path = tmp_path / "rig.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "camera_extrinsic": {
                    "camera_frame": "fixed_optical",
                    "translation_m": [0.1, 0.2, 0.9],
                    "rotation_xyzw": [1.0, 0.0, 0.0, 0.0],
                }
            }
        ),
        encoding="utf-8",
    )
    matrix, frame, problem = fixed_camera_extrinsic(path)
    assert problem is None and frame == "fixed_optical"
    assert np.allclose(matrix[:3, :3], np.diag([1.0, -1.0, -1.0]))
    assert np.allclose(matrix[:3, 3], [0.1, 0.2, 0.9])


def test_stl_loader_reads_ascii_and_binary(tmp_path):
    from arm_synthetic import write_stl

    write_stl(tmp_path / "b.stl")
    assert load_stl(tmp_path / "b.stl").shape == (3, 3)
    (tmp_path / "a.stl").write_text(
        "solid t\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\n"
        "endloop\nendfacet\nendsolid t\n"
    )
    assert load_stl(tmp_path / "a.stl").shape == (3, 3)


def masks(arm, camera, rig_from_camera, margin=0.0):
    poses = arm.link_poses(ARM_POSE)
    camera_from_base = transforms.invert(rig_from_camera) @ arm.rig_from_base
    return arm_mask(arm.geometry, poses, camera_from_base, camera, margin), poses, camera_from_base


def test_overhead_mask_covers_the_arm(arm, camera):
    mask, poses, camera_from_base = masks(arm, camera, overhead_camera())
    assert 0 < (mask > 0).mean() < 0.6
    # Every vertex of every part that projects into the image lands inside the mask.
    for part in arm.geometry.parts[::7]:
        points = transforms.apply(camera_from_base @ poses[part.link], part.vertices)
        px = np.round(camera.project(points)).astype(int)
        inside = (px[:, 0] >= 0) & (px[:, 0] < 640) & (px[:, 1] >= 0) & (px[:, 1] < 480)
        assert (mask[px[inside, 1], px[inside, 0]] > 0).all(), part.link


def test_margin_grows_the_mask(arm, camera):
    tight, *_ = masks(arm, camera, overhead_camera())
    grown, *_ = masks(arm, camera, overhead_camera(), margin=10)
    assert (grown >= tight).all() and (grown > 0).sum() > (tight > 0).sum()


def test_arm_behind_the_camera_is_not_masked(arm, camera):
    # Camera below the arm, looking down, so the arm is behind it.
    mask, *_ = masks(arm, camera, overhead_camera(height=-1.0))
    assert not mask.any()


def test_straddling_part_masks_everything(arm, camera):
    poses = arm.link_poses(ARM_POSE)
    part = max(arm.geometry.parts, key=lambda p: len(p.vertices))
    inside = transforms.apply(arm.rig_from_base @ poses[part.link], part.vertices).mean(axis=0)
    # Put the camera's optical centre inside a mesh part.
    mask, *_ = masks(
        arm,
        camera,
        overhead_camera(height=inside[2], x=inside[0])
        @ transforms.transform(np.eye(3), [0, -inside[1], 0]),
    )
    assert mask.all()


def test_part_outside_the_view_is_skipped(arm, camera):
    # Camera level with the arm, far to one side and looking away: nothing is visible.
    mask, *_ = masks(arm, camera, overhead_camera(height=0.3, x=3.0))
    assert not mask.any()


def test_missing_link_pose_raises(arm, camera):
    poses = copy.copy(arm.link_poses(ARM_POSE))
    poses.pop("left_link_007")
    with pytest.raises(ValueError, match="no pose for link"):
        camera_from_base = transforms.invert(overhead_camera()) @ arm.rig_from_base
        arm_mask(arm.geometry, poses, camera_from_base, camera, 0)


def test_geometry_has_the_toolhead(arm):
    links = {part.link for part in arm.geometry.parts}
    assert "left_toolhead_link" in links and "left_gripper_left" not in links
    assert isinstance(arm.geometry, ArmGeometry)
