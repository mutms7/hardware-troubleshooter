"""End to end on synthetic data: fixed-camera registration, then wrist alignment."""

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from conftest import ARM_POSE, CAMERA_INFO, write_arm
from hardware_troubleshooter_arm.config import ArmCalibration, TipPixel
from hardware_troubleshooter_registration.camera import CameraModel
from hardware_troubleshooter_registration.config import board_from_dict, rig_from_dict
from hardware_troubleshooter_registration.geometry import PlaneFrame, rot2
from hardware_troubleshooter_registration.synthetic import (
    SYNTHETIC_BOARD,
    SYNTHETIC_CAMERA_INFO,
    SYNTHETIC_RIG,
    render_frame,
)
from hardware_troubleshooter_vision import transforms
from hardware_troubleshooter_vision.capture import Frame, JointSample
from hardware_troubleshooter_vision.service import VisionService
from wrist_synthetic import ALIGNMENT, TARGET, TARGET_CENTER, project, render

ROOT = Path(__file__).resolve().parents[2]
TIP_OFFSET = np.array([0.0008, 0.0005])
# The wrist camera on link_007, looking along -z toward the tool tip.
WRIST_CAMERA = {"xyz_m": [0.04, 0.0, -0.10], "rpy_rad": [math.pi, 0.0, 0.0]}
PADS = [
    {
        "name": TARGET,
        "shape": "rect",
        "center_m": TARGET_CENTER.tolist(),
        "size_m": [0.0008, 0.0006],
    },
    {"name": "U1.2", "shape": "rect", "center_m": [0.0113, 0.0102], "size_m": [0.0006, 0.001]},
    {
        "name": "R1.1",
        "shape": "rect",
        "center_m": [0.009, 0.0114],
        "size_m": [0.0012, 0.0005],
        "angle_rad": 0.3,
    },
    {"name": "C1.1", "shape": "circle", "center_m": [0.0088, 0.0089], "radius_m": 0.0004},
    {"name": "TP9", "shape": "rect", "center_m": [0.0107, 0.0084], "size_m": [0.0004, 0.0004]},
]


def write_rig(directory: Path) -> Path:
    write_arm(directory / "left", wrist_camera=WRIST_CAMERA)
    (directory / "board.yaml").write_text(yaml.safe_dump(SYNTHETIC_BOARD), encoding="utf-8")
    (directory / "rig.yaml").write_text(yaml.safe_dump(SYNTHETIC_RIG), encoding="utf-8")
    (directory / "fixed.yaml").write_text(yaml.safe_dump(SYNTHETIC_CAMERA_INFO), encoding="utf-8")
    config = {
        "registration_source": "fixed",  # legacy path remains opt-in
        "board_config": "board.yaml",
        "rig_config": "rig.yaml",
        "fixed_camera_info": "fixed.yaml",
        "arms": {
            "left": {
                "arm_config": "left/left.yaml",
                "wrist_camera_info": "left/wrist.yaml",
                # Start well away from the board so the fixed camera sees all of it.
                "base_in_rig": {"xyz_m": [-3.0, 0.0, 0.0], "rpy_rad": [math.pi / 2, 0.0, 0.0]},
            }
        },
        "pads": {"source": "measured", "measured": PADS},
        "capture": {
            "max_joint_skew_s": 0.01,
            "settle_velocity_rad_s": 0.01,
            "settle_duration_s": 0.2,
        },
        "occlusion": {"margin_px": 5},
        "scene": {"diff_threshold": 25, "min_blob_area_px": 150, "board_margin_px": 10},
        "alignment": {f.name: getattr(ALIGNMENT, f.name) for f in dataclasses.fields(ALIGNMENT)},
    }
    path = directory / "vision.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def fixed_image():
    board, _ = board_from_dict(SYNTHETIC_BOARD, Path("."))
    rig, _ = rig_from_dict(SYNTHETIC_RIG)
    camera, _ = CameraModel.from_calibration_yaml(SYNTHETIC_CAMERA_INFO)
    plane = PlaneFrame.from_bench(rig.bench_normal, rig.bench_distance_m, board.thickness_m)
    axis = plane.rays_to_plane(np.array([[0.0, 0.0, 1.0]]))[0]
    x, y = axis - rot2(0.4) @ board.corners.mean(axis=0)
    return render_frame(board, camera, plane, (float(x), float(y), 0.4))


@pytest.fixture
def service(tmp_path):
    service, problems = VisionService.load(str(write_rig(tmp_path)), "test")
    assert not problems, problems
    for i in range(1000):
        service.add_joint_sample("left", JointSample(0.005 * i, ARM_POSE, (0.0,) * 7))
    assert service.process_fixed_frame(Frame("fixed", 1.0, fixed_image())).ok
    assert service.board_state(1.0).status == "valid"
    return service


def place_arm_over_target(service):
    """Move the arm's base so its wrist camera hovers over the target, and calibrate the
    tip pixel for that hover. Returns the true board <- wrist camera."""
    arm = service.arms["left"]
    _, _, rotation, translation = service.engine.board_in_base(1.0)
    rig_from_board = transforms.transform(rotation, translation)
    poses = arm.link_poses(ARM_POSE)
    base_from_camera = arm.base_from_wrist_camera(poses)
    camera_from_tip = transforms.invert(base_from_camera) @ arm.base_from_tool(poses, "probe")
    tip = camera_from_tip[:3, 3]
    board_rotation = np.diag([1.0, -1.0, -1.0])  # looking straight down
    depth = tip[2] + 0.03  # board 3 cm beyond the tip
    landing = TARGET_CENTER + TIP_OFFSET
    board_from_camera = transforms.transform(
        board_rotation, np.r_[landing, 0.0] - board_rotation @ np.r_[tip[:2], depth]
    )
    arm.rig_from_base = rig_from_board @ board_from_camera @ transforms.invert(base_from_camera)
    camera_from_board = transforms.invert(board_from_camera)
    uv = project(arm.wrist_camera, camera_from_board, landing)
    arm.calibration = ArmCalibration("cal", "left", tip_pixels={"probe": TipPixel(tuple(uv), 0.03)})
    return camera_from_board


def wrist_frame(service, camera_from_board, stamp=1.05):
    camera, _ = CameraModel.from_calibration_yaml(CAMERA_INFO)
    return Frame("left_wrist", stamp, render(camera, camera_from_board, service.pads))


def test_end_to_end_alignment(service):
    camera_from_board = place_arm_over_target(service)
    out = service.align("left", wrist_frame(service, camera_from_board), TARGET, "probe")
    assert out.ok, out.result.reasons
    assert out.registration_id.endswith("/r1")
    assert out.result.standoff_m == pytest.approx(0.03, abs=1e-9)
    assert out.result.error_board_m == pytest.approx(-TIP_OFFSET, abs=5e-5)
    base_from_board = service.base_from_board("left", 1.05)
    assert out.correction_base_m == pytest.approx(
        base_from_board[:3, :3] @ np.r_[-TIP_OFFSET, 0.0], abs=5e-5
    )


def test_alignment_refusals(service):
    camera_from_board = place_arm_over_target(service)
    frame = wrist_frame(service, camera_from_board)
    assert service.align("right", frame, TARGET, "probe").result.reasons == ["unknown_arm:right"]
    clip = service.align("left", frame, TARGET, "clip").result.reasons[0]
    assert clip.startswith("tool_not_selected:clip")
    early = service.align("left", wrist_frame(service, camera_from_board, 0.1), TARGET, "probe")
    assert early.result.reasons[0].startswith("arm_not_settled")
    stale = service.align("left", wrist_frame(service, camera_from_board, 3.0), TARGET, "probe")
    assert stale.result.reasons[0].startswith("registration_invalid")


def test_arm_without_joint_state_masks_everything(service):
    assert service.arms_mask(1.0).max() == 0  # parked far away, joints known
    assert service.arms_mask(100.0).all()  # no joint sample near this stamp


def test_unknown_arm_position_blocks_registration(tmp_path):
    service, problems = VisionService.load(str(write_rig(tmp_path)), "test")
    assert not problems
    result = service.process_fixed_frame(Frame("fixed", 1.0, fixed_image()))
    assert not result.ok and result.reasons[0].startswith("board_occluded")


def test_scene_guard_through_the_service(service):
    image = fixed_image()
    service.capture_scene_reference(Frame("fixed", 1.0, image))
    assert service.scene_clear(Frame("fixed", 1.01, image)).clear
    blocked = image.copy()
    blocked[50:120, 50:120] = 0
    assert not service.scene_clear(Frame("fixed", 1.02, blocked)).clear


def test_example_config_refuses():
    service, problems = VisionService.load(str(ROOT / "vision/config/vision.example.yaml"), "x")
    assert service is None
    assert (
        "unconfigured:board_reference.board_id" in problems
        and "unconfigured:pads.source" in problems
    )
