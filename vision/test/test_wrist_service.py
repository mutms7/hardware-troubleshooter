"""Actual arm FK -> two wrist views -> approach -> local pad alignment."""

import asyncio
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from conftest import ARM_POSE, write_arm
from hardware_troubleshooter_arm.config import ArmCalibration, TipPixel
from hardware_troubleshooter_vision import transforms
from hardware_troubleshooter_vision.capture import Frame, JointSample
from hardware_troubleshooter_vision.config import (
    BoardReference,
    CaptureConfig,
    SceneConfig,
    VisionConfig,
)
from hardware_troubleshooter_vision.homing import BoardObservation
from hardware_troubleshooter_vision.rig import ArmRuntime
from hardware_troubleshooter_vision.service import VisionService
from test_board_features import make_reference
from test_homing import CONFIG
from test_service import PADS, TIP_OFFSET, WRIST_CAMERA
from wrist_synthetic import ALIGNMENT, TARGET, TARGET_CENTER, cluster_pads, project, render


@pytest.fixture
def wrist_service(tmp_path):
    arm, problems = ArmRuntime.load(write_arm(tmp_path / "left", wrist_camera=WRIST_CAMERA))
    assert not problems
    q1 = np.array(ARM_POSE)
    q2 = q1.copy()
    q2[1] += 0.1
    arm.calibration = ArmCalibration(
        "synthetic",
        "left",
        arm.config.joint_map,
        arm.config.wrist_camera,
        {t.name: t.tip for t in arm.config.tools},
    )
    poses = arm.link_poses(q2)
    base_from_camera = arm.base_from_wrist_camera(poses)
    tip = (transforms.invert(base_from_camera) @ arm.base_from_tool(poses, "probe"))[:3, 3]
    rotation = np.diag([1.0, -1.0, -1.0])
    landing = TARGET_CENTER + TIP_OFFSET
    board_from_camera = transforms.transform(
        rotation, np.r_[landing, 0.0] - rotation @ np.r_[tip[:2], tip[2] + 0.03]
    )
    camera_from_board = transforms.invert(board_from_camera)
    rig_from_board = arm.rig_from_base @ base_from_camera @ camera_from_board
    uv = project(arm.wrist_camera, camera_from_board, landing)
    arm.calibration = dataclasses.replace(
        arm.calibration, tip_pixels={"probe": TipPixel(tuple(uv), 0.03)}
    )
    config = VisionConfig(
        homing=CONFIG,
        board_reference=BoardReference(board_id="pcb"),
        capture=CaptureConfig(0.01, 0.01, 0.2),
        alignment=ALIGNMENT,
        scene=SceneConfig(25, 100, 5),
    )
    service = VisionService(
        config,
        None,
        None,
        "",
        {"left": arm},
        cluster_pads(),
        SimpleNamespace(fingerprint="reference"),
        "wrist-test",
    )
    for t in np.arange(0.7, 1.006, 0.005):
        service.add_joint_sample("left", JointSample(float(t), tuple(q1), (0.0,) * 7))
    for t in np.arange(1.7, 2.606, 0.005):
        service.add_joint_sample("left", JointSample(float(t), tuple(q2), (0.0,) * 7))
    points = np.array(
        [[x, y, 0.0] for x in np.linspace(-0.008, 0.028, 5) for y in np.linspace(-0.008, 0.028, 5)]
    )

    def view(stamp, q=q2, board=rig_from_board):
        cam_pose = arm.rig_from_base @ arm.base_from_wrist_camera(arm.link_poses(q))
        camera_board = transforms.invert(cam_pose) @ board
        pix = arm.wrist_camera.project(transforms.apply(camera_board, points))
        obs = BoardObservation("pcb", tuple(map(str, range(len(points)))), points, pix)
        return Frame("left_wrist", stamp, render(arm.wrist_camera, camera_board, service.pads)), obs

    return service, view, q1, q2, rig_from_board


def home_service(fixture):
    service, view, q1, _, board = fixture
    assert service.begin_homing("left", TARGET, 1.0).ok
    frame, obs = view(1.0, q1)
    assert service.observe_board("left", frame, obs).phase == "first_view"
    frame, obs = view(2.0)
    result = service.observe_board("left", frame, obs)
    assert result.ok, result.reasons
    return service, view, board


def test_wrist_pose_drives_approach_and_alignment_without_fixed_camera(wrist_service):
    service, view, board = home_service(wrist_service)
    assert service.engine is None and service.fixed_camera is None
    base_board = service.base_from_board("left", 2.0)
    assert np.allclose(service.arms["left"].rig_from_base @ base_board, board, atol=1e-6)
    position, normal = service.pad_approach("left", TARGET, 0.03, 2.0)
    assert position == pytest.approx(transforms.apply(base_board, [[*TARGET_CENTER, 0.03]])[0])
    assert normal == pytest.approx(base_board[:3, 2])
    frame, obs = view(2.1)
    result = service.align("left", frame, TARGET, "probe", obs)
    assert result.ok, result.result.reasons
    assert result.result.error_board_m == pytest.approx(-TIP_OFFSET, abs=5e-5)
    assert result.registration_id == service.board_state(2.1).registration_id
    assert result.correction_base_m == pytest.approx(
        base_board[:3, :3] @ np.r_[-TIP_OFFSET, 0.0], abs=5e-5
    )


def test_target_scope_overview_and_calibration_invalidation(wrist_service):
    service, view, _ = home_service(wrist_service)
    frame, obs = view(2.1)
    assert not service.align("left", frame, "U1.2", "probe", obs).ok
    image = np.full((100, 100), 100, np.uint8)
    service.capture_scene_reference(Frame("fixed_rgb", 2.0, image))
    assert service.process_fixed_frame(Frame("fixed_rgb", 2.05, image)).clear
    image[10:30, 10:30] = 0
    assert not service.process_fixed_frame(Frame("fixed_rgb", 2.1, image)).clear
    assert not service.align("left", frame, TARGET, "probe", obs).ok
    assert service.base_from_board("left", 2.1) is None


def test_measured_joint_and_camera_identity_guards(wrist_service):
    service, view, q1, _, _ = wrist_service
    assert service.begin_homing("left", TARGET, 1.0).ok
    frame, obs = view(1.0, q1)
    assert not service.observe_board("right", frame, obs).ok
    assert not service.observe_board("left", dataclasses.replace(frame, camera_id="fixed"), obs).ok
    assert not service.observe_board("left", dataclasses.replace(frame, stamp=1.5), obs).ok
    assert not service.observe_board(
        "left", dataclasses.replace(frame, image=np.zeros((10, 10))), obs
    ).ok


def test_calibration_change_and_external_move_require_new_homing(wrist_service):
    service, view, _ = home_service(wrist_service)
    arm = service.arms["left"]
    arm.calibration = dataclasses.replace(arm.calibration, calibration_id="changed")
    assert service.base_from_board("left", 2.1) is None
    assert service.board_state(2.1).failure_reason == "calibration_changed"
    assert not service.observe_board("left", *view(2.1)).ok


@pytest.mark.asyncio
async def test_complete_homing_sequence_uses_measured_callbacks(wrist_service):
    service, view, q1, q2, _ = wrist_service
    captures = iter([(1.0, q1), (2.0, q2)])
    observations = {}
    commands = []

    async def acquire(arm):
        stamp, q = next(captures)
        frame, obs = view(stamp, q)
        observations[stamp] = obs
        return frame, []  # this fixture already fed the measured settle windows

    async def move(arm, delta):
        commands.append((arm, delta))
        return True

    service.detector.detect = lambda frame, camera: (observations[frame.stamp], None)
    result = await service.home_target("left", TARGET, acquire, move)
    assert result.ok, result.reasons
    assert len(commands) == 1
    assert np.linalg.norm(commands[0][1]) == pytest.approx(CONFIG.step_m)
    assert service.board_state(2.0).has_pose


@pytest.mark.asyncio
async def test_homing_busy_and_cancellation_revoke_pose(wrist_service):
    service, _, _ = home_service(wrist_service)
    started = asyncio.Event()
    release = asyncio.Event()

    async def acquire(arm):
        started.set()
        await release.wait()

    async def move(arm, delta):
        raise AssertionError("no move expected")

    task = asyncio.create_task(service.home_target("left", TARGET, acquire, move))
    await started.wait()
    assert not service.board_state(2.1).has_pose
    assert (await service.home_target("left", TARGET, acquire, move)).reasons == ["homing_busy"]
    assert service.begin_homing("left", TARGET, 2.1).reasons == ["homing_busy"]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not service.board_state(2.1).has_pose
    assert service.homing.reason == "homing_interrupted"
    assert not service._homing_busy


def test_loader_defaults_to_wrist_without_fixed_intrinsics_or_extrinsic(tmp_path):
    reference, camera, _ = make_reference(tmp_path)
    write_arm(tmp_path / "left")
    from hardware_troubleshooter_vision.calibration import camera_info

    (tmp_path / "reference-camera.yaml").write_text(
        yaml.safe_dump(camera_info((640, 480), camera.k, camera.d))
    )
    config = {
        "homing": dataclasses.asdict(CONFIG),
        "board_reference": {
            "board_id": "pcb",
            "image": "reference.png",
            "camera_info": "reference-camera.yaml",
            "landmarks": list(reference.landmarks),
            "ratio": 0.75,
            "ransac_px": 1.5,
            "min_matches": 12,
        },
        "arms": {
            "left": {
                "arm_config": "left/left.yaml",
                "wrist_camera_info": "left/wrist.yaml",
                "base_in_rig": {"xyz_m": [0, 0, 0], "rpy_rad": [0, 0, 0]},
            }
        },
        "pads": {"source": "measured", "measured": PADS},
    }
    path = tmp_path / "vision.yaml"
    path.write_text(yaml.safe_dump(config))
    service, problems = VisionService.load(str(path), "wrist-load")
    assert not problems, problems
    assert service.homing is not None and service.engine is None
    assert not service.begin_homing(
        "left", TARGET, 1.0
    ).ok  # nominal CAD is not hand-eye calibration


@pytest.mark.asyncio
@pytest.mark.parametrize("ack", [False, {"status": "failed"}])
async def test_motion_failure_is_not_a_usable_homing(wrist_service, ack):
    service, view, q1, _, _ = wrist_service
    frame, obs = view(1.0, q1)
    service.detector.detect = lambda frame, camera: (obs, None)

    async def acquire(arm):
        return frame, []

    async def move(arm, delta):
        return ack

    result = await service.home_target("left", TARGET, acquire, move)
    assert not result.ok and result.reasons == ["homing_motion_failed"]
    assert not service.board_state(1.0).has_pose


def test_invalid_homing_config_is_reported(tmp_path):
    from hardware_troubleshooter_vision.config import vision_from_dict

    _, problems = vision_from_dict(
        {
            "homing": {
                "min_points": 3,
                "min_lateral_baseline_m": 0.1,
                "step_m": 0.01,
                "max_baseline_m": 0.05,
            },
            "board_reference": {"board_id": 12, "ratio": 1.5, "min_matches": 3.5},
        },
        tmp_path,
    )
    text = " ".join(problems)
    for field in (
        "homing.min_points",
        "min_lateral_baseline_m <= step_m",
        "board_reference.board_id",
        "board_reference.ratio",
        "board_reference.min_matches",
    ):
        assert field in text
