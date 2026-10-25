"""Two-view metric geometry, with no fixed camera or commanded motion inputs."""

import dataclasses

import cv2
import numpy as np
import pytest
from hardware_troubleshooter_vision import transforms
from hardware_troubleshooter_vision.config import HomingConfig
from hardware_troubleshooter_vision.homing import BoardObservation, WristHoming

CONFIG = HomingConfig(
    min_points=6,
    min_lateral_baseline_m=0.01,
    step_m=0.03,
    max_baseline_m=0.08,
    min_parallax_rad=0.02,
    max_pair_interval_s=3,
    max_reprojection_error_px=0.8,
    max_pose_translation_m=0.02,
    max_pose_rotation_rad=0.35,
    min_pose_margin_px=0.1,
    max_triangulation_error_m=0.005,
    tracking_timeout_s=2,
)
POINTS = np.array(
    [[x, y, 0] for x in np.linspace(-0.045, 0.045, 6) for y in np.linspace(-0.035, 0.035, 5)]
)
BOARD = transforms.transform(
    cv2.Rodrigues(np.array([0.15, -0.1, 0.05]))[0] @ np.diag([1.0, -1.0, -1.0]), [0, 0, 0.4]
)
FIRST = np.eye(4)
SECOND = transforms.transform(cv2.Rodrigues(np.array([0.01, -0.02, 0.03]))[0], [0.03, 0.005, 0])


def observation(camera, camera_pose, board=BOARD, noise=0.0, seed=1):
    pixels = camera.project(transforms.apply(transforms.invert(camera_pose) @ board, POINTS))
    pixels += np.random.default_rng(seed).normal(0, noise, pixels.shape)
    return BoardObservation("pcb", tuple(map(str, range(len(POINTS)))), POINTS.copy(), pixels)


def start():
    engine = WristHoming(CONFIG, "session")
    engine.begin("left", "TP1", "pcb", "cal", 0.0)
    return engine


def home(camera):
    engine = start()
    assert engine.observe(observation(camera, FIRST, noise=0.05), FIRST, camera, 0.0, "cal").ok
    result = engine.observe(
        observation(camera, SECOND, noise=0.05, seed=2), SECOND, camera, 1.0, "cal"
    )
    assert result.ok, result.reasons
    return engine, result


def test_joint_pose_and_triangulation_recover_ground_truth(camera):
    engine, result = home(camera)
    snapshot = engine.snapshot(1.0)
    assert snapshot.has_pose and snapshot.board_unmoved
    assert snapshot.registration_id == "pcb/session/r1"
    assert snapshot.rig_from_board[:3, 3] == pytest.approx(BOARD[:3, 3], abs=2e-4)
    assert transforms.rotation_angle(snapshot.rig_from_board[:3, :3].T @ BOARD[:3, :3]) < 0.005
    assert result.diagnostics["rms_px"] < 0.15
    assert result.diagnostics["triangulation_rms_m"] < 0.001
    assert result.diagnostics["parallax_rad"] > 0.02
    assert result.diagnostics["baseline_m"] == pytest.approx(np.linalg.norm(SECOND[:3, 3]))


@pytest.mark.parametrize("translation", [[0, 0, 0], [0, 0, 0.025], [0.001, 0, 0], [0.1, 0, 0]])
def test_rejects_zero_axial_small_and_excessive_baselines(camera, translation):
    engine = start()
    assert engine.observe(observation(camera, FIRST), FIRST, camera, 0.0, "cal").ok
    moved = transforms.transform(cv2.Rodrigues(np.array([0.0, 0.03, 0.0]))[0], translation)
    result = engine.observe(observation(camera, moved), moved, camera, 1.0, "cal")
    assert not result.ok
    assert "baseline" in result.reasons[0]
    assert not engine.snapshot(1.0).has_pose


def test_commanded_motion_cannot_substitute_for_actual_camera_motion(camera):
    engine = start()
    engine.observe(observation(camera, FIRST), FIRST, camera, 0.0, "cal")
    # Image was taken from FIRST even though a 3 cm step was requested.
    result = engine.observe(observation(camera, FIRST), SECOND, camera, 1.0, "cal")
    assert not result.ok
    assert not engine.snapshot(1.0).has_pose


def test_board_motion_confirmation_and_no_resurrection(camera):
    engine, _ = home(camera)
    assert engine.observe(observation(camera, SECOND), SECOND, camera, 1.2, "cal").ok
    moved = BOARD.copy()
    moved[0, 3] += 0.01
    result = engine.observe(observation(camera, SECOND, moved), SECOND, camera, 1.3, "cal")
    assert not result.ok and result.reasons == ["board_moved"]
    assert not engine.observe(observation(camera, SECOND), SECOND, camera, 1.4, "cal").ok
    old = engine.registration_id
    engine.begin("left", "TP1", "pcb", "cal", 2.0)
    assert engine.registration_id != old
    assert not engine.snapshot(2.0).has_pose


def test_timeout_calibration_and_stale_observations(camera):
    engine, _ = home(camera)
    assert not engine.observe(observation(camera, SECOND), SECOND, camera, 1.0, "cal").ok
    assert not engine.observe(observation(camera, SECOND), SECOND, camera, 1.1, "newcal").ok
    assert engine.snapshot(1.1).failure_reason == "calibration_changed"
    engine, _ = home(camera)
    assert not engine.snapshot(3.1).has_pose
    assert engine.snapshot(3.1).failure_reason == "tracking_lost"
    engine = start()
    engine.tick(3.1)
    assert engine.snapshot(3.1).failure_reason == "homing_timeout"


def test_identity_geometry_and_nonfinite_inputs(camera):
    engine = start()
    obs = observation(camera, FIRST)
    bad = dataclasses.replace(obs, feature_ids=("same",) * len(POINTS))
    assert not engine.observe(bad, FIRST, camera, 0.0, "cal").ok
    bad = dataclasses.replace(obs, image_points=np.full_like(obs.image_points, np.nan))
    assert not engine.observe(bad, FIRST, camera, 0.0, "cal").ok
    assert engine.observe(obs, FIRST, camera, 0.0, "cal").ok
    obs2 = observation(camera, SECOND)
    bad = dataclasses.replace(obs2, object_points=obs2.object_points * 1.1)
    result = engine.observe(bad, SECOND, camera, 1.0, "cal")
    assert not result.ok and "feature_geometry_changed" in result.reasons[0]


def test_missing_thresholds_cannot_create_pose(camera):
    engine = WristHoming(HomingConfig(), "session")
    result = engine.observe(observation(camera, FIRST), FIRST, camera, 0.0, "cal")
    assert not result.ok and result.reasons[0].startswith("unconfigured:")


def test_noisy_or_moved_second_view_is_rejected(camera):
    engine = start()
    engine.observe(observation(camera, FIRST), FIRST, camera, 0.0, "cal")
    noisy = observation(camera, SECOND, noise=3.0)
    assert not engine.observe(noisy, SECOND, camera, 1.0, "cal").ok
    shifted_board = BOARD.copy()
    shifted_board[0, 3] += 0.03
    assert not engine.observe(
        observation(camera, SECOND, shifted_board), SECOND, camera, 1.0, "cal"
    ).ok


def test_parallax_requirement_and_ambiguous_solutions(camera, monkeypatch):
    engine = start()
    engine.config = dataclasses.replace(CONFIG, min_parallax_rad=0.2)
    engine.observe(observation(camera, FIRST), FIRST, camera, 0.0, "cal")
    result = engine.observe(observation(camera, SECOND), SECOND, camera, 1.0, "cal")
    assert not result.ok and result.reasons == ["insufficient_parallax"]
    # Exercise ambiguity refusal independently of numerical convergence of this scene.
    import hardware_troubleshooter_vision.homing as module

    original = module._refine
    count = 0

    def two_equal_solutions(seed, views, cam):
        nonlocal count
        pose, error = original(seed, views, cam)
        count += 1
        if count > 1:
            pose = pose.copy()
            pose[0, 3] += 0.001
        return pose, error

    engine = start()
    engine.config = dataclasses.replace(
        CONFIG, max_pose_translation_m=0.1, max_pose_rotation_rad=3.0
    )
    monkeypatch.setattr(module, "_refine", two_equal_solutions)
    engine.observe(observation(camera, FIRST), FIRST, camera, 0.0, "cal")
    result = engine.observe(observation(camera, SECOND), SECOND, camera, 1.0, "cal")
    assert not result.ok and result.reasons == ["pose_ambiguous"]
