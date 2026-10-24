"""Occluded (partial-edge) confirmation and the board pose in the base frame."""

import copy

import cv2
import numpy as np
import pytest
from conftest import SYNTHETIC_RIG, Scene
from hardware_troubleshooter_registration.config import rig_from_dict
from hardware_troubleshooter_registration.engine import RegistrationEngine
from hardware_troubleshooter_registration.geometry import (
    quaternion_to_rotation,
    rotation_to_quaternion,
)
from hardware_troubleshooter_registration.synthetic import render_frame
from test_engine import CAMERA_FRAME, EXTRINSIC, make_engine, occluder_across_board

FRACTION = 0.4


def tracking_scene(fraction=FRACTION):
    rig = copy.deepcopy(SYNTHETIC_RIG)
    rig["thresholds"]["min_visible_edge_fraction"] = fraction
    return Scene(rig_dict=rig)


def mask_of(scene, polygons):
    mask = np.zeros((scene.camera.height, scene.camera.width), np.uint8)
    for polygon in polygons:
        cv2.fillPoly(mask, [np.round(polygon).astype(np.int32).reshape(-1, 1, 2)], 255)
    return mask


def feed(engine, scene, pose, stamp, occluders=(), mask=None):
    image = render_frame(scene.board, scene.camera, scene.plane, pose, occluders=occluders)
    if mask is None and occluders:
        mask = mask_of(scene, occluders)
    return engine.process_frame(image, stamp, CAMERA_FRAME, scene.camera, occlusion_mask=mask)


@pytest.fixture
def tracked():
    scene = tracking_scene()
    engine = make_engine(scene)
    pose = scene.centred_pose(theta=0.4)
    assert feed(engine, scene, pose, 1.0).ok
    engine.drain_events()
    return scene, engine, pose


def test_occluded_static_board_stays_valid(tracked):
    scene, engine, pose = tracked
    occluders = occluder_across_board(scene, pose)
    for i in range(1, 15):  # longer than the tracking timeout
        result = feed(engine, scene, pose, 1.0 + 0.1 * i, occluders)
        assert result.ok, result.reasons
    snapshot = engine.snapshot(2.4)
    assert snapshot.status == "valid", snapshot.reasons
    assert snapshot.registration_id.endswith("/r1")
    assert FRACTION <= snapshot.visible_edge_fraction < 1.0
    assert snapshot.residual_px < SYNTHETIC_RIG["thresholds"]["max_fit_residual_px"]
    assert engine.drain_events() == []


def test_small_motion_under_the_arm_is_confirmed(tracked):
    scene, engine, pose = tracked
    occluders = occluder_across_board(scene, pose)
    nudged = (pose[0] + 0.0005, pose[1] - 0.0003, pose[2] + 0.002)
    result = feed(engine, scene, nudged, 1.1, occluders)
    assert result.ok
    assert result.pose[:2] == pytest.approx(nudged[:2], abs=2e-4)
    assert result.pose[2] == pytest.approx(nudged[2], abs=1e-3)  # move threshold is 1e-2
    assert engine.snapshot(1.1).status == "valid"


def test_board_moved_under_the_arm_is_caught(tracked):
    scene, engine, pose = tracked
    occluders = occluder_across_board(scene, pose)
    moved = (pose[0] + 0.003, pose[1], pose[2])
    result = feed(engine, scene, moved, 1.1, occluders)
    assert result.ok and result.pose[0] == pytest.approx(moved[0], abs=2e-4)
    events = engine.drain_events()
    assert events[-1].reasons[0].startswith("board_moved:")
    assert engine.board_transform(1.1) is None


def test_rotation_under_the_arm_is_caught(tracked):
    scene, engine, pose = tracked
    occluders = occluder_across_board(scene, pose)
    feed(engine, scene, (pose[0], pose[1], pose[2] + 0.02), 1.1, occluders)
    assert engine.drain_events()[-1].reasons[0].startswith("board_moved:")


def test_unset_fraction_keeps_the_old_behaviour():
    scene = Scene()
    engine = make_engine(scene)
    pose = scene.centred_pose(theta=0.4)
    assert feed(engine, scene, pose, 1.0).ok
    result = feed(engine, scene, pose, 1.1, occluder_across_board(scene, pose))
    assert not result.ok
    assert "unconfigured:rig.thresholds.min_visible_edge_fraction" in result.reasons
    assert not engine.snapshot(1.1).tracking_valid


def test_mostly_covered_board_is_unconfirmed():
    scene = tracking_scene(fraction=0.9)
    engine = make_engine(scene)
    pose = scene.centred_pose(theta=0.4)
    assert feed(engine, scene, pose, 1.0).ok
    result = feed(engine, scene, pose, 1.1, occluder_across_board(scene, pose))
    assert not result.ok
    assert any(r.startswith("unconfirmed:only") for r in result.reasons)
    snapshot = engine.snapshot(1.1)
    assert not snapshot.tracking_valid and snapshot.registration_id.endswith("/r1")


def test_occluded_frame_never_creates_a_revision():
    scene = tracking_scene()
    engine = make_engine(scene)
    pose = scene.centred_pose(theta=0.4)
    result = feed(engine, scene, pose, 1.0, occluder_across_board(scene, pose))
    assert not result.ok
    assert any(r.startswith("board_occluded") for r in result.reasons)
    assert engine.snapshot(1.0).registration_id == ""


def test_mask_away_from_the_board_allows_full_registration(tracked):
    scene, engine, pose = tracked
    corner = [np.array([[0, 0], [60, 0], [60, 60], [0, 60]], float)]
    result = feed(engine, scene, pose, 1.1, mask=mask_of(scene, corner))
    assert result.ok and result.visible_fraction is None  # the full fit ran
    assert engine.snapshot(1.1).status == "valid"


def test_mask_of_the_wrong_size_is_rejected(tracked):
    scene, engine, pose = tracked
    result = feed(engine, scene, pose, 1.1, mask=np.zeros((10, 10), np.uint8))
    assert not result.ok
    assert result.reasons[0].startswith("invalid:occlusion_mask size")


def test_board_in_base_composes_the_extrinsic(tracked):
    _, engine, _ = tracked
    base, board_frame, rotation, translation = engine.board_in_base(1.0)
    snapshot = engine.snapshot(1.0)
    base_from_camera = quaternion_to_rotation(EXTRINSIC[1])
    assert base == SYNTHETIC_RIG["base_frame"]
    assert board_frame == "synthetic_board"
    assert np.allclose(rotation, base_from_camera @ snapshot.rotation)
    assert np.allclose(translation, base_from_camera @ snapshot.translation + EXTRINSIC[0])


def test_board_in_base_needs_a_valid_registration(tracked):
    scene, engine, _ = tracked
    engine.tick(1.0 + SYNTHETIC_RIG["thresholds"]["tracking_loss_timeout_s"] + 0.1)
    assert engine.board_in_base(3.0) is None
    unregistered = RegistrationEngine(scene.board, [], scene.rig, [], "other")
    assert unregistered.board_in_base(0.0) is None


def test_quaternion_round_trip():
    rng = np.random.default_rng(3)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        rotation = quaternion_to_rotation(q)
        assert np.allclose(rotation @ rotation.T, np.eye(3))
        assert np.isclose(np.linalg.det(rotation), 1.0)
        assert np.allclose(quaternion_to_rotation(rotation_to_quaternion(rotation)), rotation)


def test_fraction_above_one_is_invalid():
    rig = copy.deepcopy(SYNTHETIC_RIG)
    rig["thresholds"]["min_visible_edge_fraction"] = 1.5
    config, problems = rig_from_dict(rig)
    assert "invalid:rig.thresholds.min_visible_edge_fraction (must be <= 1)" in problems
    assert config.thresholds.min_visible_edge_fraction is None
