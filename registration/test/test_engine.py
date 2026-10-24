"""Revision, invalidation and calibration behaviour over synthetic image sequences."""

import copy

import numpy as np
import pytest
from conftest import SYNTHETIC_RIG, Scene
from hardware_troubleshooter_registration.camera import CameraModel
from hardware_troubleshooter_registration.engine import RegistrationEngine
from hardware_troubleshooter_registration.synthetic import SYNTHETIC_CAMERA_INFO, render_frame

EXTRINSIC = (np.array([0.3, 0.0, 0.5]), np.array([1.0, 0.0, 0.0, 0.0]))
CAMERA_FRAME = "synthetic_camera_optical"
TIMEOUT = SYNTHETIC_RIG["thresholds"]["tracking_loss_timeout_s"]


def make_engine(scene, extrinsic=EXTRINSIC):
    engine = RegistrationEngine(scene.board, [], scene.rig, [], "test")
    engine.set_extrinsic(extrinsic, None if extrinsic else "extrinsic_unavailable:test", 0.0)
    return engine


def feed(engine, scene, pose, stamp, camera=None, **render):
    image = render_frame(scene.board, scene.camera, scene.plane, pose, **render)
    return engine.process_frame(image, stamp, CAMERA_FRAME, camera or scene.camera)


def blank(scene):
    return np.full((scene.camera.height, scene.camera.width), 60, np.uint8)


def occluder_across_board(scene, pose):
    """A rectangle in pixels covering the middle of the board, like an arm crossing it."""
    image = render_frame(scene.board, scene.camera, scene.plane, pose)
    ys, xs = np.nonzero(image > 120)
    x0, x1 = xs.min(), xs.max()
    y_mid = (ys.min() + ys.max()) / 2
    return [
        np.array(
            [
                [x0 - 50, y_mid - 40],
                [x1 + 50, y_mid - 40],
                [x1 + 50, y_mid + 40],
                [x0 - 50, y_mid + 40],
            ],
            float,
        )
    ]


@pytest.fixture
def registered(scene):
    engine = make_engine(scene)
    pose = scene.centred_pose(theta=0.4)
    assert feed(engine, scene, pose, 1.0).ok
    snapshot = engine.snapshot(1.0)
    assert snapshot.status == "valid", snapshot.reasons
    engine.drain_events()
    return engine, pose


def test_static_board_gives_valid_registration(scene, registered):
    engine, pose = registered
    for i in range(3):
        feed(engine, scene, pose, 1.1 + 0.1 * i)
    snapshot = engine.snapshot(1.3)
    assert snapshot.registration_id == "synthetic_board/test/r1"
    assert snapshot.calibration_valid and snapshot.tracking_valid and snapshot.board_unmoved
    assert snapshot.has_pose
    assert snapshot.stamp == pytest.approx(1.3)
    assert snapshot.registration_stamp == pytest.approx(1.0)
    assert snapshot.valid_until == pytest.approx(1.3 + TIMEOUT)
    assert snapshot.failure_reason == ""
    assert snapshot.calibration_revision
    assert snapshot.symmetry_resolved and snapshot.edge_points > 0
    assert engine.drain_events() == []  # confirmations alone aren't state changes
    parent, child, rotation, translation = engine.board_transform(1.3)
    assert (parent, child) == (CAMERA_FRAME, "synthetic_board")
    assert np.allclose(rotation, snapshot.rotation)
    assert np.allclose(translation, snapshot.translation)


def test_small_motion_keeps_revision(scene, registered):
    engine, pose = registered
    feed(engine, scene, (pose[0] + 0.0005, pose[1], pose[2]), 1.1)
    assert engine.snapshot(1.1).registration_id.endswith("/r1")
    assert engine.snapshot(1.1).status == "valid"


def test_pose_jump_publishes_board_moved_and_never_revalidates(scene, registered):
    engine, pose = registered
    moved = (pose[0] + 0.006, pose[1] - 0.004, pose[2] + 0.05)
    feed(engine, scene, moved, 1.1)
    events = engine.drain_events()
    assert len(events) == 1
    event = events[0]
    assert event.registration_id.endswith("/r1")
    assert event.reasons[0].startswith("board_moved:")
    assert event.status == "invalid"
    assert not event.tracking_valid and not event.board_unmoved and not event.has_pose
    assert engine.board_transform(1.1) is None

    # Back at the original pose, r1 still isn't valid; a fresh revision is made instead.
    feed(engine, scene, pose, 1.2)
    snapshot = engine.snapshot(1.2)
    assert snapshot.registration_id.endswith("/r2")
    assert snapshot.status == "valid"
    for event in engine.drain_events():
        assert not (event.registration_id.endswith("/r1") and event.status == "valid")


def test_brief_occlusion_suspends_without_invalidating(scene, registered):
    engine, pose = registered
    feed(engine, scene, pose, 1.1, occluders=occluder_across_board(scene, pose))
    snapshot = engine.snapshot(1.1)
    assert snapshot.registration_id.endswith("/r1")
    assert not snapshot.tracking_valid
    assert snapshot.has_pose
    assert any(r.startswith("unconfirmed") for r in snapshot.reasons)
    assert engine.board_transform(1.1) is None
    feed(engine, scene, pose, 1.2)
    assert engine.snapshot(1.2).status == "valid"
    assert engine.snapshot(1.2).registration_id.endswith("/r1")


def test_occlusion_longer_than_timeout_publishes_tracking_lost(scene, registered):
    engine, pose = registered
    occluders = occluder_across_board(scene, pose)
    stamp = 1.0
    while stamp < 1.0 + TIMEOUT + 0.2:
        stamp += 0.1
        feed(engine, scene, pose, stamp, occluders=occluders)
    lost = [
        e for e in engine.drain_events() if e.reasons and e.reasons[0].startswith("tracking_lost")
    ]
    assert len(lost) == 1
    assert lost[0].registration_id.endswith("/r1")
    feed(engine, scene, pose, stamp + 0.1)
    assert engine.snapshot(stamp + 0.1).registration_id.endswith("/r2")


def test_removed_board_publishes_tracking_lost(scene, registered):
    engine, _ = registered
    engine.process_frame(blank(scene), 1.2, CAMERA_FRAME, scene.camera)
    assert engine.snapshot(1.2).registration_id.endswith("/r1")
    engine.process_frame(blank(scene), 1.0 + TIMEOUT + 0.1, CAMERA_FRAME, scene.camera)
    events = engine.drain_events()
    assert events[-1].reasons[0].startswith("tracking_lost")
    assert "no_board_segment" in events[-1].reasons


def test_dropped_frames_publish_tracking_lost(scene, registered):
    engine, _ = registered
    engine.tick(1.0 + TIMEOUT / 2)
    assert engine.drain_events() == []
    engine.tick(1.0 + TIMEOUT + 0.01)
    events = engine.drain_events()
    assert len(events) == 1 and events[0].reasons[0].startswith("tracking_lost")
    assert engine.board_transform(2.0) is None


def test_missing_extrinsic_is_calibration_invalid_without_transform(scene):
    engine = make_engine(scene, extrinsic=None)
    pose = scene.centred_pose()
    assert feed(engine, scene, pose, 1.0).ok
    snapshot = engine.snapshot(1.0)
    assert snapshot.tracking_valid  # tracking and calibration are independent
    assert not snapshot.calibration_valid
    assert snapshot.status == "invalid"
    assert "extrinsic_unavailable:test" in snapshot.reasons
    assert snapshot.calibration_revision == ""
    assert engine.board_transform(1.0) is None


def test_unset_base_frame_is_unconfigured_without_transform():
    rig = copy.deepcopy(SYNTHETIC_RIG)
    rig["base_frame"] = None
    scene = Scene(rig_dict=rig)
    engine = make_engine(scene)
    feed(engine, scene, scene.centred_pose(), 1.0)
    snapshot = engine.snapshot(1.0)
    assert snapshot.status == "unconfigured"
    assert not snapshot.calibration_valid
    assert "unconfigured:rig.base_frame" in snapshot.reasons
    assert engine.board_transform(1.0) is None


def test_unset_bench_plane_fails_closed():
    rig = copy.deepcopy(SYNTHETIC_RIG)
    rig["bench_plane"] = {"normal": None, "distance_m": None}
    scene = Scene()
    from hardware_troubleshooter_registration.config import rig_from_dict

    rig_config, _ = rig_from_dict(rig)
    engine = RegistrationEngine(scene.board, [], rig_config, [], "test")
    engine.set_extrinsic(EXTRINSIC, None, 0.0)
    feed(engine, scene, scene.centred_pose(), 1.0)
    snapshot = engine.snapshot(1.0)
    assert snapshot.status == "unconfigured"
    assert not snapshot.calibration_valid and not snapshot.tracking_valid
    assert not snapshot.has_pose
    assert engine.board_transform(1.0) is None


def test_invalid_camera_info_is_calibration_invalid(scene):
    engine = make_engine(scene)
    image = render_frame(scene.board, scene.camera, scene.plane, scene.centred_pose())
    camera, problem = CameraModel.from_camera_info(1280, 960, [0] * 9, [0] * 5, "plumb_bob")
    assert camera is None
    engine.process_frame(image, 1.0, CAMERA_FRAME, camera, problem)
    snapshot = engine.snapshot(1.0)
    assert not snapshot.calibration_valid and not snapshot.tracking_valid
    assert any(r.startswith("camera_info_invalid") for r in snapshot.reasons)
    assert engine.board_transform(1.0) is None


def test_mismatched_camera_info_size_is_rejected(scene):
    engine = make_engine(scene)
    info = dict(SYNTHETIC_CAMERA_INFO, image_width=640, image_height=480)
    camera, _ = CameraModel.from_calibration_yaml(info)
    image = render_frame(scene.board, scene.camera, scene.plane, scene.centred_pose())
    engine.process_frame(image, 1.0, CAMERA_FRAME, camera)
    assert not engine.snapshot(1.0).calibration_valid


def test_extrinsic_change_publishes_calibration_changed(scene, registered):
    engine, _ = registered
    engine.set_extrinsic((EXTRINSIC[0] + 0.001, EXTRINSIC[1]), None, 1.05)
    events = engine.drain_events()
    assert events[0].reasons[0].startswith("calibration_changed")
    assert events[0].registration_id.endswith("/r1")


def test_moving_camera_never_stays_valid(scene, registered):
    """The arm camera's extrinsic changes as the arm moves; every revision must fall."""
    engine, pose = registered
    for i in range(1, 4):
        stamp = 1.0 + 0.2 * i
        engine.set_extrinsic((EXTRINSIC[0] + 0.01 * i, EXTRINSIC[1]), None, stamp)
        assert engine.snapshot(stamp).status != "valid"
        assert engine.board_transform(stamp) is None
        feed(engine, scene, pose, stamp + 0.1)  # re-registers under the new extrinsic
    changed = [
        e
        for e in engine.drain_events()
        if e.reasons and e.reasons[0].startswith("calibration_changed")
    ]
    assert [e.registration_id[-2:] for e in changed] == ["r1", "r2", "r3"]
    assert "needs the fixed camera" in changed[0].reasons[0]


def test_intrinsics_change_publishes_calibration_changed(scene, registered):
    engine, pose = registered
    info = copy.deepcopy(SYNTHETIC_CAMERA_INFO)
    info["camera_matrix"]["data"][0] += 1.0
    camera, _ = CameraModel.from_calibration_yaml(info)
    feed(engine, scene, pose, 1.1, camera=camera)
    events = engine.drain_events()
    assert events[0].reasons[0].startswith("calibration_changed")


def test_unset_thresholds_never_register():
    rig = copy.deepcopy(SYNTHETIC_RIG)
    rig["thresholds"]["move_translation_m"] = None
    rig["thresholds"]["tracking_loss_timeout_s"] = None
    scene = Scene(rig_dict=rig)
    engine = make_engine(scene)
    feed(engine, scene, scene.centred_pose(), 1.0)
    snapshot = engine.snapshot(1.0)
    assert snapshot.status == "unconfigured"
    assert snapshot.registration_id == ""
    assert not snapshot.tracking_valid
    assert "unconfigured:rig.thresholds.move_translation_m" in snapshot.reasons
    assert "unconfigured:rig.thresholds.tracking_loss_timeout_s" in snapshot.reasons


def test_no_frames_is_invalid(scene):
    engine = make_engine(scene)
    snapshot = engine.snapshot(5.0)
    assert snapshot.status == "invalid"
    assert "no_camera_frames" in snapshot.reasons
    assert "camera_info_missing" in snapshot.reasons
    assert not snapshot.has_pose


def test_out_of_order_frames_are_ignored(scene, registered):
    engine, pose = registered
    assert feed(engine, scene, pose, 0.5) is None
