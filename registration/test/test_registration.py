"""Single-frame registration against synthetic ground truth."""

import math

import numpy as np
import pytest
from hardware_troubleshooter_registration.geometry import wrap_angle
from hardware_troubleshooter_registration.registration import register_frame
from hardware_troubleshooter_registration.synthetic import render_frame

# Test-only tolerances for noiseless synthetic images. Not an accuracy claim for any rig.
POSITION_TOL_M = 0.0003
ANGLE_TOL_RAD = math.radians(0.2)


def corner_error(scene, truth, estimate):
    def corners(pose):
        x, y, theta = pose
        c, s = math.cos(theta), math.sin(theta)
        return scene.board.corners @ np.array([[c, s], [-s, c]]) + (x, y)

    return np.linalg.norm(corners(truth) - corners(estimate), axis=1).max()


@pytest.mark.parametrize(
    ("dx", "dy", "theta"),
    [
        (0.0, 0.0, 0.0),
        (0.03, -0.02, math.radians(20)),
        (-0.04, 0.03, math.radians(-75)),
        (0.01, 0.01, math.radians(180)),
        (-0.02, -0.03, math.radians(135)),
    ],
)
def test_recovers_known_pose(scene, dx, dy, theta):
    truth = scene.centred_pose(dx, dy, theta)
    image = render_frame(scene.board, scene.camera, scene.plane, truth)
    result = register_frame(
        image, scene.camera, scene.plane, scene.board, True, scene.rig.thresholds
    )
    assert result.ok, result.reasons
    assert result.symmetry_resolved
    assert corner_error(scene, truth, result.pose) < POSITION_TOL_M
    assert abs(wrap_angle(result.pose[2] - truth[2])) < ANGLE_TOL_RAD
    assert result.residual_px < scene.rig.thresholds.max_fit_residual_px
    assert result.edge_points > 100
    assert result.ippe_tilt_deg is not None


def test_recovers_pose_with_noise(scene):
    truth = scene.centred_pose(0.01, -0.01, math.radians(37))
    image = render_frame(scene.board, scene.camera, scene.plane, truth, noise_sigma=3.0)
    result = register_frame(
        image, scene.camera, scene.plane, scene.board, True, scene.rig.thresholds
    )
    assert result.ok, result.reasons
    assert corner_error(scene, truth, result.pose) < 2 * POSITION_TOL_M


def test_hidden_feature_is_ambiguous(scene):
    truth = scene.centred_pose(0.0, 0.0, math.radians(10))
    image = render_frame(scene.board, scene.camera, scene.plane, truth, show_features=False)
    result = register_frame(
        image, scene.camera, scene.plane, scene.board, True, scene.rig.thresholds
    )
    assert not result.ok
    assert not result.symmetry_resolved
    assert result.pose is None
    assert result.reasons[0].startswith("symmetry_ambiguous")


def test_feature_on_both_ends_is_ambiguous():
    from conftest import SYNTHETIC_BOARD, Scene

    board = dict(SYNTHETIC_BOARD)
    hole = dict(board["features"][0])
    board["features"] = [hole]
    scene = Scene(board_dict=board)
    # Render a board with a second, point-symmetric hole the model doesn't know about.
    render_board = dict(board)
    render_board["features"] = [hole, {**hole, "name": "mirror", "center_m": [0.092, 0.052]}]
    render_scene = Scene(board_dict=render_board)
    truth = scene.centred_pose(0.0, 0.0, 0.3)
    image = render_frame(render_scene.board, scene.camera, scene.plane, truth)
    result = register_frame(
        image, scene.camera, scene.plane, scene.board, True, scene.rig.thresholds
    )
    assert not result.ok
    assert "matched 2 of 2" in result.reasons[0]


def test_unset_feature_threshold_never_resolves(scene):
    from dataclasses import replace

    thresholds = replace(scene.rig.thresholds, feature_min_contrast=None)
    image = render_frame(scene.board, scene.camera, scene.plane, scene.centred_pose())
    result = register_frame(image, scene.camera, scene.plane, scene.board, True, thresholds)
    assert not result.ok
    assert result.reasons == ["unconfigured:rig.thresholds.feature_min_contrast"]


def test_unset_acceptance_threshold_is_not_ok(scene):
    from dataclasses import replace

    thresholds = replace(scene.rig.thresholds, max_fit_residual_px=None)
    image = render_frame(scene.board, scene.camera, scene.plane, scene.centred_pose())
    result = register_frame(image, scene.camera, scene.plane, scene.board, True, thresholds)
    assert not result.ok
    assert "unconfigured:rig.thresholds.max_fit_residual_px" in result.reasons


def test_board_off_image_edge_fails(scene):
    image = render_frame(scene.board, scene.camera, scene.plane, scene.centred_pose(dx=0.3))
    result = register_frame(
        image, scene.camera, scene.plane, scene.board, True, scene.rig.thresholds
    )
    assert not result.ok


def test_empty_image_fails(scene):
    image = np.full((scene.camera.height, scene.camera.width), 60, np.uint8)
    result = register_frame(
        image, scene.camera, scene.plane, scene.board, True, scene.rig.thresholds
    )
    assert not result.ok
