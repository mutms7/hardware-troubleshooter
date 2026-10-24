import math

import cv2
import numpy as np
import pytest
import yaml
from conftest import ARM_POSE, CAMERA_INFO
from hardware_troubleshooter_arm.config import calibration_from_dict
from hardware_troubleshooter_arm.vendor import ARM_JOINTS
from hardware_troubleshooter_registration.camera import CameraModel
from hardware_troubleshooter_vision import transforms
from hardware_troubleshooter_vision.calibration import (
    CharucoTarget,
    arm_calibration_dict,
    base_in_rig,
    bench_plane,
    calibrate_intrinsics,
    calibrate_joint_offsets,
    detect_charuco,
    eye_in_hand,
    eye_to_hand,
    pose_from_matrix,
    target_pose,
    tip_pixel_from_marks,
)

TARGET = CharucoTarget(squares_x=7, squares_y=5, square_m=0.02, marker_m=0.015)
RNG = np.random.default_rng(7)


def random_pose(rng, angle=0.4, distance=0.35):
    rvec = rng.uniform(-angle, angle, 3)
    rotation = cv2.Rodrigues(rvec)[0]
    centre = np.array([TARGET.squares_x, TARGET.squares_y, 0]) * TARGET.square_m / 2
    translation = np.array([rng.uniform(-0.08, 0.08), rng.uniform(-0.06, 0.06), distance])
    return transforms.transform(rotation, translation - rotation @ centre)


def render_target(camera, camera_from_target, px_per_m=4000, supersample=3):
    width = int(TARGET.squares_x * TARGET.square_m * px_per_m)
    height = int(TARGET.squares_y * TARGET.square_m * px_per_m)
    board_image = TARGET.board().generateImage((width, height), marginSize=0)
    h, w = camera.height * supersample, camera.width * supersample
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    raw = np.column_stack(
        [(us.ravel() + 0.5) / supersample - 0.5, (vs.ravel() + 0.5) / supersample - 0.5]
    )
    rays = camera.ideal_to_rays(camera.undistort_px(raw))
    target_from_camera = transforms.invert(camera_from_target)
    origin, directions = target_from_camera[:3, 3], rays @ target_from_camera[:3, :3].T
    points = origin[:2] + directions[:, :2] * (-origin[2] / directions[:, 2])[:, None]
    maps = (points * px_per_m - 0.5).astype(np.float32).reshape(h, w, 2)
    image = cv2.remap(board_image, maps[..., 0], maps[..., 1], cv2.INTER_LINEAR, borderValue=255)
    return cv2.resize(image, (camera.width, camera.height), interpolation=cv2.INTER_AREA)


@pytest.fixture(scope="module")
def camera():
    model, _ = CameraModel.from_calibration_yaml(CAMERA_INFO)
    return model


def test_intrinsics_from_synthetic_views(camera):
    detections = []
    for _ in range(10):
        detection = detect_charuco(render_target(camera, random_pose(RNG)), TARGET)
        assert detection is not None
        detections.append(detection)
    info, rms = calibrate_intrinsics(detections)
    recovered, problem = CameraModel.from_calibration_yaml(info)
    assert problem is None
    assert rms < 0.5
    assert recovered.k[0, 0] == pytest.approx(500, rel=0.01)
    assert recovered.k[:2, 2] == pytest.approx([320, 240], abs=3)
    # The two models agree where the views had target data (not out at the image corners).
    grid = np.array([[x, y] for x in (150, 320, 490) for y in (110, 240, 370)], float)
    assert np.abs(recovered.undistort_px(grid) - camera.undistort_px(grid)).max() < 1.0


def test_too_few_views():
    with pytest.raises(ValueError, match="at least"):
        calibrate_intrinsics([])


def test_target_pose_and_bench_plane(camera):
    truth = random_pose(np.random.default_rng(1), angle=0.2)
    detection = detect_charuco(render_target(camera, truth), TARGET)
    pose, rms = target_pose(detection, camera)
    assert rms < 0.5
    assert pose[:3, 3] == pytest.approx(truth[:3, 3], abs=1e-3)
    # A single view of a small planar target pins its tilt to a few tenths of a degree.
    assert transforms.rotation_angle(pose[:3, :3].T @ truth[:3, :3]) < math.radians(0.5)

    normal, distance = bench_plane(pose, 0.003)
    assert np.linalg.norm(normal) == pytest.approx(1.0)
    assert np.dot(normal, pose[:3, 3]) > 0
    assert distance == pytest.approx(np.dot(normal, pose[:3, 3]) + 0.003)


def random_motion(rng):
    return transforms.transform(
        cv2.Rodrigues(rng.uniform(-0.6, 0.6, 3))[0], rng.uniform(-0.2, 0.2, 3)
    )


def test_eye_in_hand_recovers_the_mount():
    rng = np.random.default_rng(2)
    link_from_camera = transforms.transform(
        cv2.Rodrigues(np.array([0.1, -0.2, 0.3]))[0], [0.03, 0.01, -0.06]
    )
    base_from_target = transforms.transform(np.eye(3), [0.3, 0.0, 0.0])
    links = [random_motion(rng) for _ in range(12)]
    views = [transforms.invert(b @ link_from_camera) @ base_from_target for b in links]
    fit = eye_in_hand(links, views)
    assert np.allclose(fit.transform, link_from_camera, atol=1e-6)
    assert fit.rms_m < 1e-6 and fit.rms_rad < 1e-6


def test_eye_to_hand_recovers_the_camera():
    rng = np.random.default_rng(3)
    base_from_camera = transforms.transform(
        cv2.Rodrigues(np.array([2.9, 0.1, 0.05]))[0], [0.2, -0.1, 0.9]
    )
    link_from_target = transforms.transform(np.eye(3), [0.0, 0.0, -0.05])
    links = [random_motion(rng) for _ in range(12)]
    views = [transforms.invert(base_from_camera) @ b @ link_from_target for b in links]
    fit = eye_to_hand(links, views)
    assert np.allclose(fit.transform, base_from_camera, atol=1e-6)


def test_joint_offsets_are_recovered(arm):
    rng = np.random.default_rng(4)
    truth = {2: 0.02, 3: -0.015, 4: 0.01, 5: 0.03}
    camera_from_base = transforms.transform(
        cv2.Rodrigues(np.array([1.2, 0.2, -0.1]))[0], [0.1, 0.0, 0.9]
    )
    link_from_target = transforms.transform(
        cv2.Rodrigues(np.array([0.0, 0.3, 0.0]))[0], [0.0, 0.02, -0.08]
    )
    samples = []
    for _ in range(25):
        q = np.array(ARM_POSE) + np.r_[rng.uniform(-0.4, 0.4, 6), 0.0]
        true_q = q.copy()
        for joint, delta in truth.items():
            true_q[joint - 1] += delta  # the joint is really at sdk + delta (identity map)
        link = arm.link_poses(true_q)["left_link_007"]
        samples.append((tuple(q), camera_from_base @ link @ link_from_target))
    fit = calibrate_joint_offsets(arm, samples)
    for joint, delta in truth.items():
        assert fit.offsets_rad[f"left_{ARM_JOINTS[joint - 1]}"] == pytest.approx(delta, abs=1e-5)
    assert fit.joint_map[1].offset_rad == pytest.approx(0.02, abs=1e-5)
    assert fit.joint_map[0].offset_rad == 0.0  # J1 isn't fitted
    assert fit.rms_m < 1e-6
    assert np.allclose(fit.camera_from_base, camera_from_base, atol=1e-5)


def test_joint_offset_fit_needs_samples(arm):
    with pytest.raises(ValueError, match="need at least"):
        calibrate_joint_offsets(arm, [])


def dot_image(centre, radius=6, noise=3.0, seed=0):
    image = np.full((480, 640), 200, np.uint8)
    cv2.circle(
        image,
        (round(centre[0] * 16), round(centre[1] * 16)),
        radius * 16,
        30,
        -1,
        cv2.LINE_AA,
        4,
    )
    jitter = np.random.default_rng(seed).normal(0, noise, image.shape)
    return np.clip(image + jitter, 0, 255).astype(np.uint8)


def test_tip_pixel_from_marks():
    images = [dot_image((331.2 + 0.2 * i, 402.7 - 0.1 * i), seed=i) for i in range(5)]
    pixel, spread = tip_pixel_from_marks(images, 0.03, 3, 12)
    assert pixel.uv == pytest.approx((331.6, 402.5), abs=0.3)
    assert pixel.standoff_m == 0.03
    assert spread < 1.0


def test_tip_pixel_rejects_a_missing_dot():
    blank = np.full((480, 640), 200, np.uint8)
    with pytest.raises(ValueError, match="no single dark dot"):
        tip_pixel_from_marks([blank] * 3, 0.03, 3, 12)


def test_pose_from_matrix_round_trips():
    rng = np.random.default_rng(5)
    for _ in range(20):
        matrix = random_motion(rng)
        assert np.allclose(transforms.from_pose(pose_from_matrix(matrix)), matrix)


def test_calibration_yaml_loads_in_the_arm_package(arm):
    from hardware_troubleshooter_arm.config import JointMap, TipPixel

    data = arm_calibration_dict(
        "left-2026-10-24",
        "left",
        joint_map=(JointMap(1.0, 0.01),) * 7,
        wrist_camera=((0.05, 0.0, -0.08), (0.0, 0.3, 0.0)),
        tool_tips={"probe": ((0.0, 0.0, -0.05), (0.0, 0.0, 0.0))},
        tip_pixels={"probe": TipPixel((320.0, 400.0), 0.03)},
    )
    calibration, problems = calibration_from_dict(yaml.safe_load(yaml.safe_dump(data)))
    assert not problems
    assert calibration.tip_pixels["probe"].standoff_m == 0.03
    assert calibration.joint_map[0].offset_rad == 0.01


def test_base_in_rig_composes():
    rig_from_camera = transforms.transform(np.diag([1.0, -1.0, -1.0]), [0.2, 0, 1.0])
    camera_from_base = transforms.invert(rig_from_camera)
    xyz, rpy = base_in_rig(rig_from_camera, camera_from_base)
    assert np.allclose(xyz, 0) and np.allclose(rpy, 0)
