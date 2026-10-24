import dataclasses

import cv2
import numpy as np
import pytest
from hardware_troubleshooter_registration.camera import CameraModel
from hardware_troubleshooter_vision import transforms
from hardware_troubleshooter_vision.board_features import BoardFeatureDetector
from hardware_troubleshooter_vision.capture import Frame
from hardware_troubleshooter_vision.config import BoardReference
from hardware_troubleshooter_vision.homing import WristHoming
from test_homing import CONFIG

BOUNDS = np.array([[-0.08, -0.06, 0], [0.08, -0.06, 0], [0.08, 0.06, 0], [-0.08, 0.06, 0]])
POSE = transforms.transform(np.diag([1.0, -1.0, -1.0]), [0.0, 0.0, 0.3])


def make_reference(tmp_path):
    camera, problem = CameraModel.from_camera_info(
        640, 480, [650, 0, 320, 0, 650, 240, 0, 0, 1], [0, 0, 0, 0, 0], "plumb_bob"
    )
    assert problem is None
    rng = np.random.default_rng(14)
    texture = np.full((600, 800), 100, np.uint8)
    for _ in range(600):
        x, y = rng.integers([5, 5], [795, 595])
        cv2.circle(
            texture, (int(x), int(y)), int(rng.integers(2, 12)), int(rng.integers(20, 240)), -1
        )
    pixels = camera.project(transforms.apply(POSE, BOUNDS))

    def image(camera_pose):
        dest = camera.project(transforms.apply(transforms.invert(camera_pose) @ POSE, BOUNDS))
        h = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [799, 0], [799, 599], [0, 599]]), dest.astype(np.float32)
        )
        return cv2.warpPerspective(texture, h, (640, 480), borderValue=30)

    path = tmp_path / "reference.png"
    cv2.imwrite(str(path), image(np.eye(4)))
    landmarks = tuple(
        {"pixel": p.tolist(), "board_m": o[:2].tolist()} for p, o in zip(pixels, BOUNDS)
    )
    config = BoardReference("pcb", path, landmarks, 0.75, 1.5, 12)
    return config, camera, image


def test_detected_surface_features_feed_two_view_pnp(tmp_path):
    config, camera, image = make_reference(tmp_path)
    detector = BoardFeatureDetector(config, camera)
    engine = WristHoming(CONFIG, "features")
    engine.begin("left", "TP1", "pcb", "cal", 0.0)
    for stamp, x in [(0.0, 0.0), (1.0, 0.03)]:
        pose = transforms.transform(np.eye(3), [x, 0, 0])
        observation, reason = detector.detect(Frame("left_wrist", stamp, image(pose)), camera)
        assert observation is not None, reason
        result = engine.observe(observation, pose, camera, stamp, "cal")
        assert result.ok, result.reasons
    snapshot = engine.snapshot(1.0)
    assert snapshot.has_pose
    assert snapshot.rig_from_board[:3, 3] == pytest.approx(POSE[:3, 3], abs=0.001)
    assert result.diagnostics["rms_px"] < 0.6


def test_untextured_wrong_size_and_invalid_reference(tmp_path):
    config, camera, _ = make_reference(tmp_path)
    detector = BoardFeatureDetector(config, camera)
    assert (
        detector.detect(Frame("left_wrist", 1.0, np.zeros((480, 640), np.uint8)), camera)[0] is None
    )
    assert (
        detector.detect(Frame("left_wrist", 1.0, np.zeros((240, 320), np.uint8)), camera)[1]
        == "camera_image_size_mismatch"
    )
    with pytest.raises(ValueError, match="landmarks"):
        BoardFeatureDetector(dataclasses.replace(config, landmarks=config.landmarks[:3]), camera)
