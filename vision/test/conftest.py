"""Shared fixtures. Every value describes an imaginary rig and is test-only."""

import copy
import math
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
for folder in ("registration", "arm", "vision", "arm/test"):
    sys.path.insert(0, str(ROOT / folder))

from arm_synthetic import SYNTHETIC_ARM, write_stl
from hardware_troubleshooter_vision import transforms
from hardware_troubleshooter_vision.config import ArmEntry
from hardware_troubleshooter_vision.rig import ArmRuntime

ARM_POSE = (0.3, -0.8, 1.2, 0.4, -0.5, 0.7, 0.0)

# A 640x480 camera with mild distortion, used for both the wrist and the fixed camera.
CAMERA_INFO = {
    "image_width": 640,
    "image_height": 480,
    "camera_matrix": {"rows": 3, "cols": 3, "data": [500.0, 0, 320.0, 0, 500.0, 240.0, 0, 0, 1]},
    "distortion_model": "plumb_bob",
    "distortion_coefficients": {"rows": 1, "cols": 5, "data": [-0.05, 0.01, 0, 0, 0]},
}

# The Maker URDF base is Y-up; this turns it Z-up in the rig frame.
BASE_IN_RIG = {"xyz_m": [0.0, 0.0, 0.0], "rpy_rad": [math.pi / 2, 0.0, 0.0]}


def write_arm(directory: Path, name="left", **changes) -> ArmEntry:
    directory.mkdir(parents=True, exist_ok=True)
    write_stl(directory / "toolhead.stl")
    arm = copy.deepcopy(SYNTHETIC_ARM)
    arm.update(name=name, **changes)
    (directory / f"{name}.yaml").write_text(yaml.safe_dump(arm), encoding="utf-8")
    (directory / "wrist.yaml").write_text(yaml.safe_dump(CAMERA_INFO), encoding="utf-8")
    return ArmEntry(
        name,
        directory / f"{name}.yaml",
        None,
        (tuple(BASE_IN_RIG["xyz_m"]), tuple(BASE_IN_RIG["rpy_rad"])),
        directory / "wrist.yaml",
    )


@pytest.fixture
def arm(tmp_path):
    runtime, problems = ArmRuntime.load(write_arm(tmp_path / "left"))
    assert not problems, problems
    return runtime


@pytest.fixture
def camera():
    from hardware_troubleshooter_registration.camera import CameraModel

    model, problem = CameraModel.from_calibration_yaml(CAMERA_INFO)
    assert problem is None
    return model


def overhead_camera(height=1.2, x=0.15):
    """rig <- camera for a camera looking straight down at the rig origin area."""
    import numpy as np

    rotation = np.diag([1.0, -1.0, -1.0])  # optical z down, image y toward rig -y
    return transforms.transform(rotation, [x, 0.0, height])
