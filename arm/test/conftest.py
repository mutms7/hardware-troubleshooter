"""Shared fixtures over arm_synthetic's test-only values."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arm_synthetic import SYNTHETIC_ARM, SYNTHETIC_CALIBRATION, write_stl
from hardware_troubleshooter_arm.config import arm_from_dict, calibration_from_dict


@pytest.fixture
def arm_dir(tmp_path):
    write_stl(tmp_path / "toolhead.stl")
    return tmp_path


@pytest.fixture
def make_arm(arm_dir):
    def make(**changes):
        data = copy.deepcopy(SYNTHETIC_ARM)
        data.update(changes)
        return arm_from_dict(data, arm_dir)

    return make


@pytest.fixture
def arm(make_arm):
    config, problems = make_arm()
    assert not problems, problems
    return config


@pytest.fixture
def calibration():
    calibration, problems = calibration_from_dict(copy.deepcopy(SYNTHETIC_CALIBRATION))
    assert not problems, problems
    return calibration
