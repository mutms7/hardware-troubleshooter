"""Shared synthetic fixtures. Test-only values; see synthetic.py."""

import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hardware_troubleshooter_registration.camera import CameraModel
from hardware_troubleshooter_registration.config import board_from_dict, rig_from_dict
from hardware_troubleshooter_registration.geometry import PlaneFrame
from hardware_troubleshooter_registration.synthetic import (
    SYNTHETIC_BOARD,
    SYNTHETIC_CAMERA_INFO,
    SYNTHETIC_RIG,
)


class Scene:
    def __init__(self, board_dict=None, rig_dict=None):
        self.board_dict = copy.deepcopy(board_dict or SYNTHETIC_BOARD)
        self.rig_dict = copy.deepcopy(rig_dict or SYNTHETIC_RIG)
        self.board, problems = board_from_dict(self.board_dict, Path("."))
        assert not problems, problems
        self.rig, problems = rig_from_dict(self.rig_dict)
        assert not problems, problems
        self.camera, problem = CameraModel.from_calibration_yaml(SYNTHETIC_CAMERA_INFO)
        assert problem is None
        self.plane = PlaneFrame.from_bench(
            self.rig.bench_normal, self.rig.bench_distance_m, self.board.thickness_m
        )

    def centred_pose(self, dx=0.0, dy=0.0, theta=0.0):
        """In-plane pose that puts the board centre near the optical axis, plus an offset."""
        axis = self.plane.rays_to_plane(np.array([[0.0, 0.0, 1.0]]))[0]
        centre = self.board.corners.mean(axis=0)
        c, s = math.cos(theta), math.sin(theta)
        rotated = np.array([c * centre[0] - s * centre[1], s * centre[0] + c * centre[1]])
        x, y = axis - rotated + (dx, dy)
        return float(x), float(y), float(theta)


@pytest.fixture
def scene():
    return Scene()
