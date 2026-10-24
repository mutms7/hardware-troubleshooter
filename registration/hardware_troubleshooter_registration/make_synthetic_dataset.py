"""Write a synthetic image folder, camera_info and configs for offline testing.

The sequence is: static board, board moved, board covered. Every value describes an
imaginary rig; the generated configs say so and must not be reused for live hardware.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import yaml

from .camera import CameraModel
from .config import board_from_dict, rig_from_dict
from .geometry import PlaneFrame
from .synthetic import SYNTHETIC_BOARD, SYNTHETIC_CAMERA_INFO, SYNTHETIC_RIG, render_frame

_HEADER = "# SYNTHETIC TEST VALUES for an imaginary rig. Never use for live hardware.\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--static", type=int, default=10, help="frames before the move")
    parser.add_argument("--moved", type=int, default=10, help="frames after the move")
    parser.add_argument("--covered", type=int, default=10, help="frames with the board covered")
    args = parser.parse_args(argv)

    board, problems = board_from_dict(SYNTHETIC_BOARD, Path("."))
    rig, rig_problems = rig_from_dict(SYNTHETIC_RIG)
    camera, problem = CameraModel.from_calibration_yaml(SYNTHETIC_CAMERA_INFO)
    if problems or rig_problems or problem:
        raise SystemExit(f"synthetic config is inconsistent: {problems + rig_problems + [problem]}")
    plane = PlaneFrame.from_bench(rig.bench_normal, rig.bench_distance_m, board.thickness_m)
    axis = plane.rays_to_plane(np.array([[0.0, 0.0, 1.0]]))[0]
    theta = math.radians(15)
    centre = board.corners.mean(axis=0)
    c, s = math.cos(theta), math.sin(theta)
    x, y = axis - np.array([c * centre[0] - s * centre[1], s * centre[0] + c * centre[1]])
    poses = [(x, y, theta)] * args.static + [(x + 0.01, y - 0.005, theta + 0.1)] * args.moved
    covered = np.array(
        [[0, 0], [camera.width, 0], [camera.width, camera.height], [0, camera.height]]
    )

    frames = args.out_dir / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    index = 0
    for i, pose in enumerate(poses):
        image = render_frame(board, camera, plane, pose, noise_sigma=2.0, seed=i)
        cv2.imwrite(str(frames / f"{index:04d}.png"), image)
        index += 1
    for i in range(args.covered):
        image = render_frame(
            board, camera, plane, poses[-1], occluders=[covered], noise_sigma=2.0, seed=i
        )
        cv2.imwrite(str(frames / f"{index:04d}.png"), image)
        index += 1

    for name, data in (
        ("camera_info.yaml", SYNTHETIC_CAMERA_INFO),
        ("board.yaml", SYNTHETIC_BOARD),
        ("rig.yaml", SYNTHETIC_RIG),
    ):
        (args.out_dir / name).write_text(_HEADER + yaml.safe_dump(data, sort_keys=False))
    print(f"wrote {index} frames and configs to {args.out_dir}")


if __name__ == "__main__":
    main()
