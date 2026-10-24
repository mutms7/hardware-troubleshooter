"""One object the measurement orchestrator talks to: the whole vision stack, wired up.

The caller owns the cameras, the arm SDK, the clock and the transport. It feeds
joint samples (add_joint_sample) and frames (process_fixed_frame, align) stamped on
one clock, and calls tick() periodically. Every answer is a result with reasons;
nothing here moves an arm, and nothing here establishes electrical contact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from hardware_troubleshooter_arm.toolhead import check_tool, tip_pixel
from hardware_troubleshooter_registration.camera import CameraModel
from hardware_troubleshooter_registration.config import load_board_config, load_rig_config
from hardware_troubleshooter_registration.engine import RegistrationEngine
from hardware_troubleshooter_registration.geometry import board_to_camera

from . import transforms
from .alignment import AlignmentResult, align, correction_in_base
from .capture import Frame, JointHistory, JointSample, tag_wrist_frame
from .config import VisionConfig, load_vision_config
from .occlusion import arm_mask
from .pads import Pad, pads_from_kicad, pads_from_specs
from .rig import ArmRuntime, fixed_camera_extrinsic
from .scene import SceneGuard, SceneResult


@dataclass
class ArmAlignment:
    """An alignment plus what it was measured against."""

    result: AlignmentResult
    arm: str
    tool: str
    image_stamp: float | None = None
    joints_stamp: float | None = None
    registration_id: str = ""
    correction_base_m: np.ndarray | None = None  # move the tool by this (arm base frame)

    @property
    def ok(self) -> bool:
        return self.result.ok


class VisionService:
    def __init__(self, config: VisionConfig, engine, fixed_camera, fixed_frame, arms, pads):
        self.config = config
        self.engine: RegistrationEngine = engine
        self.fixed_camera: CameraModel = fixed_camera
        self.fixed_frame: str = fixed_frame
        self.arms: dict[str, ArmRuntime] = arms
        self.pads: list[Pad] = pads
        self.histories = {name: JointHistory() for name in arms}
        self.scene = SceneGuard(config.scene)

    @classmethod
    def load(cls, path: str, session_id: str) -> tuple[VisionService | None, list[str]]:
        config, problems = load_vision_config(path)
        problems += [f"unconfigured:{key}" for key in config.missing_for_registration()]
        problems += [f"unconfigured:{key}" for key in config.pads.missing()]
        if problems:
            return None, problems

        board, board_problems = load_board_config(str(config.board_config))
        rig, rig_problems = load_rig_config(str(config.rig_config))
        engine = RegistrationEngine(board, board_problems, rig, rig_problems, session_id)
        try:
            info = yaml.safe_load(Path(config.fixed_camera_info).read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            return None, [f"invalid:fixed_camera_info ({exc})"]
        camera, problem = CameraModel.from_calibration_yaml(info)
        if camera is None:
            return None, [f"fixed_camera_info:{problem}"]
        rig_from_camera, frame, problem = fixed_camera_extrinsic(config.rig_config)
        extrinsic = (
            transforms.to_quaternion(rig_from_camera) if rig_from_camera is not None else None
        )
        engine.set_extrinsic(extrinsic, problem, 0.0)

        arms = {}
        for name, entry in config.arms.items():
            runtime, arm_problems = ArmRuntime.load(entry)
            problems += arm_problems
            if runtime is not None:
                arms[name] = runtime
        if not config.arms:
            problems.append("unconfigured:arms")

        if config.pads.source == "kicad":
            try:
                pads = pads_from_kicad(Path(config.pads.kicad_pcb).read_text(encoding="utf-8"))
            except (OSError, ValueError, IndexError, TypeError) as exc:
                pads = []
                problems.append(f"invalid:pads.kicad_pcb ({exc})")
        else:
            pads, pad_problems = pads_from_specs(config.pads.measured)
            problems += pad_problems
        if problems:
            return None, problems
        return cls(config, engine, camera, frame or "", arms, pads), []

    # --- inputs --------------------------------------------------------------------------

    def add_joint_sample(self, arm: str, sample: JointSample) -> None:
        self.histories[arm].add(sample)

    def process_fixed_frame(self, frame: Frame):
        """Registration on the fixed camera, with the arms masked out at the frame's stamp."""
        mask = self.arms_mask(frame.stamp)
        return self.engine.process_frame(
            frame.image, frame.stamp, self.fixed_frame, self.fixed_camera, None, mask
        )

    def tick(self, now: float) -> None:
        self.engine.tick(now)

    def capture_scene_reference(self, frame: Frame) -> None:
        self.scene.set_reference(frame.image)

    # --- outputs -------------------------------------------------------------------------

    def board_state(self, now: float):
        return self.engine.snapshot(now)

    def drain_events(self):
        return self.engine.drain_events()

    def base_from_board(self, arm: str, now: float) -> np.ndarray | None:
        """Arm base_link <- board, only while registration is valid."""
        board = self.engine.board_in_base(now)
        if board is None:
            return None
        _, _, rotation, translation = board
        rig_from_board = transforms.transform(rotation, translation)
        return transforms.invert(self.arms[arm].rig_from_base) @ rig_from_board

    def arms_mask(self, stamp: float) -> np.ndarray:
        """Fixed-camera mask of every arm at `stamp`. An arm without a joint sample within
        capture.max_joint_skew_s masks the whole image: its position is unknown."""
        shape = (self.fixed_camera.height, self.fixed_camera.width)
        mask = np.zeros(shape, np.uint8)
        rig_from_camera, _, _ = fixed_camera_extrinsic(self.config.rig_config)
        skew = self.config.capture.max_joint_skew_s
        margin = self.config.occlusion.margin_px
        if rig_from_camera is None or skew is None or margin is None:
            return np.full(shape, 255, np.uint8)
        camera_from_rig = transforms.invert(rig_from_camera)
        for name, arm in self.arms.items():
            sample = self.histories[name].nearest(stamp)
            if sample is None or abs(sample.stamp - stamp) > skew:
                return np.full(shape, 255, np.uint8)
            poses = arm.link_poses(sample.positions)
            mask |= arm_mask(
                arm.geometry, poses, camera_from_rig @ arm.rig_from_base, self.fixed_camera, margin
            )
        return mask

    def scene_clear(self, frame: Frame) -> SceneResult:
        snapshot = self.engine.snapshot(frame.stamp)
        board = None
        if snapshot.has_pose:
            board = self.fixed_camera.project(
                board_to_camera(
                    self.engine.board.outline.polyline, snapshot.rotation, snapshot.translation
                )
            )
        return self.scene.check(frame.image, self.arms_mask(frame.stamp), board)

    def align(self, arm: str, frame: Frame, target: str, tool: str) -> ArmAlignment:
        """Measure where `tool` on `arm` would land relative to pad `target`, from one
        wrist frame taken at the hover pose with the arm settled."""
        out = ArmAlignment(AlignmentResult(False, target), arm, tool, frame.stamp)

        def fail(reason):
            out.result.reasons.append(reason)
            return out

        runtime = self.arms.get(arm)
        if runtime is None:
            return fail(f"unknown_arm:{arm}")
        tagged, reason = tag_wrist_frame(frame, arm, self.histories[arm], self.config.capture)
        if tagged is None:
            return fail(reason)
        out.joints_stamp = tagged.joints.stamp
        reason = check_tool(runtime.config, tool, runtime.toolhead_angle(tagged.joints.positions))
        if reason:
            return fail(reason)
        base_from_board = self.base_from_board(arm, frame.stamp)
        if base_from_board is None:
            return fail(f"registration_invalid:{self.engine.snapshot(frame.stamp).failure_reason}")
        out.registration_id = self.engine.snapshot(frame.stamp).registration_id

        poses = runtime.link_poses(tagged.joints.positions)
        board_from_base = transforms.invert(base_from_board)
        camera_from_board = (
            transforms.invert(runtime.base_from_wrist_camera(poses)) @ base_from_board
        )
        tip_height = (board_from_base @ runtime.base_from_tool(poses, tool))[2, 3]
        gray = (
            frame.image if frame.image.ndim == 2 else cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        )
        out.result = align(
            gray,
            runtime.wrist_camera,
            camera_from_board,
            self.pads,
            target,
            tip_pixel(runtime.calibration, tool),
            float(tip_height),
            self.config.alignment,
        )
        if out.result.ok:
            out.correction_base_m = correction_in_base(out.result, base_from_board)
        return out
