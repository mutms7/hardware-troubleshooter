"""One object the measurement orchestrator talks to: the whole vision stack, wired up.

The caller owns the cameras, the arm SDK, the clock and the transport. It feeds
joint samples and wrist frames (home_target/observe_board/align) stamped on
one clock, and calls tick() periodically. Every answer is a result with reasons;
nothing here moves an arm, and nothing here establishes electrical contact.
"""

from __future__ import annotations

import hashlib
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
from .board_features import BoardFeatureDetector
from .capture import Frame, JointHistory, JointSample, tag_wrist_frame
from .config import VisionConfig, load_vision_config
from .homing import HomingResult, WristHoming
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
    def __init__(
        self,
        config: VisionConfig,
        engine,
        fixed_camera,
        fixed_frame,
        arms,
        pads,
        detector=None,
        session_id="",
    ):
        self.config = config
        self.engine: RegistrationEngine = engine
        self.fixed_camera: CameraModel = fixed_camera
        self.fixed_frame: str = fixed_frame
        self.arms: dict[str, ArmRuntime] = arms
        self.pads: list[Pad] = pads
        self.histories = {name: JointHistory() for name in arms}
        self.scene = SceneGuard(config.scene)
        self.detector = detector
        self._homing_busy = False
        self.homing = (
            WristHoming(config.homing, session_id)
            if config.registration_source == "wrist"
            else None
        )

    @classmethod
    def load(cls, path: str, session_id: str) -> tuple[VisionService | None, list[str]]:
        config, problems = load_vision_config(path)
        problems += [f"unconfigured:{key}" for key in config.missing_for_registration()]
        problems += [f"unconfigured:{key}" for key in config.pads.missing()]
        if problems:
            return None, problems

        engine, camera, frame, detector = None, None, "", None
        if config.registration_source == "fixed":
            board, board_problems = load_board_config(str(config.board_config))
            rig, rig_problems = load_rig_config(str(config.rig_config))
            engine = RegistrationEngine(board, board_problems, rig, rig_problems, session_id)
            try:
                info = (
                    yaml.safe_load(Path(config.fixed_camera_info).read_text(encoding="utf-8")) or {}
                )
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
        else:
            try:
                info = (
                    yaml.safe_load(config.board_reference.camera_info.read_text(encoding="utf-8"))
                    or {}
                )
                reference_camera, reason = CameraModel.from_calibration_yaml(info)
                if reference_camera is None:
                    raise ValueError(reason)
                detector = BoardFeatureDetector(config.board_reference, reference_camera)
            except (OSError, ValueError, TypeError, cv2.error, yaml.YAMLError) as exc:
                return None, [f"invalid:board_reference ({exc})"]

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
        return cls(config, engine, camera, frame or "", arms, pads, detector, session_id), []

    # --- inputs --------------------------------------------------------------------------

    def add_joint_sample(self, arm: str, sample: JointSample) -> None:
        self.histories[arm].add(sample)

    def process_fixed_frame(self, frame: Frame, arm_mask=None):
        """RGB overview only in wrist mode; board-edge registration in legacy fixed mode."""
        if self.homing is not None:
            # RGB overview only. A change can revoke a pose, never establish one.
            result = self.scene.check(frame.image, arm_mask)
            if not result.clear:
                self.homing.invalidate("overview_changed:" + "; ".join(result.reasons))
            return result
        mask = self.arms_mask(frame.stamp)
        return self.engine.process_frame(
            frame.image, frame.stamp, self.fixed_frame, self.fixed_camera, None, mask
        )

    def tick(self, now: float) -> None:
        (self.homing or self.engine).tick(now)

    def capture_scene_reference(self, frame: Frame) -> None:
        self.scene.set_reference(frame.image)

    # --- outputs -------------------------------------------------------------------------

    def board_state(self, now: float):
        if (
            self.homing is not None
            and self.homing.arm
            and self.homing.calibration_revision != self._calibration_revision(self.homing.arm)
        ):
            self.homing.invalidate("calibration_changed")
        return (self.homing or self.engine).snapshot(now)

    def drain_events(self):
        return (self.homing or self.engine).drain_events()

    def base_from_board(self, arm: str, now: float) -> np.ndarray | None:
        """Arm base_link <- board, only while registration is valid."""
        if self.homing is not None:
            snapshot = self.board_state(now)
            if not snapshot.has_pose or arm != snapshot.arm:
                return None
            return transforms.invert(self.arms[arm].rig_from_base) @ snapshot.rig_from_board
        board = self.engine.board_in_base(now)
        if board is None:
            return None
        _, _, rotation, translation = board
        rig_from_board = transforms.transform(rotation, translation)
        return transforms.invert(self.arms[arm].rig_from_base) @ rig_from_board

    def _calibration_revision(self, arm):
        runtime = self.arms[arm]
        payload = (
            repr(runtime.config),
            repr(runtime.calibration),
            runtime.rig_from_base.tolist(),
            runtime.wrist_camera.fingerprint(),
            self.detector.fingerprint,
        )
        return hashlib.sha256(repr(payload).encode()).hexdigest()[:16]

    def begin_homing(self, arm: str, target: str, now: float) -> HomingResult:
        if self._homing_busy:
            return HomingResult(False, "busy", ["homing_busy"])
        return self._begin_homing(arm, target, now)

    def _begin_homing(self, arm: str, target: str, now: float) -> HomingResult:
        if self.homing is None:
            return HomingResult(False, "unavailable", ["wrist_homing_disabled"])
        if self.config.homing.missing():
            return HomingResult(False, "unavailable", ["homing_unconfigured"])
        if arm not in self.arms:
            return HomingResult(False, "unavailable", [f"unknown_arm:{arm}"])
        if not any(p.name == target for p in self.pads):
            return HomingResult(False, "unavailable", [f"unknown_target:{target}"])
        if not np.isfinite(now):
            return HomingResult(False, "unavailable", ["invalid_frame_stamp"])
        runtime = self.arms[arm]
        cal = runtime.calibration
        if (
            cal is None
            or not cal.calibration_id
            or cal.joint_map is None
            or cal.wrist_camera is None
        ):
            self.homing.invalidate("arm_camera_uncalibrated")
            return HomingResult(False, "invalid", ["arm_camera_uncalibrated"])
        self.homing.begin(
            arm, target, self.config.board_reference.board_id, self._calibration_revision(arm), now
        )
        return HomingResult(True, "homing")

    def observe_board(self, arm: str, frame: Frame, observation=None) -> HomingResult:
        """PnP for first/second homing views or a fresh confirmation before alignment.

        Optional observation is for an identified-feature adapter. The automatic path
        matches the configured measured reference. No unordered contour corners.
        """
        if self.homing is None or arm != self.homing.arm:
            return HomingResult(False, "unavailable", ["homing_arm_mismatch"])
        runtime = self.arms[arm]
        tagged, reason = tag_wrist_frame(frame, arm, self.histories[arm], self.config.capture)
        if tagged is None:
            return HomingResult(False, self.homing.status, [reason])
        if frame.camera_id != f"{arm}_wrist":
            return HomingResult(False, self.homing.status, ["wrist_camera_identity_mismatch"])
        if frame.image.shape[:2] != (runtime.wrist_camera.height, runtime.wrist_camera.width):
            return HomingResult(False, self.homing.status, ["camera_image_size_mismatch"])
        if observation is None:
            observation, reason = self.detector.detect(frame, runtime.wrist_camera)
        if observation is None:
            if self.homing.status == "valid":
                self.homing.invalidate("tracking_lost:" + reason)
            return HomingResult(False, self.homing.status, [reason])
        poses = runtime.link_poses(tagged.joints.positions)
        rig_from_camera = runtime.rig_from_base @ runtime.base_from_wrist_camera(poses)
        return self.homing.observe(
            observation,
            rig_from_camera,
            runtime.wrist_camera,
            frame.stamp,
            self._calibration_revision(arm),
        )

    def homing_step_base_m(self, arm: str) -> np.ndarray:
        """Requested sideways displacement, not measured motion or permission to descend."""
        if self.homing is None or self.homing.arm != arm or self.homing.first is None:
            raise ValueError("first_homing_view_required")
        rig_step = self.homing.first.rig_from_camera[:3, 0] * self.config.homing.step_m
        return self.arms[arm].rig_from_base[:3, :3].T @ rig_step

    async def home_target(self, arm, target, acquire_settled, move_relative):
        """Complete homing through motion/capture adapters supplied by the orchestrator.

        acquire_settled(arm) -> (Frame, list[JointSample]); samples cover the settle
        window. move_relative(arm, translation_base_m) -> bool (completed/accepted).
        Both callbacks are async. Second-view measured joints establish the baseline,
        regardless of the command acknowledgement. The move adapter owns collision
        checks and guarded stop. No callback may substitute commanded joint positions.
        """
        if self._homing_busy:
            return HomingResult(False, "busy", ["homing_busy"])
        if (
            self.homing is None
            or arm not in self.arms
            or not any(p.name == target for p in self.pads)
        ):
            return HomingResult(False, "unavailable", ["homing_target_unavailable"])
        self._homing_busy = True
        self.homing.invalidate("homing_started")
        try:
            first, samples = await acquire_settled(arm)
            result = self._begin_homing(arm, target, first.stamp)
            if not result.ok:
                return result
            for sample in samples:
                self.add_joint_sample(arm, sample)
            result = self.observe_board(arm, first)
            if not result.ok:
                return result
            completed = await move_relative(arm, self.homing_step_base_m(arm))
            if completed is not True:
                self.homing.invalidate("homing_motion_failed")
                return HomingResult(False, "invalid", ["homing_motion_failed"])
            second, samples = await acquire_settled(arm)
            for sample in samples:
                self.add_joint_sample(arm, sample)
            return self.observe_board(arm, second)
        except BaseException:
            if self.homing is not None:
                self.homing.invalidate("homing_interrupted")
            raise
        finally:
            self._homing_busy = False

    def invalidate_board(self, reason="board_moved"):
        """Overview watcher/operator notification. Rehoming is required afterward."""
        if self.homing is None:
            raise ValueError("wrist_homing_disabled")
        self.homing.invalidate(reason)

    def pad_approach(self, arm: str, target: str, standoff_m: float, now: float):
        """Desired calibrated tool-tip position and board normal, in the arm base.

        The motion adapter uses its calibrated TCP and IK; this is not a joint command.
        """
        pose = self.base_from_board(arm, now)
        if pose is None or self.homing is None or target != self.homing.target:
            raise ValueError("target_homing_required")
        if not np.isfinite(standoff_m) or standoff_m <= 0:
            raise ValueError("positive_standoff_required")
        pad = next(p for p in self.pads if p.name == target)
        return transforms.apply(pose, [[*pad.center, standoff_m]])[0], pose[:3, 2].copy()

    def arms_mask(self, stamp: float) -> np.ndarray:
        """Fixed-camera mask of every arm at `stamp`. An arm without a joint sample within
        capture.max_joint_skew_s masks the whole image: its position is unknown."""
        if self.homing is not None:
            raise ValueError("fixed_camera_geometry_unavailable:provide an image-space mask")
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
        if self.homing is not None:
            return self.process_fixed_frame(frame)
        snapshot = self.engine.snapshot(frame.stamp)
        board = None
        if snapshot.has_pose:
            board = self.fixed_camera.project(
                board_to_camera(
                    self.engine.board.outline.polyline, snapshot.rotation, snapshot.translation
                )
            )
        return self.scene.check(frame.image, self.arms_mask(frame.stamp), board)

    def align(
        self, arm: str, frame: Frame, target: str, tool: str, observation=None
    ) -> ArmAlignment:
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
        if self.homing is not None:
            if runtime.calibration is None or tool not in runtime.calibration.tool_tips:
                return fail("tool_tip_uncalibrated")
            state = self.board_state(frame.stamp)
            if not state.has_pose or state.target != target or state.arm != arm:
                return fail("registration_invalid:target_homing_required")
            confirmation = self.observe_board(arm, frame, observation)
            if not confirmation.ok:
                return fail("registration_invalid:" + "; ".join(confirmation.reasons))
        base_from_board = self.base_from_board(arm, frame.stamp)
        if base_from_board is None:
            return fail(f"registration_invalid:{self.board_state(frame.stamp).failure_reason}")
        out.registration_id = self.board_state(frame.stamp).registration_id

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
