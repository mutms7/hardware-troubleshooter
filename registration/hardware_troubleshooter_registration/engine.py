"""Registration state: revisions, invalidation, calibration status.

A revision's pose is fixed when it is created. Later frames only confirm it (within
the move thresholds) or invalidate it. An invalidated revision is never reported
valid again; the next accepted frame creates a new revision.

Run it on the rig's fixed camera. The base -> camera transform must stay constant
for a revision's lifetime; on the arm-mounted camera every arm move changes it and
invalidates the revision with calibration_changed.

While an arm covers part of the board, pass its image mask to process_frame. A live
revision is then confirmed (or found moved) from the visible edges alone; see
tracking.py. No revision is ever created from an occluded frame.

failure_reason entries are "code" or "code:detail", joined with "; ". When the
latest revision was invalidated, the first entry is board_moved, tracking_lost or
calibration_changed.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import PlaneFrame, quaternion_to_rotation, wrap_angle
from .registration import FrameResult, board_corners_in_plane, failed, register_frame
from .tracking import confirm_partial, mask_overlaps_board, search_radius_px

BOARD_MOVED = "board_moved"
TRACKING_LOST = "tracking_lost"
CALIBRATION_CHANGED = "calibration_changed"
INVALIDATION_REASONS = (BOARD_MOVED, TRACKING_LOST, CALIBRATION_CHANGED)


@dataclass
class Registration:
    revision: int
    registration_id: str
    result: FrameResult
    stamp: float  # observation time of the frame that produced the pose
    camera_fingerprint: str
    extrinsic: tuple | None
    last_confirmed: float
    confirmed_latest: bool = True
    invalidated: str | None = None  # set once, never cleared


@dataclass
class Snapshot:
    """Everything RegistrationState carries, plus diagnostics it has no fields for."""

    stamp: float | None
    camera_frame: str
    board_id: str
    registration_id: str
    calibration_revision: str
    status: str  # valid | invalid | unconfigured
    calibration_valid: bool
    tracking_valid: bool
    board_unmoved: bool
    board_frame: str
    has_pose: bool
    rotation: np.ndarray | None
    translation: np.ndarray | None
    valid_until: float | None
    residual_px: float | None
    reasons: list = field(default_factory=list)
    edge_points: int = 0
    symmetry_resolved: bool = False
    registration_stamp: float | None = None
    ippe_tilt_deg: float | None = None
    ippe_offset_m: float | None = None
    visible_edge_fraction: float | None = None  # set when the latest frame was occluded

    @property
    def failure_reason(self) -> str:
        return "; ".join(self.reasons)

    def event_key(self) -> tuple:
        first = self.reasons[0].split(":")[0] if self.reasons else ""
        invalidation = first if first in INVALIDATION_REASONS else ""
        return (
            self.registration_id,
            self.status,
            self.calibration_valid,
            self.tracking_valid,
            self.has_pose,
            invalidation,
        )


class RegistrationEngine:
    def __init__(self, board, board_problems, rig, rig_problems, session_id: str):
        self.board = board
        self.rig = rig
        self.session_id = session_id
        self.config_problems = list(board_problems) + list(rig_problems)
        self.config_problems += [
            f"unconfigured:{name}"
            for name in dict.fromkeys(rig.missing_for_tracking() + rig.missing_for_calibration())
        ]
        self.plane = None
        if board is not None and rig.bench_normal is not None and rig.bench_distance_m is not None:
            try:
                self.plane = PlaneFrame.from_bench(
                    rig.bench_normal, rig.bench_distance_m, board.thickness_m
                )
            except ValueError as exc:
                self.config_problems.append(f"invalid:rig.bench_plane ({exc})")
        self.tracking_configured = (
            board is not None
            and self.plane is not None
            and not board_problems
            and not rig_problems
            and not rig.missing_for_tracking()
        )
        self.revision = 0
        self.current: Registration | None = None
        self.camera = None
        self.camera_problem = "camera_info_missing"
        self.camera_frame = ""
        self.extrinsic = None
        self.extrinsic_problem = "extrinsic_unavailable:not looked up yet"
        self.last_frame_stamp = None
        self.last_result: FrameResult | None = None
        self._events: list[Snapshot] = []
        self._last_event_key = None

    # --- inputs -------------------------------------------------------------------

    def process_frame(
        self, image, stamp, camera_frame, camera, camera_problem=None, occlusion_mask=None
    ):
        """One image with the camera_info stamped for it. Returns the FrameResult.

        occlusion_mask: optional uint8 image of the same size, non-zero where an arm (or
        anything else the caller knows about) covers the view at this stamp."""
        if self.last_frame_stamp is not None and stamp <= self.last_frame_stamp:
            return None
        self.last_frame_stamp = stamp
        self.camera_frame = camera_frame
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        h, w = gray.shape
        if camera is not None and (camera.width, camera.height) != (w, h):
            camera_problem = (
                f"camera_info_invalid:size {camera.width}x{camera.height} != image {w}x{h}"
            )
            camera = None
        self.camera, self.camera_problem = camera, camera_problem

        live = self._live()
        if (
            live is not None
            and camera is not None
            and camera.fingerprint() != live.camera_fingerprint
        ):
            self._invalidate(CALIBRATION_CHANGED, "camera intrinsics changed", stamp)

        if occlusion_mask is not None and occlusion_mask.shape != gray.shape:
            result = failed(
                f"invalid:occlusion_mask size {occlusion_mask.shape[::-1]} != image {w}x{h}"
            )
        else:
            result = self._register(gray, self._live(), occlusion_mask)
        self.last_result = result
        # A frame that invalidates a revision never also creates the next one.
        invalidated = live is not None and self._live() is None
        live = self._live()
        if not self.tracking_configured:
            pass
        elif live is not None:
            if result.ok:
                moved = self._movement(live.result.pose, result.pose)
                if moved:
                    self._invalidate(BOARD_MOVED, moved, stamp)
                else:
                    live.last_confirmed = stamp
                    live.confirmed_latest = True
            else:
                live.confirmed_latest = False
                self._check_timeout(stamp)
        elif result.ok and not invalidated:
            self.revision += 1
            self.current = Registration(
                self.revision,
                f"{self.board.board_id}/{self.session_id}/r{self.revision}",
                result,
                stamp,
                camera.fingerprint(),
                self.extrinsic,
                stamp,
            )
        self._queue_event(stamp)
        return result

    def tick(self, now: float):
        """Periodic check so a camera that stops publishing still invalidates."""
        if self.tracking_configured:
            self._check_timeout(now)
        self._queue_event(now)

    def set_extrinsic(self, extrinsic, problem, now: float):
        """extrinsic = (translation xyz, quaternion xyzw) of the camera in the base frame."""
        self.extrinsic = extrinsic
        self.extrinsic_problem = problem
        live = self._live()
        if live is not None and extrinsic is not None:
            if live.extrinsic is None:
                live.extrinsic = extrinsic
            elif np.max(np.abs(np.concatenate(live.extrinsic) - np.concatenate(extrinsic))) > 1e-9:
                self._invalidate(
                    CALIBRATION_CHANGED,
                    "camera extrinsic changed (registration needs the fixed camera)",
                    now,
                )
        self._queue_event(now)

    # --- outputs ------------------------------------------------------------------

    def drain_events(self) -> list[Snapshot]:
        events, self._events = self._events, []
        return events

    def snapshot(self, now: float) -> Snapshot:
        board = self.board
        current = self.current
        live = self._live()
        reasons = []
        if current is not None and current.invalidated:
            reasons.append(current.invalidated)
        reasons += self.config_problems

        if self.camera is None:
            reasons.append(self.camera_problem)
        if self.extrinsic is None:
            reasons.append(self.extrinsic_problem)
        calibration_valid = (
            self.camera is not None
            and self.plane is not None
            and bool(self.rig.base_frame)
            and self.extrinsic is not None
        )

        timeout = self.rig.thresholds.tracking_loss_timeout_s
        tracking_valid = False
        if self.tracking_configured and live is not None:
            age = now - live.last_confirmed
            if not live.confirmed_latest:
                reasons.append("unconfirmed:latest frame didn't confirm the registration")
            elif age > timeout:
                reasons.append(f"stale_observation:{age:.3f}s since last confirming frame")
            else:
                tracking_valid = True
        if not tracking_valid:
            if self.last_result is None:
                reasons.append("no_camera_frames")
            elif not self.last_result.ok:
                reasons += self.last_result.reasons

        unconfigured = bool(self.config_problems) or board is None
        if unconfigured:
            status = "unconfigured"
        elif calibration_valid and tracking_valid:
            status = "valid"
        else:
            status = "invalid"

        source = live.result if live is not None else self.last_result
        return Snapshot(
            stamp=live.last_confirmed if live is not None else self.last_frame_stamp,
            camera_frame=self.camera_frame,
            board_id=board.board_id if board else "",
            registration_id=current.registration_id if current else "",
            calibration_revision=self._calibration_revision(),
            status=status,
            calibration_valid=calibration_valid,
            tracking_valid=tracking_valid,
            board_unmoved=tracking_valid,
            board_frame=board.frame_id if board else "",
            has_pose=live is not None,
            rotation=live.result.rotation if live is not None else None,
            translation=live.result.translation if live is not None else None,
            valid_until=live.last_confirmed + timeout if tracking_valid else None,
            residual_px=source.residual_px if source is not None else None,
            reasons=list(dict.fromkeys(reasons)),
            edge_points=source.edge_points if source is not None else 0,
            symmetry_resolved=source.symmetry_resolved if source is not None else False,
            registration_stamp=live.stamp if live is not None else None,
            ippe_tilt_deg=source.ippe_tilt_deg if source is not None else None,
            ippe_offset_m=source.ippe_offset_m if source is not None else None,
            visible_edge_fraction=(
                self.last_result.visible_fraction if self.last_result is not None else None
            ),
        )

    def board_transform(self, now: float):
        """(parent, child, rotation, translation) only while fully valid; otherwise None."""
        snapshot = self.snapshot(now)
        if snapshot.status != "valid":
            return None
        return snapshot.camera_frame, snapshot.board_frame, snapshot.rotation, snapshot.translation

    def board_in_base(self, now: float):
        """(base_frame, board_frame, rotation, translation) of the board in rig.base_frame,
        through the extrinsic the revision was made with; None unless fully valid."""
        snapshot = self.snapshot(now)
        live = self._live()
        if snapshot.status != "valid" or live is None or live.extrinsic is None:
            return None
        translation, quaternion = live.extrinsic
        base_from_camera = quaternion_to_rotation(quaternion)
        rotation = base_from_camera @ snapshot.rotation
        translation = base_from_camera @ snapshot.translation + np.asarray(translation, float)
        return self.rig.base_frame, snapshot.board_frame, rotation, translation

    # --- internals ----------------------------------------------------------------

    def _live(self) -> Registration | None:
        if self.current is None or self.current.invalidated:
            return None
        return self.current

    def _register(self, gray, live, occlusion_mask) -> FrameResult:
        if self.board is None:
            return failed("unconfigured:board")
        if self.plane is None:
            return failed("unconfigured:rig.bench_plane")
        if self.camera is None:
            return failed(self.camera_problem)
        if self.rig.board_is_brighter is None:
            return failed("unconfigured:rig.segmentation.board_is_brighter")
        thresholds = self.rig.thresholds
        if (
            occlusion_mask is not None
            and live is not None
            and self.tracking_configured
            and mask_overlaps_board(
                occlusion_mask,
                self.board,
                self.camera,
                self.plane,
                live.result.pose,
                search_radius_px(self.camera, self.plane, thresholds),
            )
        ):
            return confirm_partial(
                gray,
                occlusion_mask,
                self.camera,
                self.plane,
                self.board,
                live.result.pose,
                self.rig.board_is_brighter,
                thresholds,
            )
        return register_frame(
            gray,
            self.camera,
            self.plane,
            self.board,
            self.rig.board_is_brighter,
            thresholds,
            occlusion_mask,
        )

    def _movement(self, old_pose, new_pose) -> str | None:
        thresholds = self.rig.thresholds
        displacement = np.linalg.norm(
            board_corners_in_plane(self.board, new_pose)
            - board_corners_in_plane(self.board, old_pose),
            axis=1,
        ).max()
        rotation = abs(wrap_angle(new_pose[2] - old_pose[2]))
        if displacement > thresholds.move_translation_m or rotation > thresholds.move_rotation_rad:
            return (
                f"corner displacement {displacement * 1000:.2f} mm, "
                f"rotation {math.degrees(rotation):.2f} deg"
            )
        return None

    def _check_timeout(self, now: float):
        live = self._live()
        if live is None:
            return
        age = now - live.last_confirmed
        if age > self.rig.thresholds.tracking_loss_timeout_s:
            self._invalidate(TRACKING_LOST, f"no confirming frame for {age:.3f} s", now)

    def _invalidate(self, reason: str, detail: str, now: float):
        self.current.invalidated = f"{reason}:{detail}"
        self.current.confirmed_latest = False
        self._queue_event(now)

    def _calibration_revision(self) -> str:
        """Short hash of the calibration inputs in use; empty when any is missing."""
        if self.camera is None or self.plane is None or self.extrinsic is None:
            return ""
        payload = repr(
            (
                self.camera.fingerprint(),
                self.rig.bench_normal,
                self.rig.bench_distance_m,
                self.board.thickness_m,
                self.rig.base_frame,
                [round(v, 9) for v in np.concatenate(self.extrinsic)],
            )
        )
        return hashlib.sha1(payload.encode()).hexdigest()[:12]

    def _queue_event(self, now: float):
        snapshot = self.snapshot(now)
        key = snapshot.event_key()
        if key != self._last_event_key:
            self._last_event_key = key
            self._events.append(snapshot)
