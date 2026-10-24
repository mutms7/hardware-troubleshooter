"""Two settled wrist views, measured camera motion, one board pose in the rig.

No arm commands and no depth input. A commanded step is never used as a baseline.
The caller supplies camera poses from measured joints and calibrated hand-eye geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import transforms
from .config import HomingConfig


@dataclass(frozen=True)
class BoardObservation:
    board_id: str
    feature_ids: tuple[str, ...]
    object_points: np.ndarray  # board top surface, metres, z=0
    image_points: np.ndarray  # raw, distorted pixels

    def validate(self, camera, minimum: int) -> None:
        obj, pix = np.asarray(self.object_points), np.asarray(self.image_points)
        if not self.board_id or obj.ndim != 2 or obj.shape[1:] != (3,):
            raise ValueError("invalid_board_points")
        if len(obj) < minimum or pix.shape != (len(obj), 2):
            raise ValueError("insufficient_board_points")
        if len(self.feature_ids) != len(obj) or len(set(self.feature_ids)) != len(obj):
            raise ValueError("feature_identity_invalid")
        if not np.isfinite(obj).all() or not np.isfinite(pix).all():
            raise ValueError("nonfinite_board_points")
        if np.max(np.abs(obj[:, 2])) > 1e-9 or np.linalg.matrix_rank(obj[:, :2] - obj[0, :2]) < 2:
            raise ValueError("board_points_must_span_a_plane")
        if (
            (pix < 0).any()
            or (pix[:, 0] >= camera.width).any()
            or (pix[:, 1] >= camera.height).any()
        ):
            raise ValueError("board_points_outside_image")


@dataclass(frozen=True)
class HomingSnapshot:
    status: str
    registration_id: str
    failure_reason: str
    stamp: float | None
    valid_until: float | None
    rig_from_board: np.ndarray | None
    arm: str
    target: str
    calibration_revision: str
    diagnostics: dict = field(default_factory=dict)

    @property
    def has_pose(self):
        return self.status == "valid" and self.rig_from_board is not None

    @property
    def tracking_valid(self):
        return self.has_pose

    @property
    def board_unmoved(self):
        return self.has_pose

    @property
    def calibration_valid(self):
        return bool(self.calibration_revision) and self.failure_reason != "calibration_changed"


@dataclass
class HomingResult:
    ok: bool
    phase: str
    reasons: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _View:
    observation: BoardObservation
    rig_from_camera: np.ndarray
    stamp: float
    candidates: tuple


def _rigid(matrix):
    matrix = np.asarray(matrix, float)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1])
        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1)
    ):
        raise ValueError("invalid_camera_transform")
    return matrix.copy()


def _errors(pose, view, camera):
    points = transforms.apply(
        transforms.invert(view.rig_from_camera) @ pose, view.observation.object_points
    )
    if (points[:, 2] <= 0).any():
        raise ValueError("board_behind_camera")
    return camera.project(points) - view.observation.image_points


def _rms(errors):
    return float(np.sqrt(np.mean(np.sum(np.asarray(errors) ** 2, axis=-1))))


def _candidates(observation, camera):
    ideal = camera.undistort_px(observation.image_points)
    _, rotations, translations, _ = cv2.solvePnPGeneric(
        np.asarray(observation.object_points, np.float64),
        ideal.astype(np.float64),
        camera.k,
        None,
        flags=cv2.SOLVEPNP_IPPE,
    )
    candidates = []
    for r, t in zip(rotations, translations):
        pose = transforms.transform(cv2.Rodrigues(r)[0], t.ravel())
        if (
            np.isfinite(pose).all()
            and (transforms.apply(pose, observation.object_points)[:, 2] > 0).all()
        ):
            candidates.append(pose)
    # IPPE's Rodrigues output can be singular for an exact 180-degree orientation.
    # Keep both planar alternatives, and add the iterative solution as another seed.
    ok, r, t = cv2.solvePnP(
        np.asarray(observation.object_points, np.float64),
        ideal.astype(np.float64),
        camera.k,
        None,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if ok:
        pose = transforms.transform(cv2.Rodrigues(r)[0], t.ravel())
        if (
            np.isfinite(pose).all()
            and (transforms.apply(pose, observation.object_points)[:, 2] > 0).all()
        ):
            candidates.append(pose)
    return tuple(candidates)


def _refine(seed, views, camera):
    """Joint least squares in first-camera coordinates; never average independent poses."""
    first_from_rig = transforms.invert(views[0].rig_from_camera)
    start = first_from_rig @ seed
    parameters = np.r_[cv2.Rodrigues(start[:3, :3])[0].ravel(), start[:3, 3]]

    def matrix(p):
        return views[0].rig_from_camera @ transforms.transform(cv2.Rodrigues(p[:3])[0], p[3:])

    def residual(p):
        pose = matrix(p)
        return np.concatenate([_errors(pose, v, camera).ravel() for v in views])

    error = residual(parameters)
    damping = 1e-3
    for _ in range(30):
        jac = np.column_stack(
            [(residual(parameters + np.eye(6)[i] * 1e-6) - error) / 1e-6 for i in range(6)]
        )
        normal = jac.T @ jac
        delta = np.linalg.solve(normal + damping * np.diag(np.diag(normal) + 1), -jac.T @ error)
        try:
            new = residual(parameters + delta)
        except ValueError:
            new = np.full_like(error, np.inf)
        if new @ new < error @ error:
            parameters, error = parameters + delta, new
            damping = max(damping / 3, 1e-9)
            if np.linalg.norm(delta) < 1e-9:
                break
        else:
            damping *= 10
            if damping > 1e10:
                break
    return matrix(parameters), _rms(error.reshape(-1, 2))


class WristHoming:
    def __init__(self, config: HomingConfig, session_id: str):
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("unique_session_id_required")
        self.config = config
        self.session_id = session_id
        self.revision = 0
        self.registration_id = ""
        self.arm = self.target = self.board_id = self.calibration_revision = ""
        self.status, self.reason = "unhomed", "homing_required"
        self.first: _View | None = None
        self.pose = None
        self.stamp = self.started = None
        self.diagnostics = {}
        self.events = []
        self.known_points = {}

    def begin(self, arm, target, board_id, calibration_revision, now):
        self.revision += 1
        self.registration_id = f"{board_id}/{self.session_id}/r{self.revision}"
        self.arm, self.target, self.board_id = arm, target, board_id
        self.calibration_revision = calibration_revision
        self.first, self.pose, self.stamp, self.started = None, None, None, now
        self.status, self.reason, self.diagnostics = "homing", "first_view_required", {}
        self.known_points = {}
        self._emit()

    def invalidate(self, reason):
        if self.status == "invalid" and self.reason == reason:
            return
        self.pose, self.first = None, None
        self.status, self.reason = "invalid", reason
        self._emit()

    def tick(self, now):
        if self.status == "valid" and now - self.stamp > self.config.tracking_timeout_s:
            self.invalidate("tracking_lost")
        if self.status == "homing" and now - self.started > self.config.max_pair_interval_s:
            self.invalidate("homing_timeout")

    def snapshot(self, now):
        self.tick(now)
        return self._snapshot()

    def _snapshot(self):
        until = self.stamp + self.config.tracking_timeout_s if self.pose is not None else None
        return HomingSnapshot(
            self.status,
            self.registration_id,
            self.reason,
            self.stamp,
            until,
            self.pose.copy() if self.pose is not None else None,
            self.arm,
            self.target,
            self.calibration_revision,
            dict(self.diagnostics),
        )

    def _emit(self):
        self.events.append(self._snapshot())

    def drain_events(self):
        events, self.events = self.events, []
        return events

    def observe(self, observation, rig_from_camera, camera, stamp, calibration_revision):
        missing = self.config.missing()
        if missing:
            return HomingResult(False, self.status, [f"unconfigured:{key}" for key in missing])
        if self.started is not None:
            self.tick(stamp)
        if self.status not in ("homing", "valid"):
            return HomingResult(False, self.status, [self.reason])
        if calibration_revision != self.calibration_revision:
            self.invalidate("calibration_changed")
            return HomingResult(False, self.status, [self.reason])
        if observation.board_id != self.board_id:
            self.invalidate("board_identity_changed")
            return HomingResult(False, self.status, [self.reason])
        if (
            not np.isfinite(stamp)
            or stamp < self.started
            or (self.stamp is not None and stamp <= self.stamp)
        ):
            return HomingResult(False, self.status, ["stale_observation"])
        try:
            observation.validate(camera, int(self.config.min_points))
            observation = BoardObservation(
                observation.board_id,
                tuple(observation.feature_ids),
                np.array(observation.object_points, dtype=float, copy=True),
                np.array(observation.image_points, dtype=float, copy=True),
            )
            if any(
                k in self.known_points
                and not np.allclose(self.known_points[k], p, atol=1e-9, rtol=0)
                for k, p in zip(observation.feature_ids, observation.object_points)
            ):
                raise ValueError("feature_geometry_changed")
            view = _View(
                observation, _rigid(rig_from_camera), stamp, _candidates(observation, camera)
            )
        except (ValueError, cv2.error, np.linalg.LinAlgError) as exc:
            if self.status == "valid":
                self.invalidate(f"tracking_lost:{exc}")
            return HomingResult(False, self.status, [f"observation_invalid:{exc}"])
        if self.status == "valid":
            try:
                error = _rms(_errors(self.pose, view, camera))
            except ValueError:
                self.invalidate("board_moved")
                return HomingResult(False, self.status, [self.reason])
            if not np.isfinite(error) or error > self.config.max_reprojection_error_px:
                self.invalidate("board_moved")
                return HomingResult(False, self.status, [self.reason], {"rms_px": error})
            self.stamp = stamp
            self.known_points.update(zip(observation.feature_ids, observation.object_points))
            return HomingResult(True, "confirmed", diagnostics={"rms_px": error})
        if not view.candidates:
            return HomingResult(False, "homing", ["pnp_failed"])
        quality = min(
            _rms(_errors(view.rig_from_camera @ p, view, camera)) for p in view.candidates
        )
        if quality > self.config.max_reprojection_error_px:
            return HomingResult(False, "homing", ["reprojection_error"], {"rms_px": quality})
        if self.first is None:
            self.first, self.stamp, self.reason = view, stamp, "second_view_required"
            self.known_points.update(zip(observation.feature_ids, observation.object_points))
            self._emit()
            return HomingResult(True, "first_view", diagnostics={"rms_px": quality})
        try:
            pose, diagnostics = self._solve_pair(self.first, view, camera)
        except (ValueError, cv2.error, np.linalg.LinAlgError) as exc:
            return HomingResult(False, "homing", [str(exc)])
        self.pose, self.stamp, self.diagnostics = pose, stamp, diagnostics
        self.known_points.update(zip(observation.feature_ids, observation.object_points))
        self.status, self.reason = "valid", ""
        self._emit()
        return HomingResult(True, "homed", diagnostics=diagnostics)

    def _solve_pair(self, first, second, camera):
        config = self.config
        displacement = second.rig_from_camera[:3, 3] - first.rig_from_camera[:3, 3]
        baseline = float(np.linalg.norm(displacement))
        optical_axis = first.rig_from_camera[:3, 2]
        lateral = float(np.linalg.norm(displacement - optical_axis * (displacement @ optical_axis)))
        if lateral < config.min_lateral_baseline_m:
            raise ValueError("insufficient_lateral_baseline")
        if baseline > config.max_baseline_m:
            raise ValueError("baseline_above_limit")
        shared = sorted(set(first.observation.feature_ids) & set(second.observation.feature_ids))
        if len(shared) < config.min_points:
            raise ValueError("insufficient_shared_features")
        i = [first.observation.feature_ids.index(k) for k in shared]
        j = [second.observation.feature_ids.index(k) for k in shared]
        objects = first.observation.object_points[i]
        if not np.allclose(objects, second.observation.object_points[j], atol=1e-9, rtol=0):
            raise ValueError("feature_geometry_changed")
        solutions = []
        for a in first.candidates:
            for b in second.candidates:
                pa, pb = first.rig_from_camera @ a, second.rig_from_camera @ b
                if np.linalg.norm(pa[:3, 3] - pb[:3, 3]) > config.max_pose_translation_m:
                    continue
                if (
                    transforms.rotation_angle(pa[:3, :3].T @ pb[:3, :3])
                    > config.max_pose_rotation_rad
                ):
                    continue
                pose, error = _refine(pa, [first, second], camera)
                if not any(
                    np.linalg.norm(pose[:3, 3] - p[:3, 3]) < 1e-5
                    and transforms.rotation_angle(pose[:3, :3].T @ p[:3, :3]) < 1e-4
                    for p, _ in solutions
                ):
                    solutions.append((pose, error))
        if not solutions:
            raise ValueError("views_inconsistent:board moved or camera motion is wrong")
        solutions.sort(key=lambda p: p[1])
        pose, error = solutions[0]
        if error > config.max_reprojection_error_px:
            raise ValueError("joint_reprojection_error")
        if len(solutions) > 1 and solutions[1][1] - error < config.min_pose_margin_px:
            raise ValueError("pose_ambiguous")
        per_view = [_rms(_errors(pose, v, camera)) for v in (first, second)]
        if max(per_view) > config.max_reprojection_error_px:
            raise ValueError("view_reprojection_error")
        pixels = [
            camera.undistort_px(v.observation.image_points[idx])
            for v, idx in ((first, i), (second, j))
        ]
        projections = [camera.k @ transforms.invert(v.rig_from_camera)[:3] for v in (first, second)]
        homogeneous = cv2.triangulatePoints(*projections, pixels[0].T, pixels[1].T)
        if np.any(np.abs(homogeneous[3]) < 1e-12):
            raise ValueError("triangulation_at_infinity")
        points = (homogeneous[:3] / homogeneous[3]).T
        for view in (first, second):
            if (transforms.apply(transforms.invert(view.rig_from_camera), points)[:, 2] <= 0).any():
                raise ValueError("triangulation_behind_camera")
        rays = [points - v.rig_from_camera[:3, 3] for v in (first, second)]
        rays = [r / np.linalg.norm(r, axis=1)[:, None] for r in rays]
        parallax = float(np.median(np.arccos(np.clip(np.sum(rays[0] * rays[1], axis=1), -1, 1))))
        if parallax < config.min_parallax_rad:
            raise ValueError("insufficient_parallax")
        triangulation_error = _rms(points - transforms.apply(pose, objects))
        if (
            not np.isfinite(triangulation_error)
            or triangulation_error > config.max_triangulation_error_m
        ):
            raise ValueError("triangulation_error")
        return pose, {
            "rms_px": error,
            "per_view_rms_px": per_view,
            "baseline_m": baseline,
            "lateral_baseline_m": lateral,
            "parallax_rad": parallax,
            "triangulation_rms_m": triangulation_error,
            "shared_features": len(shared),
        }
