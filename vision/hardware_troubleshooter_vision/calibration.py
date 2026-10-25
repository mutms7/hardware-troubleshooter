"""Bench calibration: intrinsics, bench plane, hand-eye, joint zeros and tip pixels.

All of it runs offline on captured views and writes YAML the runtime reads. A plain
checkerboard (or optional ChArUco target) is used only during calibration; the runtime
has no fiducial dependency. Every function
reports its residual, and none of them decides whether that residual is acceptable:
record it, and set acceptance limits from measurements on the real rig.

Order on a new rig:
1. intrinsics for each camera (calibrate_intrinsics)
2. bench plane from a target lying flat on the bench (bench_plane)
3. joint zeros and arm base in the rig, target on the wrist, fixed camera watching
   (calibrate_joint_offsets); this also gives base_in_rig
4. wrist camera on link_007 (eye_in_hand), using the fitted joint map
5. tip pixel per tool (tip_pixel_from_marks)
Re-zeroing a motor invalidates 3 onward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from hardware_troubleshooter_arm.config import JointMap, TipPixel
from hardware_troubleshooter_arm.kinematics import urdf_joint_positions
from hardware_troubleshooter_arm.urdf import prefixed
from hardware_troubleshooter_arm.vendor import ARM_JOINTS, WRIST_LINK

from . import transforms

MIN_CORNERS = 6
MIN_VIEWS = 5
# CharucoDetector reports corners half a pixel down and right of where projectPoints
# puts them (measured on rendered views: +0.49 px, scatter 0.04 px). Corners are shifted
# back so every pixel here uses projectPoints' convention (integer = pixel centre).
CHARUCO_PIXEL_OFFSET = 0.5


# --- targets and intrinsics ------------------------------------------------------------


@dataclass(frozen=True)
class CheckerboardTarget:
    """Inner-corner counts, not square counts; measured square spacing in metres.

    The target origin is the first inner corner, with +x along a row and +y
    along a column. Keep that physical frame consistent for pose calibration.
    """

    corners_x: int
    corners_y: int
    square_m: float

    def __post_init__(self):
        for count in (self.corners_x, self.corners_y):
            if isinstance(count, bool) or not isinstance(count, int) or count < 3:
                raise ValueError("checkerboard needs at least 3 inner corners per axis")
        if not math.isfinite(self.square_m) or self.square_m <= 0:
            raise ValueError("square_m must be finite and positive")

    def object_points(self) -> np.ndarray:
        points = np.zeros((self.corners_x * self.corners_y, 3), dtype=float)
        points[:, :2] = np.mgrid[: self.corners_x, : self.corners_y].T.reshape(-1, 2)
        return points * self.square_m


@dataclass(frozen=True)
class CharucoTarget:
    squares_x: int
    squares_y: int
    square_m: float
    marker_m: float
    dictionary: int = cv2.aruco.DICT_5X5_100

    def board(self):
        return cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_m,
            self.marker_m,
            cv2.aruco.getPredefinedDictionary(self.dictionary),
        )


@dataclass(frozen=True)
class Detection:
    image_size: tuple[int, int]  # (width, height)
    ids: np.ndarray
    image_points: np.ndarray  # (N, 2) raw pixels
    object_points: np.ndarray  # (N, 3) target frame, metres
    pose_order_verified: bool = True  # ChArUco IDs establish the target frame


def detect_checkerboard(
    gray: np.ndarray, target: CheckerboardTarget, *, reverse_order: bool | None = None
) -> Detection | None:
    """Detect a complete checkerboard, with sub-pixel corners in raw image pixels.

    None leaves ordering unverified, suitable for intrinsics only. For poses, inspect
    the returned corner order against the physical target's chosen origin and axes,
    then pass False to confirm it or True to reverse it. A plain board cannot identify
    its own origin. Reject views where reversal alone cannot establish the chosen axes
    (particularly square grids); do not guess. This is a per-image decision.
    """
    if reverse_order is not None and not isinstance(reverse_order, bool):
        raise ValueError("reverse_order must be None, False or True")
    ok, corners = cv2.findChessboardCornersSB(
        gray,
        (target.corners_x, target.corners_y),
        flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
    )
    if not ok:
        # A thin black outline can defeat the SB detector even with a white margin.
        # The classic detector handles that pattern; refine its corners explicitly.
        ok, corners = cv2.findChessboardCorners(
            gray,
            (target.corners_x, target.corners_y),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if ok:
            corners = cv2.cornerSubPix(
                gray,
                corners,
                (5, 5),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001),
            )
    if not ok:
        return None
    points = corners.reshape(-1, 2).astype(float)
    if reverse_order:
        points = points[::-1].copy()
    return Detection(
        (gray.shape[1], gray.shape[0]),
        np.arange(len(points)),  # positional indices, not physical marker IDs
        points,
        target.object_points(),
        pose_order_verified=reverse_order is not None,
    )


def detect_charuco(gray: np.ndarray, target: CharucoTarget) -> Detection | None:
    board = target.board()
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
    if ids is None or len(ids) < MIN_CORNERS:
        return None
    ids = ids.ravel()
    objects = board.getChessboardCorners()[ids]
    points = corners.reshape(-1, 2).astype(float) - CHARUCO_PIXEL_OFFSET
    return Detection((gray.shape[1], gray.shape[0]), ids, points, objects)


def calibrate_intrinsics(detections: list[Detection], fix_k3: bool = True) -> tuple[dict, float]:
    """camera_calibration-style YAML dict (plumb_bob) and the RMS reprojection error.

    k3 is held at zero by default: a target that doesn't reach the image corners can't
    constrain it, and a free k3 then bends the model wildly where there was no data."""
    if len(detections) < MIN_VIEWS:
        raise ValueError(f"need at least {MIN_VIEWS} views with a detected target")
    size = detections[0].image_size
    if any(d.image_size != size for d in detections):
        raise ValueError("views have different image sizes")
    rms, k, d, _, _ = cv2.calibrateCamera(
        [det.object_points.astype(np.float32) for det in detections],
        [det.image_points.astype(np.float32) for det in detections],
        size,
        None,
        None,
        flags=cv2.CALIB_FIX_K3 if fix_k3 else 0,
    )
    return camera_info(size, k, d.ravel()[:5]), float(rms)


def camera_info(size, k, d) -> dict:
    k = np.asarray(k, float)
    projection = np.hstack([k, np.zeros((3, 1))])
    return {
        "image_width": int(size[0]),
        "image_height": int(size[1]),
        "camera_matrix": {"rows": 3, "cols": 3, "data": k.ravel().tolist()},
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {"rows": 1, "cols": 5, "data": list(map(float, d))},
        "rectification_matrix": {"rows": 3, "cols": 3, "data": np.eye(3).ravel().tolist()},
        "projection_matrix": {"rows": 3, "cols": 4, "data": projection.ravel().tolist()},
    }


def target_pose(detection: Detection, camera) -> tuple[np.ndarray, float]:
    """camera <- target and the RMS reprojection error in pixels."""
    if not detection.pose_order_verified:
        raise ValueError("checkerboard corner order must be verified before target_pose")
    ideal = camera.undistort_px(detection.image_points)
    ok, rvec, tvec = cv2.solvePnP(
        detection.object_points.astype(np.float64),
        ideal,
        camera.k,
        None,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise ValueError("solvePnP failed")
    rotation = cv2.Rodrigues(rvec)[0]
    pose = transforms.transform(rotation, tvec.ravel())
    projected = camera.project_ideal(transforms.apply(pose, detection.object_points))
    rms = float(np.sqrt(np.mean(np.sum((projected - ideal) ** 2, axis=1))))
    return pose, rms


def bench_plane(camera_from_target: np.ndarray, target_thickness_m: float) -> tuple[list, float]:
    """rig.bench_plane (normal toward the bench, distance) in the camera frame, from a
    target lying flat on the bench with its printed face up."""
    normal = camera_from_target[:3, 2].copy()
    origin = camera_from_target[:3, 3]
    if normal @ origin < 0:
        normal = -normal
    return normal.tolist(), float(normal @ origin + target_thickness_m)


# --- hand-eye --------------------------------------------------------------------------


@dataclass(frozen=True)
class HandEyeFit:
    transform: np.ndarray
    rms_m: float  # spread of the reconstructed fixed transform across views
    rms_rad: float


def eye_in_hand(base_from_link: list, camera_from_target: list) -> HandEyeFit:
    """link <- camera for a camera on the arm, from views of a target fixed on the bench."""
    link_from_camera = _hand_eye(base_from_link, camera_from_target)
    base_from_target = [
        b @ link_from_camera @ c for b, c in zip(base_from_link, camera_from_target)
    ]
    return HandEyeFit(link_from_camera, *_spread(base_from_target))


def eye_to_hand(base_from_link: list, camera_from_target: list) -> HandEyeFit:
    """base <- camera for a fixed camera, from views of a target fixed on the arm."""
    base_from_camera = _hand_eye([transforms.invert(b) for b in base_from_link], camera_from_target)
    link_from_target = [
        transforms.invert(b) @ base_from_camera @ c
        for b, c in zip(base_from_link, camera_from_target)
    ]
    return HandEyeFit(base_from_camera, *_spread(link_from_target))


def _hand_eye(gripper_to_base, target_to_camera) -> np.ndarray:
    if len(gripper_to_base) < 3 or len(gripper_to_base) != len(target_to_camera):
        raise ValueError("need at least three matching pose pairs")
    rotation, translation = cv2.calibrateHandEye(
        [m[:3, :3] for m in gripper_to_base],
        [m[:3, 3] for m in gripper_to_base],
        [m[:3, :3] for m in target_to_camera],
        [m[:3, 3] for m in target_to_camera],
        method=cv2.CALIB_HAND_EYE_PARK,
    )
    return transforms.transform(rotation, translation.ravel())


def _spread(matrices) -> tuple[float, float]:
    positions = np.array([m[:3, 3] for m in matrices])
    rms_m = float(np.sqrt(np.mean(np.sum((positions - positions.mean(axis=0)) ** 2, axis=1))))
    reference = matrices[0][:3, :3]
    angles = [transforms.rotation_angle(reference.T @ m[:3, :3]) for m in matrices]
    return rms_m, float(np.sqrt(np.mean(np.square(angles))))


# --- kinematic calibration -------------------------------------------------------------


@dataclass(frozen=True)
class KinematicFit:
    joint_map: tuple[JointMap, ...]  # calibrated, ready for the arm calibration YAML
    offsets_rad: dict[str, float]  # fitted change per URDF joint
    camera_from_base: np.ndarray
    link_from_target: np.ndarray
    rms_m: float
    rms_rad: float


# J1 and J6 zeros aren't observable here: a J1 offset is a base yaw (it ends up in
# base_in_rig) and a J6 offset is a roll of link_007 (absorbed by the target mount, and
# later by the wrist camera's and tools' own calibration, which use the same FK).
DEFAULT_FIT_JOINTS = (2, 3, 4, 5)
_ROTATION_WEIGHT_M = 0.1  # one radian of orientation error weighs like 0.1 m


def calibrate_joint_offsets(
    arm, samples: list[tuple], fit_joints=DEFAULT_FIT_JOINTS, iterations=50
) -> KinematicFit:
    """Fit joint zero offsets from a target on link_007 seen by the fixed camera.

    samples: (sdk_positions, camera_from_target) pairs. Needs varied poses of every
    fitted joint; start from the arm config's nominal joint map.
    """
    if not 1 <= min(fit_joints) <= max(fit_joints) <= len(ARM_JOINTS):
        raise ValueError("fit_joints are 1-based arm joints (J1..J6)")
    if len(samples) < 3 * (len(fit_joints) + 2):
        raise ValueError(f"need at least {3 * (len(fit_joints) + 2)} samples")
    nominal = arm.config.joint_map
    if nominal is None:
        raise ValueError("arm.joint_map (nominal) is unset")
    link = prefixed(arm.config, WRIST_LINK)
    names = [prefixed(arm.config, name) for name in ARM_JOINTS]
    base_q = [urdf_joint_positions(arm.config, list(q)) for q, _ in samples]
    observed = [m for _, m in samples]
    fitted = [names[j - 1] for j in fit_joints]

    def forward(offsets):
        poses = []
        for q in base_q:
            moved = dict(q)
            for name, delta in zip(fitted, offsets):
                moved[name] += delta
            poses.append(arm.kinematics.link_poses(moved)[link])
        return poses

    start = eye_to_hand(forward(np.zeros(len(fitted))), observed)
    camera_from_base = transforms.invert(start.transform)
    link_from_target = (
        transforms.invert(forward(np.zeros(len(fitted)))[0]) @ start.transform @ observed[0]
    )
    params = np.concatenate(
        [np.zeros(len(fitted)), _to_vector(camera_from_base), _to_vector(link_from_target)]
    )

    def residuals(p):
        offsets = p[: len(fitted)]
        c = _from_vector(p[len(fitted) : len(fitted) + 6])
        t = _from_vector(p[len(fitted) + 6 :])
        out = []
        for link_pose, obs in zip(forward(offsets), observed):
            error = transforms.invert(obs) @ c @ link_pose @ t
            out.append(error[:3, 3])
            out.append(cv2.Rodrigues(error[:3, :3])[0].ravel() * _ROTATION_WEIGHT_M)
        return np.concatenate(out)

    params = _levenberg_marquardt(residuals, params, iterations)
    r = residuals(params).reshape(-1, 2, 3)
    offsets = params[: len(fitted)]
    joint_map = list(nominal)
    for joint, delta in zip(fit_joints, offsets):
        entry = nominal[joint - 1]
        joint_map[joint - 1] = JointMap(entry.scale, entry.offset_rad + float(delta))
    return KinematicFit(
        tuple(joint_map),
        {name: float(delta) for name, delta in zip(fitted, offsets)},
        _from_vector(params[len(fitted) : len(fitted) + 6]),
        _from_vector(params[len(fitted) + 6 :]),
        float(np.sqrt(np.mean(np.sum(r[:, 0] ** 2, axis=1)))),
        float(np.sqrt(np.mean(np.sum((r[:, 1] / _ROTATION_WEIGHT_M) ** 2, axis=1)))),
    )


def base_in_rig(rig_from_camera: np.ndarray, camera_from_base: np.ndarray) -> tuple:
    """The arm config's base_in_rig pose, from the fixed camera's extrinsic and the fit."""
    return pose_from_matrix(rig_from_camera @ camera_from_base)


def _to_vector(matrix) -> np.ndarray:
    return np.concatenate([cv2.Rodrigues(matrix[:3, :3])[0].ravel(), matrix[:3, 3]])


def _from_vector(vector) -> np.ndarray:
    return transforms.transform(cv2.Rodrigues(np.asarray(vector[:3], float))[0], vector[3:6])


def _levenberg_marquardt(fn, params, iterations, step=1e-6):
    damping = 1e-3
    r = fn(params)
    cost = r @ r
    for _ in range(iterations):
        jacobian = np.empty((len(r), len(params)))
        for i in range(len(params)):
            probe = params.copy()
            probe[i] += step
            jacobian[:, i] = (fn(probe) - r) / step
        normal = jacobian.T @ jacobian
        gradient = jacobian.T @ r
        while True:
            delta = np.linalg.solve(normal + damping * np.diag(np.diag(normal) + 1e-12), -gradient)
            candidate = params + delta
            r_new = fn(candidate)
            if r_new @ r_new < cost:
                params, r, cost = candidate, r_new, r_new @ r_new
                damping = max(damping / 10, 1e-12)
                break
            damping *= 10
            if damping > 1e8:
                return params
        if np.abs(delta).max() < 1e-12:
            break
    return params


# --- tool tip pixel ---------------------------------------------------------------------


def tip_pixel_from_marks(
    images: list[np.ndarray], standoff_m: float, min_radius_px: float, max_radius_px: float
) -> tuple[TipPixel, float]:
    """The tool's tip pixel and the spread (max distance from the median) in pixels.

    For each image: touch the tip to the centre of a dark round mark on the bench,
    retract straight along the tool's approach axis by standoff_m, and capture.
    """
    centres = []
    for i, image in enumerate(images):
        centre = find_dark_dot(image, min_radius_px, max_radius_px)
        if centre is None:
            raise ValueError(f"image {i}: no single dark dot found")
        centres.append(centre)
    if len(centres) < 3:
        raise ValueError("need at least three images")
    centres = np.array(centres)
    median = np.median(centres, axis=0)
    spread = float(np.linalg.norm(centres - median, axis=1).max())
    return TipPixel((float(median[0]), float(median[1])), float(standoff_m)), spread


def find_dark_dot(gray, min_radius_px, max_radius_px) -> tuple[float, float] | None:
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    dots = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        radius = math.sqrt(area / math.pi)
        circularity = 4 * math.pi * area / perimeter**2
        if min_radius_px <= radius <= max_radius_px and circularity > 0.8:
            moments = cv2.moments(contour)
            dots.append((moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]))
    return dots[0] if len(dots) == 1 else None


# --- YAML output ----------------------------------------------------------------------------


def pose_from_matrix(matrix: np.ndarray) -> tuple:
    """((x, y, z), (roll, pitch, yaw)) in the URDF convention R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r = matrix[:3, :3]
    pitch = math.asin(max(-1.0, min(1.0, -r[2, 0])))
    if abs(math.cos(pitch)) > 1e-9:
        roll, yaw = math.atan2(r[2, 1], r[2, 2]), math.atan2(r[1, 0], r[0, 0])
    else:  # gimbal lock: put all of the remaining rotation in yaw
        roll, yaw = 0.0, math.atan2(-r[0, 1], r[1, 1])
    return tuple(float(v) for v in matrix[:3, 3]), (roll, pitch, yaw)


def pose_dict(pose) -> dict:
    xyz, rpy = pose
    return {"xyz_m": [float(v) for v in xyz], "rpy_rad": [float(v) for v in rpy]}


def arm_calibration_dict(
    calibration_id: str,
    arm: str,
    joint_map=None,
    wrist_camera=None,
    tool_tips: dict | None = None,
    tip_pixels: dict | None = None,
) -> dict:
    """The arm package's calibration YAML layout (arm/config/calibration.example.yaml)."""
    data = {"calibration_id": calibration_id, "arm": arm}
    if joint_map is not None:
        data["joint_map"] = [{"scale": e.scale, "offset_rad": e.offset_rad} for e in joint_map]
    if wrist_camera is not None:
        data["wrist_camera"] = pose_dict(wrist_camera)
    tools = {}
    for name, pose in (tool_tips or {}).items():
        tools.setdefault(name, {})["tip"] = pose_dict(pose)
    for name, pixel in (tip_pixels or {}).items():
        tools.setdefault(name, {})["tip_pixel"] = {
            "uv": [float(v) for v in pixel.uv],
            "standoff_m": float(pixel.standoff_m),
        }
    data["tools"] = tools
    return data
