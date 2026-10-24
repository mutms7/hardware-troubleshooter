"""Camera model built only from camera_info. Unsupported or incomplete calibration is rejected."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import cv2
import numpy as np

# Coefficient counts written by the ROS camera_calibration tools for each model.
_DISTORTION_SIZES = {"plumb_bob": (5,), "rational_polynomial": (8,), "equidistant": (4,)}
_UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-10)


@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    k: np.ndarray
    d: np.ndarray
    distortion_model: str

    @classmethod
    def from_camera_info(
        cls, width, height, k, d, distortion_model, binning=(0, 0), roi=(0, 0, 0, 0)
    ) -> tuple[CameraModel | None, str | None]:
        """Returns (model, None) or (None, reason). ROI and binning aren't supported."""
        k = np.asarray(k, dtype=float).reshape(-1)
        d = np.asarray(d, dtype=float).reshape(-1)
        if int(width) <= 0 or int(height) <= 0:
            return None, "camera_info_invalid:image size unset"
        if k.size != 9 or not np.all(np.isfinite(k)) or k[0] <= 0 or k[4] <= 0:
            return None, "camera_info_invalid:K is unset or not a pinhole matrix"
        sizes = _DISTORTION_SIZES.get(distortion_model)
        if sizes is None:
            return None, f"camera_info_invalid:unsupported distortion model {distortion_model!r}"
        if d.size not in sizes or not np.all(np.isfinite(d)):
            return None, f"camera_info_invalid:{distortion_model} needs {sizes[0]} coefficients"
        if any(binning) or any(roi):
            return None, "camera_info_invalid:binning/ROI isn't supported"
        return cls(int(width), int(height), k.reshape(3, 3), d, distortion_model), None

    @classmethod
    def from_calibration_yaml(cls, data: dict) -> tuple[CameraModel | None, str | None]:
        """camera_calibration's YAML layout (image_width, camera_matrix, ...)."""
        try:
            return cls.from_camera_info(
                data["image_width"],
                data["image_height"],
                data["camera_matrix"]["data"],
                data["distortion_coefficients"]["data"],
                data["distortion_model"],
            )
        except (KeyError, TypeError) as exc:
            return None, f"camera_info_invalid:missing {exc}"

    @property
    def fisheye(self) -> bool:
        return self.distortion_model == "equidistant"

    def fingerprint(self) -> str:
        payload = repr(
            (self.width, self.height, self.distortion_model, self.k.tolist(), self.d.tolist())
        )
        return hashlib.sha1(payload.encode()).hexdigest()[:12]

    def undistort_px(self, pixels: np.ndarray) -> np.ndarray:
        """Raw image pixels to ideal pinhole pixels with the same K."""
        src = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
        if self.fisheye:
            out = cv2.fisheye.undistortPoints(src, self.k, self.d, P=self.k)
        else:
            out = cv2.undistortPointsIter(src, self.k, self.d, None, self.k, _UNDISTORT_CRITERIA)
        return out.reshape(-1, 2)

    def ideal_to_rays(self, pixels: np.ndarray) -> np.ndarray:
        pixels = np.asarray(pixels, dtype=float)
        x = (pixels[:, 0] - self.k[0, 2]) / self.k[0, 0]
        y = (pixels[:, 1] - self.k[1, 2]) / self.k[1, 1]
        return np.column_stack([x, y, np.ones(len(pixels))])

    def project_ideal(self, points: np.ndarray) -> np.ndarray:
        """Camera-frame points to ideal pinhole pixels (no distortion)."""
        points = np.asarray(points, dtype=float)
        uvw = points @ self.k.T
        return uvw[:, :2] / uvw[:, 2:3]

    def project(self, points: np.ndarray) -> np.ndarray:
        """Camera-frame points to raw (distorted) image pixels."""
        src = np.asarray(points, dtype=np.float64).reshape(-1, 1, 3)
        zero = np.zeros(3)
        if self.fisheye:
            out, _ = cv2.fisheye.projectPoints(src, zero, zero, self.k, self.d)
        else:
            out, _ = cv2.projectPoints(src, zero, zero, self.k, self.d)
        return out.reshape(-1, 2)
