"""Identify board-surface texture against a measured reference, without depth.

Reference landmarks assign metric board coordinates, not electrical pin identities.
Only reference features inside their convex hull are used. The reference must show
the flat board surface; raised parts and the probe should be outside that region.
"""

from __future__ import annotations

import hashlib

import cv2
import numpy as np

from .homing import BoardObservation


class BoardFeatureDetector:
    def __init__(self, config, reference_camera):
        self.config = config
        self.camera = reference_camera
        self.image = cv2.imread(str(config.image), cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise ValueError("board_reference:image_unreadable")
        if self.image.shape != (reference_camera.height, reference_camera.width):
            raise ValueError("board_reference:image_size_mismatch")
        try:
            pixels = np.array([p["pixel"] for p in config.landmarks], dtype=float)
            points = np.array([p["board_m"] for p in config.landmarks], dtype=float)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError("board_reference:invalid_landmarks") from exc
        if (
            len(points) < 4
            or points.shape != pixels.shape
            or points.shape[1:] != (2,)
            or not np.isfinite(points).all()
            or not np.isfinite(pixels).all()
            or np.linalg.matrix_rank(points - points[0]) < 2
            or np.linalg.matrix_rank(pixels - pixels[0]) < 2
        ):
            raise ValueError("board_reference:need_four_noncollinear_landmarks")
        if (
            (pixels < 0).any()
            or (pixels[:, 0] >= reference_camera.width).any()
            or (pixels[:, 1] >= reference_camera.height).any()
        ):
            raise ValueError("board_reference:landmark_outside_image")
        ideal = reference_camera.undistort_px(pixels)
        mapping, _ = cv2.findHomography(ideal, points, 0)
        if mapping is None or not np.isfinite(mapping).all():
            raise ValueError("board_reference:degenerate_landmarks")
        self.sift = cv2.SIFT_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        mask = np.zeros_like(self.image)
        cv2.fillConvexPoly(mask, cv2.convexHull(pixels.astype(np.float32)).astype(np.int32), 255)
        keys, self.descriptors = self.sift.detectAndCompute(self.image, mask)
        if self.descriptors is None or len(keys) < config.min_matches:
            raise ValueError("board_reference:insufficient_surface_texture")
        key_pixels = np.array([k.pt for k in keys], dtype=float)
        xy = cv2.perspectiveTransform(
            reference_camera.undistort_px(key_pixels).reshape(-1, 1, 2), mapping
        ).reshape(-1, 2)
        self.points = np.column_stack([xy, np.zeros(len(xy))])
        self.fingerprint = hashlib.sha256(
            self.image.tobytes() + repr(config).encode() + reference_camera.fingerprint().encode()
        ).hexdigest()[:16]

    def detect(self, frame, camera):
        image = frame.image
        if image.shape[:2] != (camera.height, camera.width):
            return None, "camera_image_size_mismatch"
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        keys, descriptors = self.sift.detectAndCompute(gray, None)
        if descriptors is None or len(descriptors) < 2:
            return None, "board_features_missing"
        matches, used = [], set()
        for pair in self.matcher.knnMatch(self.descriptors, descriptors, k=2):
            if len(pair) != 2:
                continue
            a, b = pair
            if a.distance < self.config.ratio * b.distance and a.trainIdx not in used:
                matches.append(a)
                used.add(a.trainIdx)
        if len(matches) < self.config.min_matches:
            return None, "board_matches_insufficient"
        objects = self.points[[m.queryIdx for m in matches]]
        pixels = np.array([keys[m.trainIdx].pt for m in matches], float)
        # Homography only filters correspondences; the pose comes from metric PnP.
        _, inliers = cv2.findHomography(
            objects[:, :2], camera.undistort_px(pixels), cv2.RANSAC, self.config.ransac_px
        )
        if inliers is None or np.count_nonzero(inliers) < self.config.min_matches:
            return None, "board_matches_inconsistent"
        keep = np.flatnonzero(inliers.ravel())
        ids = tuple(f"reference:{matches[i].queryIdx}" for i in keep)
        return BoardObservation(self.config.board_id, ids, objects[keep], pixels[keep]), None
