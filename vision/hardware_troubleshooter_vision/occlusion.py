"""Image masks of the arms, projected from their URDF collision meshes and joint state.

Each mesh part is projected through the camera and filled as its convex hull, then the
mask is grown by a margin. A hull is larger than the part's silhouette, so the mask errs
toward hiding more of the board, never less. A part that straddles the camera plane
can't be projected reliably and masks the whole image.
"""

from __future__ import annotations

import math
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from hardware_troubleshooter_arm.kinematics import rpy_matrix, transform

from . import transforms

_NEAR_M = 0.01  # points closer to the camera than this can't be projected
_EDGE_STEP_PX = 4.0  # hull edges are densified before distortion so they stay curved


def load_stl(path: Path) -> np.ndarray:
    """Unique vertices of a binary or ASCII STL, as an (N, 3) array in file units."""
    data = Path(path).read_bytes()
    if len(data) >= 84:
        count = struct.unpack("<I", data[80:84])[0]
        if len(data) == 84 + 50 * count:
            records = np.frombuffer(
                data,
                dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]),
                count=count,
                offset=84,
            )
            return np.unique(records["v"].reshape(-1, 3).astype(float), axis=0)
    text = data.decode("ascii", errors="replace")
    vertices = [
        [float(v) for v in line.split()[1:4]]
        for line in text.splitlines()
        if line.strip().startswith("vertex")
    ]
    if not vertices:
        raise ValueError(f"{path}: no vertices (neither binary nor ASCII STL)")
    return np.unique(np.array(vertices), axis=0)


@dataclass(frozen=True)
class MeshPart:
    link: str
    vertices: np.ndarray  # metres, in the link frame


class ArmGeometry:
    """Collision geometry of one built arm URDF."""

    def __init__(self, parts: list[MeshPart]):
        self.parts = parts

    @classmethod
    def from_urdf(cls, text: str, urdf_dir: str | Path) -> ArmGeometry:
        robot = ET.fromstring(text)
        cache: dict[Path, np.ndarray] = {}
        parts = []
        for link in robot.findall("link"):
            for collision in link.findall("collision"):
                mesh = collision.find("geometry/mesh")
                if mesh is None:
                    continue
                path = (Path(urdf_dir) / mesh.get("filename")).resolve()
                if path not in cache:
                    cache[path] = load_stl(path)
                scale = np.array([float(v) for v in mesh.get("scale", "1 1 1").split()])
                origin = collision.find("origin")
                xyz = (
                    [float(v) for v in origin.get("xyz", "0 0 0").split()]
                    if origin is not None
                    else [0, 0, 0]
                )
                rpy = (
                    [float(v) for v in origin.get("rpy", "0 0 0").split()]
                    if origin is not None
                    else [0, 0, 0]
                )
                points = transforms.apply(transform(rpy_matrix(*rpy), xyz), cache[path] * scale)
                parts.append(MeshPart(link.get("name"), points))
        return cls(parts)


def arm_mask(
    geometry: ArmGeometry,
    link_poses: dict[str, np.ndarray],
    camera_from_base: np.ndarray,
    camera,
    margin_px: float,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """uint8 mask (255 = arm) of one arm in a camera image.

    link_poses: arm base -> link, from Kinematics.link_poses with measured joints.
    camera_from_base: arm base -> camera optical frame.
    """
    h, w = shape or (camera.height, camera.width)
    mask = np.zeros((h, w), np.uint8)
    for part in geometry.parts:
        pose = link_poses.get(part.link)
        if pose is None:
            raise ValueError(f"no pose for link {part.link!r}")
        points = transforms.apply(camera_from_base @ pose, part.vertices)
        in_front = points[:, 2] > _NEAR_M
        if not in_front.any():
            continue  # wholly behind the camera
        if not in_front.all():
            mask[:] = 255
            return mask
        _fill_hull(mask, camera, points)
    if margin_px > 0:
        size = 2 * math.ceil(margin_px) + 1
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    return mask


def _fill_hull(mask: np.ndarray, camera, points: np.ndarray) -> None:
    """Fill the part's hull: computed in ideal pixels, densified, clipped, then distorted."""
    h, w = mask.shape
    ideal = camera.project_ideal(points)
    hull = cv2.convexHull(ideal.astype(np.float32)).reshape(-1, 2).astype(float)
    # Distortion models only hold near the image, so clip the hull to a border around it.
    border = 0.25 * max(w, h)
    lo, hi = np.array([-border, -border]), np.array([w + border, h + border])
    if (hull.max(axis=0) < lo).any() or (hull.min(axis=0) > hi).any():
        return
    dense = []
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        steps = max(1, math.ceil(np.linalg.norm(b - a) / _EDGE_STEP_PX))
        dense.append(a + np.linspace(0, 1, steps, endpoint=False)[:, None] * (b - a))
    dense = np.clip(np.vstack(dense), lo, hi)
    raw = camera.project(camera.ideal_to_rays(dense))
    cv2.fillPoly(mask, [np.round(raw).astype(np.int32).reshape(-1, 1, 2)], 255)
