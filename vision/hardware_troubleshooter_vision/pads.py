"""Copper pad geometry in the board frame, from a .kicad_pcb file or explicit measurements.

Board frame as in registration's board_model: metres, x = X_mm / 1000, y = -Y_mm / 1000,
z out of the top face. Only pads with front copper are read. Pad identity is
`<reference>.<pad number>`, e.g. R1.2; it comes from the PCB file, never from an image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from hardware_troubleshooter_registration.board_model import _child, _parse_sexpr

_CIRCLE_POINTS = 32


@dataclass(frozen=True)
class Pad:
    name: str
    center: np.ndarray  # (2,) board frame, metres
    outline: np.ndarray  # closed polygon, board frame

    @property
    def min_size(self) -> float:
        extent = self.outline.max(axis=0) - self.outline.min(axis=0)
        return float(extent.min())


def rect_outline(center, size, angle) -> np.ndarray:
    w, h = size
    corners = np.array([[-w, -h], [w, -h], [w, h], [-w, h]]) / 2
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(center, float) + corners @ np.array([[c, -s], [s, c]]).T


def circle_outline(center, radius) -> np.ndarray:
    angles = np.linspace(0, 2 * math.pi, _CIRCLE_POINTS, endpoint=False)
    return np.asarray(center, float) + radius * np.column_stack([np.cos(angles), np.sin(angles)])


def oval_outline(center, size, angle) -> np.ndarray:
    """Stadium: a rectangle with semicircular ends along its longer side."""
    w, h = size
    radius = min(w, h) / 2
    half = (max(w, h) - min(w, h)) / 2
    arc = np.linspace(-math.pi / 2, math.pi / 2, _CIRCLE_POINTS // 2)
    right = np.column_stack([half + radius * np.cos(arc), radius * np.sin(arc)])
    points = np.vstack([right, -right])
    if h > w:
        points = points[:, ::-1] * [1, -1]
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(center, float) + points @ np.array([[c, -s], [s, c]]).T


def pads_from_kicad(text: str) -> list[Pad]:
    tree = _parse_sexpr(text)
    root = next((n for n in tree if isinstance(n, list) and n and n[0] == "kicad_pcb"), None)
    if root is None:
        raise ValueError("not a kicad_pcb file")
    pads = []
    for footprint in root[1:]:
        if not (
            isinstance(footprint, list) and footprint and footprint[0] in ("footprint", "module")
        ):
            continue
        at = _child(footprint, "at")
        fx, fy = float(at[1]), float(at[2])
        fa = math.radians(float(at[3])) if len(at) > 3 else 0.0
        reference = _reference(footprint)
        for pad in footprint[1:]:
            if not (isinstance(pad, list) and pad and pad[0] == "pad"):
                continue
            number, kind, shape = pad[1], pad[2], pad[3]
            layers = _child(pad, "layers")
            if kind == "np_thru_hole" or layers is None:
                continue
            if not any(layer in ("F.Cu", "*.Cu") for layer in layers[1:]):
                continue
            pad_at = _child(pad, "at")
            px, py = float(pad_at[1]), float(pad_at[2])
            # The pad's angle in the file already includes the footprint's rotation.
            pa = math.radians(float(pad_at[3])) if len(pad_at) > 3 else fa
            # KiCad's positive rotation is counter-clockwise on screen, in y-down millimetres.
            x = fx + px * math.cos(fa) + py * math.sin(fa)
            y = fy - px * math.sin(fa) + py * math.cos(fa)
            center = np.array([x / 1000.0, -y / 1000.0])
            size = _child(pad, "size")
            w, h = float(size[1]) / 1000.0, float(size[2]) / 1000.0
            if shape == "circle":
                outline = circle_outline(center, w / 2)
            elif shape == "oval":
                outline = oval_outline(center, (w, h), pa)
            else:  # rect, roundrect, trapezoid; a custom pad uses its anchor size
                outline = rect_outline(center, (w, h), pa)
            pads.append(Pad(f"{reference}.{number}", center, outline))
    return pads


def pads_from_specs(specs) -> tuple[list[Pad], list[str]]:
    pads, problems, names = [], [], set()
    for i, spec in enumerate(specs):
        try:
            name = str(spec["name"])
            if name in names:
                raise ValueError(f"repeated name {name!r}")
            center = np.asarray(spec["center_m"], dtype=float).reshape(2)
            shape = spec.get("shape")
            if shape == "rect":
                size = np.asarray(spec["size_m"], dtype=float).reshape(2)
                outline = rect_outline(center, size, float(spec.get("angle_rad", 0.0)))
            elif shape == "circle":
                outline = circle_outline(center, float(spec["radius_m"]))
            else:
                raise ValueError(f"unsupported shape {shape!r}; use rect or circle")
            if not np.all(np.isfinite(outline)) or np.ptp(outline, axis=0).min() <= 0:
                raise ValueError("pad has no area")
        except (KeyError, TypeError, ValueError) as exc:
            problems.append(f"invalid:pads.measured[{i}] ({exc})")
            continue
        names.add(name)
        pads.append(Pad(name, center, outline))
    return pads, problems


def _reference(footprint: list) -> str:
    for item in footprint[1:]:
        if not (isinstance(item, list) and item):
            continue
        if item[0] == "fp_text" and len(item) > 2 and item[1] == "reference":
            return str(item[2])
        if item[0] == "property" and len(item) > 2 and item[1] == "Reference":
            return str(item[2])
    return "?"
