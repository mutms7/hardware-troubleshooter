"""Board outline geometry from explicit measurements or a KiCad Edge.Cuts layer.

Board frame: metres, origin and axes as in the source, z out of the top face. KiCad's
y-down millimetre coordinates become x = X/1000, y = -Y/1000 so the frame stays
right-handed. Pad and target positions must use the same frame; none come from images.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from .geometry import line_intersection, polygon_area

# Extra fraction trimmed from each end of an observed edge, beyond the model's own
# corner rounding, so blurred or segmented corners don't pull the line fit.
EDGE_TRIM_MARGIN = 0.05
_CHAIN_TOLERANCE_M = 1e-6
_SAMPLES_PER_EDGE = 40


@dataclass(frozen=True)
class Feature:
    name: str
    outline: np.ndarray  # closed polygon in the board frame
    expect: str  # "darker" or "brighter" than the board's median intensity


@dataclass(frozen=True)
class Target:
    name: str
    position: np.ndarray


@dataclass(frozen=True)
class Outline:
    edges: tuple  # ((p0, p1), ...) straight segments in counter-clockwise order
    corners: np.ndarray  # corners[i] is the intersection of edges[i-1] and edges[i]
    polyline: np.ndarray  # dense closed outline including arcs, for rendering
    trim_fraction: float


@dataclass(frozen=True)
class BoardModel:
    board_id: str
    frame_id: str
    thickness_m: float
    outline: Outline
    features: tuple
    targets: tuple

    @property
    def edges(self):
        return self.outline.edges

    @property
    def corners(self):
        return self.outline.corners


def outline_from_polygon(vertices) -> Outline:
    """Measured outline: straight edges between consecutive vertices."""
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("outline needs at least three [x, y] vertices")
    if polygon_area(points) < 0:
        points = points[::-1]
    if abs(polygon_area(points)) <= 0:
        raise ValueError("outline has zero area")
    edges = tuple((points[i], points[(i + 1) % len(points)]) for i in range(len(points)))
    return _outline(edges, [_line_samples(a, b) for a, b in edges], 0.0)


def outline_from_kicad(text: str) -> Outline:
    """Outer Edge.Cuts loop of a .kicad_pcb file. Arcs may round corners; edges stay straight."""
    items = _edge_cuts_items(_parse_sexpr(text))
    if not items:
        raise ValueError("no gr_line/gr_rect/gr_arc/gr_poly items on Edge.Cuts")
    loops = _chain_loops(items)
    loop = max(loops, key=lambda lp: abs(polygon_area(_densify(lp))))
    if polygon_area(_densify(loop)) < 0:
        loop = [_reverse(item) for item in reversed(loop)]
    lines = _merge_collinear([item for item in loop if item["kind"] == "line"])
    if len(lines) < 3:
        raise ValueError("outer outline needs at least three straight edges to fit")
    edges = tuple((item["start"], item["end"]) for item in lines)
    corners = _corners(edges)
    trim = 0.0
    for i, (p0, p1) in enumerate(edges):
        start, end = corners[i], corners[(i + 1) % len(edges)]
        length = np.linalg.norm(end - start)
        trim = max(trim, np.linalg.norm(p0 - start) / length, np.linalg.norm(end - p1) / length)
    return _outline(edges, [_item_samples(item) for item in loop], trim)


def feature_outline(spec: dict) -> np.ndarray:
    shape = spec.get("shape")
    center = np.asarray(spec["center_m"], dtype=float)
    if shape == "circle":
        angles = np.linspace(0, 2 * math.pi, 48, endpoint=False)
        return center + float(spec["radius_m"]) * np.column_stack([np.cos(angles), np.sin(angles)])
    if shape == "rect":
        w, h = (float(v) for v in spec["size_m"])
        a = float(spec["angle_rad"])
        corners = np.array([[-w, -h], [w, -h], [w, h], [-w, h]]) / 2
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        return center + corners @ rot.T
    raise ValueError(f"unsupported feature shape {shape!r}; use circle or rect")


def _outline(edges, samples, trim) -> Outline:
    trim += EDGE_TRIM_MARGIN
    if trim >= 0.4:
        raise ValueError("corner rounding leaves too little straight edge to fit")
    polyline = np.vstack(samples)
    return Outline(tuple(edges), _corners(edges), polyline, trim)


def _corners(edges) -> np.ndarray:
    corners = []
    for i, (p0, p1) in enumerate(edges):
        q0, q1 = edges[i - 1]
        corner = line_intersection(q0, q1 - q0, p0, p1 - p0)
        if corner is None:
            raise ValueError("neighbouring straight edges are parallel")
        corners.append(corner)
    return np.array(corners)


def _line_samples(a, b) -> np.ndarray:
    return a + np.linspace(0, 1, _SAMPLES_PER_EDGE, endpoint=False)[:, None] * (b - a)


# --- KiCad Edge.Cuts parsing ---------------------------------------------------------


def _parse_sexpr(text: str) -> list:
    stack: list = [[]]
    for token in re.findall(r'\(|\)|"(?:[^"\\]|\\.)*"|[^\s()"]+', text):
        if token == "(":
            stack.append([])
        elif token == ")":
            node = stack.pop()
            stack[-1].append(node)
        else:
            stack[-1].append(token[1:-1] if token.startswith('"') else token)
    return stack[0]


def _child(node: list, name: str):
    for item in node[1:]:
        if isinstance(item, list) and item and item[0] == name:
            return item
    return None


def _point(node: list, name: str) -> np.ndarray:
    item = _child(node, name)
    if item is None:
        raise ValueError(f"{node[0]} is missing {name}")
    return np.array([float(item[1]) / 1000.0, -float(item[2]) / 1000.0])


def _edge_cuts_items(tree: list) -> list:
    """Board-level graphic items on Edge.Cuts. Footprint Edge.Cuts aren't read."""
    root = next((n for n in tree if isinstance(n, list) and n and n[0] == "kicad_pcb"), None)
    if root is None:
        raise ValueError("not a kicad_pcb file")
    items = []
    for node in root[1:]:
        if not (isinstance(node, list) and node and str(node[0]).startswith("gr_")):
            continue
        layer = _child(node, "layer")
        if layer is None or layer[1] != "Edge.Cuts":
            continue
        kind = node[0]
        if kind == "gr_line":
            items.append(_line(_point(node, "start"), _point(node, "end")))
        elif kind == "gr_rect":
            (x0, y0), (x1, y1) = _point(node, "start"), _point(node, "end")
            corners = [np.array(p) for p in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
            items.extend(_line(corners[i], corners[(i + 1) % 4]) for i in range(4))
        elif kind == "gr_poly":
            pts = _child(node, "pts")
            if any(isinstance(p, list) and p[0] != "xy" for p in pts[1:]):
                raise ValueError("gr_poly with arc segments isn't supported")
            corners = [np.array([float(p[1]) / 1000.0, -float(p[2]) / 1000.0]) for p in pts[1:]]
            items.extend(
                _line(corners[i], corners[(i + 1) % len(corners)]) for i in range(len(corners))
            )
        elif kind == "gr_arc":
            items.append(_arc(node))
        # gr_circle is a closed cutout or a round board; neither has straight edges to fit.
    return items


def _line(start, end) -> dict:
    return {"kind": "line", "start": start, "end": end}


def _arc(node: list) -> dict:
    if _child(node, "mid") is not None:  # KiCad 6+
        return {
            "kind": "arc",
            "start": _point(node, "start"),
            "mid": _point(node, "mid"),
            "end": _point(node, "end"),
        }
    # KiCad 5: start is the centre, end is the first endpoint, angle in degrees (y-down).
    item = _child(node, "start")
    centre = np.array([float(item[1]), float(item[2])])
    item = _child(node, "end")
    first = np.array([float(item[1]), float(item[2])])
    angle = math.radians(float(_child(node, "angle")[1]))

    def rotate(a):
        d = first - centre
        x = centre[0] + d[0] * math.cos(a) - d[1] * math.sin(a)
        y = centre[1] + d[0] * math.sin(a) + d[1] * math.cos(a)
        return np.array([x / 1000.0, -y / 1000.0])

    return {"kind": "arc", "start": rotate(0.0), "mid": rotate(angle / 2), "end": rotate(angle)}


def _reverse(item: dict) -> dict:
    return {**item, "start": item["end"], "end": item["start"]}


def _chain_loops(items: list) -> list:
    remaining = [item for item in items if np.linalg.norm(item["end"] - item["start"]) > 0]
    loops = []
    while remaining:
        loop = [remaining.pop(0)]
        while np.linalg.norm(loop[-1]["end"] - loop[0]["start"]) > _CHAIN_TOLERANCE_M:
            tail = loop[-1]["end"]
            for i, item in enumerate(remaining):
                if np.linalg.norm(item["start"] - tail) <= _CHAIN_TOLERANCE_M:
                    loop.append(remaining.pop(i))
                    break
                if np.linalg.norm(item["end"] - tail) <= _CHAIN_TOLERANCE_M:
                    loop.append(_reverse(remaining.pop(i)))
                    break
            else:
                raise ValueError("Edge.Cuts outline isn't a closed loop")
        loops.append(loop)
    return loops


def _merge_collinear(lines: list) -> list:
    def collinear(a, b):
        da, db = a["end"] - a["start"], b["end"] - b["start"]
        cross = da[0] * db[1] - da[1] * db[0]
        touching = np.linalg.norm(a["end"] - b["start"]) <= _CHAIN_TOLERANCE_M
        return touching and abs(cross) <= 1e-9 * np.linalg.norm(da) * np.linalg.norm(db)

    merged = []
    for item in lines:
        if merged and collinear(merged[-1], item):
            merged[-1] = _line(merged[-1]["start"], item["end"])
        else:
            merged.append(item)
    if len(merged) > 1 and collinear(merged[-1], merged[0]):
        merged[0] = _line(merged[-1]["start"], merged[0]["end"])
        merged.pop()
    return merged


def _item_samples(item: dict) -> np.ndarray:
    if item["kind"] == "line":
        return _line_samples(item["start"], item["end"])
    a, m, b = item["start"], item["mid"], item["end"]
    centre = _circumcentre(a, m, b)
    radius = np.linalg.norm(a - centre)
    a0, am, a1 = (math.atan2(*(p - centre)[::-1]) for p in (a, m, b))
    ccw = (am - a0) % (2 * math.pi) < (a1 - a0) % (2 * math.pi)
    sweep = (a1 - a0) % (2 * math.pi) if ccw else -((a0 - a1) % (2 * math.pi))
    angles = a0 + np.linspace(0, sweep, _SAMPLES_PER_EDGE, endpoint=False)
    return centre + radius * np.column_stack([np.cos(angles), np.sin(angles)])


def _circumcentre(a, b, c) -> np.ndarray:
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(d) < 1e-15:
        raise ValueError("degenerate Edge.Cuts arc")
    sa, sb, sc = a.dot(a), b.dot(b), c.dot(c)
    x = (sa * (b[1] - c[1]) + sb * (c[1] - a[1]) + sc * (a[1] - b[1])) / d
    y = (sa * (c[0] - b[0]) + sb * (a[0] - c[0]) + sc * (b[0] - a[0])) / d
    return np.array([x, y])


def _densify(loop: list) -> np.ndarray:
    return np.vstack([_item_samples(item) for item in loop])
