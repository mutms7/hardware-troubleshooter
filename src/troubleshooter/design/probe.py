"""Rank the exposed copper on a net as places to land a probe tip.

Geometry only, in the board frame. The tip lands at a pad's centre, give or take the
position uncertainty. A candidate is refused when the tip, at the worst case of that
uncertainty, could reach copper on a different net: a short. Missing the pad only costs a
failed contact check, so it's reported but not refused. This says nothing about whether
the arm can reach the pad or whether a tall part blocks it; motion and vision own that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .kicad import Pad

# Lower is better: a test point is meant to be probed; an IC pin is the last resort.
_PASSIVE_PREFIXES = ("R", "C", "L", "FB", "D")
_CONNECTOR_PREFIXES = ("J", "P", "CN", "X")


@dataclass(frozen=True)
class Candidate:
    pad: Pad
    category: str
    clearance_m: float  # centre to the nearest other-net copper
    short_margin_m: float  # clearance - tip radius - uncertainty; < 0 is refused
    landing_margin_m: float  # inscribed radius - uncertainty; < 0 may miss the pad
    nearest_other: str  # pad id of that nearest other-net copper
    refused: bool
    reasons: tuple


def category(pad: Pad, testpoint: bool) -> tuple[int, str]:
    if pad.kind == "via":
        return 2, "via"
    if testpoint:
        return 0, "test_point"
    prefix = pad.ref.rstrip("0123456789")
    if prefix.upper() in _PASSIVE_PREFIXES:
        return 1, "passive"
    if prefix.upper() in _CONNECTOR_PREFIXES:
        return 2, "connector"
    return 3, "ic_pin"


def rank(
    targets: list[Pad],
    copper: list[Pad],
    testpoints: set[str],
    tip_diameter_m: float,
    uncertainty_m: float,
) -> list[Candidate]:
    """Every exposed pad in `targets`, best first. `copper` is all top copper on the board,
    covered or not: the tip can't short through solder mask, so only exposed copper of other
    nets counts against clearance."""
    exposed = [p for p in copper if p.mask_open]
    out = []
    for pad in targets:
        if not pad.mask_open:
            continue
        reasons = []
        clearance, nearest = math.inf, ""
        for other in exposed:
            if other.net == pad.net and pad.net:
                continue
            if other.pad_id == pad.pad_id:
                continue
            # A cheap reject before the polygon distance: copper beyond this can't matter.
            reach = clearance + max(other.width, other.height)
            if abs(other.x - pad.x) > reach or abs(other.y - pad.y) > reach:
                continue
            d = _distance_to_polygon((pad.x, pad.y), other.outline)
            if d < clearance:
                clearance, nearest = d, other.pad_id
        short_margin = clearance - tip_diameter_m / 2 - uncertainty_m
        landing_margin = _inscribed_radius(pad) - uncertainty_m
        if short_margin < 0:
            reasons.append(f"short_risk:{nearest}")
        if landing_margin < 0:
            reasons.append("may_miss_pad")
        _, name = category(pad, pad.ref in testpoints)
        out.append(
            Candidate(
                pad,
                name,
                clearance,
                short_margin,
                landing_margin,
                nearest,
                short_margin < 0,
                tuple(reasons),
            )
        )
    out.sort(
        key=lambda c: (
            c.refused,
            category(c.pad, c.category == "test_point")[0],
            c.landing_margin_m < 0,
            -min(c.short_margin_m, 1.0),
        )
    )
    return out


def _inscribed_radius(pad: Pad) -> float:
    return min(pad.width, pad.height) / 2


def _distance_to_polygon(point, polygon) -> float:
    if _inside(point, polygon):
        return 0.0
    px, py = point
    best = math.inf
    for i in range(len(polygon)):
        (ax, ay), (bx, by) = polygon[i], polygon[(i + 1) % len(polygon)]
        dx, dy = bx - ax, by - ay
        length = dx * dx + dy * dy
        t = 0.0 if length == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length))
        best = min(best, math.hypot(px - (ax + t * dx), py - (ay + t * dy)))
    return best


def _inside(point, polygon) -> bool:
    x, y = point
    inside = False
    for i in range(len(polygon)):
        (ax, ay), (bx, by) = polygon[i], polygon[(i + 1) % len(polygon)]
        if (ay > y) != (by > y) and x < ax + (y - ay) * (bx - ax) / (by - ay):
            inside = not inside
    return inside
