"""Read a 7-segment multimeter LCD from a camera frame.

The camera sees the meter from a taught arm pose, so the LCD lands in nearly the same place
every time. A calibration file gives the LCD's corners in a reference frame plus a grayscale
patch of it. Reading a frame:

1. Template-match the reference patch near its calibrated spot to find how far the display
   moved, and shift the corners by that much. A poor match refuses the frame.
2. Warp the LCD quad to a fixed canvas, where every segment, decimal point and annunciator
   has calibrated coordinates.
3. Judge each sample patch dark (on) or light (off) against that digit's own background, and
   mark it unknown where glare blows the pixels out. Never treat an unreadable spot as off.

Digits with unknown segments decode only when every possibility gives the same character.
OpenCV and numpy are imported lazily so the MCP server runs without them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SEGMENTS = "abcdefg"
# a top, b upper right, c lower right, d bottom, e lower left, f upper left, g middle
CHARS = {
    "abcdef": "0",
    "bc": "1",
    "abdeg": "2",
    "abcdg": "3",
    "bcfg": "4",
    "acdfg": "5",
    "acdefg": "6",
    "abc": "7",
    "abcf": "7",  # some meters light f on 7
    "abcdefg": "8",
    "abcdfg": "9",
    "abcfg": "9",  # 9 without the bottom bar
    "": " ",
    "g": "-",
    "def": "L",
    "adefg": "E",
    "eg": "r",
    "abefg": "P",
    "ceg": "n",
    "cdeg": "o",
}

Sample = tuple[float, float, float, float]  # centre x, centre y, width, height on the canvas


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DigitLayout(Strict):
    # Canvas x of the centres of the digit's (left, right) vertical segments on the upper row
    # (rows["f"]) and the lower row (rows["e"]). Measured per digit and per half: the camera
    # sees the LCD at an angle the corner warp doesn't fully undo, so the italic lean differs
    # across the display and between a digit's halves. Other rows interpolate between these.
    upper: tuple[float, float]
    lower: tuple[float, float]
    dp: Sample | None = None  # decimal point to the right of this digit, if the LCD has one


class DisplayLayout(Strict):
    reference: str  # grayscale PNG of the LCD region in the calibration frame
    reference_xy: tuple[int, int]  # where that patch's top-left sits in the calibration frame
    corners: list[tuple[float, float]] = Field(min_length=4, max_length=4)  # TL TR BR BL
    canvas: tuple[int, int] = (640, 448)
    search_px: int = Field(default=80, ge=0, le=400)
    min_match: float = Field(default=0.6, gt=0, le=1)
    rows: dict[Literal["a", "f", "g", "e", "d"], float]  # canvas y: a top, f/b upper, ...
    seg_size: tuple[float, float]  # (long, short) side of a segment sample patch
    digits: list[DigitLayout] = Field(min_length=1, max_length=8)
    minus: Sample | None = None
    annunciators: dict[str, Sample] = {}
    # Decimals a unit always shows, e.g. {"mV": 1} for a 600.0 mV range; used only when glare
    # hides every decimal point.
    fixed_decimals: dict[str, int] = {}
    glare_level: int = Field(default=245, ge=0, le=255)
    max_glare: float = Field(default=0.25, ge=0, le=1)  # share of a patch that may be blown out
    min_contrast: float = Field(default=40, ge=0)  # background minus segment, gray levels
    annunciator_fill: float = Field(default=0.15, gt=0, lt=1)  # dark share for a lit letter
    # A segment's darkness is (local background - patch mean) / (digit background - ink).
    # At or above on_level it's lit, at or below off_level it's off, in between unknown.
    on_level: float = Field(default=0.4, gt=0, le=1)
    off_level: float = Field(default=0.2, ge=0, lt=1)
    surround_px: int = Field(default=14, ge=2, le=60)  # ring around a patch for its background


def load_layout(path: str | Path) -> tuple[DisplayLayout, Path]:
    path = Path(path)
    layout = DisplayLayout.model_validate(json.loads(path.read_text(encoding="utf-8")))
    ref = Path(layout.reference)
    return layout, (ref if ref.is_absolute() else path.parent / ref).resolve()


@dataclass
class FrameReading:
    status: Literal["ok", "unreadable"]
    text: str = ""  # digits with "?" for undecodable ones and "." for decimal points
    value: float | None = None
    overload: bool = False
    decimals: int | None = None
    decimal_inferred: bool = False
    negative: bool | None = None  # None when the layout has no minus sign calibrated
    annunciators: dict[str, bool | None] = field(default_factory=dict)
    unknown_segments: list[str] = field(default_factory=list)  # e.g. "d3.c" or "dp3"
    shift_px: tuple[int, int] = (0, 0)
    match: float = 0.0
    failure_reason: str | None = None
    canvas: object | None = None  # the warped LCD (numpy array), for the caller to keep


class Reader:
    def __init__(self, layout: DisplayLayout, reference_path: Path):
        import cv2

        self.cv2 = cv2
        self.layout = layout
        self.reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
        if self.reference is None:
            raise ValueError(f"can't read display reference {reference_path}")

    # --- geometry -----------------------------------------------------------------------

    def _locate(self, gray) -> tuple[int, int, float]:
        """Shift of the display from its calibrated spot, and the match score."""
        cv2, lay = self.cv2, self.layout
        rh, rw = self.reference.shape
        x0, y0 = lay.reference_xy
        s = lay.search_px
        top, left = max(0, y0 - s), max(0, x0 - s)
        window = gray[top : min(gray.shape[0], y0 + rh + s), left : min(gray.shape[1], x0 + rw + s)]
        if window.shape[0] < rh or window.shape[1] < rw:
            return 0, 0, 0.0
        scores = cv2.matchTemplate(window, self.reference, cv2.TM_CCOEFF_NORMED)
        _, best, _, (bx, by) = cv2.minMaxLoc(scores)
        return left + bx - x0, top + by - y0, float(best)

    def _warp(self, image, dx: int, dy: int):
        import numpy as np

        cv2, lay = self.cv2, self.layout
        w, h = lay.canvas
        quad = np.float32([(x + dx, y + dy) for x, y in lay.corners])
        target = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        return cv2.warpPerspective(image, cv2.getPerspectiveTransform(quad, target), (w, h))

    def _segment_samples(self, digit: DigitLayout) -> dict[str, Sample]:
        lay = self.layout
        long_side, short_side = lay.seg_size
        r = lay.rows

        def x_at(side: float, y: float) -> float:
            """x of the left (side 0), right (1) or middle (0.5) line at canvas row y."""
            top = digit.upper[0] + side * (digit.upper[1] - digit.upper[0])
            bottom = digit.lower[0] + side * (digit.lower[1] - digit.lower[0])
            return top + (bottom - top) * (y - r["f"]) / (r["e"] - r["f"])

        vertical = (short_side, long_side)
        horizontal = (long_side, short_side)
        return {
            "a": (x_at(0.5, r["a"]), r["a"], *horizontal),
            "b": (x_at(1, r["f"]), r["f"], *vertical),
            "c": (x_at(1, r["e"]), r["e"], *vertical),
            "d": (x_at(0.5, r["d"]), r["d"], *horizontal),
            "e": (x_at(0, r["e"]), r["e"], *vertical),
            "f": (x_at(0, r["f"]), r["f"], *vertical),
            "g": (x_at(0.5, r["g"]), r["g"], *horizontal),
        }

    # --- sampling -----------------------------------------------------------------------

    @staticmethod
    def _patch(gray, sample: Sample):
        cx, cy, w, h = sample
        x0, y0 = round(cx - w / 2), round(cy - h / 2)
        return gray[max(0, y0) : max(0, y0 + round(h)), max(0, x0) : max(0, x0 + round(w))]

    def _state(
        self, gray, sample: Sample, background: float, ink: float, stroke: bool = False
    ) -> bool | None:
        """True dark (on), False light (off), None unreadable (glare, no contrast, or unsure).

        Each patch is judged against its own surroundings, not one threshold per digit: glare
        leaves a bright halo in which lit segments are still darker than the LCD around them,
        just less dark than elsewhere. A solid segment is on when its mean is darker than the
        local background by enough of the digit's contrast; in between is unsure. stroke is
        for annunciators, whose letters are thin lines: there, enough of the patch must be dark.
        """
        import numpy as np

        lay = self.layout
        patch = self._patch(gray, sample)
        if patch.size == 0:
            return None
        if np.mean(patch >= lay.glare_level) > lay.max_glare:
            return None
        contrast = background - ink
        if contrast < lay.min_contrast:
            return None
        cx, cy, w, h = sample
        margin = lay.surround_px
        ring = self._patch(gray, (cx, cy, w + 2 * margin, h + 2 * margin))
        ring = ring[ring < lay.glare_level]
        local = float(np.percentile(ring, 80)) if ring.size else background
        unblown = patch[patch < lay.glare_level]
        if stroke:
            return bool(np.mean(patch < (local + ink) / 2) > lay.annunciator_fill)
        darkness = (local - float(unblown.mean())) / contrast
        if darkness >= lay.on_level:
            return True
        if darkness <= lay.off_level:
            return False
        return None

    def _levels(self, gray, samples: list[Sample]) -> tuple[float, float]:
        """(background, ink) gray levels around these samples, ignoring glare."""
        import numpy as np

        xs = [s[0] for s in samples]
        ys = [s[1] for s in samples]
        box = gray[
            max(0, int(min(ys)) - 20) : int(max(ys)) + 20,
            max(0, int(min(xs)) - 20) : int(max(xs)) + 20,
        ]
        pixels = box[box < self.layout.glare_level]
        if pixels.size < 50:
            return 0.0, 0.0
        return float(np.percentile(pixels, 85)), float(np.percentile(pixels, 5))

    @staticmethod
    def _decode(states: dict[str, bool | None]) -> tuple[str, list[str]]:
        """The character, and the unknown segments.

        Each unknown segment could be on or off. Combinations that spell no character can't
        be what the LCD shows, so they're dropped; the digit decodes only if every remaining
        option agrees, otherwise it's "?".
        """
        unknown = [s for s in SEGMENTS if states[s] is None]
        options = set()
        for mask in range(1 << len(unknown)):
            lit = {s for s in SEGMENTS if states[s]}
            lit |= {s for i, s in enumerate(unknown) if mask >> i & 1}
            char = CHARS.get("".join(s for s in SEGMENTS if s in lit))
            if char is not None:
                options.add(char)
        return (options.pop() if len(options) == 1 else "?"), unknown

    # --- reading ------------------------------------------------------------------------

    def read(self, image) -> FrameReading:
        cv2, lay = self.cv2, self.layout
        gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        dx, dy, match = self._locate(gray_full)
        if match < lay.min_match:
            return FrameReading(
                "unreadable", match=match, shift_px=(dx, dy), failure_reason="display_not_found"
            )
        canvas = self._warp(image, dx, dy)
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) if canvas.ndim == 3 else canvas
        reading = FrameReading("ok", shift_px=(dx, dy), match=match, canvas=canvas)

        chars, dps = [], []
        for n, digit in enumerate(lay.digits, start=1):
            samples = self._segment_samples(digit)
            background, ink = self._levels(gray, list(samples.values()))
            states = {s: self._state(gray, samples[s], background, ink) for s in SEGMENTS}
            char, unknown = self._decode(states)
            chars.append(char)
            reading.unknown_segments += [f"d{n}.{s}" for s in unknown]
            if digit.dp is None:
                dps.append(False)
            else:
                dp = self._state(gray, digit.dp, background, ink)
                dps.append(dp)
                if dp is None:
                    reading.unknown_segments.append(f"dp{n}")

        everything = [s for d in lay.digits for s in self._segment_samples(d).values()]
        background, ink = self._levels(gray, everything)
        if lay.minus is not None:
            reading.negative = self._state(gray, lay.minus, background, ink)
            if reading.negative is None:
                reading.unknown_segments.append("minus")
        reading.annunciators = {
            name: self._state(gray, sample, background, ink, stroke=True)
            for name, sample in lay.annunciators.items()
        }

        text = "".join(c + ("." if dp else "") for c, dp in zip(chars, dps))
        reading.text = text.strip()
        body = "".join(chars).strip()
        if body == "" and not any(u.startswith("d") for u in reading.unknown_segments):
            return reading  # blank display: meter off (text "", value None)
        if body.replace(" ", "") in {"0L", "OL", "L"} or body.endswith("0L"):
            reading.overload = True
            return reading
        if "?" in body or not body.replace(" ", "").isdigit():
            reading.status, reading.failure_reason = "unreadable", "undecoded_digits"
            return reading

        decimals = None
        known = [i for i, dp in enumerate(dps) if dp]
        if known:
            decimals = len(chars) - 1 - known[-1]
        elif None in dps:
            unit = self.unit(reading.annunciators)
            if unit in lay.fixed_decimals:
                decimals, reading.decimal_inferred = lay.fixed_decimals[unit], True
            else:
                reading.status, reading.failure_reason = "unreadable", "decimal_point_hidden"
                return reading
        else:
            decimals = 0
        digits_only = body.replace(" ", "")
        value = int(digits_only) / 10**decimals
        if reading.negative:
            value = -value
        reading.value, reading.decimals = value, decimals
        return reading

    def vote(self, readings: list[FrameReading], min_agree: int) -> dict:
        """Combine frames: the reading most frames agree on, if at least min_agree do.

        The LCD refreshes a few times a second and a floating input wanders, so a single
        frame isn't trusted. Agreement is on everything reported: text, value, unit, flags.
        """
        votes: dict[tuple, list[FrameReading]] = {}
        for r in readings:
            if r.status != "ok":
                continue
            unit = self.unit(r.annunciators)
            key = (r.text, r.value, unit, r.overload, r.decimal_inferred)
            votes.setdefault(key, []).append(r)
        summary = {
            "frames": len(readings),
            "readable_frames": sum(len(v) for v in votes.values()),
            "candidates": sorted(
                (
                    {"text": k[0], "value": k[1], "unit": k[2], "frames": len(v)}
                    for k, v in votes.items()
                ),
                key=lambda c: -c["frames"],
            ),
            "frame_failures": sorted({r.failure_reason for r in readings if r.failure_reason}),
        }
        if not votes:
            return {"status": "failed", "failure_reason": "display_unreadable", **summary}
        key, group = max(votes.items(), key=lambda kv: len(kv[1]))
        best = group[0]
        if len(group) < min_agree:
            return {
                "status": "inconclusive",
                "failure_reason": "unstable_reading",
                "canvas": best.canvas,  # the most common frame, for a person or model to look at
                **summary,
            }
        return {
            "status": "ok",
            "failure_reason": None,
            "text": best.text,
            "value": best.value,
            "unit": key[2],
            "overload": best.overload,
            "decimal_inferred": best.decimal_inferred,
            "negative_sign_checked": best.negative is not None,
            "annunciators": best.annunciators,
            "unknown_segments": best.unknown_segments,
            "agreeing_frames": len(group),
            "canvas": best.canvas,
            **summary,
        }

    def unit(self, annunciators: dict[str, bool | None]) -> str | None:
        """The unit the lit annunciators spell, e.g. "mV" or "kΩ", None if unclear."""
        prefixes = [p for p in ("m", "u", "k", "M") if annunciators.get(p)]
        bases = [b for b in ("V", "A", "Ω", "Hz", "%", "F", "°C") if annunciators.get(b)]
        if len(prefixes) > 1 or len(bases) != 1:
            return None
        return (prefixes[0] if prefixes else "") + bases[0]
