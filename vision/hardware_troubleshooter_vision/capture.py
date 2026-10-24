"""Camera capture and stop-and-look tagging of wrist frames with joint state.

Every stamp here is on one clock. The default is time.monotonic(); joint samples from
the arm SDK must be stamped with the same clock when they're read.

A wrist frame is only useful with the joint positions at its exposure. The Maker
Arm's feedback is fast, but the camera's exposure time isn't known to the
microsecond, so a frame is accepted only when the arm was still across a window
before it: then the exact sample doesn't matter.
"""

from __future__ import annotations

import bisect
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import CaptureConfig


@dataclass(frozen=True)
class JointSample:
    stamp: float
    positions: tuple[float, ...]  # SDK joint coordinates, J1..J6 then motor 7
    velocities: tuple[float, ...]
    torques: tuple[float, ...] = ()


@dataclass(frozen=True)
class Frame:
    camera_id: str
    stamp: float
    image: np.ndarray


@dataclass(frozen=True)
class WristFrame:
    frame: Frame
    arm: str
    joints: JointSample  # the sample nearest the image stamp, from a settled window


class JointHistory:
    """Recent joint samples for one arm, in stamp order."""

    def __init__(self, max_age_s: float = 10.0):
        self.max_age_s = max_age_s
        self._samples: deque[JointSample] = deque()

    def add(self, sample: JointSample) -> None:
        if self._samples and sample.stamp <= self._samples[-1].stamp:
            return  # out of order or repeated; the history stays monotonic
        self._samples.append(sample)
        while self._samples and sample.stamp - self._samples[0].stamp > self.max_age_s:
            self._samples.popleft()

    @property
    def latest(self) -> JointSample | None:
        return self._samples[-1] if self._samples else None

    def nearest(self, stamp: float) -> JointSample | None:
        if not self._samples:
            return None
        stamps = [s.stamp for s in self._samples]
        i = bisect.bisect_left(stamps, stamp)
        candidates = [self._samples[j] for j in (i - 1, i) if 0 <= j < len(stamps)]
        return min(candidates, key=lambda s: abs(s.stamp - stamp))

    def window(self, start: float, end: float) -> list[JointSample]:
        return [s for s in self._samples if start <= s.stamp <= end]


def tag_wrist_frame(
    frame: Frame, arm: str, history: JointHistory, config: CaptureConfig
) -> tuple[WristFrame | None, str | None]:
    """The frame with its joint state, or None and why it can't be used."""
    missing = config.missing()
    if missing:
        return None, "; ".join(f"unconfigured:{key}" for key in missing)
    sample = history.nearest(frame.stamp)
    if sample is None:
        return None, "no_joint_state"
    skew = abs(sample.stamp - frame.stamp)
    if skew > config.max_joint_skew_s:
        return None, f"joint_state_skew:{skew:.4f}s from the image stamp"
    start = frame.stamp - config.settle_duration_s
    window = history.window(start, frame.stamp + config.max_joint_skew_s)
    if not window or window[0].stamp - start > config.max_joint_skew_s:
        return None, "arm_not_settled:joint history doesn't cover the settle window"
    peak = max(max((abs(v) for v in s.velocities), default=0.0) for s in window)
    if peak > config.settle_velocity_rad_s:
        return None, f"arm_not_settled:joint speed {peak:.4f} rad/s in the settle window"
    return WristFrame(frame, arm, sample), None


class OpenCVCamera:
    """A UVC camera through OpenCV, stamped with time.monotonic() as each frame is read.

    The stamp is when read() returns, not the exposure; tag_wrist_frame's settle window
    is what makes that safe. MJPG keeps three USB cameras within the RDK X5's bandwidth.
    """

    def __init__(self, camera_id: str, device, width: int, height: int, fourcc: str = "MJPG"):
        import cv2

        self.camera_id = camera_id
        self._capture = cv2.VideoCapture(device)
        if not self._capture.isOpened():
            raise OSError(f"camera {camera_id}: can't open device {device!r}")
        self._capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # don't hand back a stale queued frame

    def read(self) -> Frame | None:
        ok, image = self._capture.read()
        stamp = time.monotonic()
        return Frame(self.camera_id, stamp, image) if ok else None

    def close(self) -> None:
        self._capture.release()
