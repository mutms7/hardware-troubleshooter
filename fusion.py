"""Fuse two ways of knowing where the target points are in the camera image.

  1. Arm odometry. The arm knows how far it has moved (from its joint encoders), and the servo
     calibration says how far that moves the image. So the points can be carried forward by the arm's
     motion alone: smooth and precise over a short move, but it drifts slowly (calibration error, sag).
  2. Vision. The tracker finds the board in each frame: absolute, so it doesn't drift, but it can
     jump, drop out (glare, probe in the way), or lock onto the wrong thing (wood grain repeats).

How they are combined, and why:
  * Predict with the arm's motion, then pull the estimate toward the vision fix by `gain`.
  * The arm's prediction is also handed back to the vision step, so it can throw away feature matches
    that don't agree with where the arm says the board went (this is what stops repeating texture
    from sliding the fit).
  * Vision is late (camera plus processing). Its fix is compared with the prediction AT THE FRAME'S TIME,
    not with the present, or every move would drag the estimate backwards.
  * A vision fix that disagrees with the arm by more than the allowed gap is rejected. The gap allowed
    grows with how far the arm has moved since the last accepted fix, because odometry error is
    proportional to travel. There is NO automatic resync to vision: the arm's estimate was validated
    to a few percent, a wrong vision lock is worse than none, and you can reset by hand (c).
  * While moving, the filter also learns the arm-to-image response J from the accepted fixes.

The pixel change for a tip displacement uses the servo calibration J (pixels the probe moves in the
image per mm of jog, for "right" and "up"), rescaled with height. The scene moves opposite to the probe.
"""
import json
import threading
import time
import urllib.request
from collections import deque

import numpy as np


class ArmOdometry:
    """Polls the arm server for the tip position and camera axes, in a background thread."""

    def __init__(self, url="http://127.0.0.1:8765", hz=15.0):
        self.url, self.period, self._sample, self._stop = url.rstrip("/"), 1.0 / hz, None, threading.Event()
        self.errors = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def latest(self):
        return self._sample

    def _run(self):
        while not self._stop.is_set():
            t0 = time.time()
            try:
                with urllib.request.urlopen(f"{self.url}/api/state?measure=1", timeout=2) as r:
                    s = json.load(r)
                tip = np.array(s.get("measured_tip") or s["tip"], float) * 1000.0     # mm; the encoders' view when available
                ax = s["camera_axes"]
                right, up = np.array(ax["right"], float), np.array(ax["up"], float)
                right[2] = up[2] = 0.0                                              # jogs are level, so use the level axes
                right, up = right / max(np.linalg.norm(right), 1e-9), up / max(np.linalg.norm(up), 1e-9)
                self._sample = {"t": (t0 + time.time()) / 2, "tip": tip, "right": right, "up": up,
                                "height": float(s["height_mm"])}                    # stamped mid-request, when the encoders were read
            except Exception:
                self.errors += 1
            self._stop.wait(max(0.0, self.period - (time.time() - t0)))


class PointFuser:
    def __init__(self, J_px_per_mm, cal_height_mm, gain=0.3, gate_px=12.0, gate_frac=0.15,
                 learn=True, ridge=15.0, forget=0.995):
        self.J0 = np.array(J_px_per_mm, float)       # the calibration file's response: the starting point and the anchor
        self.J = self.J0.copy()                      # the response in use; learned while the arm moves
        self.cal_height, self.gain = cal_height_mm, gain
        self.gate, self.gate_frac = gate_px, gate_frac   # allowed gap = gate + gate_frac * (px travelled since the last fix)
        self.learn, self.ridge, self.forget = learn, ridge, forget
        self.A, self.B = np.zeros((2, 2)), np.zeros((2, 2))
        self.reset()

    def reset(self):
        self.samples = deque(maxlen=600)             # (t, tip, right, up, height)
        self.hist = deque(maxlen=600)                # (t, cumulative pixel shift caused by the arm since the start)
        self.cum = np.zeros(2)
        self.fused = self.odom = None
        self.pred_tf = None
        self.last_vis = None
        self.last_vis_tf = None
        self.travel = 0.0                            # px the arm has carried the scene since the last accepted fix
        self.rejected_in_a_row = 0
        self.frames_without_vision = 0
        self.status = "waiting for the first vision fix"

    # -- helpers -----------------------------------------------------------------------------
    def _cum_at(self, t):
        if len(self.hist) < 2 or t is None:
            return self.cum
        ts = np.array([h[0] for h in self.hist])
        cs = np.array([h[1] for h in self.hist])
        return np.array([np.interp(t, ts, cs[:, 0]), np.interp(t, ts, cs[:, 1])])

    def _tip_at(self, t):
        if not self.samples:
            return None
        ts = np.array([s[0] for s in self.samples])
        tips = np.array([s[1] for s in self.samples])
        return np.array([np.interp(t, ts, tips[:, k]) for k in range(3)])

    def _ingest(self, sample):
        """Take a new arm sample: move the estimates by the arm's motion since the previous one."""
        if sample is None or (self.samples and sample["t"] <= self.samples[-1][0]):
            return
        if self.samples:
            prev = self.samples[-1]
            d = sample["tip"] - prev[1]
            Jh = self.J * (self.cal_height / max(sample["height"], 3.0))
            dpx = -(Jh @ np.array([d @ prev[2], d @ prev[3]]))      # the scene moves opposite to the probe
            self.cum = self.cum + dpx
            self.travel += float(np.linalg.norm(dpx))
            if self.fused is not None:
                self.fused = self.fused + dpx
                self.odom = self.odom + dpx
        self.samples.append((sample["t"], sample["tip"].copy(), sample["right"], sample["up"], sample["height"]))
        self.hist.append((sample["t"], self.cum.copy()))

    def _learn(self, vis, tf):
        """Refine J from the last two accepted vision fixes and the arm's motion between them."""
        if not self.learn or self.last_vis is None or self.last_vis_tf is None:
            return
        a, b = self._tip_at(self.last_vis_tf), self._tip_at(tf)
        if a is None or b is None:
            return
        ref = min(self.samples, key=lambda s: abs(s[0] - self.last_vis_tf))
        d = b - a
        x = np.array([d @ ref[2], d @ ref[3]]) * (self.cal_height / max(self.samples[-1][4], 3.0))
        y = -np.mean(vis - self.last_vis, axis=0)       # how far the probe moved in the image
        self.A = self.forget * self.A + np.outer(x, x)
        self.B = self.forget * self.B + np.outer(y, x)
        self.J = np.linalg.solve(self.A + self.ridge * np.eye(2), (self.B + self.ridge * self.J0).T).T

    def init_from_reference(self, ref_points, ref_tip_mm, ref_height_mm, sample):
        """Start from the arm's pose when the reference photo was taken, instead of waiting for vision to lock.

        If the arm is somewhere else now, the arm's motion since then says where the points are in the live
        view, so the first vision match can be guided to the right place and just has to confirm it."""
        self._ingest(sample)
        d = sample["tip"] - np.asarray(ref_tip_mm, float)
        Jh = self.J * (self.cal_height / max(sample["height"], 3.0))
        dpx = -(Jh @ np.array([d @ sample["right"], d @ sample["up"]]))
        self.fused = np.asarray(ref_points, float) + dpx
        self.odom = self.fused.copy()
        self.travel = float(np.linalg.norm(dpx))       # so the first vision fix is judged against a gap that allows for it
        self.status = f"started from the reference pose ({np.linalg.norm(d):.0f} mm away); waiting for vision to confirm"

    # -- the two halves of a step ------------------------------------------------------------
    def predict(self, sample, frame_time=None):
        """Call before the vision step. Moves the estimate by the arm's motion and returns where the points
        should be in the frame captured at frame_time (an (N, 2) array), or None before the first fix."""
        self._ingest(sample)
        if self.fused is None:
            return None
        self.pred_tf = self.fused - (self.cum - self._cum_at(frame_time))
        return self.pred_tf

    def correct(self, vision_px, frame_time=None):
        """Call after the vision step with its result (or None). Returns dict(fused, odom, vision, status)."""
        vis = None if vision_px is None else np.asarray(vision_px, float).reshape(-1, 2)
        if self.fused is None:                                  # the first vision fix starts everything
            if vis is None or not self.samples:
                return dict(fused=None, odom=None, vision=vis, status=self.status)
            self.fused = vis + (self.cum - self._cum_at(frame_time))
            self.odom = self.fused.copy()
            self.last_vis, self.last_vis_tf, self.travel = vis.copy(), frame_time, 0.0
            self.status = "started"
            return dict(fused=self.fused, odom=self.odom, vision=vis, status=self.status)

        if vis is None:
            self.frames_without_vision += 1
            self.status = f"dead-reckoning on the arm's motion ({self.frames_without_vision} frames without vision)"
        else:
            if self.pred_tf is None:
                self.pred_tf = self.fused - (self.cum - self._cum_at(frame_time))
            gap = float(np.mean(np.linalg.norm(vis - self.pred_tf, axis=1)))
            allowed = self.gate + self.gate_frac * self.travel
            if gap > allowed:
                self.rejected_in_a_row += 1
                self.status = (f"vision disagrees with the arm by {gap:.0f} px (allowed {allowed:.0f}); "
                               f"keeping the arm's estimate [{self.rejected_in_a_row} frames, c to reset]")
            else:
                self.fused = self.fused + self.gain * (vis - self.pred_tf)
                self._learn(vis, frame_time)
                self.last_vis, self.last_vis_tf, self.travel = vis.copy(), frame_time, 0.0
                self.rejected_in_a_row = self.frames_without_vision = 0
                self.status = "fused"
        return dict(fused=self.fused, odom=self.odom, vision=vis, status=self.status)

    def step(self, sample, vision_px, frame_time=None):
        """predict + correct in one call (for when the vision step doesn't need the prediction)."""
        self.predict(sample, frame_time)
        return self.correct(vision_px, frame_time)
