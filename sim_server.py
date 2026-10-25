"""Simulation server: IK + a browser viewer. Never opens a serial port.

Serves viewer/ and urdf/ on localhost and exposes a tiny JSON API. The browser sends
tip-space nudges (meters) and tool-tilt steps; this module solves IK for the four arm
joints, clamps them to the calibration limits, rejects moves it can't make, and returns
the new joint angles.

The solver holds two things at once: the probe-tip position and the probe's tilt from
straight down (phi). Moving the tip keeps phi fixed, and changing phi keeps the tip fixed.

With --arm, this arm shares the bench with the others in config/workcell.json: every move is
checked against where the other arms are (collision.py) and `goto` plans a path around them.

With --sensors, this server owns the ESP32 sensor head (sensor_head.py): its ToF becomes a
second height estimate, and the step governor, floor stop and turret clearance use whichever of
the model and the ToF is lower. Both sensors are served at /api/sensors for the MCP.

  python sim_server.py          then open http://localhost:8765
  python sim_server.py --arm A --port 8765      and in another terminal
  python sim_server.py --arm B --port 8766
  python sim_server.py --hardware COM10 --sensors COM5
"""
import argparse
import json
import math
import mimetypes
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import pairwise
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from ikpy.chain import Chain
import serial
from scipy.optimize import least_squares

import collision
import sensor_head
from calibrate import JOINTS, PROBE_ID, TICKS, Bus, delta
from fk_test import CHAIN_INDEX, SIGN, ZERO_DEG

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
URDF_IK = ROOT / "urdf" / "debugger_arm_ik.urdf"
PORT = 8765

ARM_IDX = [CHAIN_INDEX[s] for s in JOINTS]  # pan, lift, elbow, wrist flex
TURRET_IDX = CHAIN_INDEX[PROBE_ID]
LINK_ROLL = 11                              # wrist-roll link; its z axis is the probe shaft
LINK_TIP = 12
LINK_PAN = 3

# Table = lowest point of the fixed base in the URDF root frame (meters), measured from the
# base-mount mesh. The board's top surface is assumed BOARD_THICKNESS above it; calibrate later.
TABLE_Z = -0.0554
BOARD_THICKNESS = 0.005
BOARD_Z = TABLE_Z + BOARD_THICKNESS
FLOOR_MARGIN = 0.002       # the probe never goes lower than this above the board surface
CONTACT_MAX_MM = 8.0       # ...except a deliberate touch (move contact_mm), at most this far below it
CONTACT_MIN_STEP_MM = 2.0  # and the governor doesn't shorten a touch's steps below this: the servos ignore smaller ones
# The URDF's tip point is offset from the mount and isn't known to be mm-accurate, so until the
# board height is touched off the model's height is reduced by this much (camera overrides it).
TIP_UNCERTAINTY_MM = 5.0
# Step governor: far from the board moves are big and fast, close to it they shrink.
FAR_MM, NEAR_MM = 40.0, 10.0       # above FAR full steps, below NEAR the smallest steps
CAP_FAR_MM, CAP_NEAR_MM = 10.0, 0.5
POS_TOL_MM = 1.5           # reject an IK result that misses the target by more than this
PHI_TOL_DEG = 1.5          # ...or that misses the requested tilt by more than this
MAX_JOINT_STEP_DEG = 15.0  # reject a single move that swings any joint further than this
TILT_STEP_DEG = 5.0
W_TILT = 3.0               # IK cost: mm of tip error one degree of tilt error is worth (straight down wins)
REG = 0.01                 # IK cost per degree of joint movement: prefer the solution nearest the current joints (tiny: only breaks ties)
MAX_JOG_MM = 10.0          # largest single jog command per axis; Jev loops for bigger moves
# How the wrist camera is rolled about its optical axis. 0 = image-up points away from the base.
# If a "right" jog moves the probe left in the image, try 180; if it moves up/down, try +/-90.
CAMERA_ROLL_DEG = 0.0
DOWN = np.array([0.0, 0.0, -1.0])

# Home pose: probe straight down, picked so every joint keeps plenty of headroom for tilting.
HOME_TIP = (-0.0787, -0.140, -0.020)

# Turret presets, degrees. Three probes; edit to match the real detents.
PROBE_ANGLES = (-120.0, 0.0, 120.0)
PROBE_START = 1
# Which tool head sits in each slot above (scope and logic are AD3 probes, dmm is the separate
# multimeter). Edit to match the real turret.
TOOL_HEADS = ("scope", "logic", "dmm")
# The turret swings the other tips past the board, so it only turns this far above it.
TOOL_CHANGE_CLEARANCE_MM = 20.0
# When the other arm is in the way of a move, try tilting the probe up to this far either side
# (the tip still lands on the same point).
TILT_AROUND_DEG = 20.0
# goto: lift straight up to this height, travel around the other arm, come straight down.
TRAVEL_MM = 30.0
MIN_TRAVEL_MM = 15.0       # ...unless both ends are at least this high: then it crosses at the lower end's height
PLAN_STEP_DEG = 5.0        # executed waypoints are at most this far apart on any joint
PLAN_SWEEP_MM = 10.0       # goto checks its path coarser than a jog, with that much more margin
SIM_STEP_S = 0.05          # sim only: pause between waypoints so the viewer shows the motion
# Sensor head ToF (--sensors). Its beam is assumed parallel to the probe shaft. A reading older
# than this, or taken with the probe tilted further than this, isn't used.
TOF_MAX_AGE_S = 0.3
TOF_MAX_TILT_DEG = 30.0

mimetypes.add_type("application/octet-stream", ".stl")
mimetypes.add_type("text/xml; charset=utf-8", ".urdf")


class Sim:
    def __init__(self):
        with open(ROOT / "calibration.json") as f:
            self.cal = json.load(f)
        self.chain = Chain.from_urdf_file(str(URDF_IK), base_elements=["root"])
        lo, hi = [], []
        for sid, name in JOINTS.items():
            c = self.cal["joints"][name]
            a = SIGN[sid] * c["min_delta"] * c["deg_per_tick"] + ZERO_DEG[sid]
            b = SIGN[sid] * c["max_delta"] * c["deg_per_tick"] + ZERO_DEG[sid]
            lo.append(math.radians(min(a, b)))
            hi.append(math.radians(max(a, b)))
        self.lo, self.hi = np.array(lo), np.array(hi)
        # ToF range with each head's tip touching the board. Mount geometry, so reset keeps it.
        self.tof_zero_mm = {}
        self.board_z_default = BOARD_Z  # before any touch_off; --floor-offset-mm lowers it
        self.reset()

    def pose(self, q):
        """Tip position, probe shaft direction (unit), and tilt phi in degrees from straight down.

        phi > 0 tilts the probe tip away from the base axis, phi < 0 toward it.
        """
        # The selected probe is always the one at the working position, so the controlled tip
        # does not depend on the turret angle: turning ID 5 only swaps which probe sits there.
        q = q.copy()
        q[TURRET_IDX] = 0.0
        fr = self.chain.forward_kinematics(q, full_kinematics=True)
        tip, shaft, pan = fr[LINK_TIP][:3, 3], fr[LINK_ROLL][:3, 2], fr[LINK_PAN][:3, 3]
        radial = tip - pan
        radial[2] = 0.0
        n = np.linalg.norm(radial)
        radial = radial / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
        phi = math.degrees(math.atan2(shaft @ radial, shaft @ DOWN))
        return tip, shaft, phi

    def lateral_deg(self, q):
        """How far the shaft leans sideways out of the arm's vertical plane (should be ~0 for this 4-joint arm)."""
        q = q.copy()
        q[TURRET_IDX] = 0.0
        fr = self.chain.forward_kinematics(q, full_kinematics=True)
        tip, shaft, pan = fr[LINK_TIP][:3, 3], fr[LINK_ROLL][:3, 2], fr[LINK_PAN][:3, 3]
        radial = tip - pan
        radial[2] = 0.0
        n = np.linalg.norm(radial)
        if n < 1e-6:
            return 0.0
        side = np.array([radial[1], -radial[0], 0.0]) / n          # horizontal, perpendicular to the radial direction
        return math.degrees(math.asin(float(np.clip(shaft @ side, -1.0, 1.0))))

    def solve(self, target, phi_target, q_start):
        """Joint angles that put the tip at `target` with the shaft at tilt `phi_target` (0 = straight down).

        Cost, per unit: 1 per mm of tip error, W_TILT per degree of tilt error and of sideways lean (so pointing straight
        down wins over a small position miss when the pose is at a joint limit), and REG per degree of joint movement
        (among equal solutions take the one nearest the current joints, which stops branch flips and big swings)."""
        x0 = np.clip(q_start[ARM_IDX], self.lo, self.hi)

        def residual(x):
            q = q_start.copy()
            q[ARM_IDX] = x
            tip, _, phi = self.pose(q)
            return np.r_[(tip - target) * 1000.0, W_TILT * (phi - phi_target), W_TILT * self.lateral_deg(q),
                         REG * np.degrees(x - x0)]

        r = least_squares(residual, x0, bounds=(self.lo, self.hi))
        q = q_start.copy()
        q[ARM_IDX] = r.x
        tip, _, phi = self.pose(q)
        return q, float(np.linalg.norm((tip - target) * 1000.0)), abs(phi - phi_target)

    def reset(self):
        q0 = np.zeros(13)
        q0[TURRET_IDX] = math.radians(PROBE_ANGLES[PROBE_START])
        q, pos_err, phi_err = self.solve(np.array(HOME_TIP), 0.0, q0)
        if hasattr(self, "q") and not self.claim(q):
            return
        self.q = q
        self.phi_target = 0.0
        self.probe_idx = PROBE_START
        self.note = ""
        self.retreating = False
        self.board_z = self.board_z_default
        self.calibrated = False
        self.msg = "ready" if pos_err < POS_TOL_MM and phi_err < PHI_TOL_DEG else (
            f"reset pose is off by {pos_err:.1f} mm / {phi_err:.1f} deg")

    def camera_axes(self):
        """World-frame unit vectors (right, up, depth) of the wrist camera.

        The camera looks along the probe shaft (depth, toward the board) and image-up points
        away from the base axis. Assumes that mounting; fix it by rolling the camera or by
        editing this method. Right-handed: right = depth x up.
        """
        _, shaft, _ = self.pose(self.q)
        depth = shaft / np.linalg.norm(shaft)
        tip, _, _ = self.pose(self.q)
        pan = self.chain.forward_kinematics(self.q, full_kinematics=True)[LINK_PAN][:3, 3]
        radial = tip - pan
        radial[2] = 0.0
        up = radial - (radial @ depth) * depth
        if np.linalg.norm(up) < 1e-6:  # probe pointing along the radial direction
            up = np.array([0.0, 0.0, 1.0]) - depth[2] * depth
        up /= np.linalg.norm(up)
        right = np.cross(depth, up)
        c, s = math.cos(math.radians(CAMERA_ROLL_DEG)), math.sin(math.radians(CAMERA_ROLL_DEG))
        return c * right - s * up, s * right + c * up, depth  # roll about the optical axis

    def jog_camera(self, right_mm, up_mm, in_mm, height_mm=None):
        """Move the probe tip in the camera frame. Positive = right / image-up / toward the board.

        height_mm: optional height above the board as seen by the camera; trusted over the URDF.
        """
        if max(abs(right_mm), abs(up_mm), abs(in_mm)) > MAX_JOG_MM:
            self.msg = f"blocked: jog larger than {MAX_JOG_MM:.0f} mm per command"
            return
        right, up, depth = self.camera_axes()
        # The camera's "up" is perpendicular to the probe, so with the probe tilted it has a vertical
        # part: a sideways correction would also raise or lower the tip. Keep sideways jogs level; only
        # "in" (along the probe) changes the height.
        up = np.array([up[0], up[1], 0.0])
        up = up / np.linalg.norm(up) if np.linalg.norm(up) > 1e-6 else np.array([0.0, 1.0, 0.0])
        right = np.array([right[0], right[1], 0.0])
        right = right / np.linalg.norm(right) if np.linalg.norm(right) > 1e-6 else np.array([1.0, 0.0, 0.0])
        self.move((right * right_mm + up * up_mm + depth * in_mm) / 1000.0, height_mm)

    def joints(self, q=None):
        q = self.q if q is None else q
        return {self.chain.links[i].name: float(q[i]) for i in CHAIN_INDEX.values()}

    def claim(self, q):
        """Reserve pose q against the other arms. False (and msg says why) if it isn't clear."""
        if WORKCELL is None:
            return True
        if WORKCELL.commit(self.joints(), self.joints(q)):
            return True
        self.msg = WORKCELL.msg
        return False

    def announce(self):
        """Tell the other arms where this one is, without checking: it's already there."""
        if WORKCELL is not None:
            with WORKCELL.lock():
                WORKCELL.publish(self.joints())

    def _solve_step(self, target, phi_target, label):
        q, pos_err, phi_err = self.solve(target, phi_target, self.q)
        if pos_err > POS_TOL_MM or phi_err > PHI_TOL_DEG:
            return None, f"unreachable: {label} misses by {pos_err:.1f} mm / {phi_err:.1f} deg (joint limits)"
        swing = max(abs(math.degrees(q[i] - self.q[i])) for i in ARM_IDX)
        if swing > MAX_JOINT_STEP_DEG:
            return None, f"blocked: would swing a joint {swing:.0f} deg in one step"
        return q, ""

    def _try(self, target, phi_target, label, below_mm=0.0):
        """below_mm: how far under the board plane the target may go (a deliberate touch, see move contact_mm)."""
        if target[2] < self.board_z - below_mm / 1000.0:
            self.msg = f"blocked: z {target[2]*1000:.0f} mm is below the board plane"
            return False
        q, why = self._solve_step(target, phi_target, label)
        if q is None:
            self.msg = why
            return False
        if not self.claim(q):
            # The other arm is in the way. Same tip point, different probe tilt?
            blocked = self.msg
            steps = int(TILT_AROUND_DEG // TILT_STEP_DEG)
            for k in (s * sign for s in range(1, steps + 1) for sign in (1, -1)):
                phi = phi_target + k * TILT_STEP_DEG
                q, _ = self._solve_step(target, phi, label)
                if q is not None and self.claim(q):
                    self.q, self.phi_target, self.msg = q, phi, "ok"
                    self.note = f"tilted to {phi:.0f} deg to clear the other arm"
                    return True
            self.msg = blocked
            return False
        self.q, self.phi_target, self.msg = q, phi_target, "ok"
        return True

    @staticmethod
    def step_cap_mm(height_mm):
        if height_mm >= FAR_MM:
            return CAP_FAR_MM
        if height_mm <= NEAR_MM:
            return CAP_NEAR_MM
        return CAP_NEAR_MM + (height_mm - NEAR_MM) / (FAR_MM - NEAR_MM) * (CAP_FAR_MM - CAP_NEAR_MM)

    def height_mm(self, tip=None):
        """Model's height of the tip above the board surface, minus the tip uncertainty until
        the board height has been touched off. Conservative on purpose."""
        tip = self.pose(self.q)[0] if tip is None else tip
        return (tip[2] - self.board_z) * 1000.0 - (0.0 if self.calibrated else TIP_UNCERTAINTY_MM)

    def tof(self):
        """The sensor head's ToF reading and the tip height it implies, or None without --sensors.

        height = (range - zero) * cos(phi), zero being the range read with this head's tip on the
        board. height_mm is None, and why says so, unless the reading is fresh, valid and zeroed
        and the probe isn't tilted too far."""
        if SENSORS is None:
            return None
        reading = SENSORS.tof()
        zero = self.tof_zero_mm.get(TOOL_HEADS[self.probe_idx])
        out = {"range_mm": None, "valid": False, "age_s": None, "zero_mm": zero, "height_mm": None}
        if reading is None:
            out["why"] = "no reading yet" if SENSORS.connected else f"sensor head: {SENSORS.error}"
            return out
        age = time.time() - reading.host_time
        out.update(range_mm=reading.range_mm, valid=reading.valid, age_s=round(age, 3))
        phi = self.pose(self.q)[2]
        if age > TOF_MAX_AGE_S:
            out["why"] = "stale" if SENSORS.connected else f"stale; sensor head: {SENSORS.error}"
        elif not reading.valid:
            out["why"] = "no target in range"
        elif zero is None:
            out["why"] = "not zeroed: touch_off with this head, or start with --tof-offset-mm"
        elif abs(phi) > TOF_MAX_TILT_DEG:
            out["why"] = f"probe tilted past {TOF_MAX_TILT_DEG:.0f} deg"
        else:
            out["height_mm"] = (reading.range_mm - zero) * math.cos(math.radians(phi))
            out["why"] = ""
        return out

    def tof_height_mm(self):
        return (self.tof() or {}).get("height_mm")

    def guard_height_mm(self):
        """Height the safety limits use: the model's, or the ToF's when that's lower."""
        h, tof = self.height_mm(), self.tof_height_mm()
        return h if tof is None else min(h, tof)

    def touch_off(self, tip_z=None, tof_mm=None, zero_tof=True):
        """Call when the probe tip is physically touching the board: ties the model to reality,
        and zeroes the ToF for the current head.

        tip_z (m): the tip height to use, e.g. the encoders' measured tip, which is above the
        commanded one while the probe is pushing on the board. Default: the commanded tip.
        tof_mm: the ToF zero to use, e.g. a median over many readings. Default: the latest reading.
        zero_tof=False leaves the ToF zeros alone (its reading at contact depends on the spot)."""
        self.board_z = float(self.pose(self.q)[0][2] if tip_z is None else tip_z)
        self.calibrated = True
        self.msg = f"board height set from the tip: z = {self.board_z * 1000:.1f} mm"
        head = TOOL_HEADS[self.probe_idx]
        if not zero_tof:
            return
        if tof_mm is not None:
            self.tof_zero_mm[head] = float(tof_mm)
            self.msg += f"; ToF zero for {head}: {float(tof_mm):.1f} mm"
            return
        tof = self.tof()
        if tof is not None:
            if tof["valid"] and tof["age_s"] <= TOF_MAX_AGE_S:
                self.tof_zero_mm[head] = tof["range_mm"]
                self.msg += f"; ToF zero for {head}: {tof['range_mm']} mm"
            else:
                self.msg += f"; ToF not zeroed ({tof['why'] or 'no reading'})"

    def clear_touch_off(self):
        """Forget the board height and ToF zeros from touch_off, back to the startup floors.
        Run before re-calibrating: a stale board height can sit above the real board."""
        self.board_z = self.board_z_default
        self.calibrated = False
        self.tof_zero_mm = {}
        self.msg = f"touch-off cleared: board z back to {self.board_z * 1000:.1f} mm, ToF not zeroed"

    def move(self, d, height_mm=None, contact_mm=None):
        """Move the tip by d (meters), shortened by the step governor and stopped at the floor.

        height_mm is the height above the board the camera sees. When given it overrides the
        URDF-based estimate, because the camera is trusted over the model.
        contact_mm: a deliberate touch. Once the board height is touched off, the command may go
        this far below the board (at most CONTACT_MAX_MM) instead of stopping FLOOR_MARGIN above
        it: the servos lag the command by a few mm near the board, so the caller stops on the
        encoders (measure), not on the command.
        """
        tip, _, _ = self.pose(self.q)
        d = np.array(d, dtype=float)
        self.note = ""
        h = self.height_mm(tip) if height_mm is None else float(height_mm)
        tof = self.tof_height_mm()
        if tof is not None and tof < h:
            h = tof  # the ToF can only make the limits more cautious, never less
        # Moving straight away from the board can't hurt it, so retreating isn't slowed or shortened.
        self.retreating = bool(d[2] > 0 and d[2] >= 0.9 * np.linalg.norm(d))
        cap = CAP_FAR_MM if self.retreating else self.step_cap_mm(h)
        if contact_mm is not None and self.calibrated:
            cap = max(cap, CONTACT_MIN_STEP_MM)  # the caller checks the encoders after every step
        biggest = float(np.max(np.abs(d))) * 1000.0
        if biggest > cap:
            d *= cap / biggest
            self.note = f"shortened to {cap:.1f} mm ({h:.0f} mm above board)"
        margin = FLOOR_MARGIN * 1000.0
        if contact_mm is not None and self.calibrated:
            margin = -min(max(float(contact_mm), 0.0), CONTACT_MAX_MM)
        room = (h - margin) / 1000.0                               # how far down we may still go
        room = min(room, tip[2] - (TABLE_Z + FLOOR_MARGIN))        # never below the table
        if d[2] < 0 and -d[2] > room:
            if room > 1e-4:
                d *= room / -d[2]
                self.note = "stopped at the floor"
            else:
                d[2] = 0.0
                self.note = "at the floor: not going lower"
                if not d.any():
                    self.msg = "blocked: probe is at the floor"
                    return
        self._try(tip + d, self.phi_target, "move", below_mm=-margin if margin < 0 else 0.0)

    def tilt(self, step):
        tip, _, _ = self.pose(self.q)
        self._try(tip, self.phi_target + step * TILT_STEP_DEG, "tilt")

    def point_down(self, max_step_deg=3.0, max_swing_deg=2 * MAX_JOINT_STEP_DEG - 2.0):
        """Turn the probe to straight down (tilt 0) with the tip held where it is.

        Solved in small tilt steps (each a short joint move). Holding the tip while the tilt changes swings the elbow and
        wrist by more than the tilt itself, so one call applies as far as stays under `max_swing_deg` of joint travel; if
        that is not all the way, msg says 'ok: tilt now X deg' and the call is simply repeated (Arm.point_down does this).
        Nothing is applied if even the first step is blocked by a joint limit."""
        tip, _, phi = self.pose(self.q)
        q, ph = self.q.copy(), phi
        n = max(1, int(math.ceil(abs(ph) / max_step_deg)))
        best = None
        for k in range(1, n + 1):
            want = ph * (1 - k / n)
            q, pos_err, phi_err = self.solve(tip, want, q)
            if pos_err > POS_TOL_MM or phi_err > PHI_TOL_DEG:
                if best is None:
                    self.msg = (f"point down: can't tilt toward straight down with the tip here ({pos_err:.1f} mm / "
                                f"{phi_err:.1f} deg off: a joint limit). Move the tip, then try again")
                    return False
                break
            if max(abs(math.degrees(q[i] - self.q[i])) for i in ARM_IDX) > max_swing_deg:
                break
            best = (q.copy(), want)
        if best is None:
            self.msg = "point down: even the first step would swing a joint too far; not applied"
            return False
        self.q, self.phi_target = best
        self.msg = "ok" if abs(best[1]) < 0.05 else f"ok: tilt now {best[1]:.1f} deg, call point_down again"
        return True

    def probe(self, step):
        new = self.probe_idx + step
        if not 0 <= new < len(PROBE_ANGLES):
            self.msg = f"blocked: only probes 1 to {len(PROBE_ANGLES)}"
            return
        if new != self.probe_idx and self.guard_height_mm() < TOOL_CHANGE_CLEARANCE_MM:
            self.msg = (f"blocked: lift to {TOOL_CHANGE_CLEARANCE_MM:.0f} mm above the board "
                        "before turning the turret")
            return
        # Only the turret (ID 5) turns. The arm does not move, so the tip jumps to the new probe.
        q = self.q.copy()
        q[TURRET_IDX] = math.radians(PROBE_ANGLES[new])
        if not self.claim(q):
            return
        self.probe_idx, self.q = new, q
        self.msg = "ok"

    def select_tool(self, head):
        if head not in TOOL_HEADS:
            self.msg = f"unknown tool head: {head!r} ({', '.join(TOOL_HEADS)})"
            return
        self.probe(TOOL_HEADS.index(head) - self.probe_idx)

    def _vertical(self, q, z_to, phi):
        """IK poses moving the tip of pose q straight up or down to height z_to, 2 mm apart."""
        tip = self.pose(q)[0]
        n = max(1, math.ceil(abs(z_to - tip[2]) / 0.002))
        out = []
        for k in range(1, n + 1):
            target = np.array([tip[0], tip[1], tip[2] + (z_to - tip[2]) * k / n])
            q, pos_err, phi_err = self.solve(target, phi, q)
            if pos_err > POS_TOL_MM or phi_err > PHI_TOL_DEG:
                return None
            out.append(q)
        return out

    def plan_to(self, target, phi=None, seed=0):
        """Joint poses from here to tip `target` (arm frame, m) at tilt phi, or None (msg says why).

        Lifts straight up to TRAVEL_MM above the board, crosses in joint space around the other
        arms (straight if that's clear, else RRT-Connect), then comes straight down.
        """
        phi = self.phi_target if phi is None else phi
        target = np.asarray(target, dtype=float)
        floor = self.board_z + FLOOR_MARGIN
        travel_z = self.board_z + TRAVEL_MM / 1000.0
        # Both ends already well clear of the board: cross at the lower of the two, rather than climbing to TRAVEL_MM (a
        # start pose 28 mm up was refused because its 2 mm vertical leg was out of the wrist's reach). The crossing is
        # still checked against the floor below.
        here_z = float(self.pose(self.q)[0][2])
        if min(here_z, target[2]) >= self.board_z + MIN_TRAVEL_MM / 1000.0:
            travel_z = min(travel_z, here_z, float(target[2]))
        if target[2] < floor:
            self.msg = f"blocked: target z {target[2] * 1000:.0f} mm is below the board plane"
            return None
        obstacles = {}
        if WORKCELL is not None:
            with WORKCELL.lock():
                obstacles, why = WORKCELL.obstacles()
            if obstacles is None:
                self.msg = f"blocked: {why}"
                return None

        # Samples are PLAN_SWEEP_MM of motion apart; clearing them by this much keeps the whole
        # path SWEEP_MM / 2 clear, which is what each waypoint's claim checks again.
        need = (PLAN_SWEEP_MM + collision.SWEEP_MM) / 2000.0

        def clear(qs, z_min):
            if any(self.pose(q)[0][2] < z_min for q in qs):
                return False
            if not obstacles:
                return True
            return WORKCELL.sample_margins([self.joints(q) for q in qs], obstacles)[0].min() >= need

        def chord(x0, x1):
            n = collision.sample_count(x1 - x0, PLAN_SWEEP_MM)
            out = []
            for k in range(n + 1):
                q = self.q.copy()
                q[ARM_IDX] = x0 + (x1 - x0) * k / n
                out.append(q)
            return out

        goal, pos_err, phi_err = self.solve(target, phi, self.q)
        if pos_err > POS_TOL_MM or phi_err > PHI_TOL_DEG:
            self.msg = f"unreachable: target misses by {pos_err:.1f} mm / {phi_err:.1f} deg (joint limits)"
            return None
        up, down = [self.q], [goal]
        if self.pose(self.q)[0][2] < travel_z:
            up += self._vertical(self.q, travel_z, self.phi_target) or [None]
        if target[2] < travel_z:
            down = ([goal] + (self._vertical(goal, travel_z, phi) or [None]))[::-1]
        if any(q is None for q in up + down):
            self.msg = "unreachable: can't lift straight up at the start or the goal"
            return None
        if not clear(up, floor):
            self.msg = "blocked: can't lift straight up here, the other arm is in the way"
            return None
        if not clear(down, floor):
            self.msg = "blocked: the other arm is at the goal or right above it"
            return None
        a, b = up[-1][ARM_IDX], down[0][ARM_IDX]

        def edge_ok(x0, x1):
            return clear(chord(x0, x1), min(travel_z, self.pose(up[-1])[0][2], self.pose(down[0])[0][2]) - 1e-4)

        middle = [a, b] if edge_ok(a, b) else collision.rrt_connect(
            a, b, self.lo, self.hi, edge_ok, np.random.default_rng(seed))
        if middle is None:
            self.msg = "blocked: no path around the other arm found"
            return None
        middle = collision.shortcut(middle, edge_ok, np.random.default_rng(seed))
        path = up + [q for x0, x1 in pairwise(middle) for q in chord(x0, x1)[1:]] + down[1:]
        # Thin to waypoints at most PLAN_STEP_DEG apart. The arm moves straight in joint space
        # between them, which isn't quite the path checked above, so check those chords too.
        steps = [path[0]]
        for prev, q in pairwise(path):
            if np.max(np.abs(np.degrees(q[ARM_IDX] - steps[-1][ARM_IDX]))) > PLAN_STEP_DEG:
                steps.append(prev)
        steps.append(path[-1])
        if not all(clear(chord(q0[ARM_IDX], q1[ARM_IDX]), floor) for q0, q1 in pairwise(steps)):
            steps = path
        self.msg = "ok"
        return steps[1:]

    def state(self):
        tip, shaft, phi = self.pose(self.q)
        right, up, depth = self.camera_axes()
        model_h, tof = self.height_mm(tip), self.tof()
        h = model_h if tof is None or tof["height_mm"] is None else min(model_h, tof["height_mm"])
        return {
            "camera_axes": {"right": right.tolist(), "up": up.tolist(), "depth": depth.tolist()},
            "joints": {self.chain.links[i].name: float(self.q[i]) for i in CHAIN_INDEX.values()},
            "tip": [float(v) for v in tip],
            "shaft": [float(v) for v in shaft],
            "phi": phi,
            "board_z": self.board_z,
            "board_calibrated": self.calibrated,
            # What the limits use: the lower of the model and the ToF (see tof.why if unused).
            "height_mm": h,
            "height_source": "model" if h == model_h else "tof",
            "model_height_mm": model_h,
            "tof": tof,
            "cap_mm": self.step_cap_mm(h),
            "note": self.note,
            "probe": self.probe_idx + 1,
            "probe_deg": PROBE_ANGLES[self.probe_idx],
            "probe_count": len(PROBE_ANGLES),
            "tool": TOOL_HEADS[self.probe_idx],
            "tools": list(TOOL_HEADS),
            "msg": self.msg,
            "workcell": None if WORKCELL is None else dict(
                WORKCELL.status(self.joints()), arm=WORKCELL.me,
                tip_world=(WORKCELL.models[WORKCELL.me].base @ np.r_[tip, 1.0])[:3].tolist()),
            "goto": {"active": EXEC.active, "step": EXEC.step, "steps": EXEC.steps},
        }


PRESENT_LOAD = 60   # STS/SMS register map; untested on the HX servos, so treat load as a hint only
ACCELERATION = 41   # STS/SMS register map, same assumption as calibrate.py
GOAL_POSITION = 42
GOAL_SPEED = 46


class Hardware:
    """Drives the real servos from the sim state. Nothing is written until arm(True)."""

    def __init__(self, sim, port, baud, speed, accel):
        self.sim, self.speed, self.accel = sim, speed, accel
        self.near_speed = 100       # steps/s within FAR_MM of the board
        self._speed_now = speed
        self.armed = False
        self.probes = sim.cal.get("probes", {}).get("positions", {})
        self.port_name, self.baud = port, baud
        self.bus = Bus(port, baud)
        dead = [i for i in (*JOINTS, PROBE_ID) if not self.bus.ping(i)]
        if dead:
            raise SystemExit(f"No reply from servo ID(s) {dead} on {port}.")

    def _cal(self, sid):
        return self.sim.cal["joints"][JOINTS[sid]]

    def raw_to_rad(self, sid, raw):
        c = self._cal(sid)
        return math.radians(SIGN[sid] * delta(raw, c["mid_raw"]) * c["deg_per_tick"] + ZERO_DEG[sid])

    def rad_to_raw(self, sid, rad):
        c = self._cal(sid)
        d = round((math.degrees(rad) - ZERO_DEG[sid]) / SIGN[sid] / c["deg_per_tick"])
        d = max(c["min_delta"], min(c["max_delta"], d))  # never past the calibrated range
        return (c["mid_raw"] + d) % TICKS

    def sync(self):
        """Copy the arm's real pose into the sim so the viewer shows the real arm."""
        for sid in JOINTS:
            self.sim.q[CHAIN_INDEX[sid]] = self.raw_to_rad(sid, self.bus.read_pos(sid))
        self.sim.phi_target = self.sim.pose(self.sim.q)[2]
        if self.probes:
            raw5 = self.bus.read_pos(PROBE_ID)
            slots = [self.probes[f"probe_{i + 1}"] for i in range(len(PROBE_ANGLES))]
            near = min(range(len(slots)), key=lambda i: abs(delta(raw5, slots[i])))
            self.sim.probe_idx = near
            self.sim.q[TURRET_IDX] = math.radians(PROBE_ANGLES[near])
        self.sim.announce()

    def reconnect(self):
        """The USB adapter dropped (a servo tool wrote to the bus, the board was power-cycled, a cable
        moved), which leaves our open serial handle dead. Reopen the port. We come back disarmed: the
        servos may have reset, so torque and goals are unknown until you arm again."""
        self.armed = False
        try:
            self.bus.close()
        except Exception:
            pass
        try:
            self.bus = Bus(self.port_name, self.baud)
            return True
        except SystemExit:
            return False

    def measure(self):
        """Where the tip really is, from the joint encoders, without touching the commanded pose.

        The commanded tip is what we asked for. If the probe is blocked (the board, or an
        obstacle) the joints stop short of the command and the two drift apart: that gap, the
        following error, is the contact signal. Returns the gap in mm (positive z = the real tip
        is above where it was told to be) and the measured tip."""
        q = self.sim.q.copy()
        for sid in JOINTS:
            q[CHAIN_INDEX[sid]] = self.raw_to_rad(sid, self.bus.read_pos(sid))
        measured = self.sim.pose(q)[0]
        commanded = self.sim.pose(self.sim.q)[0]
        gap = (measured - commanded) * 1000.0
        joints = []
        for sid in JOINTS:
            goal, r1, _ = self.bus.ph.read2ByteTxRx(self.bus.port, sid, GOAL_POSITION)
            load, r2, _ = self.bus.ph.read2ByteTxRx(self.bus.port, sid, PRESENT_LOAD)
            present = self.bus.read_pos(sid)
            pid = [self.bus.ph.read1ByteTxRx(self.bus.port, sid, reg) for reg in (21, 22, 23)]  # P, D, I on STS-style servos
            joints.append({"id": sid, "pid_regs_21_22_23": [v if r == 0 else None for v, r, _ in pid],
                           "goal": goal % TICKS if r1 == 0 else None, "present": present,
                           "err_ticks": delta(present, goal % TICKS) if r1 == 0 else None,
                           "load_raw": load if r2 == 0 else None})
        return {"measured_tip": [float(v) for v in measured], "joint_state": joints,
                "follow_err_mm": float(np.linalg.norm(gap)), "follow_err_z_mm": float(gap[2])}

    def _goal(self, sid, raw):
        self.bus.ph.write2ByteTxRx(self.bus.port, sid, GOAL_POSITION, raw)

    def _ids(self):
        return [*JOINTS] + ([PROBE_ID] if self.probes else [])

    def arm(self, on):
        if not on:
            self.armed = False
            return
        self.sync()
        for sid in self._ids():
            self.bus.ph.write1ByteTxRx(self.bus.port, sid, ACCELERATION, self.accel)
            self.bus.ph.write2ByteTxRx(self.bus.port, sid, GOAL_SPEED, self.speed)
            self._goal(sid, self.bus.read_pos(sid))  # hold where it is before torque comes on
            self.bus.torque(sid, True)
        self._speed_now = self.speed
        self.armed = True

    def release(self):
        """Torque OFF on every joint so the arm can be moved by hand. It will sag or fall unless it is supported."""
        self.armed = False
        for sid in self._ids():
            self.bus.torque(sid, False)
        self.sim.msg = "torque OFF: the arm is limp. Support it. Press G / POST /api/arm to hold again."

    def stop(self):
        for sid in self._ids():
            self._goal(sid, self.bus.read_pos(sid))
        self.sim.msg = "stopped: holding current position"

    def reached(self, tol_deg=2.0):
        """True once every arm joint is within tol_deg of its goal (the sim pose)."""
        for sid in JOINTS:
            goal = self.rad_to_raw(sid, self.sim.q[CHAIN_INDEX[sid]])
            if abs(delta(self.bus.read_pos(sid), goal)) * self._cal(sid)["deg_per_tick"] > tol_deg:
                return False
        return True

    def push(self):
        # Slow the servos down near the board so one command can't move fast there.
        far = self.sim.guard_height_mm() >= FAR_MM or self.sim.retreating
        speed = self.speed if far else min(self.speed, self.near_speed)
        if speed != self._speed_now:
            for sid in self._ids():
                self.bus.ph.write2ByteTxRx(self.bus.port, sid, GOAL_SPEED, speed)
            self._speed_now = speed
        for sid in JOINTS:
            self._goal(sid, self.rad_to_raw(sid, self.sim.q[CHAIN_INDEX[sid]]))
        if self.probes:
            self._goal(PROBE_ID, self.probes[f"probe_{self.sim.probe_idx + 1}"])


class Executor:
    """Runs a goto plan in the background, one waypoint at a time. Every waypoint is claimed
    against the other arms again first, so if one has moved into the way the arm stops."""

    WAYPOINT_TIMEOUT_S = 5.0

    def __init__(self):
        self.active, self.step, self.steps, self.cancel = False, 0, 0, False

    def start(self, steps, phi):
        self.active, self.step, self.steps, self.cancel = True, 0, len(steps), False
        threading.Thread(target=self._run, args=(steps, phi), daemon=True).start()

    def _run(self, steps, phi):
        try:
            for i, q in enumerate(steps):
                with LOCK:
                    if self.cancel:
                        SIM.msg = f"goto stopped at waypoint {i}/{len(steps)}"
                        return
                    if HW and not HW.armed:
                        SIM.msg = f"goto stopped at waypoint {i}/{len(steps)}: disarmed"
                        return
                    if not SIM.claim(q):
                        SIM.msg = f"goto stopped at waypoint {i}/{len(steps)}: {SIM.msg}"
                        return
                    SIM.q, self.step = q, i + 1
                    if HW:
                        HW.push()
                deadline = time.monotonic() + self.WAYPOINT_TIMEOUT_S
                while HW and not self.cancel:
                    with LOCK:
                        if HW.reached():
                            break
                    if time.monotonic() > deadline:
                        with LOCK:
                            HW.stop()
                            SIM.msg = f"goto stopped: waypoint {i + 1} not reached in time"
                        return
                    time.sleep(0.03)
                if not HW:
                    time.sleep(SIM_STEP_S)
            with LOCK:
                SIM.phi_target, SIM.msg = phi, "ok"
        except ConnectionError as e:
            SIM.msg = f"goto stopped: arm write failed: {e}"
        finally:
            self.active = False


HW = None
WORKCELL = None
SENSORS = None  # sensor_head.SensorHead with --sensors
FEED_JPG = Path(tempfile.gettempdir()) / "troubleshooter_live_feed.jpg"  # written by live_pose.py
FEED_STALE_S = 3.0
EXEC = Executor()


class TouchJob:
    """Runs touch_pad.py for the viewer's pin picker: one job at a time, its output kept for the viewer.
    Stop creates the STOP file touch_pad.py watches, so it aborts and retreats the same way as by hand."""

    def __init__(self):
        self.proc, self.target, self.mode, self.lines, self.msg = None, None, None, [], ""

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, target, align_only):
        pads = json.loads((ROOT / "pcb_geometry.json").read_text())["pads"]
        if self.running():
            self.msg = f"busy: still running {self.target}"
        elif not HW or not HW.armed:
            self.msg = "disarmed: press G to arm the real arm first"
        elif not any(f"{p['ref']}.{p['pad']}" == target or p["net"] == target for p in pads):
            self.msg = f"no pad or net called {target!r} on the board"
        else:
            (ROOT / "STOP").unlink(missing_ok=True)
            cmd = [sys.executable, "-u", str(ROOT / "touch_pad.py"), "--target", target, "--tries", "12"]
            self.proc = subprocess.Popen(cmd + (["--align-only"] if align_only else []), cwd=ROOT, text=True,
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
            self.target, self.mode, self.lines = target, "align only" if align_only else "touch", []
            self.msg = f"started: {self.mode} {target}"
            threading.Thread(target=self._read, args=(self.proc,), daemon=True).start()

    def _read(self, proc):
        for line in proc.stdout:
            if "warn" not in line.lower():      # ikpy's fixed-link warnings
                self.lines = (self.lines + [line.rstrip()])[-200:]

    def stop(self):
        if self.running():
            (ROOT / "STOP").touch()
            self.msg = f"stopping {self.target}: it aborts and retreats"

    def status(self):
        return {"running": self.running(), "target": self.target, "mode": self.mode, "msg": self.msg,
                "returncode": None if self.proc is None else self.proc.poll(), "lines": self.lines[-60:]}


TOUCH = TouchJob()
LOCK = threading.Lock()
SIM = Sim()
ALLOWED_DIRS = ("viewer", "urdf")
MOTION = ("/api/move", "/api/tilt", "/api/probe", "/api/reset", "/api/cmd")


def snapshot():
    s = SIM.state()
    s["hw"] = {"mode": "hardware" if HW else "sim", "armed": bool(HW and HW.armed)}
    return s


def goto(body):
    """Plan to body["tip"] ([x, y, z] m, arm frame, or world frame with "frame": "world") and
    start moving. Returns at once; state["goto"] shows progress."""
    tip = np.array(body.get("tip", []), dtype=float)
    if tip.shape != (3,):
        SIM.msg = "goto needs tip: [x, y, z] in meters"
        return
    if body.get("frame", "arm") == "world":
        if WORKCELL is None:
            SIM.msg = "goto: no workcell (start with --arm), so there is no world frame"
            return
        tip = (np.linalg.inv(WORKCELL.models[WORKCELL.me].base) @ np.r_[tip, 1.0])[:3]
    phi = body.get("phi")
    phi = SIM.phi_target if phi is None else float(phi)
    steps = SIM.plan_to(tip, phi)
    if steps is not None:
        EXEC.start(steps, phi)
        SIM.msg = f"moving: {len(steps)} waypoints"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(200, json.dumps(obj).encode(), "application/json")

    def _recover(self):
        """A serial error mid-request: reopen the port, say so, and answer with the current state."""
        with LOCK:
            back = HW.reconnect() if HW else False
            SIM.msg = ("serial port dropped and reopened: press G to arm again" if back
                       else f"serial port dropped and {HW.port_name if HW else 'it'} could not be reopened")
            try:
                self._json(snapshot())
            except OSError:
                pass

    def do_GET(self):
        try:
            self._get()
        except serial.SerialException:
            self._recover()

    def do_POST(self):
        try:
            self._post()
        except serial.SerialException:
            self._recover()

    def _get(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/api/workcell":
            return self._json(WORKCELL.snapshot() if WORKCELL else {"me": None, "arms": {}})
        if path == "/api/sensors":  # read-only; the sensor head has its own lock
            if SENSORS is None:
                return self._json({"enabled": False, "error": "no sensor head: start with --sensors"})
            with LOCK:
                probe = SIM.tof()
                probe["tool"] = TOOL_HEADS[SIM.probe_idx]
            return self._json({"enabled": True, **SENSORS.snapshot(), "probe": probe})
        if path == "/api/feed.jpg":  # live_pose.py's annotated camera frame, while a touch run is going
            try:
                age = time.time() - FEED_JPG.stat().st_mtime
                data = FEED_JPG.read_bytes()
            except OSError:
                return self._send(404, b"no camera feed", "text/plain")
            if age > FEED_STALE_S:
                return self._send(404, b"camera feed stale", "text/plain")
            return self._send(200, data, "image/jpeg")
        if path == "/api/pads":  # the board's pads from the KiCad file, for the viewer's pin picker
            return self._json(json.loads((ROOT / "pcb_geometry.json").read_text())["pads"])
        if path == "/api/touch":
            return self._json(TOUCH.status())
        if path == "/api/state":
            with LOCK:
                if HW and not HW.armed:  # read-only mirror: the viewer follows the real arm
                    try:
                        HW.sync()
                    except ConnectionError as e:
                        SIM.msg = f"arm read failed: {e}"
                snap = snapshot()
                if HW and "measure" in parse_qs(parsed.query):  # ?measure=1: add the encoders' view (read-only)
                    try:
                        snap.update(HW.measure())
                    except ConnectionError as e:
                        snap["measure_error"] = str(e)
                return self._json(snap)
        if path == "/":
            path = "/viewer/index.html"
        target = (ROOT / path.lstrip("/")).resolve()
        ok = target.is_file() and any(ROOT / d in target.parents for d in ALLOWED_DIRS)
        if not ok:
            return self._send(404, b"not found", "text/plain")
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") and "charset" not in ctype:
            ctype += "; charset=utf-8"
        self._send(200, target.read_bytes(), ctype)

    def _post(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        path = urlparse(self.path).path
        if path == "/api/touch":  # the job moves the arm through this server's own API, so no LOCK here
            TOUCH.start(str(body.get("target", "")).strip(), bool(body.get("align_only")))
            return self._json(TOUCH.status())
        if path == "/api/touch/stop":
            TOUCH.stop()
            return self._json(TOUCH.status())
        cmd = body.get("cmd") if path == "/api/cmd" else None
        stopping = path == "/api/stop" or cmd == "stop"
        if stopping:
            EXEC.cancel = True  # before taking LOCK, so a running goto stops at its next waypoint
        with LOCK:
            if path in MOTION and HW and not HW.armed:
                SIM.msg = "disarmed: press G to arm the real arm"
            elif path in MOTION and EXEC.active and not stopping and cmd != "state":
                SIM.msg = "busy: goto in progress (stop cancels it)"
            elif path == "/api/move":
                contact = body.get("contact_mm")
                SIM.move(body.get("d", [0, 0, 0]), contact_mm=None if contact is None else float(contact))
            elif path == "/api/tilt":
                SIM.tilt(int(body.get("step", 0)))
            elif path == "/api/probe":
                SIM.probe(int(body.get("step", 0)))
                if HW and not HW.probes:
                    SIM.msg = "no probe positions taught: ID 5 not moved (calibrate.py --probes-only 3)"
            elif path == "/api/reset":
                SIM.reset()
            elif path == "/api/cmd":
                if cmd == "jog":
                    hint = body.get("height_mm")
                    SIM.jog_camera(float(body.get("right_mm", 0)), float(body.get("up_mm", 0)),
                                   float(body.get("in_mm", 0)),
                                   None if hint is None else float(hint))
                elif cmd == "touch_off":
                    tip_z, tof_mm = body.get("tip_z"), body.get("tof_mm")
                    SIM.touch_off(None if tip_z is None else float(tip_z), None if tof_mm is None else float(tof_mm),
                                  bool(body.get("zero_tof", True)))
                elif cmd == "clear_touch_off":
                    SIM.clear_touch_off()
                elif cmd == "point_down":
                    SIM.point_down()
                elif cmd == "tilt":
                    SIM.tilt(float(body.get("deg", 0)) / TILT_STEP_DEG)
                elif cmd == "tool":
                    if HW and not HW.probes:
                        SIM.msg = "no probe positions taught: ID 5 not moved (calibrate.py --probes-only 3)"
                    else:
                        SIM.select_tool(body.get("head"))
                elif cmd == "goto":
                    goto(body)
                elif cmd == "stop":
                    if HW:
                        HW.stop()
                    else:
                        SIM.msg = "stopped"
                elif cmd != "state":
                    SIM.msg = f"unknown command: {cmd!r} (jog, goto, tilt, point_down, touch_off, clear_touch_off, tool, stop, state)"
            elif path == "/api/arm" and HW:
                try:
                    HW.arm(bool(body.get("on")))
                    SIM.msg = "ARMED: torque on, holding position" if HW.armed else "disarmed"
                except ConnectionError as e:
                    SIM.msg = f"could not arm: {e}"
            elif path == "/api/stop" and HW:
                HW.stop()
            elif path == "/api/release" and HW:
                HW.release()
            elif path not in ("/api/arm", "/api/stop", "/api/release"):
                return self._send(404, b"not found", "text/plain")
            if HW and HW.armed and path in MOTION and SIM.msg in ("ok", "ready"):
                try:
                    HW.push()
                except ConnectionError as e:
                    SIM.msg = f"arm write failed: {e}"
            self._json(snapshot())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hardware", metavar="PORT", help="drive the real arm on this COM port (e.g. COM10)")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--speed", type=int, default=300, help="servo speed limit, steps/s (4096 = one turn)")
    ap.add_argument("--accel", type=int, default=20, help="servo acceleration, 0-254")
    ap.add_argument("--port", type=int, default=PORT, help="HTTP port; give each arm its own")
    ap.add_argument("--arm", help="this arm's name in the workcell file; turns on collision checks")
    ap.add_argument("--workcell", default=str(collision.WORKCELL), help="shared bench layout")
    ap.add_argument("--sensors", metavar="PORT",
                    help="ESP32 sensor head (ToF + thermal) on this COM port, or 'auto'")
    ap.add_argument("--tof-offset-mm", type=float,
                    help="ToF range with the tip touching the board, for every head "
                         "(otherwise touch_off sets it for the current head)")
    ap.add_argument("--floor-offset-mm", type=float, default=0.0,
                    help="lower the table and board floors by this much. For finding the real needle's contact height when the "
                         "model's table height is stale or the needle sits above the model's tip. Contact is then found by "
                         "following error / PnP, not by the floor: use only with slow steps and the arm watched.")
    args = ap.parse_args()
    if args.sensors:
        SENSORS = sensor_head.SensorHead(None if args.sensors == "auto" else args.sensors)
        if args.tof_offset_mm is not None:
            SIM.tof_zero_mm = dict.fromkeys(TOOL_HEADS, args.tof_offset_mm)
        print(f"Sensor head on {SENSORS.port}: ToF and thermal at /api/sensors.")
    if args.floor_offset_mm:
        TABLE_Z -= args.floor_offset_mm / 1000.0
        SIM.board_z -= args.floor_offset_mm / 1000.0
        SIM.board_z_default -= args.floor_offset_mm / 1000.0
        print(f"FLOORS LOWERED by {args.floor_offset_mm:.0f} mm (table z {TABLE_Z * 1000:.1f} mm, board z {SIM.board_z * 1000:.1f} mm)")
    if args.arm:
        WORKCELL = collision.Workcell(args.arm, args.workcell, PROBE_ANGLES)
        if not args.hardware:
            WORKCELL.settle_s = 0.0  # a simulated arm is always exactly at its goal
        print(f"Workcell: arm {args.arm}, checking against {', '.join(WORKCELL.others) or 'nobody'}.")
    if args.hardware:
        HW = Hardware(SIM, args.hardware, args.baud, args.speed, args.accel)
        HW.sync()
        print(f"HARDWARE mode on {args.hardware}. Starts DISARMED: read-only until you press G in the viewer.")
        if not HW.probes:
            print("No probe positions in calibration.json: ID 5 will not move.")
    else:
        SIM.announce()
        print("Simulation only, no serial port.")
    print(f"Open http://localhost:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
