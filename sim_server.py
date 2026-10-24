"""Simulation server: IK + a browser viewer. Never opens a serial port.

Serves viewer/ and urdf/ on localhost and exposes a tiny JSON API. The browser sends
tip-space nudges (meters) and tool-tilt steps; this module solves IK for the four arm
joints, clamps them to the calibration limits, rejects moves it can't make, and returns
the new joint angles.

The solver holds two things at once: the probe-tip position and the probe's tilt from
straight down (phi). Moving the tip keeps phi fixed, and changing phi keeps the tip fixed.

  python sim_server.py          then open http://localhost:8765
"""
import argparse
import json
import math
import mimetypes
import threading
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from ikpy.chain import Chain
import serial
from scipy.optimize import least_squares

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

    def solve(self, target, phi_target, q_start):
        def residual(x):
            q = q_start.copy()
            q[ARM_IDX] = x
            tip, _, phi = self.pose(q)
            return np.r_[(tip - target) * 1000.0, 0.5 * (phi - phi_target)]

        x0 = np.clip(q_start[ARM_IDX], self.lo, self.hi)
        r = least_squares(residual, x0, bounds=(self.lo, self.hi))
        q = q_start.copy()
        q[ARM_IDX] = r.x
        tip, _, phi = self.pose(q)
        return q, float(np.linalg.norm((tip - target) * 1000.0)), abs(phi - phi_target)

    def reset(self):
        q0 = np.zeros(13)
        q0[TURRET_IDX] = math.radians(PROBE_ANGLES[PROBE_START])
        q, pos_err, phi_err = self.solve(np.array(HOME_TIP), 0.0, q0)
        self.q = q
        self.phi_target = 0.0
        self.probe_idx = PROBE_START
        self.note = ""
        self.retreating = False
        self.board_z = BOARD_Z
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

    def _try(self, target, phi_target, label):
        if target[2] < self.board_z:
            self.msg = f"blocked: z {target[2]*1000:.0f} mm is below the board plane"
            return False
        q, pos_err, phi_err = self.solve(target, phi_target, self.q)
        if pos_err > POS_TOL_MM or phi_err > PHI_TOL_DEG:
            self.msg = f"unreachable: {label} misses by {pos_err:.1f} mm / {phi_err:.1f} deg (joint limits)"
            return False
        swing = max(abs(math.degrees(q[i] - self.q[i])) for i in ARM_IDX)
        if swing > MAX_JOINT_STEP_DEG:
            self.msg = f"blocked: would swing a joint {swing:.0f} deg in one step"
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

    def touch_off(self):
        """Call when the probe tip is physically touching the board: ties the model to reality."""
        self.board_z = float(self.pose(self.q)[0][2])
        self.calibrated = True
        self.msg = f"board height set from the tip: z = {self.board_z * 1000:.1f} mm"

    def move(self, d, height_mm=None):
        """Move the tip by d (meters), shortened by the step governor and stopped at the floor.

        height_mm is the height above the board the camera sees. When given it overrides the
        URDF-based estimate, because the camera is trusted over the model.
        """
        tip, _, _ = self.pose(self.q)
        d = np.array(d, dtype=float)
        self.note = ""
        h = self.height_mm(tip) if height_mm is None else float(height_mm)
        # Moving straight away from the board can't hurt it, so retreating isn't slowed or shortened.
        self.retreating = bool(d[2] > 0 and d[2] >= 0.9 * np.linalg.norm(d))
        cap = CAP_FAR_MM if self.retreating else self.step_cap_mm(h)
        biggest = float(np.max(np.abs(d))) * 1000.0
        if biggest > cap:
            d *= cap / biggest
            self.note = f"shortened to {cap:.1f} mm ({h:.0f} mm above board)"
        room = (h - FLOOR_MARGIN * 1000.0) / 1000.0                # how far down we may still go
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
        self._try(tip + d, self.phi_target, "move")

    def tilt(self, step):
        tip, _, _ = self.pose(self.q)
        self._try(tip, self.phi_target + step * TILT_STEP_DEG, "tilt")

    def probe(self, step):
        new = self.probe_idx + step
        if not 0 <= new < len(PROBE_ANGLES):
            self.msg = f"blocked: only probes 1 to {len(PROBE_ANGLES)}"
            return
        # Only the turret (ID 5) turns. The arm does not move, so the tip jumps to the new probe.
        self.probe_idx = new
        self.q[TURRET_IDX] = math.radians(PROBE_ANGLES[new])
        self.msg = "ok"

    def state(self):
        tip, shaft, phi = self.pose(self.q)
        right, up, depth = self.camera_axes()
        return {
            "camera_axes": {"right": right.tolist(), "up": up.tolist(), "depth": depth.tolist()},
            "joints": {self.chain.links[i].name: float(self.q[i]) for i in CHAIN_INDEX.values()},
            "tip": [float(v) for v in tip],
            "shaft": [float(v) for v in shaft],
            "phi": phi,
            "board_z": self.board_z,
            "board_calibrated": self.calibrated,
            "height_mm": self.height_mm(tip),
            "cap_mm": self.step_cap_mm(self.height_mm(tip)),
            "note": self.note,
            "probe": self.probe_idx + 1,
            "probe_deg": PROBE_ANGLES[self.probe_idx],
            "probe_count": len(PROBE_ANGLES),
            "msg": self.msg,
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

    def stop(self):
        for sid in self._ids():
            self._goal(sid, self.bus.read_pos(sid))
        self.sim.msg = "stopped: holding current position"

    def push(self):
        # Slow the servos down near the board so one command can't move fast there.
        far = self.sim.height_mm() >= FAR_MM or self.sim.retreating
        speed = self.speed if far else min(self.speed, self.near_speed)
        if speed != self._speed_now:
            for sid in self._ids():
                self.bus.ph.write2ByteTxRx(self.bus.port, sid, GOAL_SPEED, speed)
            self._speed_now = speed
        for sid in JOINTS:
            self._goal(sid, self.rad_to_raw(sid, self.sim.q[CHAIN_INDEX[sid]]))
        if self.probes:
            self._goal(PROBE_ID, self.probes[f"probe_{self.sim.probe_idx + 1}"])


SIM = Sim()
HW = None
LOCK = threading.Lock()
ALLOWED_DIRS = ("viewer", "urdf")
MOTION = ("/api/move", "/api/tilt", "/api/probe", "/api/reset", "/api/cmd")


def snapshot():
    s = SIM.state()
    s["hw"] = {"mode": "hardware" if HW else "sim", "armed": bool(HW and HW.armed)}
    return s


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
        with LOCK:
            if path in MOTION and HW and not HW.armed:
                SIM.msg = "disarmed: press G to arm the real arm"
            elif path == "/api/move":
                SIM.move(body.get("d", [0, 0, 0]))
            elif path == "/api/tilt":
                SIM.tilt(int(body.get("step", 0)))
            elif path == "/api/probe":
                SIM.probe(int(body.get("step", 0)))
                if HW and not HW.probes:
                    SIM.msg = "no probe positions taught: ID 5 not moved (calibrate.py --probes-only 3)"
            elif path == "/api/reset":
                SIM.reset()
            elif path == "/api/cmd":
                cmd = body.get("cmd")
                if cmd == "jog":
                    hint = body.get("height_mm")
                    SIM.jog_camera(float(body.get("right_mm", 0)), float(body.get("up_mm", 0)),
                                   float(body.get("in_mm", 0)),
                                   None if hint is None else float(hint))
                elif cmd == "touch_off":
                    SIM.touch_off()
                elif cmd == "tilt":
                    SIM.tilt(float(body.get("deg", 0)) / TILT_STEP_DEG)
                elif cmd == "stop" and HW:
                    HW.stop()
                elif cmd != "state":
                    SIM.msg = f"unknown command: {cmd!r} (jog, tilt, stop, state)"
            elif path == "/api/arm" and HW:
                try:
                    HW.arm(bool(body.get("on")))
                    SIM.msg = "ARMED: torque on, holding position" if HW.armed else "disarmed"
                except ConnectionError as e:
                    SIM.msg = f"could not arm: {e}"
            elif path == "/api/stop" and HW:
                HW.stop()
            elif path not in ("/api/arm", "/api/stop"):
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
    args = ap.parse_args()
    if args.hardware:
        HW = Hardware(SIM, args.hardware, args.baud, args.speed, args.accel)
        HW.sync()
        print(f"HARDWARE mode on {args.hardware}. Starts DISARMED: read-only until you press G in the viewer.")
        if not HW.probes:
            print("No probe positions in calibration.json: ID 5 will not move.")
    else:
        print("Simulation only, no serial port.")
    print(f"Open http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
