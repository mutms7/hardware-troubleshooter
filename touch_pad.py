"""Touch the needle to a pad: one big move, then fine-tuning, all from PnP, with the live window (live_pose.py).

Needle position: needle_board = R^T (needle_cam - t) from the continuous feature pose + the hand-measured needle offset
(tip_offset_needle.json): x, y in board mm; -z = height above the board plane. The server's floor protects its model tip, not the
needle, so the floor here is PnP's.

  0. go to the start pose in --start (touch_start.json: the wrist camera sees the whole board), unless --from-here
  1. centre over the pad (burst + corrections)
  2. one descent burst to --stage-h above the board, re-centre
  3. final, straight down (z only, like the viewer's R/F keys), re-centring when more than --recentre mm off, until the joint
     encoders put the tip --press mm below the board height from calibrate_board.py (board_cal.json). The ToF at the start
     pose is a check only: more than --max-shift mm from calibration aborts (the board moved: re-calibrate). Steps are the distance left, clamped to 0.5-2 mm (smaller ones are lost in the servos' dead zone).
     If the servos fall --blocked mm behind within 2 mm of the target, the board stopped the needle: that is a touch too.
     The ToF is not used near the board: it sees beside the needle, and at J3.2 it jumped +-5 mm with no trend.
  4. hold --hold-s, save a frame, back off, retreat to ~25 mm.
The servos dither by about +-1 mm on small moves (stiction), so centring accepts --tol (0.9 mm) after a few tries instead of aborting.
Aborts and retreats on: board lost, refused jog, a crash, the window closed (q), or a file called STOP next to this script.
"""
import argparse
import json
import math
import re
import threading
import time
from pathlib import Path

import numpy as np

from arm_client import Arm
from live_pose import LivePose

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--target", default="J3.4")
ap.add_argument("--tol", type=float, default=0.9, help="accept centring within this many mm (after at most --tries corrections)")
ap.add_argument("--tries", type=int, default=6)
ap.add_argument("--recentre", type=float, default=1.0, help="in the final descent, re-centre when more than this far off (mm)")
ap.add_argument("--stage-h", type=float, default=6.0, help="the descent burst stops this high above the board (mm)")
ap.add_argument("--hold-s", type=float, default=3.0)
ap.add_argument("--start", default=str(HERE / "touch_start.json"), help="start pose: tip_m and phi_deg, as the arm's state gives them")
ap.add_argument("--align-only", action="store_true", help="centre over the pad, report, and stop there: no descent")
ap.add_argument("--from-here", action="store_true", help="skip the start pose and start from wherever the arm is")
BOARD_CAL = HERE / "board_cal.json"            # from calibrate_board.py
ap.add_argument("--contact-depth", type=float, default=6.0,
                help="the command may go this far below the board in the final descent; the servos lag it near the board (mm)")
ap.add_argument("--tof-touch", type=float, default=None,
                help="no board calibration: touching when the ToF median is at or under this (mm). A stopgap for a board that "
                     "hasn't been calibrated; the ToF sees beside the needle, so check the result by eye")
ap.add_argument("--press", type=float, default=0.5, help="go this far below the calibrated board height (mm)")
ap.add_argument("--max-shift", type=float, default=5.0, help="abort if the start-pose ToF says the board moved more than this (mm)")
ap.add_argument("--blocked", type=float, default=5.0, help="servos this far behind within 2 mm of the target = touching (mm)")
ap.add_argument("--max-cmd", type=float, default=25.0, help="give up after this much final-descent command (mm)")
ap.add_argument("--max-follow", type=float, default=8.0, help="abort if the servos lag the command by more than this (mm)")
args = ap.parse_args()

lp = LivePose(target=args.target)
arm = Arm()
goal = lp.goal
# Jog -> needle-on-board response, a full 2x2 matrix: board mm moved per mm of (right, up) jog. The two jog axes behave
# differently (right swings the base, up bends the shoulder and elbow), so one rotation + scale can't describe it: on
# 2026-10-25 two moves 90 deg apart came out 53 deg apart. Measured by one move along each axis, then refined.
J = -np.eye(2)
probes_done = 0
PROBE_MM = 10.0          # the two measuring moves, one along each jog axis
LEARN_MAX_MM = 20.0      # only moves up to this long refine J: long moves curve as the arm swings about its base
stop_file = HERE / "STOP"
if stop_file.exists():
    stop_file.unlink()
t_start = time.time()


class Abort(Exception):
    pass


def say(msg):
    print(f"[{time.time() - t_start:5.0f}s] {msg}", flush=True)
    lp.extra = [msg] + lp.extra[:2]


def rot(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def look(n=2):
    """Needle position (board mm) and fit, from frames captured after now."""
    m = lp.measure(after=time.time(), n=n, timeout=6) or lp.measure(after=time.time(), n=n, timeout=6)
    return None if m is None else (m["needle_board"], m["fit"])


def check_stop():
    if stop_file.exists() or lp.stop.is_set():
        raise Abort("STOP file found or the window was closed")


def jog(r=0.0, u=0.0, i=0.0, wait=0.3):
    """Returns the fraction of the jog the server actually sent (its governor shortens jogs near the board)."""
    s = arm.jog(right=float(r), up=float(u), in_=float(i))
    if s.get("msg") != "ok":
        raise Abort(f"jog ({r:+.1f}, {u:+.1f}, {i:+.1f}) refused: {s.get('msg')}")
    note = s.get("note") or ""
    time.sleep(wait)
    m = re.match(r"shortened to ([\d.]+) mm", note)
    if m:
        return min(1.0, float(m.group(1)) / max(abs(r), abs(u), abs(i), 1e-9))
    if note:
        say(f"   server: {note}")
    return 0.0 if "floor" in note else 1.0


def lower(mm, wait=0.8, contact_mm=None):
    """Tip straight down (negative: up) by mm, tilt held: the viewer's R/F keys. Jogging "in" goes along the tilted probe,
    which also moves the needle sideways, and near the board it stalled where straight down did not."""
    s = arm.move_z(-float(mm), contact_mm=contact_mm)
    if s.get("msg") != "ok":
        raise Abort(f"move down {mm:+.1f} mm refused: {s.get('msg')}")
    if s.get("note"):
        say(f"   server: {s['note']}")
    time.sleep(wait)


def retreat(mm=28.0):
    say(f"retreating {mm:.0f} mm")
    done = 0.0
    while done < mm - 1e-6:
        step = min(10.0, mm - done)
        arm.jog(in_=-step)
        done += step
        time.sleep(1.2)


def burst(u):
    """Jog (right, up) = u mm in <=9 mm pieces, repeating whatever the server's governor shortened, then let it settle."""
    left = np.array(u, float)
    for _ in range(60):
        n = float(np.linalg.norm(left))
        if n < 0.05:
            break
        piece = left * min(1.0, 9.0 / n)
        left -= piece * jog(piece[0], piece[1])
    time.sleep(1.2)


def level_burst(err):
    """The level move that should shift the needle by `err` board mm."""
    u = np.linalg.solve(J, err)
    burst(u)
    return u


def learn(u, before, after):
    """Refine J from a move u (jog mm) that moved the needle from `before` to `after` (board mm)."""
    global J
    d = after[:2] - before[:2]
    nu = float(np.linalg.norm(u))
    if nu < 3.0 or nu > LEARN_MAX_MM:
        return
    pred = J @ u
    if float(np.linalg.norm(d)) < 0.4 * float(np.linalg.norm(pred)):
        raise Abort(f"the needle moved {np.linalg.norm(d):.1f} mm where {np.linalg.norm(pred):.1f} mm was expected: "
                    f"the arm isn't following (a joint at its limit?)")
    J = J + np.outer(d - pred, u) / (u @ u)                      # Broyden update
    say(f"   jog response: right -> ({J[0, 0]:+.2f}, {J[1, 0]:+.2f}), up -> ({J[0, 1]:+.2f}, {J[1, 1]:+.2f}) board mm per mm")


def probe_axes():
    """One PROBE_MM move along each jog axis measures J's two columns exactly."""
    global J, probes_done
    m0 = look()
    if m0 is None:
        raise Abort("lost the board before the measuring moves")
    cols = []
    for k, name in ((0, "right"), (1, "up")):
        u = np.zeros(2)
        u[k] = PROBE_MM
        burst(u)
        m1 = look()
        if m1 is None:
            raise Abort("lost the board during the measuring moves")
        d = m1[0][:2] - m0[0][:2]
        say(f"   measuring move {name} {PROBE_MM:.0f} mm: needle moved ({d[0]:+.1f}, {d[1]:+.1f}) mm")
        if np.linalg.norm(d) < 0.4 * PROBE_MM:
            raise Abort(f"a {PROBE_MM:.0f} mm jog {name} moved the needle {np.linalg.norm(d):.1f} mm: the arm isn't following")
        cols.append(d / PROBE_MM)
        m0 = m1
    J = np.column_stack(cols)
    probes_done = 2
    if abs(np.linalg.det(J)) < 0.2:
        raise Abort("the two jog axes move the needle along nearly the same line: can't steer from here")
    say(f"   jog response: right -> ({J[0, 0]:+.2f}, {J[1, 0]:+.2f}), up -> ({J[0, 1]:+.2f}, {J[1, 1]:+.2f}) board mm per mm")


def center(tol, tries):
    m, best = None, 1e9
    for k in range(tries):
        check_stop()
        m = look()
        if m is None:
            raise Abort("lost the board while centring")
        tb = m[0]
        err = goal - tb[:2]
        e = float(np.linalg.norm(err))
        best = min(best, e)
        say(f"   needle ({tb[0]:.2f}, {tb[1]:.2f}) at {-tb[2]:.1f} mm up, {e:.2f} mm from the pad")
        if e < tol:
            return m
        if probes_done < 2:
            probe_axes()
            continue
        u = level_burst(err)
        m2 = look()
        if m2 is not None:
            learn(u, tb, m2[0])
    say(f"   accepting {np.linalg.norm(goal - m[0][:2]):.2f} mm after {tries} tries (best {best:.2f})")
    return m


def tof_mm(n=25):
    """Median of n fresh, valid ToF readings (mm), or None if fewer than half of them are valid."""
    rs = []
    for _ in range(n):
        t = arm.sensors().get("tof")
        if t and t["valid"] and t["age_s"] <= 0.5:
            rs.append(t["range_mm"])
        time.sleep(0.07)                        # the sensor gives a new range about every 33 ms
    return float(np.median(rs)) if len(rs) * 2 > n else None


def tip_z():
    """The joint encoders' tip height (mm): where the arm really is, not where it was told to be."""
    return arm.state(measure=True)["measured_tip"][2] * 1000.0


def board_height():
    """Board height (mm, arm frame) from board_cal.json, shifted by the ToF at the start pose."""
    if not BOARD_CAL.exists():
        raise Abort(f"no {BOARD_CAL.name}: run calibrate_board.py start, touch with R/F, then calibrate_board.py contact")
    cal = json.loads(BOARD_CAL.read_text())
    if "board_z_m" not in cal or "start" not in cal:
        raise Abort(f"{BOARD_CAL.name} is incomplete: run calibrate_board.py start, then contact")
    z = cal["board_z_m"] * 1000.0
    if args.from_here:
        say(f"board at z {z:.1f} mm (calibrated; no ToF check, since --from-here skips the start pose)")
        arm.touch_off(tip_z=z / 1000.0, zero_tof=False)     # also after a server restart, which forgets it
        return z
    r = tof_mm()
    # A check only: the start pose isn't reached exactly (the servos lag and sag), so the ToF lands on a slightly different
    # patch each time and reads a few mm differently (2026-10-25: 102 vs 105 with the board taped down). Shifting the board
    # height by that would be worse than not shifting it. A warning, not an abort: sliding the board on the same table
    # moves the ToF spot onto a different part (a chip, bare board, the table) without changing the board's height.
    if r is None:
        say("WARNING: no valid ToF reading at the start pose; using the calibrated board height")
    else:
        shift = r - cal["start"]["tof_mm"]
        if abs(shift) > args.max_shift:
            say(f"WARNING: the start-pose ToF reads {r:.1f} mm, {shift:+.1f} mm from calibration. Fine if the board only "
                f"slid on the same surface; re-calibrate if its height changed")
    say(f"board at z {z:.1f} mm (calibrated)")
    arm.touch_off(tip_z=z / 1000.0, zero_tof=False)     # the server's floor and model use it too
    return z


def follow(base=0.0):
    return arm.state(measure=True).get("follow_err_z_mm", 0.0) - base


def go_to_start():
    start = json.loads(Path(args.start).read_text())
    say(f"0. going to the start pose ({args.start})")
    for _ in range(10):                         # from near the board, lift first: goto would want its full travel height,
        if arm.state().get("height_mm", 99.0) >= 18.0:      # which the wrist can't reach at the start pose
            break
        arm.move_z(5.0)
        time.sleep(0.8)
    for _ in range(3):                          # the servos lag, so a goto often ends 'not reached in time': re-plan from there
        s = arm.goto(start["tip_m"], phi=start["phi_deg"], timeout=90.0)
        if s.get("msg") == "ok" or "not reached in time" not in s.get("msg", ""):
            break
    if s.get("msg") != "ok":
        if "not reached in time" not in s.get("msg", ""):
            raise Abort(f"could not reach the start pose: {s.get('msg')}")
        say(f"   {s['msg']}: carrying on, the arm is still catching up")
    time.sleep(1.5)


def run():
    if not args.from_here:
        go_to_start()
    m = look()
    if m is None:
        raise Abort("board not found at the start")
    tb, fit = m
    say(f"start: needle at board ({tb[0]:.1f}, {tb[1]:.1f}) mm, {-tb[2]:.1f} mm above the board, fit {fit:.2f} px; pad "
        f"{lp.pad['ref']}.{lp.pad['pad']} [{lp.pad['net']}] at ({goal[0]:.1f}, {goal[1]:.1f}), {np.linalg.norm(goal - tb[:2]):.1f} mm away")
    base = float(np.mean([follow() for _ in range(5)]))
    z_board = None if args.align_only or args.tof_touch is not None else board_height()

    say("1. centre over the pad")
    m = center(args.tol, args.tries)
    if args.align_only:
        m = look(n=3) or m
        off = goal - m[0][:2]
        say(f"ALIGNED (--align-only): needle {np.linalg.norm(off):.2f} mm from the pad (dx {off[0]:+.2f}, dy {off[1]:+.2f}), "
            f"{-m[0][2]:.1f} mm above the board; holding here")
        return

    h = -m[0][2]
    down = h - args.stage_h
    if down > 1.0:
        say(f"2. descent burst: {h:.1f} mm -> {args.stage_h:.0f} mm above the board")
        n = max(1, math.ceil(down / 9.0))
        for _ in range(n):
            jog(i=down / n)
        time.sleep(1.5)
        m2 = look()
        if m2 is None:
            raise Abort("lost the board after the descent burst")
        got = h + m2[0][2]
        say(f"   needle now {-m2[0][2]:.1f} mm up (asked for {down:.1f} mm of descent, got {got:.1f})")
        if got < 0.3 * down:
            raise Abort("the needle barely came down")
        m = center(args.tol, args.tries)

    if args.tof_touch is not None:
        result = tof_descent()
    else:
        result = encoder_descent(z_board, base)

    m = look(n=3) or m
    tb, fit = m
    off = goal - tb[:2]
    say(f"{result.upper()}: needle at board ({tb[0]:.2f}, {tb[1]:.2f}) mm, {-tb[2]:.2f} mm above the plane; pad is "
        f"{np.linalg.norm(off):.2f} mm away (dx {off[0]:+.2f}, dy {off[1]:+.2f})")
    json.dump({"target": f"{lp.pad['ref']}.{lp.pad['pad']}", "net": lp.pad["net"], "result": result,
               "tip_z_mm": tip_z(), "board_z_mm": z_board, "needle_board_mm": tb.tolist(), "off_pad_mm": off.tolist(),
               "off_pad_norm_mm": float(np.linalg.norm(off))},
              open(HERE / "touch_pad_result.json", "w"), indent=1)
    say(f"holding the contact for {args.hold_s:.0f} s")
    time.sleep(args.hold_s)
    arm.jog(in_=-3.5)
    time.sleep(1.5)
    retreat(25.0)
    say("done: backed off and retreated")


def tof_descent():
    """Straight down until the ToF median reaches --tof-touch (no board calibration)."""
    base = float(np.mean([follow() for _ in range(5)]))
    say(f"3. final descent: straight down until the ToF median is at or under {args.tof_touch:.0f} mm")
    cmd = 0.0
    while cmd < args.max_cmd:
        check_stop()
        m = look()
        if m is None:
            raise Abort("lost the board in the final descent")
        tb, fit = m
        err = goal - tb[:2]
        r = tof_mm(9)
        gap = follow(base)
        say(f"   cmd {cmd:5.2f}  ToF {'--' if r is None else f'{r:5.1f}'} mm  PnP {-tb[2]:5.2f} mm up  "
            f"xy off {np.linalg.norm(err):.2f} mm  follow {gap:+.2f}")
        if r is not None and r <= args.tof_touch:
            return f"touching (ToF {r:.1f} mm <= {args.tof_touch:.0f} mm)"
        if gap > args.max_follow:
            raise Abort(f"the servos are {gap:.1f} mm behind the command: stopping before they catch up all at once")
        if np.linalg.norm(err) > args.recentre:
            level_burst(err)
        step = 2.0 if -tb[2] > 4.0 else 1.0
        lower(step)
        cmd += step
    raise Abort(f"went {args.max_cmd:.0f} mm of command without the ToF reaching {args.tof_touch:.0f} mm")


def encoder_descent(z_board, base):
    """Straight down until the encoders put the tip --press below the calibrated board height."""
    target = z_board - args.press
    say(f"3. final descent: straight down until the encoders put the tip at z {target:.1f} mm ({args.press:.1f} mm into the board)")
    cmd, result = 0.0, None
    while cmd < args.max_cmd:
        check_stop()
        m = look()
        if m is None:
            raise Abort("lost the board in the final descent")
        tb, fit = m
        err = goal - tb[:2]
        z = tip_z()
        left = z - target
        gap = follow(base)
        say(f"   cmd {cmd:5.2f}  tip z {z:6.1f} mm ({left:+.2f} to go)  PnP {-tb[2]:5.2f} mm up  xy off {np.linalg.norm(err):.2f} mm"
            f"  follow {gap:+.2f}")
        if left <= 0:
            result = f"touching (encoders: tip at z {z:.1f} mm, the board is at {z_board:.1f})"
            break
        if left < 2.0 and gap > args.blocked:
            result = f"touching (stopped {left:.1f} mm short of the target with the servos {gap:.1f} mm behind: the board)"
            break
        if gap > args.max_follow:
            raise Abort(f"the servos are {gap:.1f} mm behind the command: stopping before they catch up all at once")
        if np.linalg.norm(err) > args.recentre:
            level_burst(err)
        step = float(np.clip(left, 0.5, 2.0))
        lower(step, contact_mm=args.contact_depth)
        cmd += step
    if result is None:
        raise Abort(f"went {args.max_cmd:.0f} mm of command without reaching the board height")
    return result


def worker():
    try:
        run()
    except Abort as e:
        say(f"ABORT: {e}")
        try:
            retreat(30.0)
        except Exception as e2:
            say(f"retreat failed: {e2}")
    except Exception as e:                      # never leave the arm low after a crash
        say(f"CRASH: {type(e).__name__}: {e}")
        try:
            retreat(30.0)
        except Exception:
            pass


th = threading.Thread(target=worker, daemon=True)
th.start()
lp.gui(alive=th.is_alive, linger_s=15.0)
if th.is_alive():                               # window closed mid-run: let the worker abort and retreat
    th.join(timeout=60)
lp.close()
