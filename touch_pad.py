"""Touch the needle to a pad: one big move, then fine-tuning, all from PnP, with the live window (live_pose.py).

Needle position: needle_board = R^T (needle_cam - t) from the continuous feature pose + the hand-measured needle offset
(tip_offset_needle.json): x, y in board mm; -z = height above the board plane. The server's floor protects its model tip, not the
needle, so the floor here is PnP's.

  1. centre over the pad (burst + corrections)
  2. one descent burst to ~12 mm above the board, re-centre
  3. final: 0.5 mm steps (0.25 below 4 mm), re-centring when more than --recentre mm off, until the needle reaches --stop-h above the
     board plane, or contact (command kept going, needle stopped, servo following error up > 1.5 mm)
  4. hold --hold-s, save a frame, back off, retreat to ~25 mm.
The servos dither by about +-1 mm on small moves (stiction), so centring accepts --tol (1.2 mm) after a few tries instead of aborting.
Aborts and retreats on: board lost, refused jog, a crash, the window closed (q), or a file called STOP next to this script.
"""
import argparse
import math
import threading
import time
from pathlib import Path

import numpy as np

from arm_client import Arm
from live_pose import LivePose

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--target", default="J3.4")
ap.add_argument("--tol", type=float, default=1.2, help="accept centring within this many mm (after at most --tries corrections)")
ap.add_argument("--tries", type=int, default=6)
ap.add_argument("--recentre", type=float, default=1.0, help="in the final descent, re-centre when more than this far off (mm)")
ap.add_argument("--stage-h", type=float, default=12.0, help="the descent burst stops this high above the board (mm)")
ap.add_argument("--stop-h", type=float, default=0.8, help="stop when the needle is this high above the board plane (mm)")
ap.add_argument("--hold-s", type=float, default=3.0)
args = ap.parse_args()

lp = LivePose(target=args.target)
arm = Arm()
goal = lp.goal
mapping = {"s": 1.0, "theta": 0.0}
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
    s = arm.jog(right=float(r), up=float(u), in_=float(i))
    if s.get("msg") != "ok":
        raise Abort(f"jog ({r:+.1f}, {u:+.1f}, {i:+.1f}) refused: {s.get('msg')}")
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


def level_burst(err):
    """The level move that should shift the needle by `err` board mm, as a burst of <=9 mm jogs; then let it settle."""
    u = -(rot(-mapping["theta"]) @ err) / mapping["s"]          # right, up
    n = max(1, math.ceil(float(np.linalg.norm(u)) / 9.0))
    for _ in range(n):
        jog(u[0] / n, u[1] / n)
    time.sleep(1.2)
    return u


def learn(u, before, after):
    d = after[:2] - before[:2]
    if np.linalg.norm(u) < 3.0 or np.linalg.norm(d) < 1.0:
        return
    pred = rot(mapping["theta"]) @ (-u)
    dth = math.atan2(d[1], d[0]) - math.atan2(pred[1], pred[0])
    dth = (dth + math.pi) % (2 * math.pi) - math.pi
    mapping["theta"] += 0.7 * dth
    mapping["s"] = float(np.clip(mapping["s"] + 0.7 * (np.linalg.norm(d) / np.linalg.norm(u) - mapping["s"]), 0.25, 1.6))
    say(f"   jog mapping: scale {mapping['s']:.2f}, rotation {math.degrees(mapping['theta']):+.0f} deg")


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
        u = level_burst(err)
        m2 = look()
        if m2 is not None:
            learn(u, tb, m2[0])
    say(f"   accepting {np.linalg.norm(goal - m[0][:2]):.2f} mm after {tries} tries (best {best:.2f})")
    return m


def follow(base=0.0):
    return arm.state(measure=True).get("follow_err_z_mm", 0.0) - base


def run():
    m = look()
    if m is None:
        raise Abort("board not found at the start")
    tb, fit = m
    say(f"start: needle at board ({tb[0]:.1f}, {tb[1]:.1f}) mm, {-tb[2]:.1f} mm above the board, fit {fit:.2f} px; pad "
        f"{lp.pad['ref']}.{lp.pad['pad']} [{lp.pad['net']}] at ({goal[0]:.1f}, {goal[1]:.1f}), {np.linalg.norm(goal - tb[:2]):.1f} mm away")
    base = float(np.mean([follow() for _ in range(5)]))

    say("1. centre over the pad")
    m = center(args.tol, args.tries)

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

    say("3. final descent")
    cmd, hist, stalls, result = 0.0, [], 0, None
    hist.append((0.0, -m[0][2]))
    while cmd < 18.0:
        check_stop()
        m = look()
        if m is None:
            raise Abort("lost the board in the final descent")
        tb, fit = m
        h = -tb[2]
        err = goal - tb[:2]
        gap = follow(base)
        old = next((x for x in reversed(hist) if x[0] <= cmd - 1.5), None)
        stalled = old is not None and (old[1] - h) < 0.35 * (cmd - old[0]) and gap > 1.5
        say(f"   cmd {cmd:4.2f}  needle {h:5.2f} mm up  xy off {np.linalg.norm(err):.2f} mm  follow {gap:+.2f}  fit {fit:.2f}"
            + ("  [STALLED]" if stalled else ""))
        hist.append((cmd, h))
        stalls = stalls + 1 if stalled else 0
        if h <= args.stop_h:
            result = "reached the board plane"
            break
        if stalls >= 2:
            result = "contact (the command kept going, the needle stopped)"
            break
        if np.linalg.norm(err) > args.recentre:
            level_burst(err)
        step = 0.25 if h < 4.0 else 0.5
        jog(i=step, wait=0.8)
        cmd += step
    if result is None:
        raise Abort("went 18 mm of command without reaching the board or a contact")

    m = look(n=3) or m
    tb, fit = m
    off = goal - tb[:2]
    say(f"{result.upper()}: needle at board ({tb[0]:.2f}, {tb[1]:.2f}) mm, {-tb[2]:.2f} mm above the plane; pad is "
        f"{np.linalg.norm(off):.2f} mm away (dx {off[0]:+.2f}, dy {off[1]:+.2f})")
    import json
    json.dump({"target": f"{lp.pad['ref']}.{lp.pad['pad']}", "net": lp.pad["net"], "result": result,
               "needle_board_mm": tb.tolist(), "off_pad_mm": off.tolist(), "off_pad_norm_mm": float(np.linalg.norm(off))},
              open(HERE / "touch_pad_result.json", "w"), indent=1)
    say(f"holding the contact for {args.hold_s:.0f} s")
    time.sleep(args.hold_s)
    arm.jog(in_=-3.5)
    time.sleep(1.5)
    retreat(25.0)
    say("done: backed off and retreated")


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
