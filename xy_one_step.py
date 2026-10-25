"""xy alignment over a pad: measure the needle's offset with PnP, send ONE computed level move, measure again (repeat with --steps).

A live window shows the camera with the target pad, the point under the needle, and the error (see live_pose.py).
Level jogs only (right/up): the height is never commanded, and it refuses to move if the needle is lower than --min-h.
    python xy_one_step.py [--target J3.4] [--steps 1] [--scale 1.0] [--send]      (dry run without --send: measure and print only)
Jog mapping assumed: right -> board -x, up -> board -y. --scale divides a move by the fraction of it the arm really carries out.
"""
import argparse
import math
import threading
import time

import numpy as np

from arm_client import Arm
from live_pose import LivePose

ap = argparse.ArgumentParser()
ap.add_argument("--target", default="J3.4")
ap.add_argument("--steps", type=int, default=1, help="how many measure-and-move cycles (1 = the one-shot test)")
ap.add_argument("--scale", type=float, default=1.0)
ap.add_argument("--tol", type=float, default=0.5)
ap.add_argument("--min-h", type=float, default=20.0, help="refuse to move if the needle is lower than this above the board (mm)")
ap.add_argument("--send", action="store_true")
args = ap.parse_args()

lp = LivePose(target=args.target)
result = {}


def say(msg):
    print(msg, flush=True)
    lp.extra = [msg] + lp.extra[:2]


def worker():
    try:
        t0 = time.time()
        m = lp.measure(after=time.time() - 1.0, n=3, timeout=15)
        if m is None:
            say("board not found: is the board in view?")
            return
        arm = Arm() if args.send else None
        for step in range(args.steps):
            nb = m["needle_board"]
            err = lp.goal - nb[:2]
            say(f"[{time.time() - t0:4.1f}s] step {step + 1}: needle ({nb[0]:.2f}, {nb[1]:.2f}), {-nb[2]:.1f} mm up, "
                f"{np.linalg.norm(err):.2f} mm from {args.target}  [{m['src']}]")
            if step == 0:
                result["start_err"] = float(np.linalg.norm(err))
            if np.linalg.norm(err) < args.tol:
                say("within tolerance")
                break
            u = -err / args.scale                                   # (right, up)
            if -nb[2] < args.min_h:
                say(f"needle only {-nb[2]:.1f} mm above the board (< {args.min_h}): not moving")
                break
            if np.linalg.norm(u) > 60:
                say("that move is over 60 mm: not sending it")
                break
            if not args.send:
                say(f"dry run: would send right {u[0]:+.2f}, up {u[1]:+.2f} mm; nothing moved")
                break
            if lp.stop.is_set():
                say("window closed: not moving")
                break
            n = max(1, math.ceil(float(np.linalg.norm(u)) / 9.0))   # the server caps one jog at 10 mm: a burst of jogs
            for _ in range(n):
                s = arm.jog(right=float(u[0] / n), up=float(u[1] / n))
                if s.get("msg") != "ok":
                    say(f"jog refused: {s.get('msg')}")
                    return
                time.sleep(0.25)
            t_move = time.time() + 1.5                              # let the arm settle, then use only frames after that
            time.sleep(1.5)
            m2 = lp.measure(after=t_move, n=3, timeout=8)
            if m2 is None:
                say("lost the board after the move")
                return
            moved = m2["needle_board"][:2] - nb[:2]
            say(f"   asked {np.linalg.norm(err):.2f} mm, moved {np.linalg.norm(moved):.2f} mm "
                f"({100 * np.linalg.norm(moved) / max(np.linalg.norm(err), 1e-9):.0f}%); now {np.linalg.norm(lp.goal - m2['needle_board'][:2]):.2f} mm from the pad")
            m = m2
        nb = m["needle_board"]
        result["end_err"] = float(np.linalg.norm(lp.goal - nb[:2]))
        say(f"[{time.time() - t0:4.1f}s] finished: {result['end_err']:.2f} mm from {args.target} "
            f"(started {result.get('start_err', float('nan')):.2f} mm), needle {-nb[2]:.1f} mm above the board")
    except Exception as e:                                          # never leave the window hanging
        say(f"error: {type(e).__name__}: {e}")


th = threading.Thread(target=worker, daemon=True)
th.start()
lp.gui(alive=th.is_alive, linger_s=15.0)
lp.close()
