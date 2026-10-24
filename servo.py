"""Close the loop: steer the probe onto the target using the tracker's pixel error.

  python servo.py                       dry run: prints the jogs it WOULD send, moves nothing
  python servo.py --calibrate --send    learn how jogs move the image (4 tiny moves), then save it
  python servo.py --send                run the PD loop on the real arm until the target is under the probe

Needs the arm server running and armed (G in the viewer), and target_tracker.py running with a
target drawn and the probe point set. Reads tracker_state.json, sends jogs through arm_client.

The controller. error_px = target - probe, in image pixels. We want the probe to move that way
in the image, so the wanted image motion is  m = Kp*e + Kd*(e - e_prev). Calibration measures a
2x2 matrix J (pixels of image motion per mm of jog, for right and up), so the jog that makes
image motion m is  J^-1 m. J is rescaled with height: nearer the board, one mm moves more pixels.
This also absorbs camera roll, scale, and direction, so none of those have to be guessed.

It only acts on fresh readings, waits for the arm to settle after each jog, and stops when the
target is lost, the readings go stale, or the error stays under tolerance. It never descends.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from arm_client import Arm

HERE = Path(__file__).resolve().parent


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


class Controller:
    """Pure PD logic, kept free of I/O so it can be tested against a simulated plant."""

    def __init__(self, J, cal_height_mm, kp, kd, max_step_mm):
        self.J, self.cal_height, self.kp, self.kd, self.max_step = np.array(J, float), cal_height_mm, kp, kd, max_step_mm
        self.prev = None

    def reset(self):
        self.prev = None

    def jog(self, error_px, height_mm):
        """Returns (right_mm, up_mm) for this error. height_mm scales J: closer = more px per mm."""
        e = np.asarray(error_px, float)
        de = np.zeros(2) if self.prev is None else e - self.prev
        self.prev = e
        m = self.kp * e + self.kd * de                       # wanted image motion, px
        J = self.J * (self.cal_height / max(height_mm, 3.0))  # px per mm at the current height
        u = np.linalg.solve(J, m)                            # (right, up) in mm
        return tuple(float(v) for v in np.clip(u, -self.max_step, self.max_step))


def load_calibration(path):
    d = read_json(path)
    if not d:
        return None
    return np.array(d["J_px_per_mm"], float), float(d["height_mm"])


def averaged_error(path, n=6, timeout=6.0, after=0.0):
    """Average error_px over n fresh readings taken after time `after`; None if they don't come."""
    seen, got, t_end = set(), [], time.time() + timeout
    while len(got) < n and time.time() < t_end:
        s = read_json(path)
        if s and s.get("fresh") and s.get("error_px") and s["time"] > after and s["time"] not in seen:
            seen.add(s["time"]); got.append(s["error_px"])
        time.sleep(0.03)
    return np.mean(got, axis=0) if len(got) >= max(3, n // 2) else None


def calibrate(arm, args):
    step, settle = args.cal_mm, args.settle + 0.6
    height = arm.state()["height_mm"]
    base = averaged_error(args.state_file)
    if base is None:
        raise SystemExit("No steady tracker reading to calibrate from (is the target tracked and fresh?).")
    cols = []
    for name, jog_out, jog_back in (("right", dict(right=step), dict(right=-step)), ("up", dict(up=step), dict(up=-step))):
        r = arm.jog(**jog_out)
        if r["msg"] != "ok":
            raise SystemExit(f"Calibration jog {name} refused: {r['msg']}")
        t = time.time(); time.sleep(settle)
        moved = averaged_error(args.state_file, after=t + settle - 0.2)
        arm.jog(**jog_back); time.sleep(settle)
        if moved is None:
            raise SystemExit(f"Lost the target after the {name} jog; try a smaller --cal-mm or a better target view.")
        image_motion = -(moved - base)        # the scene moves opposite to the probe
        cols.append(image_motion / step)      # px of image motion per mm of this jog
        print(f"  {name:5s} {step} mm -> the probe moved ({image_motion[0]:+.1f}, {image_motion[1]:+.1f}) px in the image")
        back = averaged_error(args.state_file, after=time.time())
        base = back if back is not None else base
    J = np.array(cols).T                      # columns: right jog, up jog
    if abs(np.linalg.det(J)) < 1e-3 or np.linalg.cond(J) > 8:
        raise SystemExit(f"The two jogs gave nearly parallel image motion (J = {J.round(2).tolist()}): not usable.")
    Path(args.cal_file).write_text(json.dumps({"J_px_per_mm": J.tolist(), "height_mm": height, "step_mm": step}, indent=2))
    ang = math.degrees(math.atan2(J[1, 0], J[0, 0]))
    print(f"Saved {args.cal_file}: J = {J.round(2).tolist()} px/mm at {height:.0f} mm height "
          f"(a 'right' jog moves the image {ang:+.0f} degrees from the image's right direction).")


def run(arm, args):
    cal = load_calibration(args.cal_file)
    if cal is None:
        if args.send:
            raise SystemExit("No calibration yet. Run:  python servo.py --calibrate --send")
        print("(dry run, no calibration file: using a guess of 0.1 mm per pixel; the real values come from --calibrate)")
        J, cal_h = np.array([[10.0, 0.0], [0.0, -10.0]]), 30.0   # right -> +x, up -> -y (image y points down)
    else:
        J, cal_h = cal
    ctl = Controller(J, cal_h, args.kp, args.kd, args.max_step)
    mode = "SENDING to the arm" if args.send else "dry run, not moving the arm"
    print(f"servo: {mode}; Kp {args.kp}, Kd {args.kd}, tolerance {args.tol_px} px, max {args.max_step} mm per jog")
    if args.send and not arm.state()["hw"]["armed"]:
        raise SystemExit("The arm isn't armed. Press G in the viewer first.")

    t0 = time.time(); last_t, ok_streak, stale_since, sent = 0.0, 0, None, 0
    while time.time() - t0 < args.timeout:
        s = read_json(args.state_file)
        time.sleep(0.03)
        usable = s and s.get("fresh") and s.get("error_px") and s["time"] > last_t and time.time() - s["time"] < 0.6
        if not usable:
            stale_since = stale_since or time.time()
            if time.time() - stale_since > args.lost_s:
                raise SystemExit(f"Stopped: no fresh tracker reading for {args.lost_s:.0f} s (target lost, or tracker not running).")
            continue
        stale_since, last_t = None, s["time"]
        e = np.array(s["error_px"]); size = math.hypot(*e)
        if size < args.tol_px:
            ok_streak += 1
            if ok_streak >= args.hold:
                print(f"Aligned: error {size:.1f} px for {args.hold} readings in a row. Done after {sent} jogs.")
                return
            continue
        ok_streak = 0
        height = arm.state()["height_mm"]
        right, up = ctl.jog(e, height)
        print(f"error ({e[0]:+6.1f}, {e[1]:+6.1f}) px  height {height:5.1f} mm  ->  jog right {right:+.2f}, up {up:+.2f} mm")
        if args.send:
            r = arm.jog(right=right, up=up)
            sent += 1
            if r["msg"] != "ok":
                raise SystemExit(f"Stopped: the arm refused the jog: {r['msg']} {r.get('note', '')}")
            if r.get("note"):
                print("  (" + r["note"] + ")")
            time.sleep(args.settle)
            last_t = max(last_t, time.time())     # ignore tracker frames from before the arm settled
            ctl.reset()                           # the error changed because WE moved, not because of drift
        else:
            time.sleep(args.settle)
    print(f"Stopped: {args.timeout:.0f} s timeout without reaching the tolerance.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--state-file", default=str(HERE / "tracker_state.json"))
    ap.add_argument("--cal-file", default=str(HERE / "servo_calibration.json"))
    ap.add_argument("--send", action="store_true", help="actually move the arm (default: dry run)")
    ap.add_argument("--calibrate", action="store_true", help="measure how jogs move the image (needs --send)")
    ap.add_argument("--cal-mm", type=float, default=3.0, help="size of the calibration jogs")
    ap.add_argument("--kp", type=float, default=0.5, help="fraction of the error corrected per step; keep it under 1")
    ap.add_argument("--kd", type=float, default=0.1)
    ap.add_argument("--tol-px", type=float, default=3.0)
    ap.add_argument("--hold", type=int, default=5, help="readings in a row under tolerance to call it aligned")
    ap.add_argument("--max-step", type=float, default=2.0, help="largest jog per axis, mm")
    ap.add_argument("--settle", type=float, default=0.4, help="seconds to wait for the arm after each jog")
    ap.add_argument("--lost-s", type=float, default=2.0, help="stop if no fresh reading for this long")
    ap.add_argument("--timeout", type=float, default=40.0)
    args = ap.parse_args()
    arm = Arm(args.url)
    if args.calibrate:
        if not args.send:
            raise SystemExit("Calibration moves the arm by a few mm. Add --send to allow that.")
        if not arm.state()["hw"]["armed"]:
            raise SystemExit("The arm isn't armed. Press G in the viewer first.")
        calibrate(arm, args)
    else:
        run(arm, args)


if __name__ == "__main__":
    main()
