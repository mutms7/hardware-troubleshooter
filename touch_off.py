"""Find the real needle's depth in the camera frame by touching the board, slowly, with PnP watching.

Go straight down along the probe ("in") in small steps. After every step PnP says how far the camera really came down.
While the needle is free, the camera follows the command (the servos undershoot a little: stiction). When the needle
meets the board the camera stops coming down while the command keeps going, and the servo following error grows.
Two things must agree before it is called contact:
  - over the last --window-mm of commanded travel the camera came down by less than 35% of that, and
  - the servo following error (z) is up by more than --follow-mm over its resting value.
At that moment the needle is ON the board plane, so its depth along the pixel ray is
    Z = (R^T t)_z / (R^T r)_z        ->  needle in the camera frame = Z * r
which is written to tip_offset_needle.json. Then it backs off and retreats.

Steps are 1.0 mm while the estimated clearance is large and 0.25 mm once either height estimate says < 20 mm.
Abort any time: create a file called STOP next to this script, or press Space in the viewer.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from arm_client import Arm
from pivot_cal import pose_here
from pnp_test import load_intrinsics

HERE = Path(__file__).resolve().parent
PCB_TOP_Z_MM = -53.8        # table z -55.4 + 1.6 mm board (model value; may be stale, so it is only one of two height estimates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-mm", type=float, default=3.0)
    ap.add_argument("--follow-mm", type=float, default=1.5)
    ap.add_argument("--max-travel-mm", type=float, default=90.0)
    ap.add_argument("--preload-mm", type=float, default=3.0, help="first move down, ignored by the detector (stiction)")
    args = ap.parse_args()

    K, dist, _ = load_intrinsics(HERE / "camera_intrinsics.json")
    geo = json.loads((HERE / "pcb_geometry.json").read_text(encoding="utf-8-sig"))
    obj = np.array([[0, 0, 0], [geo["width"], 0, 0], [geo["width"], geo["height"], 0], [0, geo["height"], 0]], np.float32)
    tp = json.loads((HERE / "probe_point.json").read_text(encoding="utf-8-sig"))
    xn = cv2.undistortPoints(np.array([[[tp["x"], tp["y"]]]], np.float32), K, dist).reshape(2)
    ray = np.array([xn[0], xn[1], 1.0])
    pivot = np.array(json.loads((HERE / "tip_offset.json").read_text())["tip_in_camera_mm"], float)

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    arm = Arm()
    prev = [None]
    warm = [25]
    stop_file = HERE / "STOP"
    if stop_file.exists():
        stop_file.unlink()

    def look(n=4):
        p = pose_here(cap, K, dist, obj, prev[0], True, n=n, drain=warm[0])
        warm[0] = 6
        if p is None:
            return None
        R = cv2.Rodrigues(p[0])[0]
        prev[0] = R
        t = p[1].ravel()
        cam = -R.T @ t
        return {"R": R, "t": t, "H": float(abs(cam[2])), "h_pnp": float(-(R.T @ (pivot - t))[2]), "fit": p[2]}

    def follow(base=0.0):
        s = arm.state(measure=True)
        return s.get("follow_err_z_mm", 0.0) - base, s

    def retreat(mm):
        done = 0.0
        while done < mm - 1e-6:
            step = min(10.0, mm - done)
            arm.jog(in_=-step)
            done += step
            time.sleep(1.0)

    s0 = arm.state(measure=True)
    if not s0["hw"]["armed"]:
        raise SystemExit("arm is not armed")
    base = np.mean([follow()[0] for _ in range(5)])
    L = look(8)
    if L is None:
        raise SystemExit("board not found")
    h_fk = lambda s: float(s["measured_tip"][2] * 1000 - PCB_TOP_Z_MM)
    print(f"start: camera {L['H']:.1f} mm up; needle clearance estimates: encoders {h_fk(s0):.1f} mm, PnP+pivot {L['h_pnp']:.1f} mm; "
          f"resting follow error {base:+.2f} mm", flush=True)

    cmd = 0.0
    hist = [(0.0, L["H"])]
    r = arm.jog(in_=args.preload_mm)
    cmd += args.preload_mm
    time.sleep(1.5)
    stalls, contact = 0, None
    while cmd < args.max_travel_mm:
        if stop_file.exists():
            print("STOP file found: stopping and retreating", flush=True)
            break
        gap, s = follow(base)
        L = look()
        if L is None:
            print("lost the board: stopping", flush=True)
            break
        hist.append((cmd, L["H"]))
        est = min(h_fk(s), L["h_pnp"])
        # progress over the last window of commanded travel
        old = next((h for h in reversed(hist) if h[0] <= cmd - args.window_mm), None)
        stalled = False
        if old is not None:
            came_down = old[1] - L["H"]
            stalled = came_down < 0.35 * (cmd - old[0]) and gap > args.follow_mm
        print(f"cmd {cmd:5.2f} mm  camera {L['H']:6.2f} mm up  needle est {est:5.1f} (enc {h_fk(s):5.1f}, pnp {L['h_pnp']:5.1f})  "
              f"follow {gap:+.2f}  fit {L['fit']:.2f}" + ("  [STALLED]" if stalled else ""), flush=True)
        stalls = stalls + 1 if stalled else 0
        if stalls >= 2:
            contact = L
            break
        step = 0.25 if est < 20 else 1.0
        r = arm.jog(in_=step)
        if r.get("msg") != "ok":
            print(f"the server refused the step: {r.get('msg')}", flush=True)
            break
        cmd += step
        time.sleep(0.9 if step >= 1 else 0.7)

    if contact is None:
        print("no contact found; retreating", flush=True)
        retreat(min(cmd, 40.0) + 5)
        cap.release()
        return
    # the pose at the stall: take a few more frames before backing off
    P = look(8) or contact
    R, t = P["R"], P["t"]
    Z = float((R.T @ t)[2] / (R.T @ ray)[2])
    needle = Z * ray
    h_check = float(-(R.T @ (needle - t))[2])
    print(f"\nCONTACT at cmd {cmd:.2f} mm. Needle depth Z = {Z:.1f} mm; needle in the camera frame ({needle[0]:.1f}, {needle[1]:.1f}, {needle[2]:.1f}) mm "
          f"(the pivot guess was {pivot[0]:.1f}, {pivot[1]:.1f}, {pivot[2]:.1f}); height above the board plane {h_check:.2f} mm", flush=True)
    ok_range = 40 <= Z <= 160
    json.dump({"tip_in_camera_mm": needle.tolist(), "depth_mm": Z, "method": "touch-off: needle on the board plane",
               "camera_height_mm": P["H"], "plausible": ok_range, "tip_pixel": [tp["x"], tp["y"]]},
              open(HERE / "tip_offset_needle.json", "w"), indent=1)
    print("saved tip_offset_needle.json" + ("" if ok_range else "  (DEPTH OUT OF THE EXPECTED RANGE: do not trust it)"), flush=True)
    arm.jog(in_=-3.5)                                  # take the pressure off first
    time.sleep(1.5)
    retreat(25.0)
    print("backed off and retreated 25 mm", flush=True)
    cap.release()


if __name__ == "__main__":
    main()
