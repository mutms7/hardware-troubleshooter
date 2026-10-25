"""Camera-to-tip offset by pivot calibration, no contact needed.

The tip is a fixed point in the camera frame, on the ray through the tip pixel (probe_point.json). Unknown: its
depth Z along the optical axis. The server tilts the wrist with the tip held at the same place, so at every tilt the
tip must land on the same board-frame point X:   R_i^T (Z * r - t_i) = X   (r = [xn, yn, 1]).
Solved by linear least squares in (Z, X) from the board pose at several tilts.

Usage: python pivot_cal.py [--step-deg 6] [--max-steps 3] [--send]     (dry run without --send)
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from arm_client import Arm
from pcb_pnp import solve_frame
from pcb_snapshot import grab  # noqa: F401  (kept so the camera helpers stay in one place)
from pnp_test import load_intrinsics

import urllib.request

HERE = Path(__file__).resolve().parent
URL = "http://127.0.0.1:8765"


def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


_BF = [None]


def _features():
    """The silkscreen-feature reference (feature_pose.py), if it has been built."""
    if _BF[0] is None:
        try:
            from feature_pose import NPZ, BoardFeatures
            _BF[0] = BoardFeatures.load() if NPZ.exists() else False
        except Exception:
            _BF[0] = False
    return _BF[0]


def pose_here(cap, K, dist, obj, prev_R, flip, n=8, drain=25):
    """Board pose from the camera: silkscreen features first (glare-proof), colour segmentation as the fallback."""
    for _ in range(drain):                                # drain the camera's buffer, and let auto-exposure settle
        cap.read()
    rv, tv, fits = [], [], []
    bf = _features()
    for _ in range(n):
        ok, frame = cap.read()
        if not ok:
            continue
        if bf:
            r = bf.solve(frame, K, dist)
            if r is not None and r[0] <= 2.5:
                fits.append(r[0])
                rv.append(r[1].ravel())
                tv.append(r[2].ravel())
                continue
        res = solve_frame(frame, obj, K, dist, prev_R, flip)
        best = None if res is None else res[0]
        if best is None or best[0] > 3.0:                 # a bad corner fit means a bad frame: drop it
            continue
        fits.append(best[0])
        rv.append(best[2].ravel())
        tv.append(best[3].ravel())
    if len(rv) < (2 if bf else 4):
        return None
    rvec = np.median(rv, axis=0).reshape(3, 1)
    tvec = np.median(tv, axis=0).reshape(3, 1)
    return rvec, tvec, float(np.median(fits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-deg", type=float, default=6.0)
    ap.add_argument("--max-steps", type=int, default=3)
    ap.add_argument("--send", action="store_true", help="actually move the arm (arms it first)")
    ap.add_argument("--no-flip", action="store_true", help="board is in the KiCad orientation (default: rotated 180)")
    args = ap.parse_args()

    K, dist, _ = load_intrinsics(HERE / "camera_intrinsics.json")
    geo = json.loads((HERE / "pcb_geometry.json").read_text(encoding="utf-8-sig"))
    W, H = geo["width"], geo["height"]
    obj = np.array([[0, 0, 0], [W, 0, 0], [W, H, 0], [0, H, 0]], np.float32)
    tip_px = json.loads((HERE / "probe_point.json").read_text(encoding="utf-8-sig"))
    xn = cv2.undistortPoints(np.array([[[tip_px["x"], tip_px["y"]]]], np.float32), K, dist).reshape(2)
    ray = np.array([xn[0], xn[1], 1.0])
    flip = not args.no_flip

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    arm = Arm(URL)

    samples, prev_R = [], None

    def record(label):
        nonlocal prev_R
        time.sleep(2.5)
        s = arm.state(measure=True)
        p = pose_here(cap, K, dist, obj, prev_R, flip)
        if p is None:
            print(f"{label}: board not found, skipped")
            return
        rvec, tvec, fit = p
        R = cv2.Rodrigues(rvec)[0]
        prev_R = R
        cam = -R.T @ tvec.ravel()
        tip = np.array(s.get("measured_tip") or s["tip"], float) * 1000
        print(f"{label}: fit {fit:.2f} px  camera {abs(cam[2]):.1f} mm up  tilt {np.degrees(np.arccos(min(1, abs(R[2, 2])))):.1f} deg  "
              f"tip (enc) {np.round(tip, 1)}  phi {s.get('phi', float('nan')):.1f}  msg {s.get('msg')}")
        if fit < 2.5:
            samples.append((R, tvec.ravel(), tip, label))

    s0 = arm.state(measure=True)
    print(f"start: armed={s0.get('hw', {}).get('armed')} height {s0['height_mm']:.1f} mm phi {s0.get('phi', float('nan')):.1f}")
    if not args.send:
        print("dry run: no motion. Measuring the board pose from where the arm is now.")
        record("here")
        return
    if not s0.get("hw", {}).get("armed"):
        print("arming (holds the current position)...")
        print(post("/api/arm", {"on": True}).get("msg"))
        time.sleep(1.5)
    record("start")
    applied = 0.0
    for sign in (+1, -1):
        for k in range(args.max_steps):
            r = arm.tilt(sign * args.step_deg)
            if r.get("msg") != "ok":
                print(f"tilt {sign * args.step_deg:+.0f} deg refused: {r.get('msg')}")
                break
            applied += sign * args.step_deg
            record(f"tilt {applied:+.0f} deg")
        while abs(applied) > 1e-6:                        # go back to where we started, one small step at a time
            back = -np.sign(applied) * min(args.step_deg, abs(applied))
            r = arm.tilt(float(back))
            if r.get("msg") != "ok":
                print(f"could not step back: {r.get('msg')}; stopping with the tilt at {applied:+.0f} deg")
                break
            applied += back
            time.sleep(1.5)
        print(f"back at tilt offset {applied:+.0f} deg")
    cap.release()

    if len(samples) < 3:
        raise SystemExit(f"only {len(samples)} usable poses, need 3 or more")
    A, b = [], []
    for R, t, tip, _ in samples:
        A.append(np.column_stack([R.T @ ray, -np.eye(3)]))      # unknowns: Z, X (3)
        b.append(R.T @ t)
    A, b = np.vstack(A), np.concatenate(b)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    Z, X = float(sol[0]), sol[1:]
    res = (A @ sol - b).reshape(-1, 3)
    rms = float(np.sqrt(np.mean(np.sum(res ** 2, axis=1))))
    tip_cam = Z * ray
    print(f"\nTip in the camera frame: ({tip_cam[0]:.1f}, {tip_cam[1]:.1f}, {tip_cam[2]:.1f}) mm  (depth Z {Z:.1f} mm)")
    print(f"Pivot point in the board frame: ({X[0]:.1f}, {X[1]:.1f}, {X[2]:.1f}) mm   residual {rms:.2f} mm over {len(samples)} poses")
    spread = np.ptp(np.array([s[2] for s in samples]), axis=0)
    print(f"Encoder tip moved by up to {np.round(spread, 1)} mm (x, y, z) across the poses (it should be ~0)")
    json.dump({"tip_in_camera_mm": tip_cam.tolist(), "depth_mm": Z, "pivot_in_board_mm": X.tolist(),
               "residual_mm": rms, "poses": len(samples), "tip_pixel": [tip_px["x"], tip_px["y"]],
               "encoder_tip_spread_mm": spread.tolist()}, open(HERE / "tip_offset.json", "w"), indent=1)
    print("saved tip_offset.json")


if __name__ == "__main__":
    main()
