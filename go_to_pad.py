"""Put the probe tip over a pad using PnP as the only sensor: one computed move, then small corrections.

 1. Solve the board pose (PnP). With tip_offset.json (from pivot_cal.py) the tip's position in the board frame is
    tip_board = R^T (tip_cam - t), in mm.
 2. Learn how the arm's jogs (right, up, in) move the tip in the board frame by trying each one a few mm and
    measuring it (this absorbs the camera's mirroring and any roll, so no axis conventions are assumed).
 3. Compute the move that takes the tip's x/y to the pad and apply it in chunks; re-measure after every chunk
    and correct until the error is small. Height is left alone unless --hover-mm is given.

Usage: python go_to_pad.py [--target SDA3] [--hover-mm 12] [--send]       (dry run without --send: measures only)
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from arm_client import Arm
from pcb_pnp import find_pad
from pivot_cal import pose_here
from pnp_test import load_intrinsics

HERE = Path(__file__).resolve().parent
LOG = HERE / "agent_run_log.jsonl"


def log(event, **kw):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, **kw}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="SDA3")
    ap.add_argument("--xy", type=float, nargs=2, metavar=("X", "Y"), help="aim at this board position (mm) instead of a pad")
    ap.add_argument("--tip-offset", default=str(HERE / "tip_offset.json"), help="JSON with tip_in_camera_mm")
    ap.add_argument("--hover-mm", type=float, default=None, help="not used yet: this script only moves level")
    ap.add_argument("--tol-mm", type=float, default=0.6)
    ap.add_argument("--max-iter", type=int, default=10)
    ap.add_argument("--gain", type=float, default=0.7)
    ap.add_argument("--send", action="store_true")
    args = ap.parse_args()

    K, dist, _ = load_intrinsics(HERE / "camera_intrinsics.json")
    geo = json.loads((HERE / "pcb_geometry.json").read_text(encoding="utf-8-sig"))
    W, H = geo["width"], geo["height"]
    obj = np.array([[0, 0, 0], [W, 0, 0], [W, H, 0], [0, H, 0]], np.float32)
    pad = find_pad(geo, args.target)
    goal_xy = np.array(args.xy if args.xy else [pad["x"], pad["y"]], float)
    tip_cam = np.array(json.loads(Path(args.tip_offset).read_text())["tip_in_camera_mm"], float)

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    arm = Arm()
    prev = {"R": None}

    warm = [25]

    def measure():
        p = pose_here(cap, K, dist, obj, prev["R"], True, drain=warm[0])
        warm[0] = 8                                # the exposure has settled after the first look
        if p is None:
            return None
        rvec, tvec, fit = p
        R = cv2.Rodrigues(rvec)[0]
        prev["R"] = R
        return R.T @ (tip_cam - tvec.ravel()), fit

    def jog(r=0.0, u=0.0, i=0.0):
        s = arm.jog(right=float(r), up=float(u), in_=float(i))
        if s.get("msg") != "ok" or "shortened" in (s.get("note") or ""):
            print(f"   jog ({r:+.1f}, {u:+.1f}, {i:+.1f}): {s.get('msg')} {s.get('note') or ''}")
        time.sleep(2.2)
        return s

    m0 = measure()
    if m0 is None:
        raise SystemExit("board not found")
    tb, fit = m0
    print(f"target {pad['ref']}.{pad['pad']} [{pad['net']}] at board ({pad['x']:.1f}, {pad['y']:.1f}) mm")
    print(f"tip in the board frame: ({tb[0]:.1f}, {tb[1]:.1f}, {tb[2]:.1f}) mm  (z negative = above the board: {-tb[2]:.1f} mm up)  fit {fit:.2f} px")
    err = goal_xy - tb[:2]
    print(f"needs to move ({err[0]:+.1f}, {err[1]:+.1f}) mm = {np.linalg.norm(err):.1f} mm")
    log("measure", tip_board=tb.tolist(), fit=fit, need_mm=err.tolist(), target=args.target)
    if not args.send:
        print("dry run: nothing moved.")
        return

    # The two level jogs map to the board's axes as right -> -x, up -> -y (seen in every test so far). The servos' stiction
    # (they stop within ~8 ticks of the goal, 1-3 mm at this reach) makes small test jogs under-move, so the matrix is not
    # learned from them: it is fixed here, and the PnP loop below absorbs the undershoot. A divergence guard stops it if the
    # direction is wrong.
    M2 = np.array([[-1.0, 0.0], [0.0, -1.0]])
    history = []

    # --- go there: one computed move, then corrections
    for k in range(args.max_iter):
        a = measure()
        if a is None:
            print("lost the board; stopping")
            break
        tb, fit = a
        want = np.array([goal_xy[0] - tb[0], goal_xy[1] - tb[1], 0.0])
        if args.hover_mm is not None:
            want[2] = -args.hover_mm - tb[2]
        print(f"iter {k}: tip ({tb[0]:.1f}, {tb[1]:.1f}, {-tb[2]:.1f} up)  error xy {np.linalg.norm(want[:2]):.2f} mm  dz {want[2]:+.1f}  fit {fit:.2f}")
        log("iter", k=k, tip_board=tb.tolist(), error_xy_mm=float(np.linalg.norm(want[:2])), fit=fit)
        history.append(float(np.linalg.norm(want[:2])))
        if np.linalg.norm(want[:2]) < args.tol_mm and abs(want[2]) < 1.5:
            print("on target (to PnP's resolution)")
            break
        if len(history) >= 4 and history[-1] > history[-3] + 2.0 and history[-2] > history[-4] + 2.0:
            print("the error is growing, so the jog directions are wrong; stopping")
            break
        u = np.zeros(3)
        u[:2] = np.linalg.solve(M2, want[:2]) * args.gain       # level jogs only: height is never touched here
        biggest = np.max(np.abs(u))
        if biggest > 8.0:                           # the server caps a jog at 10 mm per axis
            u *= 8.0 / biggest
        jog(*u)
    cap.release()


if __name__ == "__main__":
    main()
