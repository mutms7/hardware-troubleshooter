"""Does a jog move the camera where the arm's own encoders say it moved?  Compare, for one small jog:
  - the encoders' tip displacement (base frame, mm)
  - the camera's displacement from PnP (board frame, mm), C = -R^T t
Both are rigid with the tip, so their lengths must match, and a vertical jog must be vertical in both."""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from arm_client import Arm
from pivot_cal import pose_here
from pnp_test import load_intrinsics

HERE = Path(__file__).resolve().parent
axis, amount = sys.argv[1], float(sys.argv[2])            # e.g. in -5
K, dist, _ = load_intrinsics(HERE / "camera_intrinsics.json")
geo = json.loads((HERE / "pcb_geometry.json").read_text(encoding="utf-8-sig"))
obj = np.array([[0, 0, 0], [geo["width"], 0, 0], [geo["width"], geo["height"], 0], [0, geo["height"], 0]], np.float32)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
arm = Arm()
prev = [None]


def look():
    p = pose_here(cap, K, dist, obj, prev[0], True)
    if p is None:
        raise SystemExit("board not found")
    R = cv2.Rodrigues(p[0])[0]
    prev[0] = R
    s = arm.state(measure=True)
    return -R.T @ p[1].ravel(), np.array(s["measured_tip"], float) * 1000, s


c0, t0, s0 = look()
r = arm.jog(**{"right" if axis == "right" else "up" if axis == "up" else "in_": amount})
print("jog:", r.get("msg"), r.get("note") or "")
time.sleep(2.5)
c1, t1, s1 = look()
dc, dt = c1 - c0, t1 - t0
print(f"camera (PnP, board frame, Z down): moved ({dc[0]:+.1f}, {dc[1]:+.1f}, {dc[2]:+.1f}) mm  |{np.linalg.norm(dc):.1f}|  -> vertical {-dc[2]:+.1f} mm (up positive)")
print(f"tip (encoders, base frame, Z up):  moved ({dt[0]:+.1f}, {dt[1]:+.1f}, {dt[2]:+.1f}) mm  |{np.linalg.norm(dt):.1f}|  -> vertical {dt[2]:+.1f} mm")
print(f"model's depth axis (base frame): {np.round(s0['camera_axes']['depth'], 2)}  phi {s0['phi']:.1f}")
arm.jog(**{"right" if axis == "right" else "up" if axis == "up" else "in_": -amount})
time.sleep(2.0)
cap.release()
