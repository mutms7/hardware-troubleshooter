"""Needle depth from a hand-guided touch: no arm motion, no model.

Lower the needle onto a bare spot of the PCB by hand (arm limp), hold still, then run this once per spot:
    python needle_touch.py            take the frame, solve the board pose, compute the needle's depth
With the needle ON the board plane, its depth along the pixel ray is Z = (R^T t)_z / (R^T r)_z, and the needle in the
camera frame is Z * r. Each touch is appended to needle_touches.json; with two or more, the mean is written to
tip_offset_needle.json together with the spread (the spread is the honest error bar). An annotated frame is saved
so the tip pixel can be checked against the real needle tip.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from pivot_cal import pose_here
from pnp_test import load_intrinsics

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
K, dist, _ = load_intrinsics(HERE / "camera_intrinsics.json")
geo = json.loads((HERE / "pcb_geometry.json").read_text(encoding="utf-8-sig"))
obj = np.array([[0, 0, 0], [geo["width"], 0, 0], [geo["width"], geo["height"], 0], [0, geo["height"], 0]], np.float32)
tp = json.loads((HERE / "probe_point.json").read_text(encoding="utf-8-sig"))
xn = cv2.undistortPoints(np.array([[[tp["x"], tp["y"]]]], np.float32), K, dist).reshape(2)
ray = np.array([xn[0], xn[1], 1.0])

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
p = pose_here(cap, K, dist, obj, None, True, n=10, drain=30)
ok, frame = cap.read()
cap.release()
if p is None:
    raise SystemExit("board not found: is it in view, and is the camera free?")
R, t = cv2.Rodrigues(p[0])[0], p[1].ravel()
Z = float((R.T @ t)[2] / (R.T @ ray)[2])
needle = Z * ray
cam = -R.T @ t
where = R.T @ (Z * ray - t)                    # where on the board the needle is (x, y, ~0)
print(f"pose fit {p[2]:.2f} px; camera {abs(cam[2]):.1f} mm above the board, tilt {np.degrees(np.arccos(min(1, abs(R[2, 2])))):.1f} deg")
print(f"needle depth Z = {Z:.1f} mm  -> needle in the camera frame ({needle[0]:.1f}, {needle[1]:.1f}, {needle[2]:.1f}) mm")
print(f"needle is at board ({where[0]:.1f}, {where[1]:.1f}) mm  (board is {geo['width']:.0f} x {geo['height']:.0f}); "
      f"{'inside the board' if 0 <= where[0] <= geo['width'] and 0 <= where[1] <= geo['height'] else 'OUTSIDE the board: not a valid touch'}")

shot = frame.copy()
cv2.drawMarker(shot, (int(tp["x"]), int(tp["y"])), (255, 0, 255), cv2.MARKER_CROSS, 40, 2)
cv2.imwrite(str(OUT / "needle_touch_frame.jpg"), shot[int(tp["y"]) - 200:int(tp["y"]) + 200, int(tp["x"]) - 250:int(tp["x"]) + 250])

f = HERE / "needle_touches.json"
touches = json.loads(f.read_text()) if f.exists() else []
touches.append({"Z": Z, "needle_cam": needle.tolist(), "camera_height_mm": float(abs(cam[2])), "board_xy": where[:2].tolist()})
f.write_text(json.dumps(touches, indent=1))
if len(touches) >= 2:
    Zs = np.array([x["Z"] for x in touches])
    mean = np.mean([x["needle_cam"] for x in touches], axis=0)
    print(f"\n{len(touches)} touches: Z = {Zs.round(1).tolist()} mm, mean {Zs.mean():.1f}, spread (std) {Zs.std():.2f} mm")
    json.dump({"tip_in_camera_mm": mean.tolist(), "depth_mm": float(Zs.mean()), "spread_mm": float(Zs.std()),
               "method": f"hand-guided touch-off, {len(touches)} touches", "tip_pixel": [tp["x"], tp["y"]]},
              open(HERE / "tip_offset_needle.json", "w"), indent=1)
    print("saved tip_offset_needle.json")
