"""One frame: save it, segment the green board, solve PnP, save an annotated copy and print the pose.
Headless counterpart of pcb_pnp.py. Usage: python pcb_snapshot.py OUT_PREFIX [--flip]"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from pcb_pnp import HSV_HI, HSV_LO, draw_arrow, find_pad, segment, solve_frame
from pnp_test import load_intrinsics

HERE = Path(__file__).resolve().parent


def grab(camera=1, frames=25):
    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    frame = None
    for _ in range(frames):                      # let exposure settle
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()
    return frame


def analyse(frame, flip=False, lo=HSV_LO, hi=HSV_HI, target="SDA3"):
    K, dist, _ = load_intrinsics(HERE / "camera_intrinsics.json")
    geo = json.loads((HERE / "pcb_geometry.json").read_text(encoding="utf-8-sig"))
    W, H = geo["width"], geo["height"]
    obj = np.array([[0, 0, 0], [W, 0, 0], [W, H, 0], [0, H, 0]], np.float32)
    res = solve_frame(frame, obj, K, dist, None, flip)
    mask = res[2] if res else segment(frame, lo, hi)
    quad = res[1] if res else None
    out = frame.copy()
    result = {"found": False}
    if quad is not None:
        best = res[0]
        if best is not None:
            rms, k, rvec, tvec = best
            R = cv2.Rodrigues(rvec)[0]
            cam = -R.T @ tvec.reshape(3)
            outline = cv2.projectPoints(obj, rvec, tvec, K, dist)[0].reshape(-1, 2).astype(np.int32)
            cv2.polylines(out, [outline], True, (0, 255, 0), 3)
            for c in quad:
                cv2.circle(out, (int(c[0]), int(c[1])), 8, (0, 0, 255), -1)
            jp = [p for p in geo["pads"] if p["ref"].startswith("J")]
            pr = cv2.projectPoints(np.array([[p["x"], p["y"], 0] for p in jp], np.float32), rvec, tvec, K, dist)[0].reshape(-1, 2)
            for p, pad in zip(pr, jp):
                cv2.circle(out, (int(p[0]), int(p[1])), 4, (0, 220, 255), -1)
                if pad["pad"] == "1":
                    cv2.putText(out, pad["ref"], (int(p[0]) - 30, int(p[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
            tp = json.loads((HERE / "probe_point.json").read_text(encoding="utf-8-sig"))
            tgt = find_pad(geo, target)
            d = draw_arrow(out, K, dist, rvec, tvec, tgt, (tp["x"], tp["y"]))
            result = {"found": True, "target": target, "tip_to_target_board_mm": [round(float(d[0]), 1), round(float(d[1]), 1)], "corner_fit_px": rms, "camera_height_mm": float(abs(cam[2])),
                      "tilt_deg": float(np.degrees(np.arccos(min(1, abs(R[2, 2]))))),
                      "rvec": rvec.ravel().tolist(), "tvec": tvec.ravel().tolist(), "corners_px": quad.tolist(),
                      "mask_fraction": float((mask > 0).mean())}
    return out, mask, result


if __name__ == "__main__":
    prefix = sys.argv[1]
    frame = grab()
    out, mask, res = analyse(frame, flip="--flip" in sys.argv)
    cv2.imwrite(prefix + "_raw.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    cv2.imwrite(prefix + "_pnp.jpg", cv2.resize(out, None, fx=0.6, fy=0.6), [cv2.IMWRITE_JPEG_QUALITY, 85])
    cv2.imwrite(prefix + "_mask.png", cv2.resize(mask, None, fx=0.3, fy=0.3))
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items() if k not in ("rvec", "tvec", "corners_px")}))
