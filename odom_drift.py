"""How far does the arm-offset method drift on its own? No vision, no filter, no learning.

  left-click   mark the point on the live frame (this also starts the arm-offset estimate from here)
  move the arm yourself (the sim page, keys WASD etc.)
  right-click  after the arm has stopped, click where the point REALLY is now -> prints the error
  c            clear     q / Esc   quit

The cyan crosshair is pure arm odometry: the point carried along by the arm's own motion
(servo_calibration.json J, rescaled with height), compensated for camera latency. Nothing corrects it.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from fusion import ArmOdometry, PointFuser

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=1)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--vision-delay", type=float, default=0.12)
    ap.add_argument("--server-url", default="http://127.0.0.1:8765")
    ap.add_argument("--servo-cal", default=str(HERE / "servo_calibration.json"))
    ap.add_argument("--log", default=str(HERE / "odom_drift_log.csv"))
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"Could not read from camera {args.camera}.")

    cal = json.loads(Path(args.servo_cal).read_text(encoding="utf-8-sig"))
    fuser = PointFuser(cal["J_px_per_mm"], cal["height_mm"], learn=False)
    odo = ArmOdometry(args.server_url).start()

    log = open(args.log, "w", newline="")
    w = csv.writer(log)
    w.writerow(["t", "start_x", "start_y", "pred_x", "pred_y", "true_x", "true_y", "err_px", "tip_travel_mm", "path_mm", "height_mm"])

    st = {"click": None, "truth": None}
    run = {"start": None, "start_tip": None, "path": 0.0, "last_tip": None, "checks": []}

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            st["click"] = (x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            st["truth"] = (x, y)

    cv2.namedWindow("odometry drift", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("odometry drift", on_mouse)
    print("left-click the point; move the arm; stop; right-click where the point really is. q quits.")
    pred = None
    while True:
        ok, frame = cap.read()
        t_read = time.time()
        if not ok:
            break
        frame_time = t_read - args.vision_delay
        s = odo.latest()
        if st["click"] and s is not None:
            fuser.reset()
            p = np.array([st["click"]], float)
            fuser.init_from_reference(p, s["tip"], s["height"], s)
            run.update(start=p[0], start_tip=s["tip"].copy(), path=0.0, last_tip=s["tip"].copy(), checks=[])
            print(f"start ({p[0][0]:.0f}, {p[0][1]:.0f}) at tip {np.round(s['tip'], 1)} mm, height {s['height']:.1f} mm")
            st["click"] = None
        pred = None
        if run["start"] is not None and s is not None:
            pr = fuser.predict(s, frame_time)
            pred = None if pr is None else pr[0]
            run["path"] += float(np.linalg.norm(s["tip"] - run["last_tip"]))
            run["last_tip"] = s["tip"].copy()
        if st["truth"] is not None and pred is not None:
            tx, ty = st["truth"]
            err = float(np.hypot(pred[0] - tx, pred[1] - ty))
            travel = float(np.linalg.norm(s["tip"] - run["start_tip"]))
            run["checks"].append((tx, ty, pred.copy(), err))
            w.writerow([f"{t_read:.3f}", *run["start"], *pred, tx, ty, f"{err:.1f}", f"{travel:.1f}", f"{run['path']:.1f}", f"{s['height']:.1f}"])
            log.flush()
            print(f"check {len(run['checks'])}: odometry says ({pred[0]:.0f}, {pred[1]:.0f}), truth ({tx}, {ty}) "
                  f"-> off by {err:.1f} px ({err / abs(cal['J_px_per_mm'][0][0]) * (s['height'] / cal['height_mm']):.1f} mm) "
                  f"after {travel:.1f} mm straight / {run['path']:.1f} mm of path")
        st["truth"] = None

        shown = frame.copy()
        if run["start"] is not None:
            sx, sy = int(run["start"][0]), int(run["start"][1])
            cv2.circle(shown, (sx, sy), 5, (255, 255, 255), 1)
            cv2.putText(shown, "start", (sx + 8, sy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        if pred is not None:
            px, py = int(round(pred[0])), int(round(pred[1]))
            cv2.drawMarker(shown, (px, py), (255, 255, 0), cv2.MARKER_CROSS, 26, 2)
            cv2.circle(shown, (px, py), 14, (255, 255, 0), 1)
        for tx, ty, pp, err in run["checks"]:
            cv2.circle(shown, (int(tx), int(ty)), 6, (0, 255, 0), 2)
            cv2.line(shown, (int(tx), int(ty)), (int(pp[0]), int(pp[1])), (0, 200, 255), 1)
        txt = "left-click the point" if run["start"] is None else f"path {run['path']:.1f} mm   checks {len(run['checks'])}   cyan = arm offset only, green = your truth click"
        cv2.putText(shown, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(shown, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("odometry drift", shown)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("c"):
            run.update(start=None, checks=[])
    cap.release()
    cv2.destroyAllWindows()
    if run["checks"]:
        errs = [c[3] for c in run["checks"]]
        print(f"{len(errs)} checks: mean {np.mean(errs):.1f} px, worst {np.max(errs):.1f} px; log in {args.log}")


if __name__ == "__main__":
    main()
