"""PnP drift test: does a point marked on the board stay put while the camera moves?

Each frame the board (the checkerboard, or an AprilTag with --tag-mm) is detected and its pose is solved
with solvePnP using the calibrated intrinsics. A point you click is turned into board coordinates
(ray from the camera meets the board plane) and re-projected with every new pose. No arm odometry, no filter.

  left-click   mark a point on the board plane (e.g. a pin position)
  right-click  after moving, click where that same physical spot really is now -> prints the error
  c            clear     q / Esc   quit
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def load_intrinsics(path):
    d = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return (np.array(d["camera_matrix"], float), np.array(d["distortion_coefficients"], float).ravel(),
            (int(d["image_width"]), int(d["image_height"])))


class Checkerboard:
    def __init__(self, nx, ny, square_mm):
        self.size = (nx, ny)
        grid = np.zeros((ny, nx, 3), np.float32)
        grid[:, :, 0] = np.arange(nx)[None, :] * square_mm
        grid[:, :, 1] = np.arange(ny)[:, None] * square_mm
        self.grid = grid
        self.sq = square_mm
        self.prev_R = None

    def detect(self, gray):
        small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        ok, c = cv2.findChessboardCornersSB(small, self.size, cv2.CALIB_CB_ACCURACY)
        if not ok:
            return None
        c = (c * 2).astype(np.float32)
        c = cv2.cornerSubPix(gray, c, (7, 7), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
        return c.reshape(self.size[1], self.size[0], 2)

    def solve(self, gray, K, dist):
        c = self.detect(gray)
        if c is None:
            return None
        # A square grid looks the same turned by 90 degrees, so the detector's origin can jump.
        # Try the four turns and keep the pose closest to the previous one (reprojection can't tell them apart).
        best = None
        for k in range(4):
            img = np.rot90(c, k).reshape(-1, 2).astype(np.float32)
            obj = self.grid.reshape(-1, 3)
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
            if not ok:
                continue
            rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
            R = cv2.Rodrigues(rvec)[0]
            if self.prev_R is None:
                score = k                                   # first frame: arbitrary but stable
            else:
                score = np.degrees(np.arccos(np.clip((np.trace(self.prev_R.T @ R) - 1) / 2, -1, 1)))
            if best is None or score < best[0]:
                best = (score, rvec, tvec, R, obj, img)
        if best is None:
            return None
        _, rvec, tvec, R, obj, img = best
        self.prev_R = R
        return rvec, tvec, obj, img, c


class Tag:
    def __init__(self, size_mm):
        self.s = size_mm
        self.det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
                                           cv2.aruco.DetectorParameters())
        h = size_mm / 2
        self.obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)

    def solve(self, gray, K, dist):
        corners, ids, _ = self.det.detectMarkers(gray)
        if ids is None:
            return None
        img = corners[0].reshape(4, 2).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(self.obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        return (rvec, tvec, self.obj, img, img.reshape(1, 4, 2)) if ok else None


def to_board(px, K, dist, rvec, tvec):
    """Pixel -> point on the board plane, in board mm."""
    R = cv2.Rodrigues(rvec)[0]
    xn = cv2.undistortPoints(np.array([[px]], np.float32), K, dist).reshape(2)
    ray = np.array([xn[0], xn[1], 1.0])
    n, t = R[:, 2], tvec.reshape(3)
    s = (n @ t) / (n @ ray)
    return R.T @ (s * ray - t), s


def to_pixel(pt_board, K, dist, rvec, tvec):
    return cv2.projectPoints(np.array([pt_board], np.float32), rvec, tvec, K, dist)[0].reshape(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=1)
    ap.add_argument("--intrinsics", default=str(HERE / "camera_intrinsics.json"))
    ap.add_argument("--inner", type=int, nargs=2, default=(7, 7), metavar=("NX", "NY"), help="checkerboard inner corners")
    ap.add_argument("--square-mm", type=float, default=48.26)
    ap.add_argument("--tag-mm", type=float, help="use an AprilTag 36h11 of this side length instead of the checkerboard")
    ap.add_argument("--server-url", default="http://127.0.0.1:8765")
    ap.add_argument("--log", default=str(HERE / "pnp_test_log.csv"))
    args = ap.parse_args()

    K, dist, (cw, ch) = load_intrinsics(args.intrinsics)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)     # the default backend takes ~90 s to open 1080p
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cw)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ch)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"Could not read camera {args.camera}. Try --camera 0, 2.")
    h, w = frame.shape[:2]
    if (w, h) != (cw, ch):
        K = K.copy()
        K[0] *= w / cw
        K[1] *= h / ch
        print(f"WARNING: camera gives {w}x{h} but the calibration is {cw}x{ch}; intrinsics scaled (ok only if the field of view is the same).")
    print(f"camera {args.camera}: {w}x{h}; fx {K[0, 0]:.1f} fy {K[1, 1]:.1f} cx {K[0, 2]:.1f} cy {K[1, 2]:.1f}")

    target = Tag(args.tag_mm) if args.tag_mm else Checkerboard(*args.inner, args.square_mm)
    arm = None
    try:
        from fusion import ArmOdometry
        arm = ArmOdometry(args.server_url).start()
    except Exception:
        pass

    log = open(args.log, "w", newline="")
    wr = csv.writer(log)
    wr.writerow(["t", "board_x_mm", "board_y_mm", "pred_x", "pred_y", "true_x", "true_y", "err_px", "err_mm", "cam_height_mm", "tilt_deg"])
    st = {"click": None, "truth": None}
    mark = {"pt": None, "checks": []}

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            st["click"] = (x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            st["truth"] = (x, y)

    cv2.namedWindow("pnp test", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("pnp test", on_mouse)
    print("left-click a point on the board; move the arm; stop; right-click where that spot really is. q quits.")
    while True:
        ok, frame = cap.read()
        t = time.time()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = target.solve(gray, K, dist)
        shown = frame.copy()
        info = "board not found"
        if res is not None:
            rvec, tvec, obj, img, _ = res
            R = cv2.Rodrigues(rvec)[0]
            cam = -R.T @ tvec.reshape(3)
            height, tilt = abs(cam[2]), np.degrees(np.arccos(min(1.0, abs(R[2, 2]))))
            rep = cv2.projectPoints(obj, rvec, tvec, K, dist)[0].reshape(-1, 2)
            rms = float(np.sqrt(np.mean(np.sum((rep - img.reshape(-1, 2)) ** 2, axis=1))))
            cv2.drawFrameAxes(shown, K, dist, rvec, tvec, 60, 3)
            for p in img.reshape(-1, 2)[::1]:
                cv2.circle(shown, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)
            tip_z = ""
            s = arm.latest() if arm else None
            if s is not None:
                tip_z = f"   arm tip height {s['height']:.1f} mm"
            info = f"camera {height:.1f} mm above board   tilt {tilt:.1f} deg   pose rms {rms:.2f} px{tip_z}"
            if st["click"]:
                p, _ = to_board(st["click"], K, dist, rvec, tvec)
                mark.update(pt=p, checks=[])
                print(f"marked board point ({p[0]:.1f}, {p[1]:.1f}) mm  [z {p[2]:.2f}]; camera {height:.1f} mm up")
            st["click"] = None
            if mark["pt"] is not None:
                pred = to_pixel(mark["pt"], K, dist, rvec, tvec)
                cv2.drawMarker(shown, (int(pred[0]), int(pred[1])), (255, 255, 0), cv2.MARKER_CROSS, 30, 2)
                cv2.circle(shown, (int(pred[0]), int(pred[1])), 14, (255, 255, 0), 1)
                if st["truth"]:
                    tx, ty = st["truth"]
                    err = float(np.hypot(pred[0] - tx, pred[1] - ty))
                    zc = float((R @ mark["pt"] + tvec.reshape(3))[2])
                    err_mm = err * zc / K[0, 0]
                    mark["checks"].append((tx, ty, pred.copy()))
                    wr.writerow([f"{t:.3f}", mark["pt"][0], mark["pt"][1], *pred, tx, ty, f"{err:.1f}", f"{err_mm:.2f}", f"{height:.1f}", f"{tilt:.1f}"])
                    log.flush()
                    print(f"check {len(mark['checks'])}: PnP says ({pred[0]:.0f}, {pred[1]:.0f}), you say ({tx}, {ty}) -> off by {err:.1f} px = {err_mm:.2f} mm")
            st["truth"] = None
        else:
            st["click"] = st["truth"] = None
        for tx, ty, pp in mark["checks"]:
            cv2.circle(shown, (int(tx), int(ty)), 6, (0, 255, 0), 2)
        for k, line in enumerate((info, "left-click: mark   right-click: truth   c: clear   q: quit")):
            cv2.putText(shown, line, (10, 28 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(shown, line, (10, 28 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.imshow("pnp test", shown)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("c"):
            mark.update(pt=None, checks=[])
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
