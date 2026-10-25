"""Continuous board pose from the wrist camera, with a live window to watch the alignment.

  grab thread   reads frames as fast as the camera delivers them (no buffer lag)
  solve thread  PnP on the newest frame: silkscreen features first, colour segmentation as the fallback
  main thread   draws the window (OpenCV needs it); the control logic runs in a worker thread and calls
                measure(after=t) to get the median needle position from poses whose frames were captured after time t.

The window shows: the board outline and header pads, the TARGET pad (orange ring), the point on the board directly under the
needle (cyan cross; this is what gets aligned), a line between them with the error in mm, the needle's height above the board,
and which solver is giving the pose.  q / Esc closes it.

The same annotated frame is written to FEED_JPG about 5 times a second; the arm server serves it at /api/feed.jpg and the sim
viewer shows it. Without a GUI-capable OpenCV (opencv-python-headless) there is no window, only the feed.
"""
import json
import os
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from feature_pose import NPZ, BoardFeatures
from pcb_pnp import find_pad, solve_frame
from pnp_test import load_intrinsics

HERE = Path(__file__).resolve().parent
FEED_JPG = Path(tempfile.gettempdir()) / "troubleshooter_live_feed.jpg"   # read by sim_server.py
FEED_PERIOD_S = 0.2
LATENCY_S = 0.3          # frames are about this old by the time they are read: ignore poses from before (t + LATENCY_S)


class LivePose:
    def __init__(self, camera=1, width=1280, height=720, target="J3.4", needle_file="tip_offset_needle.json"):
        K, dist, (cw, ch) = load_intrinsics(HERE / "camera_intrinsics.json")
        self.K, self.dist = K.copy(), dist
        self.K[0] *= width / cw
        self.K[1] *= height / ch
        self.geo = json.loads((HERE / "pcb_geometry.json").read_text(encoding="utf-8-sig"))
        W, H = self.geo["width"], self.geo["height"]
        self.W, self.H = W, H
        self.obj = np.array([[0, 0, 0], [W, 0, 0], [W, H, 0], [0, H, 0]], np.float32)
        self.pad = find_pad(self.geo, target)
        self.goal = np.array([self.pad["x"], self.pad["y"]])
        self.needle = np.array(json.loads((HERE / needle_file).read_text())["tip_in_camera_mm"], float)
        tp = json.loads((HERE / "probe_point.json").read_text(encoding="utf-8-sig"))
        self.tip_px = (tp["x"] * width / tp.get("width", 1920), tp["y"] * height / tp.get("height", 1080))
        self.jpads = [p for p in self.geo["pads"] if p["ref"].startswith("J")]
        self.bf = BoardFeatures.load() if NPZ.exists() else None
        self.cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._lock = threading.Lock()
        self._frame, self._frame_t, self._pose = None, 0.0, None
        self._prev_R = None
        self._fs = None                     # filtered pose state: (R, t)
        self._outliers = 0
        self._prev_small, self._static_n = None, 0
        self.stop = threading.Event()
        self.extra = ["starting..."]
        threading.Thread(target=self._grab, daemon=True).start()
        threading.Thread(target=self._solve, daemon=True).start()

    # ---- background threads
    def _grab(self):
        while not self.stop.is_set():
            ok, f = self.cap.read()
            if ok:
                with self._lock:
                    self._frame, self._frame_t = f, time.time()

    def _solve(self):
        last_t = 0.0
        while not self.stop.is_set():
            with self._lock:
                frame, ft = self._frame, self._frame_t
            if frame is None or ft == last_t:
                time.sleep(0.01)
                continue
            last_t = ft
            found = None
            if self.bf is not None:
                r = self.bf.solve(frame, self.K, self.dist)
                if r is not None and r[0] <= 2.5:
                    found = (r[1], r[2], r[0], f"features ({r[3]} pts)")
            if found is None:
                res = solve_frame(frame, self.obj, self.K, self.dist, self._prev_R, True)
                if res is not None and res[0][0] <= 3.0:
                    found = (res[0][2], res[0][3], res[0][0], "colour corners")
            if found is None:
                continue
            rvec, tvec, fit, src = found
            R = cv2.Rodrigues(rvec)[0]
            self._prev_R = R
            raw_nb = R.T @ (self.needle - tvec.ravel())
            static = self._is_static(frame)
            Rf, tf = self._filter(R, tvec.ravel(), static)
            nb = Rf.T @ (self.needle - tf)
            with self._lock:
                self._pose = {"t": ft, "rvec": cv2.Rodrigues(Rf)[0], "tvec": tf.reshape(3, 1), "R": Rf, "fit": float(fit),
                              "src": src + (", still" if static else ", moving"), "needle_board": nb,
                              "raw_needle_board": raw_nb, "raw_src": src}

    # ---- filtering the pose
    def _is_static(self, frame):
        """True once the picture has stopped changing (3 solves in a row with a tiny frame-to-frame difference)."""
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.float32)
        d = 99.0 if self._prev_small is None else float(np.mean(np.abs(small - self._prev_small)))
        self._prev_small = small
        self._static_n = self._static_n + 1 if d < 2.5 else 0
        return self._static_n >= 3

    def _filter(self, R, t, static):
        """Smooth the pose. Still: heavy averaging (the overlay holds still). Moving: follows fast. A pose that jumps more than
        5 mm (at the needle) or 6 deg is ignored unless 3 in a row agree (one bad solve, e.g. a flipped planar solution)."""
        if self._fs is None:
            self._fs, self._outliers = (R.copy(), t.copy()), 0
            return self._fs
        Rf, tf = self._fs
        rel = cv2.Rodrigues(Rf.T @ R)[0].ravel()
        ang = float(np.degrees(np.linalg.norm(rel)))
        jump = float(np.linalg.norm((R.T @ (self.needle - t)) - (Rf.T @ (self.needle - tf))))
        if jump > 5.0 or ang > 6.0:
            self._outliers += 1
            if self._outliers < 3:
                return self._fs
            self._fs, self._outliers = (R.copy(), t.copy()), 0             # three in a row: a real move, take it
            return self._fs
        self._outliers = 0
        a = 0.12 if static else 0.6
        Rn = Rf @ cv2.Rodrigues(rel * a)[0]
        self._fs = (Rn, (1 - a) * tf + a * t)
        return self._fs

    # ---- for the control logic
    def latest(self):
        with self._lock:
            return self._pose

    def measure(self, after, n=3, timeout=6.0):
        """Median needle position (board mm; -z is the height) over n poses from frames captured after time `after`."""
        got, last_t, t_end = [], 0.0, time.time() + timeout
        while time.time() < t_end and len(got) < n:
            p = self.latest()
            if p is not None and p["t"] - LATENCY_S > after and p["t"] != last_t:
                got.append(p)
                last_t = p["t"]
            time.sleep(0.02)
        if not got:
            return None
        nb = np.median([g["needle_board"] for g in got], axis=0)
        return {"needle_board": nb, "fit": float(np.mean([g["fit"] for g in got])), "src": got[-1]["src"], "n": len(got)}

    # ---- the window
    def draw(self, frame, pose):
        img = frame.copy()
        cv2.putText(img, f"{self.pad['ref']}.{self.pad['pad']} [{self.pad['net']}]", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
        for i, line in enumerate(self.extra[:3]):
            cv2.putText(img, line, (12, 56 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(img, line, (12, 56 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if pose is None:
            cv2.putText(img, "no board pose", (12, img.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return img
        rvec, tvec, K, dist = pose["rvec"], pose["tvec"], self.K, self.dist

        def proj(pts):
            return cv2.projectPoints(np.array(pts, np.float32).reshape(-1, 3), rvec, tvec, K, dist)[0].reshape(-1, 2)

        cv2.polylines(img, [proj(self.obj).astype(np.int32)], True, (0, 255, 0), 2)
        for p in proj([[q["x"], q["y"], 0] for q in self.jpads]):
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 220, 255), -1)
        g = proj([[self.goal[0], self.goal[1], 0]])[0]
        nb = pose["needle_board"]
        n = proj([[nb[0], nb[1], 0]])[0]                       # the board point under the needle
        gi, ni = (int(g[0]), int(g[1])), (int(n[0]), int(n[1]))
        orange = (0, 140, 255)
        cv2.line(img, ni, gi, (255, 255, 255), 2)
        cv2.circle(img, gi, 16, (0, 0, 0), 5)
        cv2.circle(img, gi, 16, orange, 3)                      # the pad the pose is locked onto
        cv2.drawMarker(img, ni, (255, 255, 0), cv2.MARKER_CROSS, 26, 3)      # cyan: under the needle
        cv2.circle(img, (int(self.tip_px[0]), int(self.tip_px[1])), 5, (255, 0, 255), -1)   # where the needle appears in the image
        err = self.goal - nb[:2]
        age = time.time() - pose["t"]
        txt = f"error ({err[0]:+.1f}, {err[1]:+.1f}) = {np.linalg.norm(err):.1f} mm   needle {-nb[2]:.1f} mm above the board"
        cv2.putText(img, txt, (12, img.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5)
        cv2.putText(img, txt, (12, img.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        txt2 = f"pose: {pose['src']}  fit {pose['fit']:.2f} px  age {age * 1000:.0f} ms     orange = target pad, cyan = under the needle"
        cv2.putText(img, txt2, (12, img.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(img, txt2, (12, img.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return img

    def publish(self, img):
        """Write the annotated frame for the sim viewer (atomic replace, so the server never reads half a file)."""
        ok, buf = cv2.imencode(".jpg", cv2.resize(img, (960, 540)), [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            return
        tmp = FEED_JPG.with_suffix(".tmp")
        tmp.write_bytes(buf.tobytes())
        try:
            os.replace(tmp, FEED_JPG)
        except PermissionError:                 # the server is reading it right now; the next frame will do
            pass

    def gui(self, alive, linger_s=10.0):
        """Run on the main thread. Returns when q/Esc is pressed, or linger_s after `alive()` turns False."""
        win = "needle alignment (q to close)"
        try:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, 1280, 720)
            window = True
        except cv2.error:
            print(f"no OpenCV window (headless build): watch the feed in the sim viewer ({FEED_JPG})", flush=True)
            window = False
        ended, published = None, 0.0
        while True:
            with self._lock:
                frame, pose = self._frame, self._pose
            if frame is not None:
                img = self.draw(frame, pose)
                if window:
                    cv2.imshow(win, img)
                if time.time() - published >= FEED_PERIOD_S:
                    self.publish(img)
                    published = time.time()
            if window:
                if cv2.waitKey(20) & 0xFF in (ord("q"), 27):
                    self.stop.set()
                    break
            else:
                time.sleep(0.02)
            if not alive():
                ended = ended or time.time()
                if time.time() - ended > linger_s:
                    break
        if window:
            cv2.destroyAllWindows()

    def close(self):
        self.stop.set()
        time.sleep(0.2)
        self.cap.release()
