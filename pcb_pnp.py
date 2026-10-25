"""PnP from the PCB itself: segment the green board from the brown background, fit its 4 corners,
solve the pose against the known 55 x 74 mm outline (pcb_geometry.json from the KiCad file), and draw
where every header pad is. Click anywhere to read the board coordinates (mm) and the nearest pad.

  left-click  board coordinates under the cursor + nearest pad
  f           flip the board 180 deg (a rectangle can't tell; this picks the first orientation, then it stays continuous)
  m           show / hide the green mask
  q / Esc     quit
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pnp_test import load_intrinsics, to_board

HERE = Path(__file__).resolve().parent
# This camera renders the board blue-green (OpenCV hue ~95-104, saturation high) and the wood nearly grey (saturation < 45).
HSV_LO, HSV_HI = (40, 70, 40), (115, 255, 255)


def segment(frame, lo, hi):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)          # fills silkscreen, pads and holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask


def board_corners(mask, min_area, merge=False):
    """merge: the board may be cut in pieces (the probe masked out across it), so fit around every big piece together."""
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None, None
    c = max(cs, key=cv2.contourArea)
    if merge:
        big = [x for x in cs if cv2.contourArea(x) >= 0.15 * cv2.contourArea(c)]
        c = np.vstack(big) if cv2.contourArea(c) >= min_area else c
    if cv2.contourArea(c) < min_area:
        return None, None
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in np.linspace(0.01, 0.08, 15):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            break
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
    ctr = quad.mean(axis=0)
    quad = quad[np.argsort(np.arctan2(quad[:, 1] - ctr[1], quad[:, 0] - ctr[0]))]      # clockwise on screen
    quad = np.roll(quad, -int(np.argmin(quad.sum(axis=1))), axis=0)                    # start at the top-left-most
    return quad, c


def solve(quad, obj, K, dist, prev_R, flip, pick=None):
    cands = []
    for k in range(4):
        o = np.roll(obj, k, axis=0)
        ok, rvec, tvec = cv2.solvePnP(o, quad, K, dist, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            continue
        rvec, tvec = cv2.solvePnPRefineLM(o, quad, K, dist, rvec, tvec)
        rep = cv2.projectPoints(o, rvec, tvec, K, dist)[0].reshape(-1, 2)
        cands.append((float(np.sqrt(np.mean(np.sum((rep - quad) ** 2, axis=1)))), k, rvec, tvec))
    if not cands:
        return None
    cands.sort(key=lambda c: c[0])
    pair = cands[:2]                                             # the two 180-degree-apart readings fit equally well
    if pick is not None:                                         # something on the board tells the two ends apart
        best = max(pair, key=pick)
    elif prev_R is not None:
        ang = lambda c: np.arccos(np.clip((np.trace(prev_R.T @ cv2.Rodrigues(c[2])[0]) - 1) / 2, -1, 1))
        best = min(pair, key=ang)
    else:
        best = sorted(pair, key=lambda c: c[1])[1 if flip else 0]
    return best


HSV_CANDIDATES = [((35, 50, 30), (95, 255, 255)), ((40, 70, 40), (115, 255, 255)),
                  ((38, 60, 30), (105, 255, 255)), ((45, 100, 40), (110, 255, 255))]


def anchor_score(frame, K, dist, anchor):
    """How much the board point anchor = {"x", "y"} (mm) looks like a bright, unsaturated part (a metal shield) in this frame,
    for a candidate pose c = (rms, k, rvec, tvec). Used to choose between the two 180-degree readings of a rectangle."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]

    def score(c):
        p = cv2.projectPoints(np.array([[anchor["x"], anchor["y"], 0.0]], np.float32), c[2], c[3], K, dist)[0].reshape(2)
        x, y = int(round(p[0])), int(round(p[1]))
        r = int(anchor.get("r_px", 6))
        if not (r <= x < w - r and r <= y < h - r):
            return -1e9
        patch = hsv[y - r:y + r + 1, x - r:x + r + 1].reshape(-1, 3).astype(float)
        return float(np.mean(patch[:, 2]) - np.mean(patch[:, 1]))      # bright minus saturated

    return score


def solve_frame(frame, obj, K, dist, prev_R, flip, roi=None, anchor=None, ignore=None):
    """Try several colour ranges and keep the corner fit that matches the known rectangle best.
    A wrong quad (glare cutting the board, wood joining it) fits the rectangle badly, so the fit is its own check.
    roi: (x0, y0, x1, y1) as fractions of the frame to search; default skips the old bench's edge stickers.
    anchor: a bright, unsaturated board point {"x", "y"} (mm) that picks between the two 180-degree readings.
    Returns ((rms, k, rvec, tvec), quad, mask) or None."""
    best = None
    h, w = frame.shape[:2]
    fx0, fy0, fx1, fy1 = roi or (0.15, 0.05, 0.85, 0.85)                            # the bench's green stickers sit at the edges
    x0, y0, x1, y1 = int(fx0 * w), int(fy0 * h), int(fx1 * w), int(fy1 * h)
    for lo, hi in HSV_CANDIDATES:
        mask = segment(frame, lo, hi)
        inside = np.zeros_like(mask)
        inside[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
        mask = inside
        if ignore is not None:                                   # the probe: never part of the board
            mask[ignore > 0] = 0
        quad, _ = board_corners(mask, 0.005 * h * w, merge=ignore is not None)
        if quad is None:
            continue
        r = solve(quad, obj, K, dist, prev_R, flip, anchor_score(frame, K, dist, anchor) if anchor else None)
        if r is not None and (best is None or r[0] < best[0][0]):
            best = (r, quad, mask)
    return best


def find_pad(geo, spec):
    """'J3.4' -> that pad; anything else is taken as a net name (e.g. 'SDA3')."""
    if "." in spec:
        ref, num = spec.split(".", 1)
        return next((p for p in geo["pads"] if p["ref"] == ref and p["pad"] == num), None)
    same_net = [p for p in geo["pads"] if p["net"] == spec]
    return next((p for p in same_net if p["ref"].startswith("J")), same_net[0] if same_net else None)   # the header pad, not the ESP32 pin on the same net


def draw_arrow(img, K, dist, rvec, tvec, pad, tip_px):
    """Arrow from the probe tip's pixel to the target pad, with the distance on the board plane.
    The distance is exact only when the tip is at the board's height (otherwise it is the tip's line of sight)."""
    tgt = cv2.projectPoints(np.array([[pad["x"], pad["y"], 0.0]], np.float32), rvec, tvec, K, dist)[0].reshape(2)
    a, b = (int(tip_px[0]), int(tip_px[1])), (int(tgt[0]), int(tgt[1]))
    orange = (0, 140, 255)
    cv2.arrowedLine(img, a, b, (0, 0, 0), 8, tipLength=0.15)
    cv2.arrowedLine(img, a, b, orange, 4, tipLength=0.15)
    cv2.circle(img, b, 18, (0, 0, 0), 5)
    cv2.circle(img, b, 18, orange, 3)
    cv2.circle(img, a, 7, (255, 0, 255), -1)
    under, _ = to_board(tip_px, K, dist, rvec, tvec)
    d = np.array([pad["x"] - under[0], pad["y"] - under[1]])
    label = f"{pad['ref']}.{pad['pad']} {pad['net']}  {np.linalg.norm(d):.1f} mm away"
    cv2.putText(img, label, (b[0] + 24, b[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
    cv2.putText(img, label, (b[0] + 24, b[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, orange, 2)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=1)
    ap.add_argument("--intrinsics", default=str(HERE / "camera_intrinsics.json"))
    ap.add_argument("--geometry", default=str(HERE / "pcb_geometry.json"))
    ap.add_argument("--target", default="SDA3", help="pad to aim at, as REF.PAD (J3.4) or a net name (SDA3)")
    ap.add_argument("--probe-file", default=str(HERE / "probe_point.json"), help="the tip's pixel in the image")
    ap.add_argument("--hsv-lo", type=int, nargs=3, default=HSV_LO, help="lower H S V of the green board")
    ap.add_argument("--hsv-hi", type=int, nargs=3, default=HSV_HI)
    args = ap.parse_args()

    K, dist, (cw, ch) = load_intrinsics(args.intrinsics)
    geo = json.loads(Path(args.geometry).read_text(encoding="utf-8-sig"))
    W, H = geo["width"], geo["height"]
    obj = np.array([[0, 0, 0], [W, 0, 0], [W, H, 0], [0, H, 0]], np.float32)
    jpads = [p for p in geo["pads"] if p["ref"].startswith("J")]
    jxyz = np.array([[p["x"], p["y"], 0.0] for p in jpads], np.float32)
    target = find_pad(geo, args.target)
    if target is None:
        raise SystemExit(f"No pad or net called {args.target!r} in {args.geometry}.")
    tp = json.loads(Path(args.probe_file).read_text(encoding="utf-8-sig"))
    tip_px = (tp["x"] * cw / tp.get("width", cw), tp["y"] * ch / tp.get("height", ch))

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cw)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ch)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"Could not read camera {args.camera}.")
    h, w = frame.shape[:2]
    if (w, h) != (cw, ch):
        K = K.copy()
        K[0] *= w / cw
        K[1] *= h / ch
        print(f"WARNING: camera gives {w}x{h}, calibration is {cw}x{ch}; intrinsics scaled.")
    print(f"camera {args.camera}: {w}x{h}. Board {W} x {H} mm, {len(jpads)} header pads.")

    st = {"click": None}
    state = {"flip": False, "prev_R": None, "mask": False}
    cv2.namedWindow("pcb pnp", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("pcb pnp", lambda e, x, y, f, _: st.update(click=(x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)
    pose = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        mask = segment(frame, args.hsv_lo, args.hsv_hi)
        quad, contour = board_corners(mask, 0.01 * w * h)
        shown = frame.copy()
        info = "no green board found"
        pose = None
        if quad is not None:
            best = solve(quad, obj, K, dist, state["prev_R"], state["flip"])
            if best is not None:
                rms, k, rvec, tvec = best
                R = cv2.Rodrigues(rvec)[0]
                state["prev_R"] = R
                pose = (rvec, tvec)
                cam = -R.T @ tvec.reshape(3)
                tilt = np.degrees(np.arccos(min(1.0, abs(R[2, 2]))))
                info = f"camera {abs(cam[2]):.1f} mm above board   tilt {tilt:.1f} deg   corner fit {rms:.2f} px"
                outline = cv2.projectPoints(obj, rvec, tvec, K, dist)[0].reshape(-1, 2).astype(np.int32)
                cv2.polylines(shown, [outline], True, (0, 255, 0), 2)
                for c in quad:
                    cv2.circle(shown, (int(c[0]), int(c[1])), 7, (0, 0, 255), -1)
                proj = cv2.projectPoints(jxyz, rvec, tvec, K, dist)[0].reshape(-1, 2)
                for p, pad in zip(proj, jpads):
                    cv2.circle(shown, (int(p[0]), int(p[1])), 4, (0, 220, 255), -1)
                    if pad["pad"] == "1":
                        cv2.putText(shown, pad["ref"], (int(p[0]) - 30, int(p[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
                draw_arrow(shown, K, dist, rvec, tvec, target, tip_px)
                if st["click"]:
                    b, _ = to_board(st["click"], K, dist, rvec, tvec)
                    near = min(geo["pads"], key=lambda p: (p["x"] - b[0]) ** 2 + (p["y"] - b[1]) ** 2)
                    d = float(np.hypot(near["x"] - b[0], near["y"] - b[1]))
                    print(f"click {st['click']}: board ({b[0]:.2f}, {b[1]:.2f}) mm; nearest pad {near['ref']}.{near['pad']} "
                          f"[{near['net']}] {d:.2f} mm away")
        st["click"] = None
        if state["mask"]:
            small = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (w // 4, h // 4))
            shown[h - h // 4:, :w // 4] = small
        for i, line in enumerate((info, "click: board coords   f: flip 180   m: mask   q: quit")):
            cv2.putText(shown, line, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(shown, line, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.imshow("pcb pnp", shown)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("f"):
            state["flip"] = not state["flip"]
            state["prev_R"] = None
        if key == ord("m"):
            state["mask"] = not state["mask"]
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
