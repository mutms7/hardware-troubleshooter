"""Find the individual pins inside a region with no words and no positions: small bright spots that
form an evenly spaced straight row. Works at any rotation."""
import itertools

import cv2
import numpy as np


def find_pins(img, box, expect=3):
    """box = (x0, y0, x1, y1) in pixels. Returns [(x, y, area)] ordered along the row's own axis."""
    x0, y0, x1, y1 = (int(v) for v in box)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    roi = g[y0:y1, x0:x1].astype(np.float32)
    # white top-hat: keeps bright features smaller than the kernel, drops big bright areas (tape, glare)
    th = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    n, lab, stats, cent = cv2.connectedComponentsWithStats((th > 0.6 * th.max()).astype(np.uint8), connectivity=8)
    cands = []
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if 12 <= area <= 400:
            ys, xs = np.nonzero(lab == i)
            w = th[ys, xs]
            cands.append((float((xs * w).sum() / w.sum()) + x0, float((ys * w).sum() / w.sum()) + y0, area))
    if len(cands) < expect:
        return []
    # pins in a header are in a straight row with equal spacing: pick the triple that is most like that
    best, best_score = None, 1e9
    for tri in itertools.combinations(range(len(cands)), expect):
        p = np.array([[cands[i][0], cands[i][1]] for i in tri])
        c = p.mean(axis=0)
        axis = np.linalg.svd(p - c)[2][0]
        t = (p - c) @ axis
        perp = np.abs((p - c) @ np.array([-axis[1], axis[0]]))
        order = np.argsort(t)
        gaps = np.diff(t[order])
        areas = np.array([cands[i][2] for i in tri])
        score = perp.max() + 0.5 * np.abs(gaps - gaps.mean()).max() + 5.0 * (areas.max() / areas.min() - 1)
        if score < best_score:
            best, best_score = [tri[j] for j in order], score
    return [cands[i] for i in best]


if __name__ == "__main__":
    img = cv2.imread("bldc_reference_new.png")
    H, W = img.shape[:2]
    jev_box = (0.50 * W, 0.46 * H, 0.63 * W, 0.60 * H)          # exactly what Jev returned for the connector
    truth = {"A": (999.6, 588.8), "B": (1010.1, 600.3), "C": (1019.3, 613.1)}   # my hand-found centres
    pins = find_pins(img, jev_box)
    print(f"found {len(pins)} pin(s) inside Jev's box {tuple(round(v) for v in jev_box)}, in order along the row:")
    for k, (x, y, a) in enumerate(pins, 1):
        name, (tx, ty) = min(truth.items(), key=lambda kv: np.hypot(kv[1][0] - x, kv[1][1] - y))
        print(f"  pin {k}: ({x:.1f}, {y:.1f}) area {a:.0f} px   matches my hand-found {name} ({tx}, {ty}): {np.hypot(tx - x, ty - y):.2f} px apart")
    print("the same code with the whole picture rotated, and only a loose box around the row:")
    for ang in (0, 35, 100, 200, 300):
        M = cv2.getRotationMatrix2D((1010, 600), ang, 1.0)
        rot = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC)
        c = np.array([[v[0], v[1], 1.0] for v in truth.values()]) @ M.T
        b = (c[:, 0].min() - 70, c[:, 1].min() - 70, c[:, 0].max() + 70, c[:, 1].max() + 70)
        pr = find_pins(rot, b)
        err = [min(np.hypot(p[0] - q[0], p[1] - q[1]) for q in c) for p in pr]
        print(f"  rotated {ang:3d} deg: found {len(pr)} pins, worst distance to the true centre {max(err) if err else float('nan'):.2f} px")
