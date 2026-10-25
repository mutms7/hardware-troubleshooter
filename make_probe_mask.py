"""Mask the probe out of the wrist camera's picture, once. The probe is fixed to the camera, so it covers the same pixels in
every frame; live_pose.py drops them before solving the board pose (no silkscreen features on the shaft, and the shaft can't
join or cut the board's colour outline).

    python make_probe_mask.py            writes probe_mask.png (255 = probe) at 1280x720, and probe_mask_check.jpg to look at

The probe is the dark region connected to the needle tip (probe_point.json) and reaching the top of the frame. Re-run it
whenever the camera or the tool head is moved.
"""
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
W, H = 1280, 720
DARK_V = 70          # the shaft is black: HSV value under this
GROW_PX = 5          # widen the mask a little so the shaft's edge and shadow go too
BAND_PX = 90         # the shaft is searched for only in a vertical band this wide either side of the tip, above it

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
for _ in range(30):                      # let auto-exposure settle
    ok, frame = cap.read()
cap.release()
if not ok:
    raise SystemExit("no frame from camera 1")

tp = json.loads((HERE / "probe_point.json").read_text(encoding="utf-8-sig"))
tip = (int(tp["x"] * W / tp["width"]), int(tp["y"] * H / tp["height"]))

dark = (cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2] < DARK_V).astype(np.uint8)
band = np.zeros_like(dark)               # without this, cables and dark board edges touching the shaft join it
band[: tip[1] + 6, max(0, tip[0] - BAND_PX): tip[0] + BAND_PX] = 1
dark &= band
dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
n, labels = cv2.connectedComponents(dark)
# the component nearest the tip that also touches the top edge
best, best_d = 0, 1e9
for k in range(1, n):
    ys, xs = np.nonzero(labels == k)
    if ys.min() > 2:
        continue
    d = float(np.min(np.hypot(xs - tip[0], ys - tip[1])))
    if d < best_d:
        best, best_d = k, d
if best == 0 or best_d > 40:
    raise SystemExit(f"couldn't find the probe: no dark region from the top edge reaches near the tip pixel {tip}")
mask = (labels == best).astype(np.uint8) * 255
# Only the probe itself: an opening wider than a cable (about 8 px) but narrower than the shaft (about 60 px) removes the
# cable and anything thin touching the shaft.
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
n2, lab2, stats, _ = cv2.connectedComponentsWithStats(mask)       # the opening can leave dark specks: keep the shaft only
mask = ((lab2 == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))) * 255).astype(np.uint8)
mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * GROW_PX + 1, 2 * GROW_PX + 1)))
cv2.imwrite(str(HERE / "probe_mask.png"), mask)

check = frame.copy()
check[mask > 0] = (0.4 * check[mask > 0] + (0, 0, 150)).astype(np.uint8)
cv2.circle(check, tip, 5, (255, 0, 255), -1)
cv2.imwrite(str(HERE / "probe_mask_check.jpg"), check)
print(f"probe mask: {int((mask > 0).sum())} px ({100 * (mask > 0).mean():.1f} % of the frame), tip at {tip}")
