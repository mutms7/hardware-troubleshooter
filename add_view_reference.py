"""Add a second feature reference from the CURRENT view (1280x720), so the silkscreen-feature solver works from here too.

The new view's board pose comes from the existing feature reference (so it is as good as that one, not from the 4-corner colour
fit, which is weak at near-vertical views). Its keypoints are then given board-plane coordinates from that pose and appended
to board_features.npz (the previous file is kept as board_features_before_add.npz).
    python add_view_reference.py
"""
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from feature_pose import NPZ, BoardFeatures
from pnp_test import load_intrinsics

HERE = Path(__file__).resolve().parent
K, dist, _ = load_intrinsics(HERE / "camera_intrinsics.json")
K = K.copy()
K[0] *= 2.0 / 3.0
K[1] *= 2.0 / 3.0
geo = json.loads((HERE / "pcb_geometry.json").read_text(encoding="utf-8-sig"))

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
for _ in range(25):
    cap.read()
best = None
bf = BoardFeatures.load()
for _ in range(15):                                       # keep the frame the existing reference solves best
    ok, frame = cap.read()
    if not ok:
        continue
    r = bf.solve(frame, K, dist, min_inliers=15)
    if r is not None and r[0] <= 1.0 and (best is None or r[3] > best[0][3]):
        best = (r, frame.copy())
cap.release()
if best is None:
    sys.exit("the existing features don't solve this view well enough (need >= 15 inliers, rms <= 1 px); not adding")
(rms, rvec, tvec, n), frame = best
R = cv2.Rodrigues(rvec)[0]
cam = -R.T @ tvec.ravel()
print(f"pose of this view from the old features: {n} inliers, rms {rms:.2f} px, camera {abs(cam[2]):.1f} mm up, "
      f"tilt {np.degrees(np.arccos(min(1, abs(R[2, 2])))):.1f} deg")
new = BoardFeatures.build(frame, rvec, tvec, K, dist, geo["width"], geo["height"])
print(f"new view gives {len(new.pts3d)} board features (the old reference has {len(bf.pts3d)})")
shutil.copy(NPZ, HERE / "board_features_before_add.npz")
merged = BoardFeatures(np.vstack([bf.desc, new.desc]), np.vstack([bf.pts3d, new.pts3d]))
merged.save()
print(f"saved {len(merged.pts3d)} features to {NPZ.name}")
