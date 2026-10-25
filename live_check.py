"""How steady is the pose with the camera NOT moving? Collects ~10 s of poses and compares raw vs filtered needle positions.
    python live_check.py"""
import time
from collections import Counter

import numpy as np

from live_pose import LivePose

lp = LivePose()
time.sleep(4.0)                                   # let the solvers and the exposure settle
raw, filt, srcs, seen, t_end = [], [], [], set(), time.time() + 10.0
while time.time() < t_end:
    p = lp.latest()
    if p is not None and p["t"] not in seen:
        seen.add(p["t"])
        raw.append(p["raw_needle_board"])
        filt.append(p["needle_board"])
        srcs.append(p["raw_src"].split(" (")[0])
    time.sleep(0.02)
lp.close()
raw, filt = np.array(raw), np.array(filt)
print(f"{len(raw)} poses in 10 s ({len(raw) / 10:.1f} per second); solver: {dict(Counter(srcs))}")
for name, a in (("raw     ", raw), ("filtered", filt)):
    print(f"{name}: needle x std {a[:, 0].std():.2f} mm, y std {a[:, 1].std():.2f} mm, height std {a[:, 2].std():.2f} mm; "
          f"peak-to-peak x {np.ptp(a[:, 0]):.2f}, y {np.ptp(a[:, 1]):.2f} mm")
