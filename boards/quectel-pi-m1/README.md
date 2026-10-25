# Quectel Pi M1: golden configuration (2026-10-25)

The configuration that touched DBG_TX (H3.1) four times in a row, landing 0.43-1.31 mm from the pin centre.
To restore it, copy these files back to the repo root (they are the working copies `touch_pad.py` and `live_pose.py` read).

| File | What |
|---|---|
| `pcb_geometry.json` | outline 56 x 85 mm (underside up, X = 56 - y_step, Y = x_step), H3's three pins (TX = net `DBG_TX`), flip off, full-frame search, module shield as the 180 deg anchor |
| `board_features.npz` | 365 silkscreen features, built from a 0.75 px frame at the start pose |
| `touch_start.json` | start pose: probe tilt -15.8 deg, board clear of the shaft |
| `board_cal.json` | board height z = -17.7 mm (hand-guided touch on DBG_TX), start-pose ToF 119 mm |
| `tip_offset_needle.json` | needle in the camera frame, recalibrated from the DBG_TX touch (5.1 mm deeper than before) |
| `probe_mask.png` | the probe shaft's pixels (1280x720), from `make_probe_mask.py` |
| `probe_point.json`, `camera_intrinsics.json`, `calibration.json` | tool tip pixel, camera intrinsics, arm joint calibration as used |

Rig as run: arm server `sim_server.py --hardware COM6 --sensors COM5 --floor-offset-mm 5`; servo gains P 24 / I 0 (elbow P 32),
D 32; turret on the needle (slot 3). Command: `python touch_pad.py --target DBG_TX --tries 12`.
