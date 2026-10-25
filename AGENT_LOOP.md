# Agent loop: debug a board with the probe arm

The debugging LLM asks for a measurement. The arm finds the real board, solves its pose with PnP, goes to the
pad, and probes it. Judgments come from the LLM; coordinates come from the KiCad file and PnP; motion comes from code.

## The loop

| # | Stage | Who decides | Code | Status |
|---|---|---|---|---|
| 1 | The debugging LLM names a board and a target (a net or pad, e.g. `SDA3`) | LLM | (call into the loop) | not built |
| 2 | Load the board's geometry from its KiCad file | code | `extract_pcb.py` -> `pcb_geometry.json` | done |
| 3 | Survey pose: raise the camera so the table is in view and look for the board | code | | not built |
| 4 | Match the real board to the file: outline size, features, which way round (the 180 deg flip) | LLM looks, code checks the fit | `pcb_snapshot.py`, `pcb_pnp.py` | working; flip is manual |
| 5 | Segment the board (green on brown) and fit its 4 corners | LLM finds the region, colour contrast gives the corners | `pcb_pnp.py: segment, board_corners` | working |
| 6 | Solve PnP, first pass, from far away | code | `pcb_pnp.py: solve` (intrinsics: `camera_intrinsics.json`) | working |
| 7 | Choose the probe point: net -> pad -> board coordinates | LLM picks the net, KiCad gives the point | `pcb_geometry.json` | partly built |
| 8 | Move closer with the board kept in view, solve PnP again | code | | not built |
| 9 | Target in the arm's world frame, from PnP plus the camera-to-tip offset | code | pivot calibration, see below | not built |
| 10 | Navigate while PnP keeps tracking, so the target isn't lost | code | `servo.py`, `arm_client.py` | not wired to PnP |
| 11 | Descend, detect contact, hand the result back to the LLM | code | `descend.py` | untested on this setup |

Rules that keep it honest:
- A target position never comes from a model. The LLM names the pad; the file and PnP give the coordinates.
- Two solves: a coarse one from far (the board is small, about 4.6 px/mm) and a finer one from close.
- If the corner fit is above about 2 px, stop. The board doesn't match the file, or the orientation is wrong.
- Nothing moves the arm until the server is armed, and the first moves are small and slow.

## Run log

2026-10-24 (this run)
- Arm server restarted from the worktree. Two blockers found and fixed: `calibration.json` had a byte-order mark
  (written by PowerShell) that the server's loader rejected; `COM10` was held by ServoStudio, which had to be closed.
  The camera was also held by another program until the user closed it. Camera 1 needs the DirectShow backend
  (`cv2.CAP_DSHOW`): the default backend takes about 90 s to open 1080p.
- Turret positions taught: probe 1 = 1177, probe 2 = 2530, probe 3 = 3900 (`calibration.json`). The server started
  with probe 3 as the nearest slot.
- Camera intrinsics (1920 x 1080): fx 676.7, fy 676.6, cx 990.1, cy 547.3, 0.83 px RMS over 18 images
  (PR #8, `camera_calibration.yaml`).
- **First board on the table did not match the KiCad file** (about 100 x 78 mm, three driver sockets, logo at the
  bottom right, PnP corner fit 21 px). The file is the 55.0 x 74.0 mm "Controller" layout: ESP32-S3 footprint,
  headers J1 to J6, logo at the bottom left. Nothing was moved.
- **The correct board was then put on the table.** Corner fit 0.94 px unflipped, 0.70 px with the 180 deg flip.
  The flip is the right reading: the ESP32 pin columns sit at 21 % and 63 % of the width measured from the right,
  and the logo is at the top right. Camera height 133.9 mm above the board, tilted 25 deg from the board normal.
  Tip pixel in the image (probe_point.json): (894, 696) at 1920 x 1080.

- PnP is stable once the camera's auto-exposure has settled (wait about 25 frames after opening it): corner fit
  0.2 to 1 px, camera height repeatable to about 1 mm. The first frames after opening give fits of 5 to 90 px.
- Probe tip pixel: the saved (894, 696) was for another probe. The needle tip now in the working position (turret
  slot 3) is at (1022, 447) at 1920 x 1080 (`probe_point.json`). It must be re-marked whenever the turret changes.
- The net `SDA3` is on two pads: ESP32 pin U1.12 and header pad J3.4 (board 11.0, 38.5 mm). The target is the
  header pad. Arrow overlay (tip pixel -> target pad, distance on the board plane) is in `pcb_pnp.py` / `pcb_snapshot.py`.
- Pivot calibration (`pivot_cal.py`, tilts of -2..-8 deg; tilting up was refused or lost the board): tip at
  (4.7, -14.8, 99.6) mm in the camera frame, residual 1.5 mm, but see the problem below.
- `go_to_pad.py` (PnP as the only sensor): measured the jog -> board-frame matrix and moved the tip 18 mm of the 75 mm
  to the target, then stopped being useful: the server's step governor kept shortening the moves because its own
  model says the tip is 6 to 10 mm above the board.
- **Problem found.** The server's tip is a virtual point from the URDF, not the probe's needle. Evidence: after a
  5 mm "in" retreat PnP says the camera moved DOWN 3.9 mm while the encoders say the tip moved UP 1.4 mm (the
  real arm also lags its command by about half, from friction and sag); PnP puts the tip 29 to 36 mm above the board
  while the model says about 15 mm. The pivot method holds the model's tip still, so it measured the model's tip
  point, not the needle. The needle's depth along its pixel ray is still unknown.
- State left behind: the arm is armed and holding, about 15 mm above the PCB by the encoders, no contact.

- Hand-guided touch 1 (user on the bottom-right GND pin): needle depth Z = 124.5 mm along the pixel ray (the pivot guess
  of 99.6 mm was 25 mm short); the computed needle position (42.8, 61.8) mm is 0.76 mm from pad U1.23 (43.5, 61.5).
- Hand-guided touch 2 (user on pad U1.2, +3.3V): Z = 126.7 mm; computed needle position (20.4, 11.1) mm is 0.47 mm from
  U1.2 (20.64, 10.7). Two touches: Z = 125.6 mm, spread 1.1 mm; needle in the camera frame (5.9, -18.6, 125.6) mm,
  saved in `tip_offset_needle.json` (supersedes the pivot guess in `tip_offset.json`).
- **Safety finding.** With the needle on the board the server's model tip reads z -38 mm, about 16 mm ABOVE the board
  surface: the needle is lower than the model tip. So the server's floor does not protect the needle, and the earlier
  "15 mm above the board" readings were the needle on the board (the user saw the probe on the ground). The server is
  now started with `--floor-offset-mm -20` (floors raised 20 mm) until the needle is modelled as the tip.
- IK: the solver now weights tilt and sideways lean (W_TILT = 3 per degree) and prefers the joints nearest the current
  ones (REG = 0.01). In simulation, straight down holds to 0.01 deg and 0.02 mm over 147 targets. New `point_down`
  command (`Arm.point_down()`): turns the probe to tilt 0 with the tip held, in safe increments (about 20 to 40 deg of
  joint travel for a 13 to 21 deg tilt, since holding the tip swings the elbow and wrist more than the tilt itself).
  Not yet run on the real arm; the server must be restarted to load it.

## Next: needle depth (stage 9)

Plan: one slow, supervised touch-off. With the needle over a bare part of the PCB, descend in 0.25 mm steps until
contact (following-error stop, `descend.py`). At contact the needle is on the board plane, so its depth along the
pixel ray is Z = (R^T t)_z / (R^T r)_z from the PnP pose at that moment. That gives the needle's position in the
camera frame exactly, with no model in between. Then redo `go_to_pad.py` with that offset.

## Earlier plan: camera-to-tip offset (stage 9)

The tip is a fixed point in the camera frame, along the ray through the tip pixel. Unknown: the distance L along
that ray. Method: pivot calibration. Tilt the wrist with the tip held fixed, solve PnP at several tilts, and find the
L (and board point) for which the tip stays at one place in the board frame. No contact is needed. The tilt axis is
horizontal and the tip pixel ray is not parallel to it, so L is observable.
