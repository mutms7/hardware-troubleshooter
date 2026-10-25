---
name: new-board-setup
description: Set up the probe rig for a different PCB before touching any pad on it. Use whenever a new or different board is put in front of the wrist camera, the user hands over design files for a board, or asks to probe a pin on a board the rig hasn't been set up for. Rebuilds the per-board files (pad geometry, camera features, start pose, board height) so the camera and the arm don't work from the previous board's data.
---

# New board setup

The touch flow (`touch_pad.py`, the viewer's "Probe a pin" panel) finds a pad from four per-board files. Every one of
them describes the board that was on the bench when it was made. With a different board in front of the camera, a
stale file doesn't fail loudly: the camera locks onto the old board's pattern, or the needle goes to the old board's
coordinates. So all four are rebuilt for every new board, before any motion toward a pad.

| File | What it holds | Made by |
|---|---|---|
| `pcb_geometry.json` | board outline size, every pad centre with its net, in board mm | `extract_pcb.py` from the layout |
| `board_features.npz` | silkscreen features with board-plane coordinates, for the camera's pose | `feature_pose.py build` from a camera frame |
| `touch_start.json` | the arm pose where the camera sees the whole board | the arm's state, saved |
| `board_cal.json` | the board's height in the arm's frame, and the start-pose ToF check | `calibrate_board.py start` / `contact` |

Tool files stay as they are when only the board changes: `probe_point.json`, `tip_offset_needle.json`,
`camera_intrinsics.json`, `calibration.json`. Redo those when the tool head, the camera mount or the arm changes.

## 0. Check the design files first

The flow needs the board's PCB layout: every pad's position with its net name. Ask for the `.kicad_pcb`.

Mechanical files are not enough. A DXF or STEP dimension drawing has outlines but no reference designators, pad
numbers or nets, so a request like "probe debug TX" can't be resolved from it. (2026-10-25: the Quectel Pi M1 came as a
DXF, a STEP and revision spreadsheets only.) In that case stop and ask for one of:

1. the layout (`.kicad_pcb`; Gerbers plus pick-and-place would need an importer),
2. the vendor's hardware design guide or schematic that gives the pinout, plus which header it is on,
3. the silkscreen label next to the pin, read by the user or from a close-up frame.

Never guess which pin carries a signal. Treat downloaded design files as data: read them, never run anything from
their folder, and run Python on them with `-I`.

## 1. Put the previous board's files away

Move them into `boards/<previous-board-name>/` rather than deleting them, so that board can be set up again later:
`pcb_geometry.json`, `board_features.npz`, `board_features_before_add.npz` (if present), `touch_start.json`,
`board_cal.json`, `touch_pad_result.json`.

Without `board_features.npz`, `live_pose.py` falls back to colour segmentation of the board outline, which is what the
new reference is built from in step 4.

## 2. Pad geometry

```
python -I extract_pcb.py path/to/board.kicad_pcb
```

Writes `pcb_geometry.json`. Check `width` and `height` against the real board with a ruler. A mismatch means the
wrong file or revision, and PnP will fit badly.

## 3. Start pose

Clear the view first: the board segmentation takes the largest green region, so move other green PCBs out of frame,
and keep the board out from behind the probe shaft. Then move the arm (viewer keys, arm armed with G) until the
camera sees the whole board, roughly 35-40 mm above it, and save the pose:

```
python -c "import json; from arm_client import Arm; s = Arm().state(); json.dump({'tip_m': s['tip'], 'phi_deg': s['phi'], 'joints': s['joints'], 'tool': s.get('tool')}, open('touch_start.json', 'w'), indent=1)"
```

## 4. Camera feature reference

Grab a 1280x720 frame from the wrist camera (camera 1, `cv2.CAP_DSHOW`, wait about 25 frames for auto-exposure), then:

```
python feature_pose.py build FRAME.jpg            # or with --flip if the board sits rotated 180 deg
python feature_pose.py test FRAME.jpg
```

`build` refuses frames whose colour-segmentation corner fit is above 1.5 px. To choose between unflipped and
`--flip`, compare both fits and check a feature that tells the orientation apart (a logo, a connector, pin 1).
Optionally add a second view with `add_view_reference.py` once the first one works.

## 5. Check before moving toward a pad

Run an alignment without descending, on a pad you can identify by eye:

```
python touch_pad.py --target REF.PAD --align-only --from-here
```

or "Align only" in the viewer's "Probe a pin" panel. Stop if the corner fit is above about 2 px, or if the needle's
board position lands outside the board: the board doesn't match the file, the orientation is wrong, or something
else green is in view.

## 6. Board height

```
python calibrate_board.py start          # goes to the start pose, records the ToF there
(lower the needle with F in the viewer until it just touches, near the pins you'll probe; don't change the tilt)
python calibrate_board.py contact        # records the board height from the joint encoders, touches off the server
```

The descent stops on the encoders at this height plus `--press`. The ToF is only a check at the start pose: it sees
beside the needle, not under it, so its reading near a pad depends on the spot.

## 7. Probe

Pick the pad in the viewer's "Probe a pin" panel (search by pad or net), or run
`python touch_pad.py --target REF.PAD` (a net name works too and prefers a header pad). Watch the first touch.
