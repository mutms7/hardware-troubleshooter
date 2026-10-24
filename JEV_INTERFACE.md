# Jev <-> target_tracker.py file contract

`target_tracker.py` shows the wrist-camera feed, lets a person draw the target box and the fixed
probe point, and displays whatever box Jev reports. All exchange is through JSON files.

## 1. Jev reads the starting box

When you finish drawing a box, the tool writes two files into `selection/`:

- `selection/selection.png`: the exact frame the box was drawn on.
- `selection/selection.json`:

```json
{
  "time": 1761300000.0,
  "image": "selection.png",
  "frame": {"width": 640, "height": 480},
  "bbox": [170, 220, 60, 40],
  "target_px": [200.0, 260.0],
  "target_point": "middle of the bottom edge of the box",
  "bbox_format": "x, y, width, height in pixels, origin top-left",
  "probe_px": [300.0, 250.0]
}
```

Start tracking from `selection.png` + `bbox`. Re-read the file when `time` changes (a new box was drawn).

## 2. Jev writes its tracked box, once per frame

Write a small JSON file (any path; pass it as `--jev-file`):

```json
{"bbox": [x, y, width, height], "ok": true}
```

- Same pixel space as the camera frame, origin top-left, y pointing down.
- `ok: false` (or a missing/unreadable file) counts as a miss. After 8 misses in a row the
  tool shows the target as lost.
- Write atomically (write a temp file, then rename) so the tool never reads half a file.

## 3. What the tool writes for the control loop

`tracker_state.json`, updated every frame:

```json
{
  "tracking": true,
  "fresh": true,
  "source": "jev",
  "probe_px": [300.0, 250.0],
  "target_bbox": [210.0, 130.0, 50.0, 40.0],
  "target_centre_px": [235.0, 150.0],
  "error_px": [-65.0, -100.0]
}
```

`error_px` = target centre - probe point. `error_px[0] > 0`: the target is right of the probe.
`error_px[1] > 0`: the target is below it. It is `null` unless `fresh` is true, meaning this very frame
produced the position. After a missed frame the box is held at its last good value for up to 8 frames
(`tracking` stays true), but `fresh` is false and `error_px` is `null`, so a control loop never steers on
stale data. Local modes (`--tracker homography|vit|csrt`) also add `tracker_score`: the number of matched
features for homography, or the tracking score for vit.

## Run it

```powershell
.\.venv\Scripts\python.exe target_tracker.py --camera 1 --jev-file C:\path\to\jev_out.json
```

Left-drag draws the target box, right-click sets the probe point (saved to `probe_point.json`,
fixed at that pixel), `c` clears the target, `q` or Esc quits.
