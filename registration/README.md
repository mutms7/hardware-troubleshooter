# Board registration (CV)

Board-edge registration of a clamped PCB from the rig's fixed camera, not the camera
mounted on the arm. It's a plain Python library (numpy, OpenCV, PyYAML) with no
middleware. There's no AprilTag, motion, measurement, or MCP code here, and no
electrical identity is inferred from images.

## Method

1. Otsu-segment the board and take its outer contour (`rig.segmentation.board_is_brighter`).
2. Undistort the contour with the camera_info model (plumb_bob, rational_polynomial,
   or equidistant; anything else is rejected).
3. Split the contour into the model's N straight edges, trim the corners, and
   Huber-fit a line per edge. Neighbouring lines intersect at sub-pixel corners.
   Rounded KiCad corners give virtual corners the same way.
4. Intersect the corner rays with the board's top surface: the calibrated bench plane
   lifted by `board.thickness_m`. Fit an in-plane rigid pose (x, y, theta), not 6-DOF.
5. Each cyclic corner correspondence is an orientation hypothesis. Hypotheses above
   `max_fit_residual_px` are dropped. The configured asymmetric feature must pass
   `feature_min_contrast` in exactly one remaining hypothesis. Otherwise the frame is
   `symmetry_ambiguous` and nothing is guessed.
6. `SOLVEPNP_IPPE` runs on the same corners as a diagnostic cross-check only.

`reprojection_error_px` is the RMS distance from the fitted edge points to the model
outline projected at the solved pose.

## Using the engine

`RegistrationEngine` holds the state. The caller owns the camera, the clock and the
transport:

```python
from hardware_troubleshooter_registration.camera import CameraModel
from hardware_troubleshooter_registration.config import load_board_config, load_rig_config
from hardware_troubleshooter_registration.engine import RegistrationEngine

board, board_problems = load_board_config("board.yaml")
rig, rig_problems = load_rig_config("rig.yaml")
camera, camera_problem = CameraModel.from_calibration_yaml(camera_info_dict)
engine = RegistrationEngine(board, board_problems, rig, rig_problems, session_id)
engine.set_extrinsic((translation_xyz, quaternion_xyzw), None, now)

engine.process_frame(image, stamp, camera_frame, camera, camera_problem)
engine.tick(now)  # call periodically so a stalled camera times out
state = engine.snapshot(now)  # current Snapshot
changes = engine.drain_events()  # Snapshots queued on state changes
pose = engine.board_transform(now)  # (camera_frame, board_frame, R, t) or None
```

| Output | Notes |
| --- | --- |
| `snapshot(now)` | Current `Snapshot`: status, validity flags, pose, `failure_reason`, diagnostics. |
| `drain_events()` | Snapshots queued whenever the reported state changes. |
| `board_transform(now)` | Camera -> board pose, only while `status=valid`. |
| `overlay.draw_overlay(...)` | Debug drawing: outline, corners, +x arrow, feature, targets. Not verification. |

Pass each image with the camera model calibrated for it. `stamp` and `now` must use
the same clock.

## Semantics

- A revision's pose is fixed when it's created. Later frames confirm it (within the
  move thresholds) or invalidate it. An invalidated revision is never reported valid
  again; the next accepted frame creates a new one. `registration_id` is
  `<board_id>/<session_id>/r<N>`, so IDs don't repeat across restarts.
- `stamp` is the latest frame that confirmed the revision. The frame that created
  the pose is `registration_stamp`.
- `rotation`/`translation` are the board pose in the camera optical frame
  (`camera_frame`). `has_pose` is false once the revision is invalidated.
- `tracking_valid` needs the latest frame to confirm the revision within
  `tracking_loss_timeout_s`. A failed frame inside the timeout makes tracking invalid
  (`unconfirmed`) without invalidating the revision. Past the timeout it is
  `tracking_lost`. `board_unmoved` equals `tracking_valid`.
- `calibration_valid` needs valid camera_info, the bench plane, `base_frame`, and an
  extrinsic `base_frame <- camera` from `set_extrinsic()`. A change in intrinsics or
  extrinsic invalidates the revision with `calibration_changed`.
  `calibration_revision` hashes those inputs.
- `valid_until` is the last confirmation plus the timeout, or None.
- Board height isn't estimated. Board height for the approach belongs to the motion
  adapter's calibration.
- `failure_reason` is `code` or `code:detail` entries joined by `; `. After an
  invalidation the first entry is `board_moved`, `tracking_lost` or
  `calibration_changed`. Other codes: `unconfigured:<key>`, `invalid:<key>`,
  `camera_info_missing`, `camera_info_invalid`, `extrinsic_unavailable`,
  `no_camera_frames`, `no_board_segment`, `board_at_image_border`, `outline_mismatch`,
  `fit_failed`, `residual_above_limit`, `symmetry_ambiguous`, `unconfirmed`,
  `stale_observation`.
- `status=unconfigured` while any required config value is unset.

## Configuration

`config/board.example.yaml` and `config/rig.example.yaml` are entirely unset. Each value
needs a real measurement or calibration before live use:

| Key | Needed for |
| --- | --- |
| `board.board_id`, `board.thickness_m` | identity; top-surface plane |
| `board.outline_source` + `kicad_pcb` or `outline_m` | outline model |
| `board.features` | symmetry breaking |
| `rig.base_frame` | calibration |
| `rig.bench_plane.normal`, `distance_m` | in-plane pose, calibration |
| `rig.segmentation.board_is_brighter` | segmentation |
| `rig.thresholds.max_fit_residual_px` | acceptance |
| `rig.thresholds.move_translation_m`, `move_rotation_rad` | `board_moved` |
| `rig.thresholds.tracking_loss_timeout_s` | `tracking_lost`, `valid_until` |
| `rig.thresholds.feature_min_contrast` | symmetry decision |
| camera_info (K, D) | intrinsic calibration, as a camera_info-style YAML |
| `rig.camera_extrinsic` | hand-eye calibration, passed to `set_extrinsic()` by the caller |

## Synthetic data

A synthetic sequence (static, moved, covered) with matching test-only configs:

```bash
python -m hardware_troubleshooter_registration.make_synthetic_dataset /tmp/reg_synth
```

Run from this folder. It writes `frames/`, `camera_info.yaml`, `board.yaml` and
`rig.yaml`. Every value describes an imaginary rig.

## Tests

```bash
python3 -m pytest -q registration/test
```

The core tests need only numpy, OpenCV, PyYAML and pytest. They render synthetic
boards through a known distorted camera and check: recovered pose against ground
truth, the ambiguous (hidden or mirrored feature) case, `board_moved` on a pose jump,
`tracking_lost` on occlusion, removal and dropped frames, `calibration_changed`, and
that missing calibration gives `calibration_valid=false` with no board transform.
The tolerances are test-only and say nothing about accuracy on a real rig.

## Known limits

- Fixed camera only. The rig also has a camera on the arm, and feeding its frames to
  the engine fails closed: each arm move changes `base_frame <- camera` and invalidates
  the revision with `calibration_changed`. Registering from the arm camera would need
  the bench plane and the revision pose kept in the base frame, with the extrinsic
  sampled at each image stamp. That isn't implemented.
- Convex-ish outlines whose straight edges dominate. Notches shorter than the
  corner trim, round boards, and footprint-level Edge.Cuts aren't supported.
- Contour points sit on boundary pixels, a sub-pixel inward bias. A gradient-based
  edge refinement would remove it.
- The visible PCB side wall isn't modelled. The outline is assumed to be the top edge.
- A board flipped upside down isn't detected beyond the feature check.
