# Board registration (CV)

Board-edge registration of a clamped PCB from one fixed camera. It serves the shared
`GetRegistration`/`RegistrationState` contract from `hardware_troubleshooter_interfaces`
unchanged. There's no AprilTag, motion, measurement, or MCP code here, and no
electrical identity is inferred from images.

## Method

1. Otsu-segment the board and take its outer contour (`rig.segmentation.board_is_brighter`).
2. Undistort the contour with the `camera_info` model (plumb_bob, rational_polynomial,
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

## Endpoints

| Name | Type | Notes |
| --- | --- | --- |
| `image_raw` + `camera_info` | subscribed | Exact stamp match. An image without its own camera_info isn't processed. |
| `/registration/get` | `GetRegistration` | Current `RegistrationState`. An unknown `board_id` returns invalid. |
| `/registration/events` | `RegistrationState` | Reliable, transient-local, depth 1. Published on state changes. |
| `/registration/overlay` | `sensor_msgs/Image` | Debug drawing: outline, corners, +x arrow, feature, targets. Not verification. |
| `/registration/diagnostics` | `diagnostic_msgs/DiagnosticArray` | Edge points, symmetry resolution, per-hypothesis residuals, IPPE tilt. |
| TF `camera -> board_frame` | dynamic | Published only while `status=valid`. |

The `/registration` prefix is the `endpoint_prefix` parameter.

## Semantics

- A revision's pose is fixed when it's created. Later frames confirm it (within the
  move thresholds) or invalidate it. An invalidated revision is never reported valid
  again; the next accepted frame creates a new one. `registration_id` is
  `<board_id>/<node session>/r<N>`, so IDs don't repeat across restarts.
- `header.stamp` is the latest frame that confirmed the revision. The frame that
  created the pose is `registration_stamp` in diagnostics.
- `board_in_camera` is in the camera optical frame (`header.frame_id`).
  `has_pose` is false once the revision is invalidated.
- `tracking_valid` needs the latest frame to confirm the revision within
  `tracking_loss_timeout_s`. A failed frame inside the timeout makes tracking invalid
  (`unconfirmed`) without invalidating the revision. Past the timeout it is
  `tracking_lost`. `board_unmoved` equals `tracking_valid`.
- `calibration_valid` needs valid camera_info, the bench plane, `base_frame`, and a TF
  lookup `base_frame <- camera frame`. A change in intrinsics or extrinsic invalidates
  the revision with `calibration_changed`. `calibration_revision` hashes those inputs.
- `valid_until` is the last confirmation plus the timeout, or zero.
- `has_board_height` is always false. Board height for the approach belongs to the
  motion adapter's calibration.
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
| `rig.base_frame` | calibration / TF |
| `rig.bench_plane.normal`, `distance_m` | in-plane pose, calibration |
| `rig.segmentation.board_is_brighter` | segmentation |
| `rig.thresholds.max_fit_residual_px` | acceptance |
| `rig.thresholds.move_translation_m`, `move_rotation_rad` | `board_moved` |
| `rig.thresholds.tracking_loss_timeout_s` | `tracking_lost`, `valid_until` |
| `rig.thresholds.feature_min_contrast` | symmetry decision |
| camera_info (K, D) | from the ROS `camera_calibration` tools |
| `base_frame <- camera` TF | hand-eye calibration, published by whoever owns it |

The launch file can publish that TF from `rig.camera_extrinsic` with
`publish_extrinsic:=true`. Leave it off if another node publishes it.

## Build and run (ROS 2 Humble)

```bash
cd ros2_ws
colcon build --packages-select hardware_troubleshooter_interfaces hardware_troubleshooter_registration
source install/setup.bash
ros2 launch hardware_troubleshooter_registration registration.launch.py \
  board_config:=/path/board.yaml rig_config:=/path/rig.yaml \
  image_topic:=/camera/image_raw camera_info_topic:=/camera/camera_info
```

### Playback without a camera

Synthetic sequence (static, moved, covered) with matching test-only configs:

```bash
ros2 run hardware_troubleshooter_registration make_synthetic_dataset /tmp/reg_synth
ros2 launch hardware_troubleshooter_registration registration.launch.py \
  board_config:=/tmp/reg_synth/board.yaml rig_config:=/tmp/reg_synth/rig.yaml \
  publish_extrinsic:=true image_folder:=/tmp/reg_synth/frames \
  camera_info_file:=/tmp/reg_synth/camera_info.yaml playback_frame_id:=synthetic_camera_optical
ros2 topic echo /registration/events
ros2 service call /registration/get hardware_troubleshooter_interfaces/srv/GetRegistration "{}"
```

Your own image folder works the same way with your camera_info YAML. For a rosbag, run
`ros2 bag play <bag> --clock` and launch with `use_sim_time:=true`, so the timeout
compares image stamps with bag time.

## Tests

```bash
python3 -m pytest -q ros2_ws/src/hardware_troubleshooter_registration/test
```

The core tests need only numpy, OpenCV, PyYAML and pytest. They render synthetic
boards through a known distorted camera and check: recovered pose against ground
truth, the ambiguous (hidden or mirrored feature) case, `board_moved` on a pose jump,
`tracking_lost` on occlusion, removal and dropped frames, `calibration_changed`, and
that missing calibration gives `calibration_valid=false` with no TF.
`test_node_ros.py` runs the node over real topics on unique test names. It's skipped
unless rclpy and the built interfaces are importable. The tolerances are test-only
and say nothing about accuracy on a real rig.

## Known limits

- One fixed camera. A camera on the arm would change the extrinsic every frame and
  trigger `calibration_changed` continuously.
- Convex-ish outlines whose straight edges dominate. Notches shorter than the
  corner trim, round boards, and footprint-level Edge.Cuts aren't supported.
- Contour points sit on boundary pixels, a sub-pixel inward bias. A gradient-based
  edge refinement would remove it.
- The visible PCB side wall isn't modelled. The outline is assumed to be the top edge.
- A board flipped upside down isn't detected beyond the feature check.
