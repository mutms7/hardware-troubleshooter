# Vision stack

The CV around the measurement orchestrator. There are three cameras: the fixed camera
over the bench and one wrist camera on each Maker Arm's `link_007`. The AMG8833 thermal
sensor adds hints on top. It's a plain Python library (numpy, OpenCV, PyYAML) that
uses [registration/](../registration/README.md) and [arm/](../arm/README.md). It never
moves an arm, and nothing it reports is evidence of electrical contact.

| Camera | Job | Module |
| --- | --- | --- |
| Fixed | Where the board is in the rig, and whether it moved, even with an arm over it | registration engine, `occlusion.py` |
| Fixed | Anything in the workspace that shouldn't be | `scene.py` |
| Wrist | Where this tool will land relative to the target pad, and the correction | `alignment.py` |
| Thermal | Which part of the registered board runs hot | `thermal.py` |

The wrist cameras never register the board. Every arm move changes their extrinsic, so
they only measure locally: target pad versus tool tip in the same image. Most of the
registration and kinematic error cancels out of that comparison.

## Using it

`VisionService` loads everything from one YAML and is the orchestrator's single
API. The caller owns the cameras, the arm SDK, the clock and the transport. Stamp
every joint sample and frame with the same clock (`time.monotonic()` by default).

```python
from hardware_troubleshooter_vision.capture import Frame, JointSample, OpenCVCamera
from hardware_troubleshooter_vision.service import VisionService

vision, problems = VisionService.load("vision.yaml", session_id)  # problems lists every unset value

vision.add_joint_sample("left", JointSample(stamp, positions, velocities, torques))
vision.process_fixed_frame(fixed_camera.read())  # registration, both arms masked
vision.tick(now)                                 # so a stalled camera still times out

vision.board_state(now)               # registration Snapshot: status, reasons, pose
vision.base_from_board("left", now)   # 4x4 left base_link <- board, None unless valid
vision.scene_clear(fixed_camera.read())  # SceneResult(clear, reasons, blobs)

out = vision.align("left", wrist_camera.read(), "U1.1", "probe")
out.ok, out.result.error_m, out.result.aligned  # target - tip landing point
out.correction_base_m                           # move the tool by this, base frame
```

### Fixed camera

`process_fixed_frame` projects each arm's URDF collision meshes, at its joint sample
nearest the frame stamp, into the fixed image. It passes that mask to registration,
which then confirms or invalidates the board from the edges still in view. A board
pushed under an arm is still caught as `board_moved`.

- **Mask construction:** each mesh part is filled as its convex hull and grown by
  `occlusion.margin_px`, so the mask errs toward hiding more. A part wholly outside
  the view frustum is skipped. One straddling the camera plane inside it masks
  everything.
- **Missing joint state:** an arm without a joint sample within
  `capture.max_joint_skew_s` masks the whole image. Its position is unknown, so
  registration can't confirm through it.
- **Threshold:** set `rig.thresholds.min_visible_edge_fraction` in the registration
  rig YAML, or occluded frames stay unconfirmed.

### Wrist alignment

`align` refuses unless the frame was taken with the arm settled, the requested tool is
selected, and registration is valid. Settled means every joint was below
`capture.settle_velocity_rad_s` for `capture.settle_duration_s`, with a joint sample
within `max_joint_skew_s` of the image. The tool check uses the measured toolhead
angle. Then it:

1. predicts board -> wrist camera from registration and measured joints at the stamp,
2. resamples the wrist image onto the board plane around the target at
   `alignment.resolution_m`,
3. renders the copper within `patch_radius_m` from the pad geometry and finds it by
   normalised correlation within `search_radius_m`,
4. maps the tool's calibrated tip pixel through the same prediction to its landing point,
5. returns target minus landing point, in the board plane and in the arm base.

The tip pixel is valid only at the stand-off it was calibrated at. Hover there (within
`max_standoff_error_m`) or the alignment refuses. Repeat align, move, align until
`aligned` (error within `converged_m`). On synthetic views with a 0.8 mm / 0.3°
registration error, the error is measured to 15 µm and one correction converges.

Failure reasons: `arm_not_settled`, `joint_state_skew`, `no_joint_state`,
`tool_not_selected`, `unknown_tool`, `registration_invalid`, `unknown_target`,
`tip_pixel_uncalibrated`, `standoff_mismatch`, `target_not_in_view`,
`no_pad_contrast`, `match_weak`, `match_ambiguous`, `match_at_search_edge`,
`unconfigured:<key>`.

`match_ambiguous` is the important one. A regular row of identical pads, such as IC
pins, matches nearly as well one pitch over. The best match must beat the best match
elsewhere by `min_uniqueness_margin`. A larger `patch_radius_m` takes in the row's
ends or other parts and breaks the tie.

Pads come from the board's `.kicad_pcb` (front copper, named `REF.number`) or a
measured list. They never come from an image.

### Contact

`contact.TouchdownDetector` flags a sustained step in measured torque on the watched
joints, relative to a baseline averaged before the descent. It says the tip met
something. The alignment before descent says where, and only the AD3 can say the
contact is electrical. There's no image check at contact: the tip covers the pad then,
and the tip pixel is only valid at its stand-off.

### Thermal

Calibrate `thermal.homography` once with `thermal_homography`, from four or more warm
points seen by both the thermal sensor and the fixed camera. With the board registered,
`cells_on_board` gives every cell a board position, and `hot_spot` names the pads
nearest the hottest cell that stands `min_delta_c` above the board's median. A cell is
centimetres wide, so this is a hint, not a measurement.

## Calibration

`calibration.py` holds the offline steps, in this order on a new rig:

1. **Intrinsics:** run `calibrate_intrinsics` per camera on ChArUco views
   (`detect_charuco`). It writes camera_calibration YAML. k3 is held at zero unless the
   views reach the image corners.
2. **Bench plane:** run `bench_plane` from `target_pose` of a target lying flat on the
   bench, seen by the fixed camera. The result goes in the registration rig YAML.
3. **Joint zeros:** run `calibrate_joint_offsets` with a target on `link_007`, 25 or
   more varied poses, and the fixed camera watching. It fits J2–J5 offsets and gives
   camera <- base. `base_in_rig` turns that into the arm's `base_in_rig`. J1 and J6
   offsets aren't observable this way. J1 ends up in `base_in_rig`, and J6 is absorbed
   consistently by everything mounted on `link_007`.
4. **Wrist camera:** run `eye_in_hand` with views of a fixed target, using the fitted
   joint map.
5. **Tip pixel per tool:** run `tip_pixel_from_marks`. Touch the tip to a dark dot,
   retract straight up by the stand-off, and capture. Repeat three or more times.

`arm_calibration_dict` writes the result in the arm package's calibration layout. Every
function reports a residual, but none decides whether it's acceptable. Record the
residuals and set limits from the real rig. Re-zeroing a motor (`maker-arm zero`)
invalidates step 3 onward.

OpenCV's `CharucoDetector` reports corners 0.5 px down and right of where its own
`projectPoints` puts them (measured +0.49 px, scatter 0.04 px). `detect_charuco` shifts
them back so every pixel here uses one convention.

## Configuration

[`config/vision.example.yaml`](config/vision.example.yaml) is entirely unset, and
`VisionService.load` lists every missing value. Each one needs a calibration or a
measurement. The test values in `test/` describe an imaginary rig and must not be
copied into a live config.

| Section | Needed for |
| --- | --- |
| `board_config`, `rig_config`, `fixed_camera_info` | registration (the rig YAML's `camera_extrinsic` is the fixed camera's) |
| `arms.<name>` | arm config, optional calibration, wrist camera_info, `base_in_rig` |
| `pads` | wrist alignment |
| `capture` | settled wrist frames, arm masks |
| `occlusion` | arm masks |
| `alignment` | wrist alignment acceptance |
| `scene`, `contact`, `thermal` | their modules |

Tool names in the arm configs should match the instrument service's turret heads
(`scope`, `logic`, `dmm`), so `select_tool` and `align` mean the same tool.

## Tests

```bash
python -m pytest -q vision/test
```

Run each folder's suite on its own. The registration, arm and vision suites each
have a `conftest`, and one pytest run over all three mixes them up. The tests render
synthetic views: ChArUco targets, wrist copper and the registration board. They check
the camera intrinsics, hand-eye mounts and joint offsets recovered against ground
truth. They check alignment accuracy and convergence, ambiguity and refusals, and arm
masks against projected mesh vertices. They also run registration end to end through
the service into wrist alignment. Passing them says nothing about accuracy on the real
rig.

## Known limits

- Stop-and-look only. The read stamp, not the exposure, is the frame's stamp. The
  settle window is what makes that safe.
- Alignment assumes the target is on the board's top surface and roughly planar
  around it. Tall parts within `patch_radius_m` are rendered as flat copper or mask.
- Custom KiCad pads use their anchor size. Back-side pads aren't read.
- Arm masks are only as good as the URDF meshes (the vendor export lacks `part_003`)
  and the joint map. Keep `margin_px` generous until the arms are calibrated.
- The scene guard compares against one reference frame. Recapture it when the lighting
  or the bench setup changes.
