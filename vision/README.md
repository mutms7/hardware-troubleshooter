# Wrist-camera homing and pad alignment

The arm-mounted camera establishes the PCB pose. Before approaching a pad, capture
one settled wrist view, move sideways, capture another settled view, and jointly fit
one metric board pose against both images and the **measured** camera movement.
Triangulation of shared features independently checks the fit. Then approach the pad
and finish with local wrist-camera alignment.

The fixed camera is a depth-capable camera beside the bench at roughly 45 degrees.
Its RGB/intensity stream is an optional overview. Neither depth nor fixed-camera PnP
is used in the default flow. No fixed camera intrinsics, bench plane or camera-to-rig
extrinsic is required for wrist homing.

This Python library uses numpy, OpenCV 4.x and PyYAML. It doesn't open hardware or
command motors by itself. `home_target` runs capture/move/capture through async
adapters supplied by the orchestrator. The motion adapter owns bounded movement,
collision checks and guarded stop. Nothing here proves electrical contact.

## Configuration and board identification

Start with [config/vision.example.yaml](config/vision.example.yaml). Physical values
and acceptance limits stay unset until measured on the rig.

- `registration_source: wrist` is the default. Explicit `registration_source: fixed`
  retains the standalone legacy board-edge engine for old callers and regression tests.
- Each arm needs camera intrinsics, a measured `joint_map` and a measured
  `wrist_camera` hand-eye transform in its calibration YAML. Nominal CAD mounts don't
  authorize homing. `base_in_rig` locates the arm in a common frame; use an explicit
  identity if this arm's base defines that frame.
- Pads come from KiCad or a measured registry, in the same board frame as the homing
  reference. Electrical identities aren't inferred from photos.
- `homing` configures measured lateral baseline, requested step, maximum baseline,
  minimum parallax, shared feature count, pose consistency, per-view/joint reprojection
  limits, triangulation error, ambiguity margin and tracking lifetime. The requested
  step must be between the minimum and maximum baseline limits.
- `board_reference` provides a photo and its camera intrinsics, plus at least four
  identified, noncollinear `pixel` -> `board_m` landmarks. Use actual corresponding
  physical features, not an unordered cyclic contour. An asymmetric feature resolves
  orientation. Coordinates describe the top surface, z=0, in metres.

`BoardFeatureDetector` matches SIFT texture inside the reference landmarks' convex
hull. It removes reference lens distortion to assign metric board coordinates, then
matches reference features to wrist images. A RANSAC homography rejects inconsistent
matches; metric PnP estimates pose. Select a region of the flat PCB surface. Raised
components, probe pixels, glare and repetitive pads can invalidate the planar model.
Texture needs to be spread over the board. This is reference matching, not learned
PCB segmentation, and doesn't depend on the board being brighter than the bench.

An identified-feature adapter may instead pass `BoardObservation` to `observe_board`
and `align`: stable physical feature IDs, known board points and raw image pixels.

## Homing, approach and final alignment

```python
vision, problems = VisionService.load("vision.yaml", session_id)
if problems:
    raise ValueError(problems)

# async acquire_settled(arm) -> (Frame, list[JointSample]) covering the settle window
# async move_relative(arm, delta_base_m) -> bool; the adapter guards the move
result = await vision.home_target("left", "U1.1", acquire_settled, move_relative)
if not result.ok:
    raise RuntimeError(result.reasons)

point_base_m, board_normal_base = vision.pad_approach("left", "U1.1", standoff_m, now)
# Orchestrator: move the calibrated tool TIP to point_base_m at the right orientation
# using calibrated TCP/IK; feed measured joint feedback continuously.
out = vision.align("left", settled_wrist_frame, "U1.1", "probe")
if out.ok:
    correction_base_m = out.correction_base_m
# Correct, settle, capture and align again until out.result.aligned.
```

All stamps use one monotonic clock. Wrist camera IDs are `<arm>_wrist`; resolution
must match the camera matrix. Each accepted image needs seven finite measured joint
positions/velocities with continuous feedback across the settle window.

The lower-level sequence is `begin_homing`, `observe_board`, `homing_step_base_m`,
then `observe_board` again. One view never authorizes a transform. The requested step
is along camera +x expressed in the arm base; it isn't evidence of movement. Actual
camera translation AND rotation come from measured joints and calibrated hand-eye.

The solver retains IPPE alternatives and an iterative seed for singular front-facing
orientations, checks pose consistency in the rig, then jointly refines a shared pose.
It rejects similar competing solutions, poor per-view/joint fit, inadequate shared
features, changed geometry, insufficient lateral translation/parallax, negative depth
and excessive triangulation error. It never averages PnP poses or treats a commanded
displacement as ground truth. The board must stay stationary across the two views.

`board_state(now)` reports validity, revision, expiry, arm/target ownership,
calibration revision, baseline, parallax, per-view RMS and triangulation RMS.
`base_from_board` only returns a valid pose for the homing arm. Each homing stage gets
a new revision scoped to one target; a different pad requires another homing stage.
Concurrent `home_target` calls are rejected.

`align` first confirms the board against the current measured camera pose. A missing
observation blocks alignment. Board movement, expiry, calibration changes or explicit
`invalidate_board()` require fresh homing; a revoked revision cannot become valid
through another confirmation. Local pad alignment additionally requires a selected
tool, calibrated landing pixel, matching stand-off and a strong, unique copper match.
The correction is target minus tool landing point, expressed in the arm base.

## Fixed RGB overview

`capture_scene_reference(frame)` stores an overview reference. In wrist mode,
`process_fixed_frame(frame, arm_mask=None)` and `scene_clear(frame)` only compare
images. A changed/unobservable scene revokes wrist homing; a clear frame cannot create
or restore it. The PCB isn't excluded from the comparison. No depth is read.

A caller-supplied image-space arm mask may suppress known arm pixels. Without it,
arm movement can conservatively invalidate homing. Automatic URDF masks aren't used
because no fixed-camera geometry is assumed. Overview checks aren't a claim of
continuous monitoring. Feed them if used, call `invalidate_board("board_moved")` on
fixture movement, and call `tick(now)` to expire stale wrist evidence. An unchanged
overview cannot confirm a PCB hidden by an arm.

## Calibration

The images supplied on 2026-10-24 calibrate **one arm-mounted camera** at 1920x1080,
not the fixed camera or the other wrist camera. Its matrices are saved in
[calibrations/wrist-camera-20261024.yaml](calibrations/wrist-camera-20261024.yaml);
the companion JSON records all 18 view errors. The target has 8x8 squares, 7x7 inner
corners and 1.9-inch (0.04826 m) spacing. Overall RMS is 0.830340 pixels, k3 fixed at
zero. Assign this file only to the camera that took those photos, at the same image
resolution and lens/focus settings. It doesn't establish hand-eye, TCP or metric
positioning accuracy.

`calibration.py` supplies these offline steps:

1. `CheckerboardTarget(corners_x, corners_y, square_m)`, `detect_checkerboard` and
   `calibrate_intrinsics` for each camera's lens calibration. Count inner corners,
   not squares. Vary distance, tilt and position, with coverage near the image edges.
   The five-view minimum is an input check, not evidence of adequate coverage.
2. `target_pose` for identified target poses. For a plain checkerboard, inspect the
   corner order against an external origin/axis mark in every image. Pass
   `reverse_order=False` to confirm it or `True` to reverse it. Omit it for intrinsics
   only. Reject views where neither ordering matches the physical axes, especially
   square grids. The mark isn't detected automatically.
3. `eye_in_hand` for link <- camera from a stationary target and varied measured arm
   poses. `calibrate_joint_offsets`/`eye_to_hand` remain available if the fixed RGB
   camera is used for offline joint-zero calibration; that optional procedure is
   separate from runtime board localization.
4. `tip_pixel_from_marks`: touch a dark dot, retract along the approach axis by the
   known stand-off, capture, and repeat at least three times per tool. The motion
   adapter also needs measured tool-tip transforms (TCPs).

Use a flat, rigid checkerboard. The physical board's outer corners may be rounded;
printed intersections should be sharp. A thin black border followed by a white band
is supported; keep the complete pattern and white band visible. Difficult photos
may need reduced-resolution detection followed by full-resolution refinement, as
in the saved real calibration run. ChArUco stays optional; its detector-specific
half-pixel correction isn't applied to checkerboard detections.

Changing mounts or re-zeroing motors invalidates the affected calibration.

## Contact, thermal and tests

`TouchdownDetector` detects sustained torque steps, not electrical contact. Thermal
helpers remain for separately calibrated callers; wrist mode doesn't invent a
fixed-camera board pose or a depth mapping.

Install numpy, PyYAML, pytest and `opencv-contrib-python>=4.8,<5`. Run separately:

```powershell
python -m pytest -q vision/test
python -m pytest -q registration/test
python -m pytest -q arm/test
```

Tests cover two-view recovery, actual feature matching, arm FK through homing and pad
alignment, triangulation, ambiguity, feedback/camera guards, movement and calibration
invalidation, plus the explicit legacy mode. Synthetic values describe an imaginary
rig. Real PCB accuracy needs validation at the working distance.
