# Arm description (Maker Arm v1 with the rotating toolhead)

Builds each arm's URDF and Maker SDK profile from the vendored Maker Arm v1 files
([vendor/maker_arm/](vendor/maker_arm/UPSTREAM.md)). Our rig keeps the six arm joints and
motor 7, but swaps the stock gripper for a rotating toolhead and mounts a camera on the
wrist. It's a plain Python library (numpy, PyYAML). There's no IK, motion, camera or
MCP code here. Hand-taught bench moves over USB-CAN on Windows are in [bench/](bench/README.md).

## What gets built

The URDF is the vendor chain `base_link` → `link_007` unchanged, plus:

| Frame | Parent | From |
| --- | --- | --- |
| `flange` (fixed) | `link_007` | `flange` |
| `toolhead_link` (revolute `toolhead_joint`) | `flange` | `toolhead.joint_origin`, `axis`, limits, mesh |
| `tool_<name>` (fixed), one per tool | `toolhead_link` | `toolhead.tools[].tip`, or the calibrated tip |
| `wrist_camera_optical` (fixed) | `link_007` | `wrist_camera`, or the calibrated hand-eye result |

The stock gripper links and joints (`gripper_left`, `gripper_right`, `grasp_center`)
are dropped, along with the `link_007` meshes listed in `remove_link_007_meshes`. Every
link and joint gets the arm's `name` as a prefix (`left_link_007`), so two arms can share
one tree. Mesh paths are written relative to the output directory. A header comment
records the calibration ID and which values are still nominal CAD values.

The camera sits on `link_007`, so turning the toolhead never moves it. One hand-eye
calibration per arm covers every tool. Turning J6 moves both.

The SDK profile is the vendored `maker_arm_v1.yaml` with motor 7's entry replaced by
`sdk_joint7`. The motor ID and RS00 model stay. The stock entry (kp 20, kd 0.5) is
deliberately soft so the gripper limits its own grip force. A toolhead needs to be
stiffer, or a tip offset from the axis twists under probe load.

```powershell
python -m hardware_troubleshooter_arm.build left.yaml --out build/ [--calibration left_cal.yaml]
```

Run from this folder. It writes `build/left.urdf` and `build/left_sdk_profile.yaml`, or
prints every unset or invalid value and writes nothing. Give the profile to the SDK with
`maker-arm ... --config build/left_sdk_profile.yaml` or `Arm.from_yaml(path, ...)`.

## Using the library

```python
from hardware_troubleshooter_arm.config import load_arm_calibration, load_arm_config
from hardware_troubleshooter_arm.kinematics import Kinematics, urdf_joint_positions
from hardware_troubleshooter_arm.toolhead import check_tool, tip_pixel
from hardware_troubleshooter_arm.urdf import build_urdf

config, problems = load_arm_config("left.yaml")
calibration, cal_problems = load_arm_calibration("left_cal.yaml")
urdf, build_problems = build_urdf(config, calibration, output_dir)
kinematics = Kinematics.from_urdf(urdf)

q = urdf_joint_positions(config, arm.get_joint_positions(), calibration)  # 7 SDK values
poses = kinematics.link_poses(q)  # 4x4 per link, in left_base_link
reason = check_tool(config, "probe", q["left_toolhead_joint"])  # None when selected
pixel = tip_pixel(calibration, "probe")  # TipPixel(uv, standoff_m) or None
```

- `urdf_joint_positions` maps the SDK's seven positions (J1–J6, then motor 7) through
  `urdf = scale * sdk + offset_rad`. The calibrated `joint_map` wins over the nominal one.
  The vendored URDF and the SDK disagree on joint zeros, so this map is required. FK
  raises without it.
- Pass **measured** joint positions sampled at the image or contact stamp. The arm runs
  PD control with low wrist gains, so the commanded pose isn't where the arm is.
- `check_tool` returns `unknown_tool:<name>`, `toolhead_angle_invalid`,
  `tool_not_selected:<name> (...)` or `unconfigured:<key>`. It never guesses.
- `tip_pixel` only comes from a calibration. There's no CAD fallback.

## Configuration

[`config/arm.example.yaml`](config/arm.example.yaml) and
[`config/calibration.example.yaml`](config/calibration.example.yaml) are entirely unset.
Copy the arm file once per arm. Each value needs CAD or a measurement:

| Key | Needed for | Source |
| --- | --- | --- |
| `name` | prefixes, calibration match | choose (`left`, `right`) |
| `remove_link_007_meshes` | URDF | which stock parts the toolhead replaces |
| `flange`, `toolhead.*` geometry, `tools[].tip` | URDF | toolhead CAD |
| `tools[].select_rad`, `toolhead.select_tolerance_rad` | `check_tool` | toolhead design, then measured |
| `wrist_camera` | URDF | camera mount CAD |
| `joint_map` | FK from SDK positions | kinematic calibration |
| `sdk_joint7.*` | SDK profile | toolhead travel, tuned gains |

The calibration file overrides nominal values: `joint_map` (kinematic calibration),
`wrist_camera` (hand-eye), and per tool, `tip` (TCP) and `tip_pixel` (where the tip
appears in the wrist image at a known stand-off). Change `calibration_id` whenever any
value changes. Re-zeroing a motor with `maker-arm zero` invalidates the joint map and
everything measured after it.

Candidates for `remove_link_007_meshes`, from their extents in the `link_007` frame:
`part_026` and `part_027` form a 166 mm-wide plate at the far end, probably the jaw rail.
`part_022`, `part_048` and `part_049` are on the axis, probably the motor stack. That's
a guess from geometry; confirm against the CAD before using it.

## Tests

```bash
python -m pytest -q arm/test
```

They need numpy, PyYAML and pytest. The values in `test/arm_synthetic.py` describe an
imaginary toolhead. Tests check that the gripper is replaced and the vendor chain is
unchanged (FK matches the vendor URDF), that a toolhead turn moves tool tips but not the
camera, and that calibration overrides apply and unset values fail closed. They also check
that only motor 7 changes in the SDK profile. With the Maker SDK importable, one more test
loads the generated profile through the SDK's own `ArmConfig.from_yaml`.

## Known limits

- The toolhead is direct drive or a fixed gear ratio: `scale` is linear. A linkage like
  the stock gripper's crank isn't modelled.
- The toolhead link has no inertial; the URDF is for kinematics and geometry, not
  dynamics. `toolhead_joint` reuses J6's placeholder effort/velocity; the SDK profile
  is what limits the motor.
- The vendor caveats still apply: unverified joint zeros and limits, and a missing
  `part_003`.
