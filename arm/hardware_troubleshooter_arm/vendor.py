"""Names and paths in the vendored Maker Arm v1 description (see vendor/maker_arm/UPSTREAM.md)."""

from pathlib import Path

VENDOR_DIR = Path(__file__).resolve().parents[1] / "vendor" / "maker_arm"
VENDOR_URDF = VENDOR_DIR / "urdf" / "robot.urdf"
VENDOR_PROFILE = VENDOR_DIR / "profiles" / "maker_arm_v1.yaml"

# SDK joints J1..J6 drive these URDF joints in order; J7 (motor 7) drives the toolhead.
ARM_JOINTS = (
    "link_002_joint",
    "link_003_joint",
    "link_004_joint",
    "link_005_joint",
    "link_006_joint",
    "link_007_joint",
)
SDK_JOINT_COUNT = len(ARM_JOINTS) + 1
TOOLHEAD_MOTOR_ID = 7

WRIST_LINK = "link_007"
GRIPPER_LINKS = ("gripper_left", "gripper_right", "grasp_center")
GRIPPER_JOINTS = ("gripper_left_joint", "gripper_right_joint", "grasp_center_joint")
