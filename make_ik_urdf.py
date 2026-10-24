"""Turn the Onshape export (urdf/debugger_arm.urdf) into the copy ikpy can load.

Onshape writes every arm joint as `continuous` and keeps two dead-end fixed branches that
make ikpy pick the wrong chain. This switches the joints to `revolute` with wide limits
(the real limits come from calibration.json) and drops the two branches.

Run it after every re-export:  python make_ik_urdf.py
"""
import xml.etree.ElementTree as ET
from pathlib import Path

URDF_DIR = Path(__file__).resolve().parent / "urdf"
DEAD_END_JOINTS = {"base__1_", "fastened_5__1_"}


def main():
    tree = ET.parse(URDF_DIR / "debugger_arm.urdf")
    robot = tree.getroot()
    for joint in list(robot.findall("joint")):
        if joint.get("name") in DEAD_END_JOINTS:
            robot.remove(joint)
        elif joint.get("type") == "continuous":
            joint.set("type", "revolute")
            ET.SubElement(joint, "limit", lower="-3.1416", upper="3.1416", effort="1", velocity="1")
    tree.write(URDF_DIR / "debugger_arm_ik.urdf", encoding="utf-8", xml_declaration=True)
    print("wrote urdf/debugger_arm_ik.urdf")


if __name__ == "__main__":
    main()
