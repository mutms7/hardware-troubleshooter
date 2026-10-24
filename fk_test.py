"""Read-only check: print where the URDF says the probe tip is while you move the arm by hand.

Never writes to the servos (no torque change, no goals). Move the arm and compare the
printed tip position to where the tip physically is.

The URDF's joint zeros and axis directions are NOT guaranteed to match the calibration's
zero pose. SIGN flips a joint's direction; ZERO_DEG is the URDF angle of the pose you held
during calibration. Tune both until the printed tip tracks the real tip.

Usage:
  python fk_test.py --port COM10 --once
  python fk_test.py --port COM10
"""
import argparse
import json
import math
import time
import warnings

import numpy as np
from ikpy.chain import Chain

from calibrate import Bus, delta, JOINTS, PROBE_ID

warnings.filterwarnings("ignore")

URDF = "urdf/debugger_arm_ik.urdf"
CALIBRATION = "calibration.json"

# servo ID -> index in the ikpy chain
CHAIN_INDEX = {1: 3, 2: 5, 3: 7, 4: 9, PROBE_ID: 11}
SIGN = {1: 1, 2: -1, 3: -1, 4: -1}
ZERO_DEG = {1: 6.0, 2: 5.2, 3: 2.3, 4: 8.1}


def joint_angles_rad(bus, cal):
    q = np.zeros(13)
    for sid, name in JOINTS.items():
        c = cal["joints"][name]
        deg = delta(bus.read_pos(sid), c["mid_raw"]) * c["deg_per_tick"]
        q[CHAIN_INDEX[sid]] = math.radians(SIGN[sid] * deg + ZERO_DEG[sid])
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    with open(CALIBRATION) as f:
        cal = json.load(f)
    chain = Chain.from_urdf_file(URDF, base_elements=["root"])
    bus = Bus(args.port, args.baud)
    try:
        while True:
            q = joint_angles_rad(bus, cal)
            x, y, z = chain.forward_kinematics(q)[:3, 3] * 1000
            degs = " ".join(f"{math.degrees(q[CHAIN_INDEX[s]]):+7.1f}" for s in JOINTS)
            print(f"\rjoints(deg) {degs}  |  tip mm  x {x:+8.1f}  y {y:+8.1f}  z {z:+8.1f}", end="", flush=True)
            if args.once:
                print()
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        print()
    finally:
        bus.close()


if __name__ == "__main__":
    main()
