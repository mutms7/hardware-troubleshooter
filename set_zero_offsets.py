"""Set ZERO_DEG so the arm's current pose reads as 0 degrees on every joint.

Pose the real arm exactly like the Onshape assembly was when you exported the URDF, then run
this. Reads IDs 1-4 only, never writes to the servos. Patches ZERO_DEG in fk_test.py.

  python set_zero_offsets.py --port COM10
"""
import argparse
import json
import re
import statistics
import time
from pathlib import Path

from calibrate import Bus, JOINTS, delta
from fk_test import SIGN

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=1_000_000)
    args = ap.parse_args()

    cal = json.loads((HERE / "calibration.json").read_text())["joints"]
    bus = Bus(args.port, args.baud)
    try:
        samples = {sid: [] for sid in JOINTS}
        for _ in range(10):
            for sid in JOINTS:
                samples[sid].append(bus.read_pos(sid))
            time.sleep(0.05)
    finally:
        bus.close()

    zero = {}
    for sid, name in JOINTS.items():
        c = cal[name]
        raw = round(statistics.median(samples[sid]))
        d = delta(raw, c["mid_raw"])
        spread = max(samples[sid]) - min(samples[sid])
        zero[sid] = round(-SIGN[sid] * d * c["deg_per_tick"], 1)
        note = "  (arm was moving, hold it still)" if spread > 6 else ""
        edge = "  (outside calibrated range)" if not c["min_delta"] <= d <= c["max_delta"] else ""
        print(f"ID {sid} {name:14s} raw {raw:4d}  delta {d:+5d}  ZERO_DEG {zero[sid]:+7.1f}{note}{edge}")

    path = HERE / "fk_test.py"
    text = path.read_text()
    new = "ZERO_DEG = {" + ", ".join(f"{s}: {zero[s]}" for s in JOINTS) + "}"
    text, n = re.subn(r"^ZERO_DEG = \{.*\}$", new, text, flags=re.M)
    if n != 1:
        raise SystemExit("could not find the ZERO_DEG line in fk_test.py")
    path.write_text(text)
    print("\nWrote", new)


if __name__ == "__main__":
    main()
