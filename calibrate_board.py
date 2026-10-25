"""Calibrate where the board is, for touch_pad.py. Two steps, with the arm server running (--hardware, --sensors) and armed:

    python calibrate_board.py start      go to the start pose (touch_start.json) and record the ToF there
    (lower the needle onto the board with the viewer's R/F keys until it just touches)
    python calibrate_board.py contact    record the needle's height from the joint encoders: that is the board, and
                                         touch_off the server with it, so its model, floor and viewer agree with reality

Both go into board_cal.json. Before each job, touch_pad.py reads the ToF at the start pose again: if the board has moved up or
down, the ToF changes by the same amount and the board height is shifted to match.

Why the ToF only at the start pose: it sees the board beside the needle, not under it, so near a pad its reading depends on
what is there (2026-10-25: 64 mm at one spot, 71-82 mm at J3.2, jumping +-5 mm). At the start pose it always looks at the
same patch of board, which makes it a steady check of the board's height.

Why the encoders, not the command: the servos lag and sag, up to several mm (following error). The encoders say where the
arm really is. Keep the tilt fixed (R/F keys hold it), since the encoders' tip is the model's tip, not the needle itself.
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from arm_client import Arm

HERE = Path(__file__).resolve().parent
CAL = HERE / "board_cal.json"

ap = argparse.ArgumentParser()
ap.add_argument("step", choices=("start", "contact"))
ap.add_argument("--start", default=str(HERE / "touch_start.json"), help="start pose: tip_m and phi_deg")
args = ap.parse_args()

arm = Arm()


def tof_median(n=25):
    rs = []
    for _ in range(n):
        t = arm.sensors().get("tof")
        if t and t["valid"] and t["age_s"] <= 0.5:
            rs.append(t["range_mm"])
        time.sleep(0.07)
    if len(rs) * 2 <= n:
        raise SystemExit("no steady ToF reading: is the sensor head connected (--sensors)?")
    return float(np.median(rs)), float(np.std(rs))


def measured_z(timeout=4.0):
    """The encoders' tip height (m), once the joints have stopped moving."""
    prev, t0 = None, time.time()
    while True:
        z = arm.state(measure=True)["measured_tip"][2]
        if prev is not None and abs(z - prev) < 0.00015 or time.time() - t0 > timeout:
            return z
        prev = z
        time.sleep(0.3)


def go_to_start():
    start = json.loads(Path(args.start).read_text())
    for _ in range(3):                          # the servos lag, so a goto often ends 'not reached in time'
        s = arm.goto(start["tip_m"], phi=start["phi_deg"], timeout=90.0)
        if s.get("msg") == "ok" or "not reached in time" not in s.get("msg", ""):
            break
    if s.get("msg") != "ok" and "not reached in time" not in s.get("msg", ""):
        raise SystemExit(f"could not reach the start pose: {s.get('msg')}")
    time.sleep(2.0)


if not arm.state()["hw"]["armed"]:
    raise SystemExit("the arm is not armed: press G in the viewer")
cal = json.loads(CAL.read_text()) if CAL.exists() else {}

if args.step == "start":
    go_to_start()
    tof, sd = tof_median()
    z = measured_z()
    cal["start"] = {"saved": datetime.now().isoformat(timespec="seconds"), "tof_mm": tof, "tof_sd_mm": sd,
                    "measured_z_m": z}
    CAL.write_text(json.dumps(cal, indent=1))
    print(f"start pose: ToF {tof:.1f} mm (sd {sd:.1f}), encoders' tip z {z * 1000:.1f} mm. Saved to {CAL.name}.")
    print("Now lower the needle onto the board with R/F until it just touches, then: python calibrate_board.py contact")
else:
    if "start" not in cal:
        raise SystemExit("run 'python calibrate_board.py start' first")
    st = arm.state()
    z = measured_z()
    cal["board_z_m"] = z
    msg = arm.touch_off(tip_z=z, zero_tof=False)["msg"]     # the server's model now agrees with the real board
    cal["contact"] = {"saved": datetime.now().isoformat(timespec="seconds"), "measured_z_m": z,
                      "commanded_tip_m": st["tip"], "phi_deg": st["phi"], "tool": st.get("tool")}
    CAL.write_text(json.dumps(cal, indent=1))
    print(f"board at encoders' z {z * 1000:.1f} mm (commanded tip z {st['tip'][2] * 1000:.1f} mm, tilt {st['phi']:.1f} deg); "
          f"{(cal['start']['measured_z_m'] - z) * 1000:.1f} mm below the start pose. Saved to {CAL.name}.")
    print(f"server: {msg}")
