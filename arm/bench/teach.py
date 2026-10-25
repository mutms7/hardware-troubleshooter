"""Torque-off teach mode: move the arm by hand, record named points, track the swept envelope.

Usage: python -X utf8 teach.py <workdir>
Commands (python send.py <workdir> "<command>"):
  record <name>   save the current pose as point <name>
  record <name> J6 [J7 ...]   save only those joints (a partial point, e.g. a knob angle)
  delete <name>   forget point <name>
  reset           forget the swept envelope (keeps home and the current pose)
  quit            write teach.yaml and exit
Never enables torque: the only frames sent are the SDK's stop-frame probes.
"""

import math
import sys
import time
from pathlib import Path

import yaml

from bench_common import Inbox, fmt, log, write_state
from gsusb_backend import GsUsbBackend
from maker_arm.arm import Arm
from maker_arm.config import ArmConfig, JointConfig

workdir = Path(sys.argv[1]).resolve()
workdir.mkdir(parents=True, exist_ok=True)
out = workdir / "teach.yaml"

# Wide placeholder limits around the new zero so the SDK applies no 2π shift; torque stays off.
cfg = ArmConfig(joints=[JointConfig(motor_id=i, direction=1, offset=0.0, lo=-math.pi, hi=math.pi,
                                    kp=0.0, kd=0.0, model="RS02" if i in (2, 3) else "RS00")
                        for i in range(1, 8)])
arm = Arm(cfg, GsUsbBackend())

saved = yaml.safe_load(out.read_text(encoding="utf-8")) if out.exists() else {}
points = saved.get("points", {}) or {}
lo = saved.get("envelope", {}).get("min", [0.0] * 7)
hi = saved.get("envelope", {}).get("max", [0.0] * 7)


def save():
    out.write_text(yaml.safe_dump({
        "zero_note": "motor zeros written 2026-10-25 at the resting pose",
        "points": points,
        "envelope": {"min": [round(x, 4) for x in lo], "max": [round(x, 4) for x in hi]},
    }, sort_keys=False), encoding="utf-8")


inbox = Inbox(workdir)
arm.connect()
log(workdir, "teach: connected, torque OFF - move the arm by hand")
try:
    running = True
    last_print = 0.0
    while running:
        fresh = arm.refresh(wait=True, timeout=0.05)
        q = arm.get_joint_positions()
        if all(fresh) and all(math.isfinite(x) for x in q):
            lo = [min(a, b) for a, b in zip(lo, q)]
            hi = [max(a, b) for a, b in zip(hi, q)]
        for cmd in inbox.poll():
            parts = cmd.split()
            if parts[0] == "record" and len(parts) >= 2:
                if not all(fresh):
                    log(workdir, f"record {parts[1]}: REFUSED, stale feedback {fresh}")
                    continue
                if len(parts) == 2:
                    points[parts[1]] = [round(x, 4) for x in q]
                    log(workdir, f"recorded {parts[1]}: {fmt(q)}")
                else:  # record <name> J6 [J7 ...]: a partial point that only moves those joints
                    try:
                        idx = [int(p.upper().lstrip("J")) - 1 for p in parts[2:]]
                        assert all(0 <= i < 7 for i in idx)
                    except (ValueError, AssertionError):
                        log(workdir, f"bad joint list in {cmd!r} (use J1..J7)")
                        continue
                    points[parts[1]] = {f"J{i + 1}": round(q[i], 4) for i in idx}
                    log(workdir, f"recorded {parts[1]}: {points[parts[1]]}")
                save()
            elif parts[0] == "reset":  # forget the swept envelope: keep only home and the current pose
                lo = [min(0.0, x) for x in q]
                hi = [max(0.0, x) for x in q]
                save()
                log(workdir, f"envelope reset to home + current pose {fmt(q)}")
            elif parts[0] == "delete" and len(parts) == 2:
                points.pop(parts[1], None)
                save()
                log(workdir, f"deleted {parts[1]}")
            elif parts[0] == "quit":
                running = False
            else:
                log(workdir, f"unknown command: {cmd!r}")
        write_state(workdir, {"mode": "teach", "t": time.time(), "fresh": fresh,
                              "q": [round(x, 4) for x in q], "min": lo, "max": hi,
                              "points": points, "faults": arm.get_faults(),
                              "temps": arm.get_temperatures()})
        if time.monotonic() - last_print > 2.0:
            print(fmt(q), flush=True)
            last_print = time.monotonic()
        time.sleep(0.05)
finally:
    save()
    arm.disconnect()
    log(workdir, f"teach: saved {out}")
