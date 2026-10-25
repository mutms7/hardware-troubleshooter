"""Send one command to run.py and wait until it arrives (or times out), then report.

Usage: python step.py <workdir> <timeout_s> <command...>
Exit 0 = arrived, 2 = timed out (arm still holding its goal), 3 = fault / runner gone.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

workdir, timeout, cmd = Path(sys.argv[1]), float(sys.argv[2]), " ".join(sys.argv[3:])


def state():
    for _ in range(10):
        try:
            return json.loads((workdir / "state.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            time.sleep(0.02)
    return None


subprocess.run([sys.executable, str(Path(__file__).with_name("send.py")), str(workdir), cmd], check=True)
t0 = time.time()
time.sleep(0.4)
code = 2
while time.time() - t0 < timeout:
    s = state()
    if s is None or time.time() - s["t"] > 1.0:
        code = 3
        break
    if s["state"] != "ENABLED":
        code = 3
        break
    if s["arrived"] and not s.get("queue") and s["t"] > t0 + 0.3:
        code = 0
        break
    time.sleep(0.1)
s = state() or {}
print(f"{cmd}: {['arrived', '', 'TIMED OUT', 'FAULT/STOPPED'][code]} after {time.time() - t0:.1f}s "
      f"state={s.get('state')} target={s.get('target')}")
for key in ("q", "error", "torque"):
    if s.get(key) is not None:
        print(f"  {key:6} " + " ".join(f"{x:+.3f}" for x in s[key]))
print("  temps  " + " ".join(f"{x:.0f}" for x in s.get("temps", [])), " faults", s.get("faults"))
sys.exit(code)
