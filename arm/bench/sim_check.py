"""Hardware-free check of teach.py and run.py against simulated motors that track MIT commands.

Usage: python -X utf8 sim_check.py   (prints SIM OK; never opens the USB adapter)
The motors sag under gravity while the arm is raised and accept injected torques, so the run
covers waypoint sequencing, sag correction, the contact guard, jog/save and the home route.
"""

import json
import runpy
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import gsusb_backend
from maker_arm import protocol as p
from maker_arm.transport.mock import MockBackend


def feedback_frame(motor_id, pos=0.0, vel=0.0, tau=0.0, temp=30.0, fault=0, mode=2, params=None):
    """A private-protocol feedback frame (same layout as the SDK's test helper)."""
    pr = params or p.RS00
    cid = (p.COMM_FEEDBACK << 24) | (mode << 22) | (fault << 16) | (motor_id << 8) | p.HOST_CAN_ID
    data = struct.pack(">HHHH",
                       p.float_to_u16(pos, pr.p_min, pr.p_max),
                       p.float_to_u16(vel, pr.v_min, pr.v_max),
                       p.float_to_u16(tau, pr.t_min, pr.t_max),
                       int(temp * 10))
    return cid, data

PARAMS = {i: (p.RS02 if i in (2, 3) else p.RS00) for i in range(1, 8)}
pos = {i: 0.0 for i in range(1, 8)}
sent_enable = []
SAG = {2: 0.023, 3: 0.035}  # rad the real arm sat short of its command at "above" (gravity, PD only)
TAU = {}  # injected joint torques (Nm), e.g. a contact


class SimBackend(MockBackend):
    def __init__(self, *a, **k):
        super().__init__()
        self.responder = self._respond

    def _respond(self, cid, data):
        mid, comm = cid & 0xFF, (cid >> 24) & 0x1F
        if mid not in pos:
            return None
        pr = PARAMS[mid]
        if comm == p.COMM_READ_PARAM:
            idx = struct.unpack("<H", data[:2])[0]
            value = struct.pack("<B3x", 0) if idx == p.ParamIndex.RUN_MODE else struct.pack("<I", 4000)
            return [((p.COMM_READ_PARAM << 24) | (mid << 8) | p.HOST_CAN_ID, struct.pack("<H2x", idx) + value)]
        if comm == p.COMM_ENABLE:
            sent_enable.append(mid)
        if comm == p.COMM_MIT:
            sag = SAG.get(mid, 0.0) if pos[2] < -0.8 else 0.0  # gravity sag only while raised
            pos[mid] = p.u16_to_float(struct.unpack(">H", data[:2])[0], pr.p_min, pr.p_max) - sag
        return [feedback_frame(mid, pos=pos[mid], tau=TAU.get(mid, 0.0), params=pr, mode=2 if sent_enable else 0)]


gsusb_backend.GsUsbBackend = SimBackend
here = Path(__file__).resolve().parent
work = Path(tempfile.mkdtemp(prefix="arm_bench_sim_"))  # never the real work/ folder


def send(cmd, delay=0.3):
    subprocess.run([sys.executable, str(here / "send.py"), str(work), cmd], check=True)
    time.sleep(delay)


def run_script(name, args):
    sys.argv = [name, *args]
    runpy.run_path(str(here / name), run_name="__main__")


# Teach: hand-move to a knob pose, record it, then wrist and jaw partial points.
t = threading.Thread(target=run_script, args=("teach.py", [str(work)]), daemon=True)
t.start()
time.sleep(0.5)
pos.update({1: 0.8, 2: -0.6, 3: 1.2, 4: 0.4, 5: 0.3, 6: 0.0, 7: -0.5})
send("record knob")
pos[6] = 0.6
send("record range_v J6")
pos[6] = -0.4
send("record range_ohm J6")
pos[7] = -1.2
send("record jaw_closed J7")
pos[7] = -0.3
send("record jaw_open J7")
pos.update({6: 0.6, 7: -0.9})  # knob turned by hand, jaws loosened (as on the real arm)
send("record knob_v")
pos.update({2: -1.0, 3: 0.6, 7: -0.3})
send("record above")
pos.update({i: 0.0 for i in pos})  # back to rest before powering
send("quit")
t.join(timeout=5)
print((work / "teach.yaml").read_text())

# Run: home -> knob -> close jaw -> turn wrist -> release refused -> quit (homes, releases).
t = threading.Thread(target=run_script, args=("run.py", [str(work), "--max-velocity", "2.0"]), daemon=True)
t.start()
time.sleep(1.0)
send("goto knob", 1.5)
send("goto jaw_closed", 1.0)
send("goto range_v", 1.0)
q = [round(pos[i], 2) for i in range(1, 8)]
print("after knob+jaw+range_v:", q)
assert q == [0.8, -0.6, 1.2, 0.4, 0.3, 0.6, -1.2], q
send("goto range_ohm", 1.0)
assert round(pos[6], 2) == -0.4 and round(pos[7], 2) == -1.2
send("release", 0.5)
assert t.is_alive(), "release away from home must be refused"
send("jog J7 0.05", 0.5)
assert round(pos[7], 2) == -1.15, pos[7]
send("jog J7 0.5", 0.3)
assert round(pos[7], 2) == -1.15, "oversized jog must be refused"
send("save jaw_tight J7", 0.3)
assert (work / "teach.yaml").read_text().count("jaw_tight") == 1
send("home", 3.0)  # must route jaw_open -> above -> home
send("approach knob", 4.0)
assert [round(pos[i], 2) for i in range(1, 7)] == [0.8, -0.6, 1.2, 0.4, 0.3, 0.0]

# turn: grip before the twist, keep gripping through it (jaw_closed, not the pose's loose J7), end above.
send("turn knob knob_v", 6.0)
turn_log = [l for l in (work / "log.txt").read_text().splitlines() if " goto " in l][-8:]
assert [l.split()[2].rstrip(":") for l in turn_log] == [
    "above", "above@knob", "knob@jaw_open", "knob@jaw_closed", "knob_v@jaw_closed", "knob_v@jaw_open",
    "above@knob_v", "above"], turn_log
assert "J7=-1.200" in turn_log[4], "twist must keep the jaw_closed grip, not knob_v's loose J7"
# hover = above's arm joints with the knob's wrist angle and open jaws
assert "J1=+0.800 J2=-1.000 J3=+0.600 J4=+0.400 J5=+0.300 J6=+0.000 J7=-0.300" in turn_log[1], turn_log[1]
assert [round(pos[i], 2) for i in range(1, 8)] == [0.8, -1.0, 0.6, 0.4, 0.3, 0.6, -0.3], pos
# Regression (real arm 2026-10-25): "home" from "above" stalled at its jaw_open step because a
# jaw-only move was treated as contact, dropping the sag correction. It must reach home now.
send("home", 6.0)
assert all(abs(pos[i]) < 0.02 for i in range(1, 7)), f"home from above must arrive: {pos}"
# Contact guard: a J1 push during the descent to the knob must freeze the arm and clear the queue.
send("approach knob", 0.1)
deadline = time.time() + 8
def runner_target():
    try:
        return json.loads((work / "state.json").read_text(encoding="utf-8"))["target"]
    except (OSError, ValueError, KeyError):
        return None
while time.time() < deadline and not (runner_target() == "knob@jaw_open" and -0.9 < pos[2] < -0.7):
    time.sleep(0.01)  # mid-descent from the hover to the knob
TAU[1] = 4.0
time.sleep(0.5)
TAU.clear()
assert "GUARD: J[1]" in (work / "log.txt").read_text(), "guard must trip on a 4 Nm J1 push"
frozen = dict(pos)
time.sleep(0.5)
assert all(abs(frozen[i] - pos[i]) < 0.01 for i in pos), "arm must stay frozen after the guard"
send("quit", 6.0)
t.join(timeout=5)
assert not t.is_alive()
assert all(abs(x) < 0.05 for x in pos.values()), pos
log_text = (work / "log.txt").read_text()
print(log_text)
gotos = [(l.split()[0], l.split()[2].rstrip(":")) for l in log_text.splitlines() if " goto " in l]
names = [n for _, n in gotos]
routes = [i for i in range(len(names) - 3) if names[i:i + 4] == ["jaw_open", "above@here", "above", "home"]]
assert len(routes) == 3, names  # home from the knob, home from above, and the final "quit"
# The first home starts gripping (J7 -1.15): "above" must wait ~0.43 s for the jaws to open.
i = routes[0]
h, m, s = (float(x) for x in gotos[i][0].split(":"))
h2, m2, s2 = (float(x) for x in gotos[i + 1][0].split(":"))
gap = (h2 - h) * 3600 + (m2 - m) * 60 + (s2 - s)
assert gap >= 0.3, f"above must wait for the jaws to open (0.85 rad at 2 rad/s): gap {gap:.3f}s"
print("SIM OK")
