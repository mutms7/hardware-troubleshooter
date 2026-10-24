"""Calibrate servos 1-4 (pan, shoulder lift, elbow flex, wrist flex).

ID 5 (probe turret) is never touched unless --probes is given, and ID 6 (gripper)
is never touched at all.

Nothing is written to servo EEPROM. Positions are only read, and the result goes
to a JSON file that the control code applies in software.

Steps:
  1. Torque is switched off on IDs 1-4 so you can move the arm by hand.
     THE ARM WILL GO LIMP. Support it before you start.
  2. Put every joint at the middle of its range -> Enter. This is the zero pose.
  3. Sweep every joint through its full safe range -> Enter. Min/max are recorded.
  4. (--probes) Rotate ID 5 to each probe detent by hand and record its position.

Assumes the STS/SMS register map (Torque_Enable=40, Present_Position=56). The
Hiwonder fork reuses that table, but run --check first to confirm the servos
answer before trusting the numbers.

Usage:
  python calibrate.py --port COM5 --check
  python calibrate.py --port COM5
  python calibrate.py --port COM5 --probes 4
"""
import argparse
import json
import sys
import time

from scservo_sdk import PacketHandler, PortHandler, COMM_SUCCESS

TORQUE_ENABLE = 40
PRESENT_POSITION = 56
TICKS = 4096
JOINTS = {1: "shoulder_pan", 2: "shoulder_lift", 3: "elbow_flex", 4: "wrist_flex"}
PROBE_ID = 5


class Bus:
    def __init__(self, port, baud):
        self.port = PortHandler(port)
        self.ph = PacketHandler(0)
        if not self.port.openPort():
            sys.exit(f"Could not open {port}. Is another program using it?")
        self.port.setBaudRate(baud)

    def ping(self, sid):
        _, result, _ = self.ph.ping(self.port, sid)
        return result == COMM_SUCCESS

    def read_pos(self, sid, retries=3):
        for _ in range(retries):
            val, result, _ = self.ph.read2ByteTxRx(self.port, sid, PRESENT_POSITION)
            if result == COMM_SUCCESS:
                return val % TICKS
        raise ConnectionError(f"No reply from servo {sid}")

    def torque(self, sid, on):
        self.ph.write1ByteTxRx(self.port, sid, TORQUE_ENABLE, 1 if on else 0)

    def close(self):
        self.port.closePort()


def delta(raw, mid):
    """Signed ticks from mid, wrapped to [-2048, 2047] so a joint crossing 0/4095 stays continuous."""
    return ((raw - mid + TICKS // 2) % TICKS) - TICKS // 2


def check(bus, ids):
    ok = True
    for sid in ids:
        if not bus.ping(sid):
            print(f"  ID {sid}: NO REPLY")
            ok = False
            continue
        print(f"  ID {sid}: ok, position {bus.read_pos(sid)}")
    return ok


def calibrate_joints(bus):
    ids = list(JOINTS)
    for sid in ids:
        bus.torque(sid, False)
    print("\nTorque is OFF on IDs 1-4. Support the arm; it can fall.")

    input("\nMove every joint to the MIDDLE of its range, then press Enter... ")
    mid = {sid: bus.read_pos(sid) for sid in ids}
    print("  zero pose:", {JOINTS[s]: p for s, p in mid.items()})

    print("\nNow move each joint slowly through its FULL safe range. Enter when done.")
    lo = {sid: 0 for sid in ids}
    hi = {sid: 0 for sid in ids}
    import threading
    done = threading.Event()
    threading.Thread(target=lambda: (input(), done.set()), daemon=True).start()
    while not done.is_set():
        line = []
        for sid in ids:
            d = delta(bus.read_pos(sid), mid[sid])
            lo[sid], hi[sid] = min(lo[sid], d), max(hi[sid], d)
            line.append(f"{JOINTS[sid][:5]} {lo[sid]:+5d}..{hi[sid]:+5d}")
        print("\r  " + " | ".join(line), end="", flush=True)
        time.sleep(0.03)
    print()

    out = {}
    for sid in ids:
        if hi[sid] - lo[sid] < 50:
            print(f"  WARNING: {JOINTS[sid]} barely moved ({lo[sid]}..{hi[sid]} ticks)")
        out[JOINTS[sid]] = {
            "id": sid,
            "mid_raw": mid[sid],
            "min_delta": lo[sid],
            "max_delta": hi[sid],
            "deg_per_tick": 360.0 / TICKS,
        }
    return out


def teach_probes(bus, n):
    bus.torque(PROBE_ID, False)
    print(f"\nTorque is OFF on ID {PROBE_ID} only. Rotate the turret by hand.")
    probes = {}
    for i in range(1, n + 1):
        input(f"  Rotate to probe {i}'s detent, press Enter... ")
        probes[f"probe_{i}"] = bus.read_pos(PROBE_ID)
        print(f"    probe_{i} = {probes[f'probe_{i}']}")
    return probes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="e.g. COM5")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--out", default="calibration.json")
    ap.add_argument("--check", action="store_true", help="ping and read IDs 1-5, then exit")
    ap.add_argument("--probes", type=int, metavar="N", help="also teach N probe positions for ID 5")
    ap.add_argument("--probes-only", type=int, metavar="N",
                    help="keep the existing calibration file and only (re)teach N probe positions")
    args = ap.parse_args()

    bus = Bus(args.port, args.baud)
    try:
        print("Checking servos:")
        if not check(bus, [*JOINTS, PROBE_ID]):
            sys.exit("Some servos did not answer. Check power, port, baud and IDs.")
        if args.check:
            return
        if args.probes_only:
            with open(args.out) as f:
                result = json.load(f)
            result["probes"] = {"id": PROBE_ID, "positions": teach_probes(bus, args.probes_only)}
            with open(args.out, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nSaved probes into {args.out}")
            return
        result = {"joints": calibrate_joints(bus)}
        if args.probes:
            result["probes"] = {"id": PROBE_ID, "positions": teach_probes(bus, args.probes)}
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved {args.out}")
    finally:
        bus.close()


if __name__ == "__main__":
    main()
