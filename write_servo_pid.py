"""Set P and I on chosen servos (EEPROM, so it persists), after checking the current values are the expected ones.
Stop the arm server first (it holds the serial port).

    python write_servo_pid.py --ids 1 4 --p 40 --i 4              apply
    python write_servo_pid.py --ids 1 4 --p 24 --i 0              revert to what they were (P 24, I 0; D untouched at 32)
Registers: 21 = P, 22 = D (never written here), 23 = I, 55 = EEPROM lock (opened for the write, closed again).
"""
import argparse
import sys

from scservo_sdk import COMM_SUCCESS

from calibrate import Bus

ap = argparse.ArgumentParser()
ap.add_argument("--port", default="COM10")
ap.add_argument("--ids", type=int, nargs="+", required=True)
ap.add_argument("--p", type=int, required=True)
ap.add_argument("--i", type=int, required=True)
ap.add_argument("--expect-p", type=int, default=None, help="refuse unless P currently has this value")
args = ap.parse_args()

bus = Bus(args.port, 1_000_000)


def rd(sid, addr):
    v, r, _ = bus.ph.read1ByteTxRx(bus.port, sid, addr)
    if r != COMM_SUCCESS:
        sys.exit(f"ID {sid}: could not read register {addr}")
    return v


def wr(sid, addr, val):
    r, e = bus.ph.write1ByteTxRx(bus.port, sid, addr, val)
    if r != COMM_SUCCESS:
        sys.exit(f"ID {sid}: write to register {addr} failed")


for sid in args.ids:
    before = (rd(sid, 21), rd(sid, 22), rd(sid, 23), rd(sid, 55), rd(sid, 40))
    print(f"ID {sid} before: P {before[0]}, D {before[1]}, I {before[2]}, EEPROM lock {before[3]}, torque {before[4]}")
    if args.expect_p is not None and before[0] != args.expect_p:
        sys.exit(f"ID {sid}: P is {before[0]}, expected {args.expect_p}; not writing")
    wr(sid, 55, 0)                         # open the EEPROM lock
    wr(sid, 21, args.p)
    wr(sid, 23, args.i)
    wr(sid, 55, 1)                         # close it again
    after = (rd(sid, 21), rd(sid, 22), rd(sid, 23), rd(sid, 55), rd(sid, 40))
    print(f"ID {sid} after:  P {after[0]}, D {after[1]}, I {after[2]}, EEPROM lock {after[3]}, torque {after[4]}")
    if (after[0], after[2]) != (args.p, args.i) or after[1] != before[1]:
        sys.exit(f"ID {sid}: read-back does not match what was written")
bus.close()
print("done")
