"""Read-only dump of the settings that decide how finely a servo can hold a position (writes nothing).
Stop the arm server first: it holds the serial port.   python read_servo_regs.py [COM10]"""
import sys

from scservo_sdk import COMM_SUCCESS

from calibrate import JOINTS, PROBE_ID, Bus

# STS/SMS-style register map (the P, D, I registers 21-23 were already confirmed on these HX servos)
ONE = {"P coeff (21)": 21, "D coeff (22)": 22, "I coeff (23)": 23, "CW dead zone (26)": 26, "CCW dead zone (27)": 27,
       "angular resolution (30)": 30, "mode (33)": 33, "speed P (37)": 37, "speed I (39)": 39, "torque enable (40)": 40,
       "acceleration (41)": 41}
TWO = {"min start force (24)": 24, "protect current (28)": 28, "max torque limit (16)": 16, "goal speed (46)": 46,
       "torque limit (48)": 48, "present load (60)": 60}

bus = Bus(sys.argv[1] if len(sys.argv) > 1 else "COM10", 1_000_000)
for sid in (*JOINTS, PROBE_ID):
    if not bus.ping(sid):
        print(f"ID {sid}: no reply")
        continue
    row = {}
    for name, addr in ONE.items():
        v, r, _ = bus.ph.read1ByteTxRx(bus.port, sid, addr)
        row[name] = v if r == COMM_SUCCESS else "?"
    for name, addr in TWO.items():
        v, r, _ = bus.ph.read2ByteTxRx(bus.port, sid, addr)
        row[name] = v if r == COMM_SUCCESS else "?"
    print(f"ID {sid} ({JOINTS.get(sid, 'probe turret')}):")
    print("   " + ", ".join(f"{k} = {v}" for k, v in row.items()))
bus.close()
