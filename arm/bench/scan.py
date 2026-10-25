"""Read-only bus scan (same probe as `maker-arm scan`) over the gs_usb adapter."""

import sys
import time

from gsusb_backend import GsUsbBackend
from maker_arm import protocol as p
from maker_arm.errors import fault_text

max_id = int(sys.argv[1]) if len(sys.argv) > 1 else 14
found, other = {}, []


def on_frame(cid, d):
    msg = p.parse_frame(cid, d)
    if isinstance(msg, p.MotorFeedback):
        found[msg.motor_id] = msg
    else:
        other.append((hex(cid), d.hex()))


be = GsUsbBackend()
be.set_recv_callback(on_frame)
be.open()
print("adapter open at 1 Mbit/s")
try:
    for mid in range(1, max_id + 1):
        for _ in range(3):
            be.send(*p.encode_disable(mid))
            deadline = time.monotonic() + 0.05
            while mid not in found and time.monotonic() < deadline:
                time.sleep(0.001)
            if mid in found:
                break
finally:
    be.close()

if not found:
    print("no motors found - check arm power, CAN wiring, termination")
print(f"{'ID':>3} {'pos_rad':>10} {'vel':>8} {'torque':>8} {'temp_C':>8} fault")
for mid in sorted(found):
    fb = found[mid]
    print(f"{mid:>3} {fb.position:>10.3f} {fb.velocity:>8.3f} {fb.torque:>8.3f} "
          f"{fb.temperature:>8.1f} {fault_text(fb.fault_bits)}")
print(f"other frames: {len(other)} {other[:5]}")
