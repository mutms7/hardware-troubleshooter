"""Host reader for the ESP32-S3 AMG8833 USB thermal camera. See ../src/protocol.h.

    from thermal_cam import ThermalCamera
    with ThermalCamera() as cam:          # auto-detects the port
        frame = cam.read()                # ThermalFrame, celsius is an 8x8 float32 array

CLI:
    python thermal_cam.py                 # print frames
    python thermal_cam.py --view          # live OpenCV window
    python thermal_cam.py --port COM7 --fps 1
"""

from __future__ import annotations

import argparse
import struct
import time
from dataclasses import dataclass

import numpy as np
import serial
from serial.tools import list_ports

SYNC = b"\xa5\x5a"
TYPE_FRAME = 0x01
TYPE_STATUS = 0x02
MAX_PAYLOAD = 256
FRAME_STRUCT = struct.Struct("<IQh64h")
ESPRESSIF_VID = 0x303A
# Statuses the device sends on its own, never as a command reply.
UNSOLICITED = ("err sensor_not_found", "ok sensor_found", "err i2c_read")


@dataclass(frozen=True)
class ThermalFrame:
    seq: int
    device_time_us: int
    host_time: float
    thermistor_c: float
    celsius: np.ndarray  # (8, 8) float32, sensor row-major


def crc16(data: bytes, crc: int = 0xFFFF) -> int:
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else crc << 1
            crc &= 0xFFFF
    return crc


def find_port() -> str:
    candidates = [p.device for p in list_ports.comports() if p.vid == ESPRESSIF_VID]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one Espressif USB device, found {candidates or 'none'}; pass port="
        )
    return candidates[0]


class ThermalCamera:
    def __init__(self, port: str | None = None, timeout: float = 2.0):
        self.port = port or find_port()
        self.timeout = timeout
        self.statuses: list[str] = []
        self.dropped = 0
        self._last_seq: int | None = None
        self._buf = bytearray()
        self._ser = serial.Serial(self.port, timeout=0.05)
        self._ser.reset_input_buffer()

    def close(self) -> None:
        self._ser.close()

    def __enter__(self) -> ThermalCamera:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def command(self, line: str) -> str:
        """Sends one command and returns the device's next status reply."""
        self.statuses.clear()
        self._ser.write(line.strip().encode() + b"\n")
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            self._pump()
            for i, text in enumerate(self.statuses):
                if text.startswith(UNSOLICITED):
                    continue
                if text.startswith(("ok", "err", "info", "scan")):
                    del self.statuses[: i + 1]
                    return text
        raise TimeoutError(f"no reply to {line!r}")

    def read(self) -> ThermalFrame:
        """Blocks until the next frame. Raises TimeoutError if none arrives."""
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            frame = self._pump()
            if frame is not None:
                return frame
        raise TimeoutError(f"no frame from {self.port}; last status: {self.statuses[-1:]}")

    def _pump(self) -> ThermalFrame | None:
        self._buf += self._ser.read(max(1, self._ser.in_waiting))
        while True:
            start = self._buf.find(SYNC)
            if start < 0:
                del self._buf[:-1]
                return None
            del self._buf[:start]
            if len(self._buf) < 5:
                return None
            ptype = self._buf[2]
            (length,) = struct.unpack_from("<H", self._buf, 3)
            if length > MAX_PAYLOAD:
                del self._buf[:2]
                continue
            total = 5 + length + 2
            if len(self._buf) < total:
                return None
            packet = bytes(self._buf[:total])
            (crc,) = struct.unpack_from("<H", packet, total - 2)
            if crc16(packet[2:-2]) != crc:
                # False sync inside a payload; resync one byte later.
                del self._buf[:1]
                continue
            del self._buf[:total]
            payload = packet[5:-2]
            if ptype == TYPE_STATUS:
                self.statuses.append(payload.decode(errors="replace"))
            elif ptype == TYPE_FRAME and length == FRAME_STRUCT.size:
                return self._decode_frame(payload)

    def _decode_frame(self, payload: bytes) -> ThermalFrame:
        seq, t_us, therm, *pixels = FRAME_STRUCT.unpack(payload)
        if self._last_seq is not None and seq > self._last_seq + 1:
            self.dropped += seq - self._last_seq - 1
        self._last_seq = seq
        celsius = np.asarray(pixels, dtype=np.float32).reshape(8, 8) * 0.25
        return ThermalFrame(seq, t_us, time.time(), therm * 0.0625, celsius)


def _view(cam: ThermalCamera) -> None:
    import cv2

    while True:
        frame = cam.read()
        lo, hi = float(frame.celsius.min()), float(frame.celsius.max())
        norm = ((frame.celsius - lo) / max(hi - lo, 1e-3) * 255).astype(np.uint8)
        img = cv2.applyColorMap(
            cv2.resize(norm, (320, 320), interpolation=cv2.INTER_CUBIC), cv2.COLORMAP_INFERNO
        )
        cv2.putText(img, f"{lo:.1f}-{hi:.1f} C", (5, 20), 0, 0.6, (255, 255, 255), 1)
        cv2.imshow("AMG8833", img)
        if cv2.waitKey(1) in (27, ord("q")):
            break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port")
    parser.add_argument("--fps", type=int, choices=(1, 10))
    parser.add_argument("--avg", choices=("on", "off"))
    parser.add_argument("--view", action="store_true", help="show a live OpenCV window")
    args = parser.parse_args()

    with ThermalCamera(args.port) as cam:
        print(cam.port, cam.command("info"))
        if args.fps:
            print(cam.command(f"fps {args.fps}"))
        if args.avg:
            print(cam.command(f"avg {args.avg}"))
        if args.view:
            _view(cam)
            return
        while True:
            f = cam.read()
            print(
                f"seq={f.seq} therm={f.thermistor_c:.2f}C "
                f"min={f.celsius.min():.2f} max={f.celsius.max():.2f} "
                f"mean={f.celsius.mean():.2f} dropped={cam.dropped}"
            )


if __name__ == "__main__":
    main()
