"""Host driver for the ESP32-S3 sensor head: an AMG8833 8x8 thermal camera and a VL53L0X
(or VL53L1X) time-of-flight range sensor. Wire format: firmware/amg8833_usb/src/protocol.h.

A background thread keeps reading, so the latest value of each sensor is always at hand, and
it reopens the port if the board resets or the cable is pulled.

    from sensor_head import SensorHead
    with SensorHead() as head:          # auto-detects the port
        head.tof()                      # latest RangeReading, or None
        head.thermal()                  # latest ThermalFrame, or None; .celsius is 8x8
        head.wait_thermal()             # blocks for the next frame

Only one process can open the port. On the rig that's the arm server
(sim_server.py --sensors), which serves both sensors at /api/sensors.

CLI:
    python sensor_head.py               # print both streams
    python sensor_head.py --view        # live thermal window with the range overlaid
    python sensor_head.py --port COM5 --fps 1
"""

from __future__ import annotations

import argparse
import contextlib
import struct
import threading
import time
from dataclasses import dataclass

import numpy as np
import serial
from serial.tools import list_ports

SYNC = b"\xa5\x5a"
TYPE_FRAME = 0x01
TYPE_STATUS = 0x02
TYPE_RANGE = 0x03
MAX_PAYLOAD = 256
FRAME_STRUCT = struct.Struct("<IQh64h")
RANGE_STRUCT = struct.Struct("<IQHBB")
ESPRESSIF_VID = 0x303A
REOPEN_S = 1.0
# Statuses the device sends on its own, never as a command reply.
UNSOLICITED = (
    "err sensor_not_found",
    "ok sensor_found",
    "err i2c_read",
    "err tof_",
    "ok tof_found",
)


@dataclass(frozen=True)
class ThermalFrame:
    seq: int
    device_time_us: int
    host_time: float
    thermistor_c: float
    celsius: np.ndarray  # (8, 8) float32, sensor row-major


@dataclass(frozen=True)
class RangeReading:
    seq: int
    device_time_us: int
    host_time: float
    range_mm: int
    valid: bool  # False: no target in range, too much ambient light, or a weak return
    raw_status: int  # the sensor model's own status code, for diagnostics


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


def parse_info(text: str) -> dict[str, str]:
    """'info fw=2.0.0 tof=VL53L0X ...' -> {'fw': '2.0.0', 'tof': 'VL53L0X', ...}"""
    return dict(field.split("=", 1) for field in text.split()[1:] if "=" in field)


class SensorHead:
    def __init__(self, port: str | None = None, timeout: float = 2.0):
        self.port = port or find_port()
        self.timeout = timeout
        self.info: dict[str, str] = {}
        self.error: str | None = None  # why the port is closed, while it is
        self.last_status: str | None = None
        self.dropped_frames = 0
        self.dropped_ranges = 0
        self._frame: ThermalFrame | None = None
        self._range: RangeReading | None = None
        self._replies: list[str] = []
        self._last_seq: dict[int, int] = {}
        self._buf = bytearray()
        self._cond = threading.Condition()
        self._closed = False
        self._ser: serial.Serial | None = None
        self._open()  # fail loudly on a wrong port; later drops are retried quietly
        self._thread = threading.Thread(target=self._run, name="sensor-head", daemon=True)
        self._thread.start()

    def _open(self) -> None:
        self._ser = serial.Serial(self.port, timeout=0.05)
        self._ser.reset_input_buffer()
        self._buf.clear()
        self._last_seq.clear()
        self.error = None

    def close(self) -> None:
        self._closed = True
        self._thread.join(timeout=1.0)
        if self._ser is not None:
            self._ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def connected(self) -> bool:
        return self._ser is not None

    def tof(self) -> RangeReading | None:
        return self._range

    def thermal(self) -> ThermalFrame | None:
        return self._frame

    def wait_thermal(self) -> ThermalFrame:
        """Blocks until a frame newer than the current one arrives."""
        return self._wait(lambda: self._frame, "frame")

    def wait_tof(self) -> RangeReading:
        return self._wait(lambda: self._range, "range")

    def _wait(self, latest, what: str):
        with self._cond:
            before = latest()
            if not self._cond.wait_for(lambda: latest() is not before, self.timeout):
                raise TimeoutError(f"no {what} from {self.port}; last status: {self.last_status}")
            return latest()

    def command(self, line: str) -> str:
        """Sends one command and returns the device's reply."""
        with self._cond:
            if self._ser is None:
                raise ConnectionError(f"{self.port} is closed: {self.error}")
            self._replies.clear()
            self._ser.write(line.strip().encode() + b"\n")
            if not self._cond.wait_for(lambda: self._replies, self.timeout):
                raise TimeoutError(f"no reply to {line!r}")
            return self._replies.pop(0)

    def snapshot(self, now: float | None = None) -> dict:
        """JSON-ready view of both sensors, with each reading's age."""
        now = time.time() if now is None else now
        frame, reading = self._frame, self._range
        thermal = tof = None
        if frame is not None:
            c = frame.celsius
            row, col = divmod(int(np.argmax(c)), 8)
            thermal = {
                "seq": frame.seq,
                "age_s": round(now - frame.host_time, 3),
                "thermistor_c": frame.thermistor_c,
                "min_c": float(c.min()),
                "max_c": float(c.max()),
                "mean_c": round(float(c.mean()), 2),
                "hottest_cell": [row, col],
                "celsius": c.tolist(),
            }
        if reading is not None:
            tof = {
                "seq": reading.seq,
                "age_s": round(now - reading.host_time, 3),
                "range_mm": reading.range_mm,
                "valid": reading.valid,
                "raw_status": reading.raw_status,
            }
        return {
            "connected": self.connected,
            "port": self.port,
            "error": self.error,
            "firmware": self.info.get("fw"),
            "tof_model": self.info.get("tof"),
            "last_status": self.last_status,
            "dropped": {"frames": self.dropped_frames, "ranges": self.dropped_ranges},
            "tof": tof,
            "thermal": thermal,
        }

    def _run(self) -> None:
        asked_info = False
        next_open = 0.0
        while not self._closed:
            if self._ser is None:
                if time.monotonic() < next_open:
                    time.sleep(0.05)
                    continue
                try:
                    with self._cond:
                        self._open()
                    asked_info = False
                except serial.SerialException as exc:
                    self.error = str(exc)
                    next_open = time.monotonic() + REOPEN_S
                    continue
            try:
                if not asked_info:
                    self._ser.write(b"info\n")
                    asked_info = True
                self._buf += self._ser.read(max(1, self._ser.in_waiting))
                self._parse()
            except (serial.SerialException, OSError) as exc:
                with self._cond:
                    self.error = f"lost: {exc}"
                    with contextlib.suppress(Exception):  # the port is already gone
                        self._ser.close()
                    self._ser = None
                next_open = time.monotonic() + REOPEN_S

    def _parse(self) -> None:
        while True:
            start = self._buf.find(SYNC)
            if start < 0:
                del self._buf[:-1]
                return
            del self._buf[:start]
            if len(self._buf) < 5:
                return
            ptype = self._buf[2]
            (length,) = struct.unpack_from("<H", self._buf, 3)
            if length > MAX_PAYLOAD:
                del self._buf[:2]
                continue
            total = 5 + length + 2
            if len(self._buf) < total:
                return
            packet = bytes(self._buf[:total])
            (crc,) = struct.unpack_from("<H", packet, total - 2)
            if crc16(packet[2:-2]) != crc:
                # False sync inside a payload; resync one byte later.
                del self._buf[:1]
                continue
            del self._buf[:total]
            self._handle(ptype, packet[5:-2])

    def _gap(self, ptype: int, seq: int) -> int:
        last = self._last_seq.get(ptype)
        self._last_seq[ptype] = seq
        return seq - last - 1 if last is not None and seq > last + 1 else 0

    def _handle(self, ptype: int, payload: bytes) -> None:
        now = time.time()
        with self._cond:
            if ptype == TYPE_STATUS:
                text = payload.decode(errors="replace")
                self.last_status = text
                if text.startswith("info"):
                    self.info = parse_info(text)
                if not text.startswith(UNSOLICITED):
                    self._replies.append(text)
            elif ptype == TYPE_FRAME and len(payload) == FRAME_STRUCT.size:
                seq, t_us, therm, *pixels = FRAME_STRUCT.unpack(payload)
                self.dropped_frames += self._gap(ptype, seq)
                celsius = np.asarray(pixels, dtype=np.float32).reshape(8, 8) * 0.25
                self._frame = ThermalFrame(seq, t_us, now, therm * 0.0625, celsius)
            elif ptype == TYPE_RANGE and len(payload) == RANGE_STRUCT.size:
                seq, t_us, mm, valid, raw = RANGE_STRUCT.unpack(payload)
                self.dropped_ranges += self._gap(ptype, seq)
                self._range = RangeReading(seq, t_us, now, mm, bool(valid), raw)
            else:
                return
            self._cond.notify_all()


def _view(head: SensorHead) -> None:
    import cv2

    while True:
        frame = head.wait_thermal()
        lo, hi = float(frame.celsius.min()), float(frame.celsius.max())
        norm = ((frame.celsius - lo) / max(hi - lo, 1e-3) * 255).astype(np.uint8)
        img = cv2.applyColorMap(
            cv2.resize(norm, (320, 320), interpolation=cv2.INTER_CUBIC), cv2.COLORMAP_INFERNO
        )
        reading = head.tof()
        tof = "--" if reading is None or not reading.valid else f"{reading.range_mm} mm"
        cv2.putText(img, f"{lo:.1f}-{hi:.1f} C  ToF {tof}", (5, 20), 0, 0.6, (255, 255, 255), 1)
        cv2.imshow("Sensor head", img)
        if cv2.waitKey(1) in (27, ord("q")):
            break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port")
    parser.add_argument("--fps", type=int, choices=(1, 10))
    parser.add_argument("--avg", choices=("on", "off"))
    parser.add_argument("--scan", action="store_true", help="check both buses' wiring and exit")
    parser.add_argument("--view", action="store_true", help="show a live OpenCV window")
    args = parser.parse_args()

    with SensorHead(args.port) as head:
        print(head.port, head.command("info"))
        if args.scan:
            print(head.command("scan"))
            print(head.command("scan tof"))
            return
        if args.fps:
            print(head.command(f"fps {args.fps}"))
        if args.avg:
            print(head.command(f"avg {args.avg}"))
        if args.view:
            _view(head)
            return
        while True:
            f = head.wait_thermal()
            r = head.tof()
            tof = "none" if r is None else f"{r.range_mm}mm{'' if r.valid else ' (invalid)'}"
            print(
                f"seq={f.seq} therm={f.thermistor_c:.2f}C "
                f"min={f.celsius.min():.2f} max={f.celsius.max():.2f} "
                f"mean={f.celsius.mean():.2f} tof={tof} "
                f"dropped={head.dropped_frames}/{head.dropped_ranges}"
            )


if __name__ == "__main__":
    main()
