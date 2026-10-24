"""Analog Discovery 3 driver over the WaveForms SDK (dwf.dll, installed with WaveForms).

Read-only: the wavegen, power supplies, and DIO outputs are reset to off on open and never
enabled. Only one process can own the device, so close the WaveForms app while this runs.
Calls block; the instrument service runs them in a worker thread, one at a time.
"""

import ctypes
import sys
import time
from dataclasses import dataclass, field
from typing import Literal, Protocol

from .rig import LogicType

# dwf.h constants used here.
TRIGSRC_NONE = 0
TRIGSRC_DETECTOR_ANALOG_IN = 2
TRIGSRC_DETECTOR_DIGITAL_IN = 3
TRIGTYPE_EDGE = 0
SLOPE = {"rising": 0, "falling": 1}
ACQMODE_SINGLE = 0  # acqmodeSingle; 1 is ScanShift, which never reaches Done
STATE_DONE = 2
# Digilent: the input offset needs time to settle after its configuration changes.
OFFSET_SETTLE_S = 2.0


class InstrumentUnavailable(RuntimeError):
    pass


class InstrumentError(RuntimeError):
    pass


@dataclass(frozen=True)
class AnalogRequest:
    channel: int  # 1 or 2
    range_v: float
    offset_v: float
    sample_rate_hz: float
    samples: int
    trigger_level_v: float | None
    trigger_slope: Literal["rising", "falling"]
    auto_timeout_s: float


@dataclass
class AnalogCapture:
    samples: list[float]
    sample_rate_hz: float
    range_v: float
    offset_v: float
    triggered: bool | None  # None when no trigger was requested


@dataclass(frozen=True)
class DigitalRequest:
    pins: tuple[int, ...]
    sample_rate_hz: float
    samples: int
    trigger_pin: int
    trigger_edge: Literal["rise", "fall"]
    auto_timeout_s: float
    # Context only; the real device ignores it. The simulator uses it to synthesize signals.
    roles: dict[int, str] = field(default_factory=dict)
    logic: LogicType | None = None


@dataclass
class DigitalCapture:
    words: list[int]  # one 16-bit word per sample, bit n = DIO n
    sample_rate_hz: float
    triggered: bool


class Instrument(Protocol):
    mode: Literal["live", "simulated"]

    def info(self) -> dict: ...
    def capture_analog(self, request: AnalogRequest) -> AnalogCapture: ...
    def capture_digital(self, request: DigitalRequest) -> DigitalCapture: ...
    def close(self) -> None: ...


def load_dwf():
    try:
        if sys.platform.startswith("win"):
            return ctypes.cdll.LoadLibrary("dwf")
        if sys.platform == "darwin":
            return ctypes.cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
        return ctypes.cdll.LoadLibrary("libdwf.so")
    except OSError as exc:
        raise InstrumentUnavailable(f"WaveForms SDK not found: {exc}") from exc


class AD3:
    mode = "live"

    def __init__(self, serial: str | None = None, dwf=None):
        self.serial = serial
        self.dwf = dwf or load_dwf()
        self.hdwf: ctypes.c_int | None = None
        self.name = ""
        self.device_serial = ""
        self._analog_config = None

    def _error(self) -> str:
        msg = ctypes.create_string_buffer(512)
        self.dwf.FDwfGetLastErrorMsg(msg)
        return msg.value.decode(errors="replace").strip() or "unknown WaveForms error"

    def _call(self, name: str, *args):
        if not getattr(self.dwf, name)(*args):
            raise InstrumentError(f"{name}: {self._error()}")

    def open(self):
        if self.hdwf is not None:
            return
        count = ctypes.c_int()
        self._call("FDwfEnum", ctypes.c_int(0), ctypes.byref(count))
        found = []
        for index in range(count.value):
            name, sn = ctypes.create_string_buffer(64), ctypes.create_string_buffer(32)
            self.dwf.FDwfEnumDeviceName(ctypes.c_int(index), name)
            self.dwf.FDwfEnumSN(ctypes.c_int(index), sn)
            found.append((index, name.value.decode(), sn.value.decode()))
        if self.serial:
            found = [entry for entry in found if self.serial in entry[2]]
        if not found:
            wanted = f" with serial {self.serial}" if self.serial else ""
            raise InstrumentUnavailable(f"no WaveForms device{wanted} connected")
        index, self.name, self.device_serial = found[0]
        hdwf = ctypes.c_int()
        if not self.dwf.FDwfDeviceOpen(ctypes.c_int(index), ctypes.byref(hdwf)) or not hdwf.value:
            raise InstrumentUnavailable(f"couldn't open {self.name}: {self._error()}")
        self.hdwf = hdwf
        self._analog_config = None
        # Outputs off: this driver only reads.
        self._call("FDwfAnalogOutReset", hdwf, ctypes.c_int(-1))
        self._call("FDwfAnalogOutConfigure", hdwf, ctypes.c_int(-1), ctypes.c_int(0))
        self._call("FDwfDigitalIOReset", hdwf)
        self._call("FDwfDigitalIOConfigure", hdwf)

    def info(self) -> dict:
        self.open()
        return {"name": self.name, "serial": self.device_serial}

    def _wait(self, status: str, deadline: float):
        state = ctypes.c_ubyte()
        while True:
            self._call(status, self.hdwf, ctypes.c_int(1), ctypes.byref(state))
            if state.value == STATE_DONE:
                return
            if time.monotonic() > deadline:
                raise InstrumentError("capture_timeout")
            time.sleep(0.002)

    def capture_analog(self, request: AnalogRequest) -> AnalogCapture:
        self.open()
        h, ch = self.hdwf, ctypes.c_int(request.channel - 1)
        self._call("FDwfAnalogInReset", h)
        self._call("FDwfAnalogInChannelEnableSet", h, ctypes.c_int(-1), ctypes.c_int(0))
        self._call("FDwfAnalogInChannelEnableSet", h, ch, ctypes.c_int(1))
        self._call("FDwfAnalogInChannelRangeSet", h, ch, ctypes.c_double(request.range_v))
        self._call("FDwfAnalogInChannelOffsetSet", h, ch, ctypes.c_double(request.offset_v))
        self._call("FDwfAnalogInAcquisitionModeSet", h, ctypes.c_int(ACQMODE_SINGLE))
        self._call("FDwfAnalogInFrequencySet", h, ctypes.c_double(request.sample_rate_hz))
        low, high = ctypes.c_int(), ctypes.c_int()
        self._call("FDwfAnalogInBufferSizeInfo", h, ctypes.byref(low), ctypes.byref(high))
        n = max(low.value, min(request.samples, high.value))
        self._call("FDwfAnalogInBufferSizeSet", h, ctypes.c_int(n))
        if request.trigger_level_v is None:
            self._call("FDwfAnalogInTriggerSourceSet", h, ctypes.c_ubyte(TRIGSRC_NONE))
        else:
            self._call(
                "FDwfAnalogInTriggerSourceSet", h, ctypes.c_ubyte(TRIGSRC_DETECTOR_ANALOG_IN)
            )
            self._call(
                "FDwfAnalogInTriggerAutoTimeoutSet", h, ctypes.c_double(request.auto_timeout_s)
            )
            self._call("FDwfAnalogInTriggerChannelSet", h, ch)
            self._call("FDwfAnalogInTriggerTypeSet", h, ctypes.c_int(TRIGTYPE_EDGE))
            self._call("FDwfAnalogInTriggerLevelSet", h, ctypes.c_double(request.trigger_level_v))
            self._call(
                "FDwfAnalogInTriggerConditionSet", h, ctypes.c_int(SLOPE[request.trigger_slope])
            )
        rate, span, offset = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
        self._call("FDwfAnalogInFrequencyGet", h, ctypes.byref(rate))
        self._call("FDwfAnalogInChannelRangeGet", h, ch, ctypes.byref(span))
        self._call("FDwfAnalogInChannelOffsetGet", h, ch, ctypes.byref(offset))
        config = (request.channel, span.value, offset.value)
        if config != self._analog_config:
            self._call("FDwfAnalogInConfigure", h, ctypes.c_int(1), ctypes.c_int(0))
            time.sleep(OFFSET_SETTLE_S)
            self._analog_config = config
        self._call("FDwfAnalogInConfigure", h, ctypes.c_int(1), ctypes.c_int(1))
        timeout = request.auto_timeout_s if request.trigger_level_v is not None else 0.0
        self._wait("FDwfAnalogInStatus", time.monotonic() + timeout + n / rate.value + 2.0)
        data = (ctypes.c_double * n)()
        self._call("FDwfAnalogInStatusData", h, ch, data, ctypes.c_int(n))
        triggered = None
        if request.trigger_level_v is not None:
            auto = ctypes.c_int()
            self._call("FDwfAnalogInStatusAutoTriggered", h, ctypes.byref(auto))
            triggered = not auto.value
        return AnalogCapture(list(data), rate.value, span.value, offset.value, triggered)

    def capture_digital(self, request: DigitalRequest) -> DigitalCapture:
        self.open()
        h = self.hdwf
        self._call("FDwfDigitalInReset", h)
        clock = ctypes.c_double()
        self._call("FDwfDigitalInInternalClockInfo", h, ctypes.byref(clock))
        divider = max(1, round(clock.value / request.sample_rate_hz))
        self._call("FDwfDigitalInDividerSet", h, ctypes.c_uint(divider))
        self._call("FDwfDigitalInSampleFormatSet", h, ctypes.c_int(16))
        self._call("FDwfDigitalInAcquisitionModeSet", h, ctypes.c_int(ACQMODE_SINGLE))
        high = ctypes.c_int()
        self._call("FDwfDigitalInBufferSizeInfo", h, ctypes.byref(high))
        n = min(request.samples, high.value)
        self._call("FDwfDigitalInBufferSizeSet", h, ctypes.c_int(n))
        self._call("FDwfDigitalInTriggerSourceSet", h, ctypes.c_ubyte(TRIGSRC_DETECTOR_DIGITAL_IN))
        self._call("FDwfDigitalInTriggerAutoTimeoutSet", h, ctypes.c_double(request.auto_timeout_s))
        mask = 1 << request.trigger_pin
        rise, fall = (mask, 0) if request.trigger_edge == "rise" else (0, mask)
        self._call(
            "FDwfDigitalInTriggerSet",
            h,
            ctypes.c_uint(0),
            ctypes.c_uint(0),
            ctypes.c_uint(rise),
            ctypes.c_uint(fall),
        )
        self._call("FDwfDigitalInTriggerPositionSet", h, ctypes.c_uint(n - n // 10))
        actual = ctypes.c_uint()
        self._call("FDwfDigitalInDividerGet", h, ctypes.byref(actual))
        rate = clock.value / actual.value
        self._call("FDwfDigitalInConfigure", h, ctypes.c_int(1), ctypes.c_int(1))
        self._wait(
            "FDwfDigitalInStatus", time.monotonic() + request.auto_timeout_s + n / rate + 2.0
        )
        data = (ctypes.c_uint16 * n)()
        self._call("FDwfDigitalInStatusData", h, data, ctypes.c_int(2 * n))
        auto = ctypes.c_int()
        self._call("FDwfDigitalInStatusAutoTriggered", h, ctypes.byref(auto))
        return DigitalCapture(list(data), rate, not auto.value)

    def close(self):
        if self.hdwf is not None:
            self.dwf.FDwfDeviceClose(self.hdwf)
            self.hdwf = None
