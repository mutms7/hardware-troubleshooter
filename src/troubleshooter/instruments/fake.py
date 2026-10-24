"""Software-only AD3 and arms. Every capture is labelled mode=simulated.

The simulated bench always has the signal the request asks for: a 1 kHz 0-3.3 V square on
each scope channel and a matching UART/PWM/I2C/SPI bus on the logic pins. Scenarios inject
the failures the service must handle.
"""

import math
from typing import Literal

from .ad3 import (
    AnalogCapture,
    AnalogRequest,
    DigitalCapture,
    DigitalRequest,
    InstrumentError,
    InstrumentUnavailable,
)

FakeScenario = Literal[
    "success",
    "clipped",
    "no_trigger",
    "no_activity",
    "instrument_missing",
    "capture_error",
    "tool_change_blocked",
    "tool_not_confirmed",
    "arm_unreachable",
]
UART_MESSAGE = b"OK\r\n"
I2C_WRITE = (0x48, [0x01, 0x60])  # address, data bytes
SPI_BURSTS = ([0x9F], [0xEF, 0x40])


class FakeAD3:
    mode = "simulated"

    def __init__(self, scenario: str = "success"):
        self.scenario = scenario

    def _check(self):
        if self.scenario == "instrument_missing":
            raise InstrumentUnavailable("no WaveForms device connected")
        if self.scenario == "capture_error":
            raise InstrumentError("capture_timeout")

    def info(self) -> dict:
        self._check()
        return {"name": "Analog Discovery 3 (simulated)", "serial": "SIM0000"}

    def capture_analog(self, request: AnalogRequest) -> AnalogCapture:
        self._check()
        rate, n = request.sample_rate_hz, request.samples
        low, high = (1.2, 1.2) if self.scenario == "no_trigger" else (0.0, 3.3)
        if self.scenario == "clipped":
            low, high = -20.0, 20.0
        floor = request.offset_v - request.range_v / 2
        ceiling = request.offset_v + request.range_v / 2
        samples = []
        for i in range(n):
            phase = (i / rate * 1000.0) % 1.0
            v = (high if phase < 0.5 else low) + 0.005 * math.sin(i * 0.7)
            samples.append(min(ceiling, max(floor, v)))
        triggered = None
        if request.trigger_level_v is not None:
            triggered = low < request.trigger_level_v < high
        return AnalogCapture(samples, rate, request.range_v, request.offset_v, triggered)

    def capture_digital(self, request: DigitalRequest) -> DigitalCapture:
        self._check()
        n, rate = request.samples, request.sample_rate_hz
        by_role = {role: pin for pin, role in request.roles.items()}
        lines = {}
        if self.scenario != "no_activity":
            lines = _synthesize(request.logic, by_role, n, rate)
        words = [0] * n
        for pin in request.pins:
            line = lines.get(pin, [1] * n)
            for i in range(n):
                words[i] |= line[i] << pin
        return DigitalCapture(words, rate, self.scenario != "no_activity")

    def close(self):
        pass


def _synthesize(logic, by_role: dict[str, int], n: int, rate: float):
    protocol, lead = logic.protocol, n // 10
    if protocol == "uart":
        spb = rate / logic.baud
        line = [1] * n
        t = float(lead)
        for byte in UART_MESSAGE:
            frame = [0] + [(byte >> k) & 1 for k in range(8)] + [1]
            for b in frame:
                for i in range(int(t), min(n, int(t + spb))):
                    line[i] = b
                t += spb
        return {by_role["rx"]: line}
    if protocol == "pwm":
        period = rate / 1000.0
        return {by_role["signal"]: [1 if (i % period) < 0.25 * period else 0 for i in range(n)]}
    if protocol == "i2c":
        return dict(zip((by_role["scl"], by_role["sda"]), _i2c(n, rate, lead)))
    if protocol == "spi":
        lines = _spi(n, rate, lead, logic.cpol, logic.cpha)
        return dict(zip((by_role["sck"], by_role["data"]), lines))
    raise ValueError(f"no simulated signal for {protocol!r}")


def _i2c(n, rate, lead):
    quarter = max(1, int(rate / 100_000 / 4))
    scl, sda = [1] * n, [1] * n
    t = lead

    def hold(c, d, quarters=1):
        nonlocal t
        for i in range(t, min(n, t + quarters * quarter)):
            scl[i], sda[i] = c, d
        t += quarters * quarter

    hold(1, 1, 2)
    hold(1, 0, 2)  # START: SDA falls while SCL is high
    address, data = I2C_WRITE
    for byte in [address << 1, *data]:
        for b in [(byte >> (7 - k)) & 1 for k in range(8)] + [0]:  # target ACKs each byte
            hold(0, b)
            hold(1, b, 2)
            hold(0, b)
    hold(0, 0)
    hold(1, 0, 2)
    hold(1, 1, 2)  # STOP: SDA rises while SCL is high
    return scl, sda


def _spi(n, rate, lead, cpol, cpha):
    half = max(1, int(rate / 1_000_000 / 2))
    sck, mosi = [cpol] * n, [0] * n
    t = lead
    for burst in SPI_BURSTS:
        for byte in burst:
            for bit in [(byte >> (7 - k)) & 1 for k in range(8)]:
                # CPHA=0 samples on the leading edge mid-bit, CPHA=1 on the trailing edge.
                for level in (cpol, 1 - cpol) if cpha == 0 else (1 - cpol, cpol):
                    for i in range(t, min(n, t + half)):
                        sck[i], mosi[i] = level, bit
                    t += half
        t += 20 * half  # gap between bursts
    return sck, mosi


class FakeArm:
    def __init__(self, scenario: str = "success"):
        self.scenario = scenario
        self.tool = "dmm"

    def state(self) -> dict:
        if self.scenario == "arm_unreachable":
            raise ConnectionError("connection refused")
        return {"tool": self.tool, "msg": "ok", "hw": {"mode": "sim", "armed": False}}

    def select_tool(self, head: str) -> dict:
        state = self.state()
        if self.scenario == "tool_change_blocked" and head != self.tool:
            state["msg"] = "blocked: lift to 20 mm above the board before turning the turret"
            return state
        if self.scenario != "tool_not_confirmed":
            self.tool = head
        return self.state()
