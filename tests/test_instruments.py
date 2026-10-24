import ctypes
import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from troubleshooter.instruments import ad3, analysis, fake
from troubleshooter.instruments.rig import RigConfig, load_rig
from troubleshooter.instruments.service import InstrumentService
from troubleshooter.settings import Settings

ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "config" / "rig.example.json"


def make_service(tmp_path, scenario="success", **overrides):
    settings = Settings(
        instruments="fake",
        rig_config=str(EXAMPLE),
        fake_scenario=scenario,
        captures_dir=str(tmp_path / "captures"),
        journal=str(tmp_path / "operations.sqlite3"),
        **overrides,
    )
    return InstrumentService(settings)


@pytest.fixture
async def service(tmp_path):
    instruments = make_service(tmp_path)
    yield instruments
    await instruments.close()


# --- rig config ---------------------------------------------------------------------------


def test_example_rig_is_valid():
    rig = load_rig(EXAMPLE)
    assert {arm.scope.channel for arm in rig.arms.values()} == {1, 2}


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c["arms"]["B"]["scope"].update(channel=1),  # shared scope channel
        lambda c: c["arms"]["B"]["logic"].update(dio=0),  # shared DIO pin
        lambda c: c["logic_types"]["uart_9600_8n1"].pop("baud"),
        lambda c: c["logic_types"]["uart_9600_8n1"].update(sample_rate_hz=10000),
        lambda c: c["arms"].update({"bad name": c["arms"]["A"]}),
        lambda c: c["arms"]["A"].update(dmm={"port": "COM3"}),  # unknown key
    ],
)
def test_rig_rejects_bad_wiring(change):
    config = json.loads(EXAMPLE.read_text())
    change(config)
    with pytest.raises(ValidationError):
        RigConfig.model_validate(config)


# --- waveform -----------------------------------------------------------------------------


@pytest.mark.parametrize(("arm", "channel"), [("A", 1), ("B", 2)])
async def test_waveform_selects_scope_head_and_its_channel(service, arm, channel):
    result = await service.capture_waveform(arm, "general")
    assert result.status == "ok", result
    assert result.mode == "simulated"
    assert result.instrument["channel"] == channel
    assert result.arms == {arm: {"tool": "scope", "arm_server": "sim"}}
    assert service.arms[arm].tool == "scope"
    m = result.measurements
    assert m["frequency_hz"] == pytest.approx(1000, rel=0.01)
    assert m["duty_cycle"] == pytest.approx(0.5, abs=0.01)
    assert m["vpp_v"] == pytest.approx(3.3, abs=0.02)
    assert not m["flat"]
    assert result.contact_verified is False
    saved = json.loads(Path(result.capture_file).read_text())
    assert saved["result"]["capture_id"] == result.capture_id
    assert len(saved["raw"]["samples_v"]) == 8192


@pytest.mark.parametrize(
    ("scenario", "profile", "status", "reason"),
    [
        ("clipped", "general", "inconclusive", "clipped"),
        ("no_trigger", "fast_edges", "inconclusive", "no_trigger"),
        ("no_trigger", "general", "ok", None),
        ("instrument_missing", "general", "failed", "instrument_unavailable"),
        ("capture_error", "general", "failed", "instrument_error"),
        ("arm_unreachable", "general", "failed", "arm_unavailable"),
        ("tool_not_confirmed", "general", "failed", "tool_not_confirmed"),
    ],
)
async def test_waveform_failures(tmp_path, scenario, profile, status, reason):
    instruments = make_service(tmp_path, scenario)
    try:
        result = await instruments.capture_waveform("A", profile)
    finally:
        await instruments.close()
    assert (result.status, result.failure_reason) == (status, reason)
    if status != "ok":
        assert result.measurements is None
    if status == "failed":
        assert result.mode is None
    if scenario == "no_trigger" and status == "ok":
        assert result.measurements["flat"]
        assert result.triggered is False


async def test_tool_change_blocked_reports_arm_message(tmp_path):
    instruments = make_service(tmp_path, "tool_change_blocked")
    try:
        result = await instruments.capture_waveform("A", "general")
    finally:
        await instruments.close()
    assert result.failure_reason.startswith("tool_change_failed:blocked")
    assert result.mode is None


# --- logic --------------------------------------------------------------------------------


@pytest.mark.parametrize("logic_type", ["uart_9600_8n1", "uart_115200_8n1"])
async def test_uart(service, logic_type):
    result = await service.capture_logic(logic_type, {"rx": "B"})
    assert result.status == "ok", result
    decoded = result.measurements["decoded"]
    assert decoded["bytes_hex"] == ["4f", "4b", "0d", "0a"]
    assert decoded["text"] == "OK.."
    assert decoded["framing_errors"] == 0
    assert result.instrument["dio"] == {"rx": 1}
    assert result.arms["B"] == {"tool": "logic", "arm_server": "sim", "role": "rx"}


@pytest.mark.parametrize("lines", [{"scl": "A", "sda": "B"}, {"scl": "B", "sda": "A"}])
async def test_i2c_uses_both_arms(service, lines):
    result = await service.capture_logic("i2c_100k", lines)
    assert result.status == "ok", result
    (transaction,) = result.measurements["decoded"]["transactions"]
    assert transaction["address"] == "0x48"
    assert transaction["rw"] == "write"
    assert transaction["address_ack"] is True
    assert transaction["data_hex"] == ["01", "60"]
    assert transaction["stop"] and transaction["complete"]
    assert {arm for arm in result.arms} == {"A", "B"}
    assert all(info["tool"] == "logic" for info in result.arms.values())


async def test_spi_two_lines(service):
    result = await service.capture_logic("spi_mode0", {"sck": "A", "data": "B"})
    assert result.status == "ok", result
    decoded = result.measurements["decoded"]
    assert decoded["bytes_hex"] == ["9f", "ef", "40"]
    assert decoded["bits_per_burst"] == [8, 16]
    assert decoded["sck_frequency_hz"] == pytest.approx(1e6, rel=0.01)


async def test_pwm(service):
    result = await service.capture_logic("pwm", {"signal": "A"})
    assert result.status == "ok", result
    line = result.measurements["lines"]["signal"]
    assert line["frequency_hz"] == pytest.approx(1000, rel=0.01)
    assert line["duty_cycle"] == pytest.approx(0.25, abs=0.01)


@pytest.mark.parametrize(
    ("logic_type", "lines", "reason"),
    [
        ("i2c_100k", {"scl": "A"}, "invalid_lines"),
        ("i2c_100k", {"scl": "A", "sda": "A"}, "invalid_lines"),
        ("uart_9600_8n1", {"tx": "A"}, "invalid_lines"),
        ("uart_9600_8n1", {"rx": "C"}, "unknown_arm"),
        ("can_500k", {"rx": "A"}, "unknown_logic_type"),
    ],
)
async def test_logic_refusals(service, logic_type, lines, reason):
    result = await service.capture_logic(logic_type, lines)
    assert (result.status, result.failure_reason) == ("failed", reason)
    assert result.mode is None
    assert all(arm.tool == "dmm" for arm in service.arms.values())  # nothing moved


async def test_no_activity_is_inconclusive(tmp_path):
    instruments = make_service(tmp_path, "no_activity")
    try:
        result = await instruments.capture_logic("uart_9600_8n1", {"rx": "A"})
    finally:
        await instruments.close()
    assert (result.status, result.failure_reason) == ("inconclusive", "no_activity")
    assert result.measurements is None
    assert result.diagnostics["lines"]["rx"]["idle_level"] == 1


async def test_busy_and_status(service):
    async with service.lock:
        busy = await service.capture_waveform("A", "general")
    assert busy.failure_reason == "busy"
    status = await service.status()
    assert status["instrument"]["connected"]
    assert status["arms"]["B"] == {
        "scope_channel": 2,
        "logic_dio": 1,
        "reachable": True,
        "tool": "dmm",
        "arm_server": "sim",
    }
    profiles = service.profiles()
    assert profiles["logic_types"]["i2c_100k"]["roles"] == ["scl", "sda"]


async def test_unset_rig_refuses(tmp_path):
    instruments = InstrumentService(Settings(instruments="fake"))
    try:
        result = await instruments.capture_logic("pwm", {"signal": "A"})
        assert result.failure_reason == "configuration_required:rig"
        assert (await instruments.status())["failure_reason"] == "configuration_required:rig"
    finally:
        await instruments.close()


# --- decoders -----------------------------------------------------------------------------


@pytest.mark.parametrize(("cpol", "cpha"), [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_spi_modes(cpol, cpha):
    sck, data = fake._spi(4000, 20e6, 100, cpol, cpha)
    decoded = analysis.decode_spi_partial(sck, data, 20e6, cpol, cpha)
    assert decoded["bytes_hex"] == ["9f", "ef", "40"]


def test_uart_framing_error():
    rate, baud = 200_000, 9600
    spb = rate / baud
    line = [1] * 400
    for i in range(20, int(20 + 10 * spb)):
        line[i] = 0  # start bit, 0x00, then a missing stop bit
    decoded = analysis.decode_uart(line, rate, baud)
    assert decoded["frames"] == 1
    assert decoded["framing_errors"] == 1
    assert decoded["valid_frames"] == 0


def test_flat_waveform_has_no_frequency():
    summary = analysis.waveform_summary([1.0] * 100, 1e6, 0.01)
    assert summary["flat"] and summary["frequency_hz"] is None


# --- AD3 driver against a recording dwf -----------------------------------------------------


class FakeDwf:
    """Stands in for dwf.dll: records calls and fills the out-parameters the driver reads."""

    def __init__(self, serial="SN:210415B12345", open_ok=True):
        self.calls = []
        self.serial, self.open_ok = serial, open_ok
        self.divider = 1

    @staticmethod
    def _set(ref, value):
        ref._obj.value = value

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def call(*args):
            self.calls.append((name, [getattr(a, "value", a) for a in args]))
            handler = getattr(self, f"_{name}", None)
            return handler(*args) if handler else 1

        return call

    def names(self):
        return [name for name, _ in self.calls]

    def args(self, name):
        return [args for called, args in self.calls if called == name]

    def _FDwfEnum(self, _, count):
        self._set(count, 1)
        return 1

    def _FDwfEnumDeviceName(self, _, buf):
        buf.value = b"Analog Discovery 3"
        return 1

    def _FDwfEnumSN(self, _, buf):
        buf.value = self.serial.encode()
        return 1

    def _FDwfDeviceOpen(self, _, handle):
        if not self.open_ok:
            return 0
        self._set(handle, 7)
        return 1

    def _FDwfGetLastErrorMsg(self, buf):
        buf.value = b"Device is busy (WaveForms open?)"
        return 1

    def _FDwfAnalogInBufferSizeInfo(self, _, low, high):
        self._set(low, 16)
        self._set(high, 4096)
        return 1

    def _FDwfAnalogInFrequencyGet(self, _, rate):
        self._set(rate, 1e6)
        return 1

    def _FDwfAnalogInChannelRangeGet(self, _, ch, span):
        self._set(span, 50.0)  # the device picked its low-gain range
        return 1

    def _FDwfAnalogInStatus(self, _, read, state):
        self._set(state, ad3.STATE_DONE)
        return 1

    def _FDwfAnalogInStatusData(self, _, ch, buf, n):
        for i in range(n.value):
            buf[i] = 1.25
        return 1

    def _FDwfAnalogInStatusAutoTriggered(self, _, auto):
        self._set(auto, 1)
        return 1

    def _FDwfDigitalInInternalClockInfo(self, _, hz):
        self._set(hz, 100e6)
        return 1

    def _FDwfDigitalInDividerSet(self, _, divider):
        self.divider = divider.value
        return 1

    def _FDwfDigitalInDividerGet(self, _, divider):
        self._set(divider, self.divider)
        return 1

    def _FDwfDigitalInBufferSizeInfo(self, _, high):
        self._set(high, 4096)
        return 1

    def _FDwfDigitalInStatus(self, _, read, state):
        self._set(state, ad3.STATE_DONE)
        return 1

    def _FDwfDigitalInStatusData(self, _, buf, nbytes):
        for i in range(nbytes.value // 2):
            buf[i] = 0b10
        return 1

    def _FDwfDigitalInStatusAutoTriggered(self, _, auto):
        self._set(auto, 0)
        return 1


ANALOG = ad3.AnalogRequest(
    channel=2,
    range_v=10,
    offset_v=0,
    sample_rate_hz=1e6,
    samples=8192,
    trigger_level_v=1.5,
    trigger_slope="falling",
    auto_timeout_s=0.5,
)


def test_driver_analog_capture(monkeypatch):
    monkeypatch.setattr(ad3, "OFFSET_SETTLE_S", 0)
    dwf = FakeDwf()
    device = ad3.AD3(dwf=dwf)
    capture = device.capture_analog(ANALOG)
    # Outputs are reset off on open, and nothing ever enables them.
    assert {"FDwfAnalogOutReset", "FDwfDigitalIOReset"} <= set(dwf.names())
    assert not [n for n in dwf.names() if "OutputEnable" in n or "NodeEnable" in n]
    assert dwf.args("FDwfAnalogInChannelEnableSet")[-1] == [7, 1, 1]  # channel 2 is index 1
    assert dwf.args("FDwfAnalogInBufferSizeSet") == [[7, 4096]]  # clamped to the device
    assert dwf.args("FDwfAnalogInAcquisitionModeSet") == [[7, 0]]  # acqmodeSingle
    assert dwf.args("FDwfAnalogInTriggerConditionSet") == [[7, ad3.SLOPE["falling"]]]
    assert len(capture.samples) == 4096 and capture.samples[0] == 1.25
    assert capture.range_v == 50.0
    assert capture.triggered is False  # auto-triggered
    device.capture_analog(ANALOG)
    assert dwf.names().count("FDwfDeviceOpen") == 1
    # Same channel configuration: the offset settle start isn't repeated.
    assert dwf.args("FDwfAnalogInConfigure").count([7, 1, 0]) == 1


def test_driver_digital_capture():
    dwf = FakeDwf()
    request = ad3.DigitalRequest(
        pins=(1, 4),
        sample_rate_hz=2e6,
        samples=16384,
        trigger_pin=4,
        trigger_edge="fall",
        auto_timeout_s=1.0,
    )
    capture = ad3.AD3(dwf=dwf).capture_digital(request)
    assert dwf.divider == 50
    assert dwf.args("FDwfDigitalInAcquisitionModeSet") == [[7, 0]]
    assert capture.sample_rate_hz == 2e6
    assert dwf.args("FDwfDigitalInTriggerSet") == [[7, 0, 0, 0, 1 << 4]]
    assert dwf.args("FDwfDigitalInBufferSizeSet") == [[7, 4096]]
    assert dwf.args("FDwfDigitalInStatusData")[0][2] == 2 * 4096
    assert capture.words[0] == 0b10 and capture.triggered


def test_driver_open_failures():
    with pytest.raises(ad3.InstrumentUnavailable, match="serial X"):
        ad3.AD3(serial="X", dwf=FakeDwf()).info()
    with pytest.raises(ad3.InstrumentUnavailable, match="WaveForms open"):
        ad3.AD3(dwf=FakeDwf(open_ok=False)).info()


@pytest.mark.skipif(not hasattr(ctypes, "WinDLL"), reason="WaveForms SDK check is Windows-only")
def test_real_sdk_exports_every_function_used():
    try:
        dwf = ad3.load_dwf()
    except ad3.InstrumentUnavailable:
        pytest.skip("WaveForms isn't installed")
    source = (ROOT / "src" / "troubleshooter" / "instruments" / "ad3.py").read_text()
    used = set(re.findall(r'"(FDwf\w+)"', source)) | set(re.findall(r"dwf\.(FDwf\w+)", source))
    assert [name for name in sorted(used) if not hasattr(dwf, name)] == []


# --- arm server turret ----------------------------------------------------------------------


def test_sim_server_tool_selection(monkeypatch):
    pytest.importorskip("ikpy")
    pytest.importorskip("scservo_sdk")
    monkeypatch.syspath_prepend(str(ROOT))
    import sim_server

    sim = sim_server.Sim()
    assert sim.state()["tool"] == sim_server.TOOL_HEADS[sim_server.PROBE_START]
    sim.select_tool("scope")
    assert (sim.msg, sim.state()["tool"]) == ("ok", "scope")
    sim.select_tool("oscilloscope")
    assert sim.msg.startswith("unknown tool head")
    sim.board_z = sim.pose(sim.q)[0][2] - 0.010  # tip 10 mm above the board
    sim.select_tool("logic")
    assert sim.msg.startswith("blocked: lift")
    assert sim.state()["tool"] == "scope"
    sim.select_tool("scope")  # already there: no turret motion, so not blocked
    assert sim.msg == "ok"
