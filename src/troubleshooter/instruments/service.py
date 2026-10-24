"""Tool-head selection plus AD3 capture and analysis for the MCP instrument tools.

Asking for a probe type turns that arm's turret to the head and reads the AD3 input wired to
it (scope head -> scope channel, logic head -> DIO pin). Positioning and contact aren't
managed here: a capture is taken wherever the probe already is, and says so.
"""

import asyncio
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from ..contracts import Mode
from ..settings import Settings
from . import analysis
from .ad3 import AD3, AnalogRequest, DigitalRequest, InstrumentError, InstrumentUnavailable
from .arm import ArmLink
from .fake import FakeAD3, FakeArm
from .rig import ROLES, RigConfig, load_rig

LOG = logging.getLogger(__name__)
Head = Literal["scope", "logic", "dmm"]
CLIP_MARGIN = 0.005  # samples this close to the range edge (fraction of span) count as clipped


class CaptureResult(BaseModel):
    capture_id: str
    kind: Literal["waveform", "logic"]
    status: Literal["ok", "inconclusive", "failed"]
    failure_reason: str | None = None
    # Instrument mode. None when no capture happened, never inferred from configuration.
    mode: Mode | None = None
    profile: str
    arms: dict[str, dict[str, Any]] = {}
    instrument: dict[str, Any] | None = None
    settings: dict[str, Any] | None = None
    triggered: bool | None = None
    # Only status=ok carries measurements. Diagnostics explain other outcomes and aren't evidence.
    measurements: dict[str, Any] | None = None
    diagnostics: dict[str, Any] | None = None
    contact_verified: bool = False
    placement: Literal["unmanaged"] = "unmanaged"
    capture_file: str | None = None
    duration_ms: int = 0


class Refused(Exception):
    def __init__(self, reason: str, diagnostics: dict | None = None):
        super().__init__(reason)
        self.reason, self.diagnostics = reason, diagnostics


class InstrumentService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.rig: RigConfig | None = None
        self.config_error: str | None = None
        if not settings.rig_config:
            self.config_error = "configuration_required:rig"
        else:
            try:
                self.rig = load_rig(settings.rig_config)
            except (OSError, ValueError, ValidationError) as exc:
                LOG.error("Rig config %s rejected: %s", settings.rig_config, exc)
                self.config_error = "configuration_invalid:rig"
        fake = settings.instruments == "fake"
        self.instrument = (
            (FakeAD3(settings.fake_scenario) if fake else AD3(serial=settings.ad3_serial))
            if self.rig
            else None
        )
        self.arms = {
            name: FakeArm(settings.fake_scenario)
            if fake
            else ArmLink(arm.url, settings.arm_timeout_s)
            for name, arm in (self.rig.arms.items() if self.rig else [])
        }
        # Every device call goes through one thread, so a cancelled MCP request can't overlap
        # the capture still running underneath it.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ad3")
        self.lock = asyncio.Lock()

    async def _device(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self.executor, fn, *args)

    def _require_rig(self) -> RigConfig:
        if self.rig is None:
            raise Refused(self.config_error)
        return self.rig

    def _arm(self, name: str):
        if name not in self._require_rig().arms:
            raise Refused("unknown_arm", {"arms": sorted(self.rig.arms)})
        return self.rig.arms[name]

    async def _ensure_tool(self, arm: str, head: Head) -> dict:
        link = self.arms[arm]
        try:
            state = await asyncio.to_thread(link.state)
            if state.get("tool") != head:
                state = await asyncio.to_thread(link.select_tool, head)
                if state.get("msg") != "ok":
                    raise Refused(f"tool_change_failed:{state.get('msg')}")
        except ConnectionError as exc:
            raise Refused("arm_unavailable", {"arm": arm, "detail": str(exc)}) from exc
        if state.get("tool") != head:
            raise Refused("tool_not_confirmed", {"arm": arm, "reported_tool": state.get("tool")})
        return {"tool": head, "arm_server": (state.get("hw") or {}).get("mode")}

    async def select_tool(self, arm: str, head: Head) -> dict:
        try:
            self._arm(arm)
            if self.lock.locked():
                raise Refused("busy")
            async with self.lock:
                selected = await self._ensure_tool(arm, head)
            return {"status": "ok", "failure_reason": None, "arm": arm, **selected}
        except Refused as refused:
            return {
                "status": "failed",
                "failure_reason": refused.reason,
                "arm": arm,
                "diagnostics": refused.diagnostics,
            }

    async def _run(self, kind: str, profile: str, work) -> CaptureResult:
        started = time.monotonic()
        capture_id = f"cap-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        result = CaptureResult(capture_id=capture_id, kind=kind, status="failed", profile=profile)
        raw = None
        try:
            if self.lock.locked():
                raise Refused("busy")
            async with self.lock:
                raw = await work(result)
        except Refused as refused:
            result.status, result.failure_reason = "failed", refused.reason
            result.diagnostics = refused.diagnostics
        except InstrumentUnavailable as exc:
            result.status, result.failure_reason = "failed", "instrument_unavailable"
            result.diagnostics = {"detail": str(exc)}
        except InstrumentError as exc:
            result.status, result.failure_reason = "failed", "instrument_error"
            result.diagnostics = {"detail": str(exc)}
            await self._device(self.instrument.close)  # reopen cleanly next time
        result.duration_ms = int((time.monotonic() - started) * 1000)
        if raw is not None:
            result.capture_file = self._save(result, raw)
        return result

    def _save(self, result: CaptureResult, raw: dict) -> str | None:
        try:
            folder = Path(self.settings.captures_dir)
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{result.capture_id}.json"
            body = {"result": result.model_dump(mode="json"), "raw": raw}
            path.write_text(json.dumps(body, allow_nan=False), encoding="utf-8")
            return str(path.resolve())
        except (OSError, ValueError):
            LOG.exception("Couldn't save capture %s", result.capture_id)
            return None

    async def capture_waveform(self, arm: str, profile_id: str) -> CaptureResult:
        async def work(result: CaptureResult):
            wiring = self._arm(arm)
            profile = self.rig.scope_profiles.get(profile_id)
            if profile is None:
                raise Refused("unknown_profile", {"profiles": sorted(self.rig.scope_profiles)})
            # Check the AD3 is there before turning any turret.
            info = await self._device(self.instrument.info)
            result.arms = {arm: await self._ensure_tool(arm, "scope")}
            request = AnalogRequest(
                channel=wiring.scope.channel,
                range_v=profile.range_v,
                offset_v=profile.offset_v,
                sample_rate_hz=profile.sample_rate_hz,
                samples=profile.samples,
                trigger_level_v=profile.trigger_level_v,
                trigger_slope=profile.trigger_slope,
                auto_timeout_s=profile.auto_timeout_s,
            )
            capture = await self._device(self.instrument.capture_analog, request)
            result.mode = self.instrument.mode
            result.instrument = {**info, "channel": wiring.scope.channel}
            result.settings = {
                "sample_rate_hz": capture.sample_rate_hz,
                "samples": len(capture.samples),
                "range_v": capture.range_v,
                "offset_v": capture.offset_v,
                "trigger_level_v": profile.trigger_level_v,
            }
            result.triggered = capture.triggered
            samples = capture.samples
            floor = capture.offset_v - capture.range_v / 2
            ceiling = capture.offset_v + capture.range_v / 2
            margin = CLIP_MARGIN * capture.range_v
            noise = max(0.01, 0.002 * capture.range_v)
            summary = analysis.waveform_summary(samples, capture.sample_rate_hz, noise)
            if min(samples) <= floor + margin or max(samples) >= ceiling - margin:
                result.status, result.failure_reason = "inconclusive", "clipped"
                result.diagnostics = {
                    "min_v": min(samples),
                    "max_v": max(samples),
                    "window_v": [floor, ceiling],
                }
            elif profile.require_trigger and not capture.triggered:
                result.status, result.failure_reason = "inconclusive", "no_trigger"
                result.diagnostics = summary
            else:
                result.status, result.measurements = "ok", summary
            return {"samples_v": samples, "sample_rate_hz": capture.sample_rate_hz}

        return await self._run("waveform", profile_id, work)

    async def capture_logic(self, logic_type: str, lines: dict[str, str]) -> CaptureResult:
        async def work(result: CaptureResult):
            rig = self._require_rig()
            logic = rig.logic_types.get(logic_type)
            if logic is None:
                raise Refused("unknown_logic_type", {"logic_types": sorted(rig.logic_types)})
            roles = ROLES[logic.protocol]
            if set(lines) != set(roles) or len(set(lines.values())) != len(lines):
                raise Refused(
                    "invalid_lines",
                    {
                        "required_roles": list(roles),
                        "note": "map each role to a different arm's logic probe",
                    },
                )
            pins = {role: self._arm(arm).logic.dio for role, arm in lines.items()}
            info = await self._device(self.instrument.info)
            result.arms = {}
            for role, arm in lines.items():
                result.arms[arm] = {**await self._ensure_tool(arm, "logic"), "role": role}
            trigger_role, edge = {
                "uart": ("rx", "fall"),  # start bit
                "pwm": ("signal", "rise"),
                "i2c": ("sda", "fall"),  # START condition
                "spi": ("sck", "rise" if logic.cpol == 0 else "fall"),
            }[logic.protocol]
            request = DigitalRequest(
                pins=tuple(pins.values()),
                sample_rate_hz=logic.sample_rate_hz,
                samples=logic.samples,
                trigger_pin=pins[trigger_role],
                trigger_edge=edge,
                auto_timeout_s=logic.auto_timeout_s,
                roles={pin: role for role, pin in pins.items()},
                logic=logic,
            )
            capture = await self._device(self.instrument.capture_digital, request)
            rate = capture.sample_rate_hz
            result.mode = self.instrument.mode
            result.instrument = {**info, "dio": pins}
            result.settings = {"sample_rate_hz": rate, "samples": len(capture.words)}
            result.triggered = capture.triggered
            signals = {role: analysis.bits(capture.words, pin) for role, pin in pins.items()}
            stats = {role: analysis.line_stats(line, rate) for role, line in signals.items()}
            raw = {
                "sample_rate_hz": rate,
                "dio": pins,
                "lines": {r: "".join(map(str, line)) for r, line in signals.items()},
            }
            if not any(s["edges"] for s in stats.values()):
                result.status, result.failure_reason = "inconclusive", "no_activity"
                result.diagnostics = {"lines": stats}
                return raw
            if logic.protocol == "uart":
                decoded = analysis.decode_uart(
                    signals["rx"], rate, logic.baud, logic.data_bits, logic.parity, logic.stop_bits
                )
                usable = decoded["valid_frames"] > 0
            elif logic.protocol == "i2c":
                decoded = analysis.decode_i2c(signals["scl"], signals["sda"])
                usable = decoded["transaction_count"] > 0
            elif logic.protocol == "spi":
                decoded = analysis.decode_spi_partial(
                    signals["sck"], signals["data"], rate, logic.cpol, logic.cpha, logic.bit_order
                )
                usable = decoded["clock_edges"] >= 8
            else:
                decoded = {}
                usable = stats["signal"]["frequency_hz"] is not None
            if usable:
                result.status = "ok"
                result.measurements = {
                    "protocol": logic.protocol,
                    "lines": stats,
                    "decoded": decoded,
                }
            else:
                result.status, result.failure_reason = "inconclusive", "decode_failed"
                result.diagnostics = {"lines": stats, "decoded": decoded}
            return raw

        return await self._run("logic", logic_type, work)

    def profiles(self) -> dict:
        rig = self._require_rig()
        return {
            "scope_profiles": {
                name: p.model_dump(
                    include={
                        "description",
                        "range_v",
                        "offset_v",
                        "sample_rate_hz",
                        "samples",
                        "trigger_level_v",
                    }
                )
                for name, p in rig.scope_profiles.items()
            },
            "logic_types": {
                name: {
                    **t.model_dump(exclude_defaults=True, exclude={"samples", "auto_timeout_s"}),
                    "roles": list(ROLES[t.protocol]),
                }
                for name, t in rig.logic_types.items()
            },
        }

    async def status(self) -> dict:
        if self.rig is None:
            return {"status": "failed", "failure_reason": self.config_error, "mode": None}
        result = {
            "status": "ok",
            "failure_reason": None,
            "mode": self.instrument.mode,
            "busy": self.lock.locked(),
            "arms": {},
            "dmm": "separate device, not driven by this server yet",
        }
        try:
            result["instrument"] = {"connected": True, **await self._device(self.instrument.info)}
        except (InstrumentUnavailable, InstrumentError) as exc:
            result["instrument"] = {"connected": False, "detail": str(exc)}
        for name, wiring in self.rig.arms.items():
            arm = {"scope_channel": wiring.scope.channel, "logic_dio": wiring.logic.dio}
            try:
                state = await asyncio.to_thread(self.arms[name].state)
                arm.update(
                    reachable=True,
                    tool=state.get("tool"),
                    arm_server=(state.get("hw") or {}).get("mode"),
                )
            except ConnectionError as exc:
                arm.update(reachable=False, detail=str(exc))
            result["arms"][name] = arm
        return result

    async def close(self):
        if self.instrument is not None:
            await self._device(self.instrument.close)
        self.executor.shutdown(wait=False)
