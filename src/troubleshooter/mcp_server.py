import asyncio
import logging
import math
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

from mcp.server.fastmcp import Context, FastMCP, Image
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .backend import BackendUnavailable, UnavailableBackend
from .contracts import Mode
from .design.service import DesignService
from .instruments.dmm_knob import DmmKnob
from .instruments.service import InstrumentService, Refused
from .service import MeasurementService
from .settings import Settings

LOG = logging.getLogger(__name__)
INSTRUCTIONS = """Investigate circuit problems using measured evidence.
Read get_rig_status and list_test_points before requesting a measurement. Choose a
registered pad, component-pin identity, or net. Don't invent coordinates or pin maps.
measure_voltage requests a complete task from the orchestrator, including approach and retraction.
Read each result before requesting another measurement. One bench runs one task at a time.
Only status=ok with a non-null reading is usable circuit evidence. Inconclusive contact
isn't evidence of zero volts or of a component fault. Cite request IDs and preserve mode.
Simulated readings demonstrate software only. Missing hardware or registration must
never trigger a fallback to simulated evidence. Stop on uncertain motion, timeout,
abort, protocol_violation, or failed calibration. Never silently retry a movement.

AD3 tools: each robot arm has a turret with a scope probe, a logic probe, and a multimeter
probe (the multimeter is a separate device the AD3 tools don't drive). capture_waveform and
capture_logic turn the turret to the needed head and read the AD3 input wired to it. They
don't position the probe or verify contact: they capture wherever the probe already is, so
contact_verified is always false. A flat line or no_activity can mean the probe isn't
touching anything. Read get_instrument_status and list_capture_profiles first, choose a
profile or logic type by name, and cite capture IDs. Only status=ok carries measurements.
Two-line protocols (I2C, SPI) use both arms' logic probes, one line each. SPI is partial:
SCK plus one data line, no chip select, so its bytes are tentative.
read_digital is a declared extension point and returns not_implemented.

Sensor head tools read an arm's ESP32 sensor head and never move anything. read_tof gives the
time-of-flight range and, once zeroed, the probe tip's height above whatever the beam hits
(the board, or a tall part under it). It's a proximity reading, not contact evidence.
read_thermal gives an 8x8 grid in degrees C across a ~60 degree field of view: each cell
averages a wide area, so it shows what runs hot, never a pad or junction temperature.

Design tools work without the rig. ingest_design reads a KiCad board (and the schematics
beside it) into a local store; query_design and find_probe_points answer from that store
exactly. find_probe_points ranks exposed copper on a net and refuses pads where the tip could
short to another net; prefer its first unrefused candidate. locate_on_image asks vision
where pads appear in a camera; it predicts, it doesn't confirm contact. search_datasheets
returns cited passages for reading only: never take coordinates, nets or pin numbers
from them. After reading a pin table, record_pinout stores it with its citation.

Multimeter knob: a bench arm turns the meter's rotary switch. Read list_dmm_settings first
and choose a setting by name. set_dmm_setting turns from the last recorded setting unless you
pass current_setting; pass it only when the operator has told you where the knob is. If a
turn fails after the arm moved, the knob position is unknown: ask the operator before trying
again. set_dmm_setting can't see the knob, so setting_verified is always false; read_dmm
reads the display afterwards, and its setting_check says whether the unit fits the setting.
Only status=ok from read_dmm is a reading; if decimal_inferred is true, say so when citing it.
No model selection service or voice provider runs inside this server.
"""


class TargetInfo(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    id: str
    net: str
    notes: str


class RegistryReply(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    status: str
    mode: Mode
    failure_reason: str | None = None
    targets: list[TargetInfo]


class RigReply(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    status: str
    mode: Mode
    board_id: str
    registration_valid: bool
    calibration_valid: bool
    registration_age_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    registration_id: str
    arm_state: str
    instrument_connected: bool
    busy: bool
    capabilities: list[str]
    failure_reason: str | None = None
    note: str | None = None


def build_backend(settings: Settings):
    if settings.backend == "stub":
        from .stub import StubBackend

        return StubBackend(settings)
    LOG.error("No hardware backend is implemented; measurements will fail closed")
    return UnavailableBackend("No hardware backend is implemented")


@dataclass
class Services:
    measurement: MeasurementService
    instruments: InstrumentService
    design: DesignService
    knob: DmmKnob


def create_mcp(settings: Settings | None = None) -> FastMCP:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(_):
        services = Services(
            MeasurementService(settings, build_backend(settings)),
            InstrumentService(settings),
            DesignService(settings.design_dir, settings.moss_project_id, settings.moss_project_key),
            DmmKnob(settings),
        )
        try:
            yield services
        finally:
            await services.measurement.close()
            await services.instruments.close()
            services.design.close()

    mcp = FastMCP("Hardware troubleshooter", instructions=INSTRUCTIONS, lifespan=lifespan)
    readonly = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)
    movement = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)
    local = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)
    online = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True)

    def service(ctx: Context) -> MeasurementService:
        return ctx.request_context.lifespan_context.measurement

    def instruments(ctx: Context) -> InstrumentService:
        return ctx.request_context.lifespan_context.instruments

    def design(ctx: Context) -> DesignService:
        return ctx.request_context.lifespan_context.design

    def knob(ctx: Context) -> DmmKnob:
        return ctx.request_context.lifespan_context.knob

    @mcp.tool(annotations=movement)
    async def measure_voltage(
        target: str, ctx: Context, request_id: str | None = None
    ) -> dict[str, Any]:
        """Request a complete DC voltage measurement at a registered pad, pin, or net.

        The orchestrator handles localization, approach, verification, acquisition, and retraction.
        Only status=ok returns a usable reading. Inconclusive contact is NOT 0 V and
        doesn't establish a fault. Failed results explain why execution stopped.
        Only one measurement can run. Optional request_id deduplicates exact retries;
        omitting it generates a new ID and therefore a new request.
        """
        progress = 0

        async def feedback(stage: str, attempt: int):
            nonlocal progress
            progress += 1
            LOG.info("stage=%s attempt=%s", stage, attempt)
            try:
                await ctx.report_progress(progress, message=f"{stage} (attempt {attempt})")
            except Exception:  # noqa: BLE001 - progress delivery mustn't interrupt physical work
                LOG.warning("Couldn't deliver MCP progress; measurement result still required")

        result = await service(ctx).measure(target, request_id, feedback)
        LOG.info(
            "mcp_result request_id=%s target=%s status=%s reason=%s mode=%s duration_ms=%s",
            result.request_id,
            target,
            result.status,
            result.failure_reason,
            result.mode,
            result.duration_ms,
        )
        return result.model_dump(mode="json")

    @mcp.tool(annotations=readonly)
    async def list_test_points(ctx: Context) -> dict[str, Any]:
        """Read registered electrical targets from the orchestrator. The registry is the source of truth."""
        try:
            raw = await service(ctx).backend.list_test_points()
            result = RegistryReply.model_validate(raw)
            if result.status not in {"ok", "failed"}:
                raise ValueError("Invalid registry status")
            return result.model_dump(mode="json")
        except (BackendUnavailable, asyncio.TimeoutError):
            return {
                "status": "failed",
                "mode": None,
                "targets": [],
                "failure_reason": "orchestrator_unavailable",
            }
        except (ValidationError, ValueError):
            LOG.exception("Malformed registry response")
            return {
                "status": "failed",
                "mode": None,
                "targets": [],
                "failure_reason": "protocol_violation",
            }

    @mcp.tool(annotations=readonly)
    async def get_rig_status(ctx: Context) -> dict[str, Any]:
        """Read orchestrator hardware/registration state and the MCP execution latch. No movement."""
        boundary = service(ctx)
        try:
            raw = await boundary.backend.get_rig_status()
            age = raw.get("registration_age_s")
            if isinstance(age, float) and math.isnan(age):
                raw["registration_age_s"] = None
            result = RigReply.model_validate(raw).model_dump(mode="json")
            if result["status"] not in {"ok", "failed"}:
                raise ValueError("Invalid rig status")
        except (BackendUnavailable, asyncio.TimeoutError):
            result = {
                "status": "failed",
                "mode": None,
                "failure_reason": "orchestrator_unavailable",
            }
        except (ValidationError, ValueError):
            LOG.exception("Malformed rig status")
            result = {"status": "failed", "mode": None, "failure_reason": "protocol_violation"}
        result.update(
            mcp_busy=boundary.active is not None,
            execution_blocked=boundary.blocked,
            implemented_tools=[
                "measure_voltage",
                "capture_waveform",
                "capture_logic",
                "select_tool",
                "set_dmm_setting",
                "read_tof",
                "read_thermal",
            ],
            planned_tools=["read_digital"],
        )
        return result

    @mcp.tool(annotations=movement)
    async def abort(ctx: Context) -> dict[str, Any]:
        """Cancel the active goal. Acknowledgement alone doesn't confirm physical stopping.

        Returns confirmed final checks when available; otherwise execution remains
        uncertain and later measurements are blocked. This isn't a hardware E-stop.
        """
        return (await service(ctx).abort()).model_dump(mode="json")

    @mcp.tool(annotations=readonly)
    def get_operation(request_id: str, ctx: Context) -> dict[str, Any]:
        """Read a stored result or unresolved operation. Never repeats movement."""
        return service(ctx).get_operation(request_id)

    @mcp.tool(annotations=readonly)
    async def get_instrument_status(ctx: Context) -> dict[str, Any]:
        """Read AD3 connection, each arm's reachability and current tool head, and the wiring.

        Opens the AD3 if needed (read-only). No movement.
        """
        return await instruments(ctx).status()

    @mcp.tool(annotations=readonly)
    def list_capture_profiles(ctx: Context) -> dict[str, Any]:
        """List the operator-configured scope profiles and logic types, with each logic type's
        signal roles. capture_waveform and capture_logic accept only these names."""
        try:
            return {"status": "ok", "failure_reason": None, **instruments(ctx).profiles()}
        except Refused as refused:
            return {"status": "failed", "failure_reason": refused.reason}

    @mcp.tool(annotations=movement)
    async def select_tool(
        arm: str, head: Literal["scope", "logic", "dmm"], ctx: Context
    ) -> dict[str, Any]:
        """Turn one arm's turret so this tool head is at the working position.

        Only the turret turns; the arm doesn't move. The arm server refuses unless the tip is
        well above the board. The capture tools select their head themselves.
        """
        return await instruments(ctx).select_tool(arm, head)

    @mcp.tool(annotations=movement)
    async def capture_waveform(arm: str, profile_id: str, ctx: Context) -> dict[str, Any]:
        """Capture an AD3 scope trace from this arm's oscilloscope probe.

        Selects the scope head (turret only) and reads its AD3 channel with the named profile.
        Returns Vpp, min/max, mean, RMS, frequency, and duty cycle. clipped or no_trigger is
        inconclusive. Probe placement and contact aren't verified.
        """
        result = await instruments(ctx).capture_waveform(arm, profile_id)
        LOG.info(
            "capture %s status=%s reason=%s",
            result.capture_id,
            result.status,
            result.failure_reason,
        )
        return result.model_dump(mode="json")

    @mcp.tool(annotations=movement)
    async def capture_logic(logic_type: str, lines: dict[str, str], ctx: Context) -> dict[str, Any]:
        """Capture and decode digital signals from the arms' logic probes.

        lines maps each signal role of the logic type to an arm, e.g. {"rx": "A"} for UART,
        {"scl": "A", "sda": "B"} for I2C, {"sck": "A", "data": "B"} for SPI, {"signal": "A"}
        for PWM. Selects each arm's logic head (turret only). Placement isn't verified.
        """
        result = await instruments(ctx).capture_logic(logic_type, lines)
        LOG.info(
            "capture %s status=%s reason=%s",
            result.capture_id,
            result.status,
            result.failure_reason,
        )
        return result.model_dump(mode="json")

    @mcp.tool(annotations=readonly)
    async def read_tof(arm: str, ctx: Context) -> dict[str, Any]:
        """Read this arm's time-of-flight sensor: range in mm and, once zeroed with touch_off,
        height_above_board_mm for the current tool head's tip (null with height_note saying why).

        No movement. no_target (inconclusive) means nothing is in range. It measures to
        whatever the beam hits, which may be a tall part rather than the board, and it isn't
        evidence of contact.
        """
        return await instruments(ctx).read_tof(arm)

    @mcp.tool(annotations=readonly)
    async def read_thermal(arm: str, ctx: Context) -> dict[str, Any]:
        """Read this arm's AMG8833 thermal camera: an 8x8 grid in degrees C (row-major, as the
        sensor sees it), min/max/mean, the hottest cell [row, col], and the sensor's own
        temperature.

        No movement. Cells span the sensor's ~60 degree view, so a hot cell points at a part
        running hot; it isn't that part's temperature. Compare against the mean, not 25 C.
        """
        return await instruments(ctx).read_thermal(arm)

    @mcp.tool(annotations=readonly)
    def list_dmm_settings(ctx: Context) -> dict[str, Any]:
        """List the multimeter knob settings the arm can turn to, and the last recorded one.

        current_setting is the last setting commanded (or told by the operator), not a reading
        of the knob; null means unknown. Also reports whether the arm runner is up. No movement.
        """
        return knob(ctx).status()

    @mcp.tool(annotations=movement)
    async def set_dmm_setting(
        setting: str, ctx: Context, current_setting: str | None = None
    ) -> dict[str, Any]:
        """Turn the multimeter's rotary switch to a setting from list_dmm_settings.

        The arm lifts above the meter, descends onto the knob, grips, twists, releases, and waits
        above the meter. It turns from the last recorded setting; pass current_setting only when
        the operator has said where the knob is. Refuses unless the arm runner is up and the arm
        is parked. After a failure with moved=true the knob position is unknown. The display
        isn't read: setting_verified is always false.
        """
        result = await knob(ctx).set_setting(setting, current_setting)
        LOG.info(
            "dmm_knob setting=%s from=%s status=%s reason=%s",
            setting,
            result.get("previous_setting"),
            result["status"],
            result["failure_reason"],
        )
        return result

    @mcp.tool(annotations=readonly)
    async def read_dmm(ctx: Context) -> list[Any]:
        """Read the multimeter's display through the bench arm's wrist camera. No movement.

        Needs the arm parked above the meter (where set_dmm_setting leaves it). Several frames
        are decoded locally (7-segment) and only a reading most frames agree on is status=ok.
        Returns value and unit as shown, value_base in V or ohms, text, overload ("OL"),
        decimal_inferred (glare hid the decimal point and the unit's fixed range placed it),
        and setting_check: whether the unit fits the recorded knob setting. Also returns the
        straightened LCD image so you can check the digits yourself.
        """
        import json

        result, png = await knob(ctx).read_display()
        LOG.info(
            "dmm_read %s status=%s reason=%s text=%r unit=%s",
            result.get("reading_id"),
            result["status"],
            result["failure_reason"],
            result.get("text"),
            result.get("unit"),
        )
        out: list[Any] = [json.dumps(result, default=str)]
        if png:
            out.append(Image(data=png, format="png"))
        return out

    @mcp.tool(annotations=readonly)
    def read_digital(group_id: str) -> dict[str, Any]:
        """Scaffold only: future static reads from prewired AD3 input groups. Not implemented."""
        return {
            "status": "failed",
            "reading": None,
            "mode": None,
            "failure_reason": "not_implemented",
            "tool": "read_digital",
            "target": group_id,
            "note": "No acquisition was requested.",
        }

    @mcp.tool(annotations=local)
    async def ingest_design(path: str, ctx: Context) -> dict[str, Any]:
        """Read a .kicad_pcb (or a KiCad project folder) and its schematics into the design
        store and make it the active design. Re-ingest after any board edit."""
        return await design(ctx).ingest_design(path)

    @mcp.tool(annotations=readonly)
    def list_designs(ctx: Context) -> dict[str, Any]:
        """Ingested designs and which one is active."""
        return design(ctx).list_designs()

    @mcp.tool(annotations=local)
    def set_active_design(design_id: str, ctx: Context) -> dict[str, Any]:
        """Make an ingested design the active one."""
        return design(ctx).set_active_design(design_id)

    @mcp.tool(annotations=readonly)
    def query_design(
        ctx: Context,
        ref: str | None = None,
        net: str | None = None,
        pad: str | None = None,
        mpn: str | None = None,
    ) -> dict[str, Any]:
        """Exact lookup in the active design. Give one of: ref (U3: part and every pin's
        net), net (3V3: every pad on it), pad (U3.7: its net and net-mates), mpn (parts with
        that MPN and any recorded pinout). Net names match on their last path segment."""
        return design(ctx).query(ref=ref, net=net, pad=pad, mpn=mpn)

    @mcp.tool(annotations=readonly)
    def find_probe_points(
        ctx: Context,
        net: str | None = None,
        pad: str | None = None,
        tip_diameter_mm: float = 0.5,
        position_uncertainty_mm: float = 0.2,
        limit: int = 5,
    ) -> dict[str, Any]:
        """Rank exposed copper on a net (or check one pad) as probe landing points: test
        points, then passives, vias and connectors, then IC pins. A candidate is refused when
        the tip, off by the position uncertainty, could touch another net."""
        return design(ctx).find_probe_points(
            net, pad, tip_diameter_mm, position_uncertainty_mm, max(1, min(limit, 50))
        )

    @mcp.tool(annotations=readonly)
    async def locate_on_image(pad_ids: list[str], ctx: Context, camera: str = "fixed") -> list[Any]:
        """Ask vision where pads appear in a camera image: "fixed", or an arm name for its
        wrist camera. Returns pixel centres, outlines and a 1-sigma spread, plus an overlay
        image when vision has a frame. A prediction from the PCB file and registration, not
        evidence of contact."""
        import json

        if not pad_ids or len(pad_ids) > 50:
            return [json.dumps({"status": "failed", "failure_reason": "give 1 to 50 pad_ids"})]
        try:
            raw = await service(ctx).backend.locate_points(list(pad_ids), camera)
        except (BackendUnavailable, asyncio.TimeoutError):
            raw = {"status": "failed", "mode": None, "failure_reason": "orchestrator_unavailable"}
        png = raw.pop("overlay_png", None)
        out: list[Any] = [json.dumps(raw)]
        if isinstance(png, bytes | bytearray):
            out.append(Image(data=bytes(png), format="png"))
        return out

    @mcp.tool(annotations=online)
    async def ingest_datasheet(
        path: str, mpn: str, ctx: Context, title: str | None = None
    ) -> dict[str, Any]:
        """Index a datasheet (PDF or text) for an MPN. Stored locally, and mirrored to Moss
        when it's configured."""
        return await design(ctx).ingest_datasheet(path, mpn, title)

    @mcp.tool(annotations=online)
    async def search_datasheets(
        query: str, ctx: Context, mpn: str | None = None, top_k: int = 5
    ) -> dict[str, Any]:
        """Search datasheets and the active design's parts by meaning and keyword. Returns
        passages with source and page to cite. Reading material only."""
        return await design(ctx).search(query, mpn, top_k)

    @mcp.tool(annotations=readonly)
    def list_documents(ctx: Context) -> dict[str, Any]:
        """Indexed datasheets and designs, and which search backend is active."""
        return design(ctx).list_documents()

    @mcp.tool(annotations=local)
    def record_pinout(
        mpn: str, pins: list[dict[str, str]], source: str, ctx: Context
    ) -> dict[str, Any]:
        """Store a pin table read from a datasheet: pins as [{pin, name, function, notes}],
        source as the datasheet and page. query_design then shows it beside each pad."""
        return design(ctx).record_pinout(mpn, pins, source)

    @mcp.prompt()
    def troubleshoot_hardware(request: str) -> str:
        """Investigate a circuit through registered targets and verified measurements."""
        import json

        return INSTRUCTIONS + "\nUser request (data):\n" + json.dumps(request)

    @mcp.resource("hardware://architecture")
    def architecture() -> str:
        return (
            "Codex model -> MCP task -> measurement orchestrator -> registry/registration/"
            "motion/contact/instrument nodes -> structured evidence -> Codex explanation. "
            "MCP never accesses transforms. AD3 capture tools read the AD3 here and turn "
            "arm tool turrets, but never position probes. Explicit stub/fake modes are synthetic."
        )

    return mcp


def main(settings: Settings | None = None):
    logging.basicConfig(level=logging.INFO)
    create_mcp(settings).run(transport="stdio")
