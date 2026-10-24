import asyncio
import logging
import math
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .backend import BackendUnavailable, UnavailableBackend
from .contracts import Mode
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
probe (the multimeter is a separate device these tools don't drive). capture_waveform and
capture_logic turn the turret to the needed head and read the AD3 input wired to it. They
don't position the probe or verify contact: they capture wherever the probe already is, so
contact_verified is always false. A flat line or no_activity can mean the probe isn't
touching anything. Read get_instrument_status and list_capture_profiles first, choose a
profile or logic type by name, and cite capture IDs. Only status=ok carries measurements.
Two-line protocols (I2C, SPI) use both arms' logic probes, one line each. SPI is partial:
SCK plus one data line, no chip select, so its bytes are tentative.
read_digital is a declared extension point and returns not_implemented.
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


def create_mcp(settings: Settings | None = None) -> FastMCP:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(_):
        services = Services(
            MeasurementService(settings, build_backend(settings)), InstrumentService(settings)
        )
        try:
            yield services
        finally:
            await services.measurement.close()
            await services.instruments.close()

    mcp = FastMCP("Hardware troubleshooter", instructions=INSTRUCTIONS, lifespan=lifespan)
    readonly = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)
    movement = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)

    def service(ctx: Context) -> MeasurementService:
        return ctx.request_context.lifespan_context.measurement

    def instruments(ctx: Context) -> InstrumentService:
        return ctx.request_context.lifespan_context.instruments

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
