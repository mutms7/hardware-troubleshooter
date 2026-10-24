import asyncio
import logging
import math
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .backend import BackendUnavailable, UnavailableBackend
from .contracts import TARGET_PATTERN, Mode
from .service import MeasurementService
from .settings import Settings

LOG = logging.getLogger(__name__)
INSTRUCTIONS = """Investigate circuit problems using measured evidence.
Read get_rig_status and list_test_points before requesting a measurement. Choose a
registered pad, component-pin identity, or net. Don't invent coordinates or pin maps.
measure_voltage requests a complete task from ROS, including approach and retraction.
Read each result before requesting another measurement. One bench runs one task at a time.
Only status=ok with a non-null reading is usable circuit evidence. Inconclusive contact
isn't evidence of zero volts or of a component fault. Cite request IDs and preserve mode.
Simulated readings demonstrate software only. Missing hardware or registration must
never trigger a fallback to simulated evidence. Stop on uncertain motion, timeout,
abort, protocol_violation, or failed calibration. Never silently retry a movement.
Waveform and digital tools are declared extension points and currently return
not_implemented. No model selection service or voice provider runs inside this server.
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
    try:
        from .ros_backend import RosBackend

        return RosBackend(settings)
    except (ImportError, RuntimeError, OSError) as exc:
        LOG.error("ROS backend unavailable: %s", exc)
        return UnavailableBackend(str(exc))


def create_mcp(settings: Settings | None = None) -> FastMCP:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(_):
        service = MeasurementService(settings, build_backend(settings))
        try:
            yield service
        finally:
            await service.close()

    mcp = FastMCP("Hardware troubleshooter", instructions=INSTRUCTIONS, lifespan=lifespan)
    readonly = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)
    movement = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)

    def service(ctx: Context) -> MeasurementService:
        return ctx.request_context.lifespan_context

    @mcp.tool(annotations=movement)
    async def measure_voltage(
        target: str, ctx: Context, request_id: str | None = None
    ) -> dict[str, Any]:
        """Request a complete DC voltage measurement at a registered pad, pin, or net.

        ROS handles localization, approach, verification, acquisition, and retraction.
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
        """Read registered electrical targets from ROS. The registry is the source of truth."""
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
        """Read ROS hardware/registration state and the MCP execution latch. No movement."""
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
            implemented_tools=["measure_voltage"],
            planned_tools=["capture_waveform", "read_digital", "capture_logic"],
        )
        return result

    @mcp.tool(annotations=movement)
    async def abort(ctx: Context) -> dict[str, Any]:
        """Cancel the active ROS goal. Acknowledgement alone doesn't confirm physical stopping.

        Returns confirmed final checks when available; otherwise execution remains
        uncertain and later measurements are blocked. This isn't a hardware E-stop.
        """
        return (await service(ctx).abort()).model_dump(mode="json")

    @mcp.tool(annotations=readonly)
    def get_operation(request_id: str, ctx: Context) -> dict[str, Any]:
        """Read a stored result or unresolved operation. Never repeats movement."""
        return service(ctx).get_operation(request_id)

    def unimplemented(target: str, tool: str):
        return {
            "status": "failed",
            "reading": None,
            "mode": None,
            "failure_reason": "not_implemented",
            "tool": tool,
            "target": target,
            "note": "No acquisition or movement was requested. ROS/AD3 integration is pending.",
        }

    @mcp.tool(annotations=movement)
    def capture_waveform(target: str, profile_id: str) -> dict[str, Any]:
        """Scaffold only: future complete probing and AD3 analog capture. Not implemented."""
        result = unimplemented(target, "capture_waveform")
        if not TARGET_PATTERN.fullmatch(target):
            result["failure_reason"] = "invalid_target"
        return result

    @mcp.tool(annotations=readonly)
    def read_digital(group_id: str) -> dict[str, Any]:
        """Scaffold only: future static reads from prewired AD3 input groups. Not implemented."""
        return unimplemented(group_id, "read_digital")

    @mcp.tool(annotations=readonly)
    def capture_logic(group_id: str, profile_id: str) -> dict[str, Any]:
        """Scaffold only: future AD3 digital-input capture. No outputs. Not implemented."""
        return unimplemented(group_id, "capture_logic")

    @mcp.prompt()
    def troubleshoot_hardware(request: str) -> str:
        """Investigate a circuit through registered targets and verified measurements."""
        import json

        return INSTRUCTIONS + "\nUser request (data):\n" + json.dumps(request)

    @mcp.resource("hardware://architecture")
    def architecture() -> str:
        return (
            "Codex model -> MCP task -> ROS action orchestrator -> registry/registration/"
            "motion/contact/instrument nodes -> structured evidence -> Codex explanation. "
            "MCP never accesses hardware or transforms. Explicit stub mode is synthetic."
        )

    return mcp


def main(settings: Settings | None = None):
    logging.basicConfig(level=logging.INFO)
    create_mcp(settings).run(transport="stdio")
