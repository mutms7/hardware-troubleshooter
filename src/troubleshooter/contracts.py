import json
import logging
import math
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

LOG = logging.getLogger(__name__)
Mode = Literal["simulated", "live"]
TARGET_PATTERN = re.compile(r"[A-Za-z0-9_+./:#@\-]{1,96}\Z")
REQUEST_PATTERN = re.compile(r"[A-Za-z0-9_\-]{1,64}\Z")


class Checks(BaseModel):
    move_completed: bool = False
    position_ok: bool = False
    contact: Literal["verified", "not_verified", "unknown"] = "unknown"
    board_unmoved: bool = False
    calibration_valid: bool = False
    retraction_status: Literal["completed", "failed", "unknown", "not_attempted"] = "unknown"
    stop_confirmed: bool = False


class TargetResult(BaseModel):
    requested: str
    test_point: str | None = None
    net: str | None = None


class Measurement(BaseModel):
    request_id: str
    status: Literal["ok", "inconclusive", "failed"]
    reading: float | None = Field(default=None, allow_inf_nan=False)
    units: str = "V"
    target: TargetResult
    contact_verified: bool = False
    checks: Checks = Field(default_factory=Checks)
    attempts: int = 0
    recovery_log: list[Any] = Field(default_factory=list)
    failure_reason: str | None = None
    # No result from ROS means mode is unknown. Never infer it from MCP configuration.
    mode: Mode | None = None
    registration_id: str | None = None
    duration_ms: int = 0
    execution_uncertain: bool = False


class WireResult(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    request_id: str
    status: Literal["ok", "inconclusive", "failed"]
    reading: float
    units: str
    requested_target: str
    test_point: str
    net: str
    contact_verified: bool
    move_completed: bool
    position_ok: bool
    contact: Literal["verified", "not_verified", "unknown"]
    board_unmoved: bool
    calibration_valid: bool
    registration_id: str
    retraction_status: Literal["completed", "failed", "unknown", "not_attempted"]
    stop_confirmed: bool
    attempts: int = Field(ge=0)
    recovery_log: list[str]
    failure_reason: str
    mode: Mode
    duration_ms: int = Field(ge=0)


def failure(request_id: str, target: str, reason: str, *, uncertain=False) -> Measurement:
    return Measurement(
        request_id=request_id,
        status="failed",
        target=TargetResult(requested=target),
        failure_reason=reason,
        execution_uncertain=uncertain,
    )


def normalize_result(raw: dict, request_id: str, target: str) -> Measurement:
    """ROS NaN is never emitted as JSON; inconsistent evidence is never usable."""
    try:
        wire = WireResult.model_validate(raw)
    except (ValidationError, TypeError):
        LOG.error("protocol_violation request_id=%s raw_result=%r", request_id, raw)
        invalid = failure(request_id, target, "protocol_violation", uncertain=True)
        invalid.status = "inconclusive"
        return invalid
    checks = Checks(**{key: getattr(wire, key) for key in Checks.model_fields})
    log = []
    for item in wire.recovery_log:
        try:
            log.append(json.loads(item))
        except (ValueError, TypeError):
            log.append(item)
    numeric = math.isfinite(wire.reading)
    success_checks = (
        wire.contact_verified
        and checks.contact == "verified"
        and checks.move_completed
        and checks.position_ok
        and checks.board_unmoved
        and checks.calibration_valid
        and checks.retraction_status == "completed"
        and checks.stop_confirmed
        and bool(wire.registration_id)
        and bool(wire.test_point)
    )
    violated = (
        wire.request_id != request_id
        or wire.requested_target != target
        or wire.units != "V"
        or (wire.status == "ok" and (not numeric or not success_checks or wire.failure_reason))
        or (wire.status != "ok" and (not math.isnan(wire.reading) or not wire.failure_reason))
    )
    status = "inconclusive" if violated else wire.status
    reason = "protocol_violation" if violated else (wire.failure_reason or None)
    if violated:
        LOG.error("protocol_violation request_id=%s raw_result=%r", request_id, raw)
    return Measurement(
        request_id=request_id,
        status=status,
        reading=wire.reading if status == "ok" else None,
        target=TargetResult(
            requested=target, test_point=wire.test_point or None, net=wire.net or None
        ),
        contact_verified=wire.contact_verified,
        checks=checks,
        attempts=wire.attempts,
        recovery_log=log,
        failure_reason=reason,
        mode=wire.mode,
        registration_id=wire.registration_id or None,
        duration_ms=wire.duration_ms,
        execution_uncertain=not checks.stop_confirmed
        or checks.retraction_status in {"unknown", "failed"}
        or violated,
    )
