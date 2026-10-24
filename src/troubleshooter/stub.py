"""Explicit software simulation. No ROS, cameras, arms, or AD3 are accessed."""

import asyncio
import math
from dataclasses import dataclass

from .backend import BackendUnavailable, Feedback
from .settings import Settings

STAGES = (
    "PRECHECK",
    "RESOLVE",
    "PLAN",
    "APPROACH",
    "VERIFY_POSITION",
    "VERIFY_CONTACT",
    "ACQUIRE",
    "RETRACT",
)
TARGETS = [{"id": "TP_DEMO", "net": "DEMO_RAIL", "notes": "Synthetic target, not a real pad"}]


def scripted_stages(scenario: str, target: str, state) -> tuple[str, ...]:
    if not state.registration_valid or scenario in {"stale_calibration", "unset_config"}:
        return ("PRECHECK",)
    if target not in {"TP_DEMO", "DEMO_RAIL"}:
        return ("PRECHECK", "RESOLVE")
    if scenario in {"board_moved", "stage_timeout"}:
        return STAGES[:4] + ("STOP", "RETRACT")
    if scenario in {"contact_failed", "zero_unverified"}:
        return STAGES[:6] + ("RETRACT",)
    return STAGES


@dataclass
class StubState:
    registration_valid: bool = True


def stub_result(scenario: str, target: str, request_id: str, state: StubState) -> dict:
    """Shared scripted result for the in-process simulator and ROS action stub."""
    raw = {
        "request_id": request_id,
        "status": "ok",
        "reading": 3.29,
        "units": "V",
        "requested_target": target,
        "test_point": "TP_DEMO",
        "net": "DEMO_RAIL",
        "contact_verified": True,
        "move_completed": True,
        "position_ok": True,
        "contact": "verified",
        "board_unmoved": True,
        "calibration_valid": True,
        "registration_id": "synthetic-registration",
        "retraction_status": "completed",
        "stop_confirmed": True,
        "attempts": 1,
        "recovery_log": [],
        "failure_reason": "",
        "mode": "simulated",
        "duration_ms": 0,
    }
    reason = None
    if target not in {"TP_DEMO", "DEMO_RAIL"}:
        reason = "unknown_target"
    elif not state.registration_valid:
        reason = "precheck_failed:registration_invalidated"
    elif scenario in {"contact_failed", "zero_unverified"}:
        raw.update(contact_verified=False, contact="not_verified", status="inconclusive")
        reason = "contact_unverified"
    elif scenario == "board_moved":
        state.registration_valid = False
        raw.update(board_unmoved=False, calibration_valid=False, status="inconclusive")
        reason = "registration_invalidated"
    elif scenario == "stale_calibration":
        raw.update(
            calibration_valid=False,
            move_completed=False,
            position_ok=False,
            contact_verified=False,
            contact="unknown",
            attempts=0,
            retraction_status="not_attempted",
        )
        reason = "precheck_failed:stale_calibration"
    elif scenario == "unset_config":
        reason = "precheck_failed:live_config_unset"
    elif scenario == "stage_timeout":
        reason = "timeout"
    elif scenario == "aborted":
        reason = "aborted"
    elif scenario == "malformed":
        raw.update(contact_verified=False, contact="not_verified")
    elif scenario == "bad_position":
        raw["position_ok"] = False
    elif scenario == "bad_mode":
        raw["mode"] = "invented"
    if reason:
        raw.update(reading=math.nan, failure_reason=reason)
        if raw["status"] == "ok":
            raw["status"] = "failed"
    if scenario == "zero_unverified":
        # Deliberately malformed hardware report: boundary must reject even 0 V.
        raw["reading"] = 0.0
    return raw


class StubHandle:
    def __init__(self, backend, target: str, request_id: str, feedback: Feedback):
        self.backend, self.target, self.request_id, self.feedback = (
            backend,
            target,
            request_id,
            feedback,
        )
        self.cancelled = asyncio.Event()

    async def result(self) -> dict:
        self.backend.active = True
        try:
            for stage in scripted_stages(
                self.backend.settings.stub_scenario, self.target, self.backend.state
            ):
                await self.feedback(stage, 1)
                delay = self.backend.settings.stub_step_s
                if self.backend.settings.stub_scenario in {"slow", "cancel_unconfirmed"}:
                    delay *= 100
                try:
                    await asyncio.wait_for(self.cancelled.wait(), delay)
                    return stub_result("aborted", self.target, self.request_id, self.backend.state)
                except asyncio.TimeoutError:
                    pass
                if self.backend.settings.stub_scenario == "lost_connection":
                    raise BackendUnavailable("Synthetic connection loss")
            return stub_result(
                self.backend.settings.stub_scenario,
                self.target,
                self.request_id,
                self.backend.state,
            )
        finally:
            self.backend.active = False

    async def cancel(self) -> bool:
        if self.backend.settings.stub_scenario == "cancel_unconfirmed":
            return False
        self.cancelled.set()
        return True


class StubBackend:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.state = StubState()
        self.active = False

    async def available(self, timeout_s: float) -> bool:
        return self.settings.stub_scenario != "unavailable"

    def start(self, target: str, request_id: str, feedback: Feedback) -> StubHandle:
        return StubHandle(self, target, request_id, feedback)

    async def list_test_points(self) -> dict:
        if not await self.available(0):
            raise BackendUnavailable("Synthetic action server unavailable")
        return {"status": "ok", "mode": "simulated", "targets": TARGETS}

    async def get_rig_status(self) -> dict:
        if not await self.available(0):
            raise BackendUnavailable("Synthetic action server unavailable")
        return {
            "status": "ok",
            "mode": "simulated",
            "board_id": "synthetic-board",
            "registration_valid": self.state.registration_valid,
            "calibration_valid": self.state.registration_valid
            and self.settings.stub_scenario != "stale_calibration",
            "registration_age_s": 0.0,
            "registration_id": "synthetic-registration",
            "arm_state": "simulated_idle",
            "instrument_connected": True,
            "busy": self.active,
            "capabilities": ["measure_voltage"],
            "note": "All hardware state is simulated. AD3 tools aren't implemented yet.",
        }

    async def close(self):
        pass
