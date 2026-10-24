import asyncio
import logging
import time
from dataclasses import dataclass
from uuid import uuid4

import anyio

from .backend import Backend, BackendUnavailable, Feedback, GoalHandle, GoalRejected
from .contracts import REQUEST_PATTERN, TARGET_PATTERN, Measurement, failure, normalize_result
from .journal import Journal
from .settings import Settings

LOG = logging.getLogger(__name__)


@dataclass
class Active:
    request_id: str
    target: str
    started: float
    handle: GoalHandle | None = None
    result_task: asyncio.Task | None = None
    abort_requested: bool = False


class MeasurementService:
    """MCP boundary only. The measurement orchestrator owns every physical execution stage."""

    def __init__(self, settings: Settings, backend: Backend):
        self.settings, self.backend = settings, backend
        self.journal = Journal(settings.journal)
        self.blocked = self.journal.blocked()
        self.active: Active | None = None

    def get_operation(self, request_id: str) -> dict:
        row = self.journal.get(request_id)
        if not row:
            return {"status": "failed", "failure_reason": "unknown_request", "reading": None}
        if row[1]:
            return Measurement.model_validate_json(row[1]).model_dump(mode="json")
        in_flight = self.active is not None and self.active.request_id == request_id
        return {
            "request_id": request_id,
            "target": row[0],
            "reading": None,
            "status": "executing" if in_flight else "failed",
            "failure_reason": None if in_flight else "uncertain_motion_state",
            "execution_uncertain": not in_flight,
        }

    async def measure(self, target: str, request_id: str | None, feedback: Feedback) -> Measurement:
        request_id = request_id or str(uuid4())
        if not TARGET_PATTERN.fullmatch(target):
            return failure(request_id, target, "invalid_target")
        if not REQUEST_PATTERN.fullmatch(request_id):
            return failure(request_id, target, "invalid_request_id")
        previous = self.journal.get(request_id)
        if previous:
            if previous[0] != target:
                return failure(request_id, target, "request_id_conflict")
            if previous[1]:
                return Measurement.model_validate_json(previous[1])
            return failure(
                request_id,
                target,
                "busy" if self.active else "uncertain_motion_state",
                uncertain=self.active is None,
            )
        if self.active:
            return failure(request_id, target, "busy")
        if self.blocked:
            return failure(request_id, target, "uncertain_motion_state", uncertain=True)
        active = Active(request_id, target, time.monotonic())
        self.active = active
        dispatched = False
        try:
            if not await self.backend.available(self.settings.availability_timeout_s):
                return failure(request_id, target, "orchestrator_unavailable")
            if active.abort_requested:
                return failure(request_id, target, "aborted")
            timeout = self.settings.effective_outer_timeout
            if timeout is None or self.settings.effective_cancel_grace is None:
                return failure(request_id, target, "configuration_required:timeouts")
            self.journal.begin(request_id, target)
            dispatched = True
            # start() is non-blocking and retains the pending goal acceptance future.
            active.handle = self.backend.start(target, request_id, feedback)
            active.result_task = asyncio.create_task(active.handle.result())
            try:
                raw = await asyncio.wait_for(asyncio.shield(active.result_task), timeout)
                result = normalize_result(raw, request_id, target)
                if active.abort_requested:
                    result = self._without_reading(result, "aborted")
            except asyncio.TimeoutError:
                result = await self._cancel_and_settle(active, "timeout")
            return self._finish(active, result)
        except asyncio.CancelledError:
            # MCP cancellation may cancel an AnyIO scope as well as the asyncio task.
            with anyio.CancelScope(shield=True):
                if dispatched:
                    self._finish(active, await self._cancel_and_settle(active, "aborted"))
            raise
        except GoalRejected:
            return self._finish(active, failure(request_id, target, "goal_rejected"))
        except Exception as exc:
            LOG.exception("orchestrator_error request_id=%s", request_id)
            if dispatched:
                result = await self._cancel_and_settle(active, "orchestrator_connection_lost")
                return self._finish(active, result)
            reason = (
                "orchestrator_unavailable"
                if isinstance(exc, BackendUnavailable)
                else ("orchestrator_error")
            )
            return failure(request_id, target, reason)
        finally:
            self.active = None

    def _finish(self, active: Active, result: Measurement) -> Measurement:
        result.duration_ms = max(
            result.duration_ms, int((time.monotonic() - active.started) * 1000)
        )
        # An unconfirmed concurrent abort must survive a later result and restart.
        result.execution_uncertain |= self.blocked
        self.blocked |= result.execution_uncertain
        self.journal.finish(result)
        LOG.info(
            "measurement request_id=%s target=%s status=%s reason=%s mode=%s duration_ms=%s",
            result.request_id,
            active.target,
            result.status,
            result.failure_reason,
            result.mode,
            result.duration_ms,
        )
        return result

    @staticmethod
    def _without_reading(result: Measurement, reason: str) -> Measurement:
        return result.model_copy(
            update={"status": "failed", "reading": None, "failure_reason": reason}
        )

    async def _cancel_and_settle(self, active: Active, reason: str) -> Measurement:
        grace = self.settings.effective_cancel_grace or 0.1
        deadline = time.monotonic() + grace
        try:
            if active.handle is None or active.result_task is None:
                raise RuntimeError("Goal submission outcome unknown")
            # Cancel acknowledgement alone isn't proof of stopping or retraction.
            await asyncio.wait_for(active.handle.cancel(), grace / 2)
            remaining = max(0.001, deadline - time.monotonic())
            raw = await asyncio.wait_for(asyncio.shield(active.result_task), remaining)
            return self._without_reading(
                normalize_result(raw, active.request_id, active.target), reason
            )
        except Exception:  # noqa: BLE001 - unresolved cancellation must fail closed
            if active.result_task:
                # Retain the goal and late-cancel intent, but consume eventual errors.
                active.result_task.add_done_callback(_consume_result)
            return failure(active.request_id, active.target, reason, uncertain=True)

    async def abort(self) -> Measurement:
        active = self.active
        if active is None:
            return failure("", "", "no_active_operation")
        active.abort_requested = True
        if active.handle is None:
            return failure(active.request_id, active.target, "abort_requested")
        result = await self._cancel_and_settle(active, "aborted")
        # An unresolved abort must block later dispatch even if result arrives later.
        self.blocked |= result.execution_uncertain
        return result

    async def close(self):
        if self.active:
            await self.abort()
        await self.backend.close()
        self.journal.close()


def _consume_result(task):
    if not task.cancelled():
        task.exception()
