from collections.abc import Awaitable, Callable
from typing import Protocol

Feedback = Callable[[str, int], Awaitable[None]]


class BackendUnavailable(RuntimeError):
    pass


class GoalRejected(RuntimeError):
    pass


class GoalHandle(Protocol):
    async def result(self) -> dict: ...
    async def cancel(self) -> bool: ...


class Backend(Protocol):
    async def available(self, timeout_s: float) -> bool: ...
    def start(self, target: str, request_id: str, feedback: Feedback) -> GoalHandle: ...
    async def list_test_points(self) -> dict: ...
    async def get_rig_status(self) -> dict: ...
    async def close(self) -> None: ...


class UnavailableBackend:
    def __init__(self, detail: str):
        self.detail = detail

    async def available(self, timeout_s: float) -> bool:
        return False

    def start(self, target: str, request_id: str, feedback: Feedback):
        raise BackendUnavailable(self.detail)

    async def list_test_points(self) -> dict:
        raise BackendUnavailable(self.detail)

    async def get_rig_status(self) -> dict:
        raise BackendUnavailable(self.detail)

    async def close(self) -> None:
        pass
