import asyncio
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from troubleshooter.ros_backend import RosHandle, ros_future


async def test_ros_future_callback_crosses_thread_to_asyncio():
    future = Future()
    waiting = asyncio.create_task(ros_future(future))
    await asyncio.to_thread(future.set_result, "completed")
    assert await waiting == "completed"


async def test_goal_accepted_after_cancel_is_still_cancelled():
    acceptance = Future()
    backend = SimpleNamespace(
        action_type=SimpleNamespace(Goal=lambda **args: SimpleNamespace(**args)),
        action=SimpleNamespace(send_goal_async=lambda *args, **kwargs: acceptance),
    )

    async def feedback(stage, attempt):
        pass

    handle = RosHandle(backend, "TP_DEMO", "late-goal", feedback)
    cancelling = asyncio.create_task(handle.cancel())
    await asyncio.sleep(0)
    cancelling.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelling
    cancellations = []
    native_handle = SimpleNamespace(
        accepted=True, cancel_goal_async=lambda: cancellations.append("cancel")
    )
    await asyncio.to_thread(acceptance.set_result, native_handle)
    await asyncio.sleep(0)
    assert cancellations == ["cancel"]
