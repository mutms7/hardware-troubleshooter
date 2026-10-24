"""Optional rclpy transport. No hardware calls, TF access, or stage execution."""

import asyncio
import logging
import os
import threading

from .backend import BackendUnavailable, Feedback, GoalRejected
from .contracts import WireResult
from .settings import Settings

LOG = logging.getLogger(__name__)


async def ros_future(future):
    """rclpy futures aren't concurrent.futures; bridge callbacks to the asyncio thread."""
    loop = asyncio.get_running_loop()
    waiter = loop.create_future()

    def settle():
        if waiter.done():
            return
        try:
            waiter.set_result(future.result())
        except asyncio.CancelledError:
            waiter.cancel()
        except Exception as exc:  # noqa: BLE001 - native future exceptions cross this boundary
            waiter.set_exception(exc)

    def done(_):
        if not loop.is_closed():
            loop.call_soon_threadsafe(settle)

    future.add_done_callback(done)
    try:
        return await asyncio.shield(waiter)
    except asyncio.CancelledError:
        waiter.add_done_callback(_consume_orphan)
        raise


class RosHandle:
    def __init__(self, backend, target: str, request_id: str, feedback: Feedback):
        self.backend = backend
        self.loop = asyncio.get_running_loop()
        self.cancel_requested = False
        goal = backend.action_type.Goal(target=target, request_id=request_id)

        def feedback_callback(message):
            def publish():
                task = self.loop.create_task(
                    feedback(message.feedback.stage, message.feedback.attempt)
                )
                task.add_done_callback(_consume_feedback)

            if not self.loop.is_closed():
                self.loop.call_soon_threadsafe(publish)

        self.acceptance = backend.action.send_goal_async(goal, feedback_callback=feedback_callback)

        def accepted(future):
            # A timeout while accepting must still cancel a goal accepted later.
            try:
                handle = future.result()
                if self.cancel_requested and handle.accepted:
                    handle.cancel_goal_async()
            except Exception:
                LOG.exception("late_goal_acceptance_failed request_id=%s", request_id)

        self.acceptance.add_done_callback(accepted)

    async def result(self) -> dict:
        handle = await ros_future(self.acceptance)
        if not handle.accepted:
            raise GoalRejected("ROS orchestrator rejected the measurement")
        wrapped = await ros_future(handle.get_result_async())
        raw = {key: getattr(wrapped.result, key) for key in WireResult.model_fields}
        # ROS action state and the application result must agree.
        if wrapped.status not in {4, 5, 6} or (raw["status"] == "ok" and wrapped.status != 4):
            raw["transport_protocol_violation"] = True
        return raw

    async def cancel(self) -> bool:
        self.cancel_requested = True
        handle = await ros_future(self.acceptance)
        if not handle.accepted:
            return True
        response = await ros_future(handle.cancel_goal_async())
        return bool(response.goals_canceling)


def _consume_feedback(task):
    if not task.cancelled() and task.exception():
        LOG.warning("MCP progress notification failed: %s", task.exception())


def _consume_orphan(future):
    if not future.cancelled():
        future.exception()


class RosBackend:
    def __init__(self, settings: Settings):
        # rclpy and generated messages come from the sourced ROS installation.
        # They aren't pip dependencies and are never needed for the explicit stub.
        import rclpy
        from hardware_troubleshooter_interfaces.action import MeasureVoltage
        from hardware_troubleshooter_interfaces.srv import GetRigStatus, ListTestPoints
        from rclpy.action import ActionClient
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.context import Context
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node

        os.environ.setdefault("RCUTILS_LOGGING_USE_STDOUT", "0")
        self.settings, self.rclpy = settings, rclpy
        self.context = Context()
        rclpy.init(args=[], context=self.context)
        self.node = Node("hardware_mcp_client", context=self.context)
        self.group = ReentrantCallbackGroup()
        self.action_type = MeasureVoltage
        self.action = ActionClient(
            self.node, MeasureVoltage, settings.action_name, callback_group=self.group
        )
        self.registry_type, self.status_type = ListTestPoints, GetRigStatus
        self.registry = self.node.create_client(
            ListTestPoints, settings.registry_service, callback_group=self.group
        )
        self.status = self.node.create_client(
            GetRigStatus, settings.status_service, callback_group=self.group
        )
        self.executor = MultiThreadedExecutor(num_threads=2, context=self.context)
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self._spin, name="ros-mcp-executor", daemon=True)
        self.thread.start()

    def _spin(self):
        try:
            self.executor.spin()
        except Exception:
            LOG.exception("ROS executor stopped")

    async def available(self, timeout_s: float) -> bool:
        if not self.thread.is_alive():
            return False
        return await asyncio.to_thread(self.action.wait_for_server, timeout_sec=timeout_s)

    def start(self, target: str, request_id: str, feedback: Feedback) -> RosHandle:
        return RosHandle(self, target, request_id, feedback)

    async def _call_service(self, client, request):
        available = await asyncio.to_thread(
            client.wait_for_service, timeout_sec=self.settings.availability_timeout_s
        )
        if not available:
            raise BackendUnavailable("ROS service unavailable")
        return await asyncio.wait_for(
            ros_future(client.call_async(request)), self.settings.service_timeout_s
        )

    async def list_test_points(self) -> dict:
        response = await self._call_service(self.registry, self.registry_type.Request())
        return {
            "status": response.status,
            "mode": response.mode,
            "failure_reason": response.failure_reason or None,
            "targets": [
                {"id": item.id, "net": item.net, "notes": item.notes} for item in response.targets
            ],
        }

    async def get_rig_status(self) -> dict:
        response = await self._call_service(self.status, self.status_type.Request())
        return {
            key: getattr(response, key)
            for key in (
                "status",
                "mode",
                "board_id",
                "registration_valid",
                "calibration_valid",
                "registration_age_s",
                "registration_id",
                "arm_state",
                "instrument_connected",
                "busy",
                "capabilities",
                "failure_reason",
            )
        }

    async def close(self):
        await asyncio.to_thread(self.executor.shutdown, timeout_sec=2.0)
        self.thread.join(timeout=2.0)
        self.node.destroy_node()
        self.context.try_shutdown()
