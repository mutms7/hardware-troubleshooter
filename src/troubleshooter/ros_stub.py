"""ROS action/service stub. Synthetic results only; no hardware can be connected here."""

import argparse
import threading
import time

from .settings import Settings
from .stub import TARGETS, StubState, scripted_stages, stub_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="success")
    args = parser.parse_args()
    settings = Settings(backend="stub", stub_scenario=args.scenario)
    if settings.stub_scenario == "lost_connection":
        parser.error("lost_connection is an in-process transport fault scenario only")
    # Intentionally optional imports so the package runs without ROS in stub mode.
    import rclpy
    from hardware_troubleshooter_interfaces.action import MeasureVoltage
    from hardware_troubleshooter_interfaces.msg import TestPoint
    from hardware_troubleshooter_interfaces.srv import GetRigStatus, ListTestPoints
    from rclpy.action import ActionServer, CancelResponse, GoalResponse
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node

    class StubNode(Node):
        def __init__(self):
            super().__init__("hardware_measurement_stub")
            self.state = StubState()
            self.busy = False
            self.lock = threading.Lock()
            self.group = ReentrantCallbackGroup()
            self.registry = self.create_service(
                ListTestPoints,
                settings.registry_service,
                self.list_targets,
                callback_group=self.group,
            )
            self.status = self.create_service(
                GetRigStatus, settings.status_service, self.rig_status, callback_group=self.group
            )
            self.server = None
            if settings.stub_scenario != "unavailable":
                self.server = ActionServer(
                    self,
                    MeasureVoltage,
                    settings.action_name,
                    self.execute,
                    goal_callback=self.accept_goal,
                    cancel_callback=self.cancel_goal,
                    callback_group=self.group,
                )
            self.get_logger().warning("SIMULATED ROS STUB: no arm, camera, contact sensor, or AD3")

        def accept_goal(self, _):
            with self.lock:
                if self.busy:
                    return GoalResponse.REJECT
                self.busy = True
                return GoalResponse.ACCEPT

        def cancel_goal(self, _):
            if settings.stub_scenario == "cancel_unconfirmed":
                return CancelResponse.REJECT
            return CancelResponse.ACCEPT

        def execute(self, handle):
            started = time.monotonic()
            try:
                cancelled = False
                for stage in scripted_stages(
                    settings.stub_scenario, handle.request.target, self.state
                ):
                    handle.publish_feedback(MeasureVoltage.Feedback(stage=stage, attempt=1))
                    delay = settings.stub_step_s
                    if settings.stub_scenario in {"slow", "cancel_unconfirmed"}:
                        delay *= 100
                    deadline = time.monotonic() + delay
                    while time.monotonic() < deadline:
                        if handle.is_cancel_requested:
                            cancelled = True
                            break
                        time.sleep(min(0.01, max(0, deadline - time.monotonic())))
                    if cancelled:
                        break
                scenario = "aborted" if cancelled else settings.stub_scenario
                raw = stub_result(
                    scenario, handle.request.target, handle.request.request_id, self.state
                )
                raw["duration_ms"] = int((time.monotonic() - started) * 1000)
                result = MeasureVoltage.Result(**raw)
                if cancelled:
                    handle.canceled()
                elif raw["status"] == "ok":
                    handle.succeed()
                else:
                    handle.abort()
                return result
            finally:
                with self.lock:
                    self.busy = False

        def list_targets(self, _, response):
            response.status, response.mode = "ok", "simulated"
            response.targets = [TestPoint(**item) for item in TARGETS]
            return response

        def rig_status(self, _, response):
            response.status, response.mode = "ok", "simulated"
            response.board_id = "synthetic-board"
            response.registration_valid = self.state.registration_valid
            response.calibration_valid = (
                self.state.registration_valid and settings.stub_scenario != "stale_calibration"
            )
            response.registration_age_s = 0.0
            response.registration_id = "synthetic-registration"
            response.arm_state = "simulated_idle"
            response.instrument_connected = True
            response.busy = self.busy
            response.capabilities = ["measure_voltage"]
            return response

    rclpy.init(args=[])
    node = StubNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
