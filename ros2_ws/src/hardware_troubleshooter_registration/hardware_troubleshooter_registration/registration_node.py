"""ROS 2 board registration node. Geometry and state live in the ROS-free engine.

Subscribes image_raw + camera_info (exact stamp match), serves <prefix>/get,
publishes <prefix>/events, <prefix>/overlay, <prefix>/diagnostics, and the
camera -> board TF only while the registration is fully valid.
"""

from __future__ import annotations

import secrets
import time

import message_filters
import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import TransformStamped
from hardware_troubleshooter_interfaces.msg import RegistrationState
from hardware_troubleshooter_interfaces.srv import GetRegistration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time as RosTime
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener

from .camera import CameraModel
from .config import load_board_config, load_rig_config
from .engine import RegistrationEngine
from .geometry import rotation_to_quaternion
from .overlay import draw_overlay

# Agreed with the orchestrator side: the latest state survives late subscribers.
EVENTS_QOS = QoSProfile(
    depth=1,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class RegistrationNode(Node):
    def __init__(self, **kwargs):
        super().__init__("board_registration", **kwargs)
        board_config = self.declare_parameter("board_config", "").value
        rig_config = self.declare_parameter("rig_config", "").value
        prefix = self.declare_parameter("endpoint_prefix", "/registration").value.rstrip("/")
        tick_period = self.declare_parameter("tick_period_s", 0.1).value

        board, board_problems = load_board_config(board_config)
        rig, rig_problems = load_rig_config(rig_config)
        session = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + secrets.token_hex(2)
        self.engine = RegistrationEngine(board, board_problems, rig, rig_problems, session)
        self.base_frame = rig.base_frame
        self.bridge = CvBridge()

        image_sub = message_filters.Subscriber(
            self, Image, "image_raw", qos_profile=qos_profile_sensor_data
        )
        info_sub = message_filters.Subscriber(
            self, CameraInfo, "camera_info", qos_profile=qos_profile_sensor_data
        )
        # Exact stamps only: an image without its own camera_info is never processed.
        self.sync = message_filters.TimeSynchronizer([image_sub, info_sub], 10)
        self.sync.registerCallback(self.on_frame)

        self.events = self.create_publisher(RegistrationState, f"{prefix}/events", EVENTS_QOS)
        self.overlay = self.create_publisher(Image, f"{prefix}/overlay", 1)
        self.diagnostics = self.create_publisher(DiagnosticArray, f"{prefix}/diagnostics", 10)
        self.service = self.create_service(GetRegistration, f"{prefix}/get", self.on_get)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.timer = self.create_timer(tick_period, self.on_tick)

        for problem in board_problems + rig_problems:
            self.get_logger().warning(problem)
        self.engine.tick(self.now())
        self.publish_events()

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_frame(self, image_msg: Image, info_msg: CameraInfo):
        camera, problem = CameraModel.from_camera_info(
            info_msg.width,
            info_msg.height,
            info_msg.k,
            info_msg.d,
            info_msg.distortion_model,
            (info_msg.binning_x, info_msg.binning_y),
            (info_msg.roi.x_offset, info_msg.roi.y_offset, info_msg.roi.width, info_msg.roi.height),
        )
        image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        stamp = image_msg.header.stamp.sec + image_msg.header.stamp.nanosec * 1e-9
        result = self.engine.process_frame(image, stamp, image_msg.header.frame_id, camera, problem)
        if result is None:
            return
        self.publish_events()
        snapshot = self.engine.snapshot(self.now())
        if self.overlay.get_subscription_count():
            drawn = draw_overlay(image, result, snapshot, self.engine.board, camera)
            msg = self.bridge.cv2_to_imgmsg(drawn, encoding="bgr8")
            msg.header = image_msg.header
            self.overlay.publish(msg)
        self.publish_diagnostics(snapshot, result, image_msg.header)

    def on_tick(self):
        self.update_extrinsic()
        now = self.now()
        self.engine.tick(now)
        self.publish_events()
        transform = self.engine.tf_to_publish(now)
        if transform is None:
            return
        parent, child, rotation, translation = transform
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = parent
        msg.child_frame_id = child
        msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = (
            float(v) for v in translation
        )
        q = rotation_to_quaternion(rotation)
        r = msg.transform.rotation
        r.x, r.y, r.z, r.w = q
        self.tf_broadcaster.sendTransform(msg)

    def update_extrinsic(self):
        camera_frame = self.engine.camera_frame
        now = self.now()
        if not self.base_frame:
            self.engine.set_extrinsic(None, "extrinsic_unavailable:rig.base_frame unset", now)
            return
        if not camera_frame:
            self.engine.set_extrinsic(None, "extrinsic_unavailable:no camera frame yet", now)
            return
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, camera_frame, RosTime())
        except TransformException as exc:
            reason = (
                f"extrinsic_unavailable:{self.base_frame} <- {camera_frame} ({type(exc).__name__})"
            )
            self.engine.set_extrinsic(None, reason, now)
            return
        t, q = tf.transform.translation, tf.transform.rotation
        self.engine.set_extrinsic(
            (np.array([t.x, t.y, t.z]), np.array([q.x, q.y, q.z, q.w])), None, now
        )

    def on_get(self, request, response):
        snapshot = self.engine.snapshot(self.now())
        msg = to_msg(snapshot)
        if request.board_id and request.board_id != snapshot.board_id:
            msg = RegistrationState(
                board_id=request.board_id,
                status="invalid",
                failure_reason=f"unknown_board:{request.board_id} (configured: {snapshot.board_id or 'none'})",
            )
        response.registration = msg
        return response

    def publish_events(self):
        for snapshot in self.engine.drain_events():
            self.events.publish(to_msg(snapshot))

    def publish_diagnostics(self, snapshot, result, header):
        status = DiagnosticStatus(name="board_registration", hardware_id=snapshot.board_id)
        status.level = DiagnosticStatus.OK if snapshot.status == "valid" else DiagnosticStatus.WARN
        status.message = snapshot.failure_reason or "valid"
        values = {
            "registration_id": snapshot.registration_id,
            "frame_ok": result.ok,
            "frame_reasons": "; ".join(result.reasons),
            "frame_residual_px": result.residual_px,
            "frame_edge_points": result.edge_points,
            "frame_symmetry_resolved": result.symmetry_resolved,
            "hypothesis_residuals_px": [round(h.residual_px, 4) for h in result.hypotheses],
            "hypothesis_feature_contrast": [h.feature_contrast for h in result.hypotheses],
            "ippe_tilt_deg": result.ippe_tilt_deg,
            "ippe_offset_m": result.ippe_offset_m,
            "registration_residual_px": snapshot.residual_px,
            "registration_edge_points": snapshot.edge_points,
            "registration_symmetry_resolved": snapshot.symmetry_resolved,
            "registration_stamp": snapshot.registration_stamp,
            "calibration_revision": snapshot.calibration_revision,
        }
        status.values = [KeyValue(key=k, value=str(v)) for k, v in values.items()]
        self.diagnostics.publish(DiagnosticArray(header=header, status=[status]))


def to_time(seconds: float | None) -> Time:
    if seconds is None:
        return Time()
    nanoseconds = round(seconds * 1e9)
    return Time(sec=nanoseconds // 1_000_000_000, nanosec=nanoseconds % 1_000_000_000)


def to_msg(snapshot) -> RegistrationState:
    msg = RegistrationState()
    msg.header.stamp = to_time(snapshot.stamp)
    msg.header.frame_id = snapshot.camera_frame
    msg.board_id = snapshot.board_id
    msg.registration_id = snapshot.registration_id
    msg.calibration_revision = snapshot.calibration_revision
    msg.status = snapshot.status
    msg.calibration_valid = snapshot.calibration_valid
    msg.tracking_valid = snapshot.tracking_valid
    msg.board_unmoved = snapshot.board_unmoved
    msg.board_frame = snapshot.board_frame
    msg.has_pose = snapshot.has_pose
    if snapshot.has_pose:
        p = msg.board_in_camera.position
        p.x, p.y, p.z = (float(v) for v in snapshot.translation)
        o = msg.board_in_camera.orientation
        o.x, o.y, o.z, o.w = rotation_to_quaternion(snapshot.rotation)
    msg.valid_until = to_time(snapshot.valid_until)
    msg.has_reprojection_error = snapshot.residual_px is not None
    msg.reprojection_error_px = (
        float(snapshot.residual_px) if snapshot.residual_px is not None else float("nan")
    )
    # Board height above the arm base belongs to the motion adapter's calibration.
    msg.has_board_height = False
    msg.board_height_m = float("nan")
    msg.failure_reason = snapshot.failure_reason
    return msg


def main(args=None):
    rclpy.init(args=args)
    node = RegistrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
