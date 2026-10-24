"""Node checks over real ROS topics on unique test names. Never uses the production endpoints."""

import copy
import time
from uuid import uuid4

import pytest
import yaml

rclpy = pytest.importorskip("rclpy", reason="ROS isn't installed; node checks are unverified here")
pytest.importorskip("cv_bridge")
try:
    from hardware_troubleshooter_interfaces.msg import RegistrationState
    from hardware_troubleshooter_interfaces.srv import GetRegistration
except ImportError:
    pytest.skip(
        "Build and source hardware_troubleshooter_interfaces first", allow_module_level=True
    )

from conftest import Scene
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from hardware_troubleshooter_registration.image_folder_player import (
    camera_info_from_yaml,
)
from hardware_troubleshooter_registration.registration_node import (
    EVENTS_QOS,
    RegistrationNode,
)
from hardware_troubleshooter_registration.synthetic import (
    SYNTHETIC_BOARD,
    SYNTHETIC_CAMERA_INFO,
    SYNTHETIC_RIG,
    render_frame,
)
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from sensor_msgs.msg import CameraInfo, Image
from tf2_msgs.msg import TFMessage
from tf2_ros import StaticTransformBroadcaster


@pytest.fixture
def ros():
    rclpy.init()
    yield
    rclpy.try_shutdown()


class Harness:
    def __init__(self, tmp_path, publish_extrinsic=True):
        uid = uuid4().hex[:8]
        namespace = f"/test_only_{uid}"
        self.prefix = f"{namespace}/registration"
        self.board_frame = f"test_board_{uid}"
        self.camera_frame = f"test_camera_{uid}"
        board = dict(SYNTHETIC_BOARD, frame_id=self.board_frame)
        rig = copy.deepcopy(SYNTHETIC_RIG)
        rig["base_frame"] = f"test_base_{uid}"
        (tmp_path / "board.yaml").write_text(yaml.safe_dump(board))
        (tmp_path / "rig.yaml").write_text(yaml.safe_dump(rig))
        self.scene = Scene()
        self.bridge = CvBridge()
        self.info = camera_info_from_yaml(SYNTHETIC_CAMERA_INFO)

        self.node = RegistrationNode(
            namespace=namespace,
            parameter_overrides=[
                Parameter("board_config", value=str(tmp_path / "board.yaml")),
                Parameter("rig_config", value=str(tmp_path / "rig.yaml")),
                Parameter("endpoint_prefix", value=self.prefix),
            ],
        )
        self.peer = rclpy.create_node(f"harness_{uid}", namespace=namespace)
        self.image_pub = self.peer.create_publisher(Image, "image_raw", 10)
        self.info_pub = self.peer.create_publisher(CameraInfo, "camera_info", 10)
        self.events = []
        self.peer.create_subscription(
            RegistrationState, f"{self.prefix}/events", self.events.append, EVENTS_QOS
        )
        self.tf_children = set()
        self.peer.create_subscription(TFMessage, "/tf", self.on_tf, 100)
        self.client = self.peer.create_client(GetRegistration, f"{self.prefix}/get")
        if publish_extrinsic:
            tf = TransformStamped()
            tf.header.stamp = self.peer.get_clock().now().to_msg()
            tf.header.frame_id = rig["base_frame"]
            tf.child_frame_id = self.camera_frame
            tf.transform.translation.z = 0.5
            tf.transform.rotation.x = 1.0
            self.static = StaticTransformBroadcaster(self.peer)
            self.static.sendTransform(tf)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.executor.add_node(self.peer)

    def on_tf(self, msg):
        self.tf_children.update(t.child_frame_id for t in msg.transforms)

    def spin(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=0.02)

    def publish(self, pose, frames=3, **render):
        image = render_frame(self.scene.board, self.scene.camera, self.scene.plane, pose, **render)
        for _ in range(frames):
            msg = self.bridge.cv2_to_imgmsg(image, encoding="mono8")
            msg.header.stamp = self.peer.get_clock().now().to_msg()
            msg.header.frame_id = self.camera_frame
            self.info.header = msg.header
            self.image_pub.publish(msg)
            self.info_pub.publish(self.info)
            self.spin(0.15)

    def get(self) -> RegistrationState:
        assert self.client.wait_for_service(timeout_sec=5.0)
        future = self.client.call_async(GetRegistration.Request(board_id=""))
        deadline = time.monotonic() + 5.0
        while not future.done() and time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=0.02)
        assert future.done()
        return future.result().registration

    def close(self):
        self.executor.shutdown()
        self.node.destroy_node()
        self.peer.destroy_node()


@pytest.fixture
def harness(tmp_path, ros):
    h = Harness(tmp_path)
    yield h
    h.close()


def test_register_then_board_moved(harness):
    pose = harness.scene.centred_pose(theta=0.3)
    harness.spin(0.5)
    harness.publish(pose)
    harness.spin(0.3)
    state = harness.get()
    assert state.status == "valid", state.failure_reason
    assert state.registration_id.endswith("/r1")
    assert state.header.frame_id == harness.camera_frame
    assert state.has_pose and state.has_reprojection_error
    assert not state.has_board_height
    assert harness.board_frame in harness.tf_children

    harness.publish((pose[0] + 0.01, pose[1], pose[2]), frames=1)
    moved = [e for e in harness.events if e.failure_reason.startswith("board_moved")]
    assert moved and moved[0].registration_id.endswith("/r1")
    assert moved[0].status == "invalid" and not moved[0].board_unmoved

    harness.publish((pose[0] + 0.01, pose[1], pose[2]))
    state = harness.get()
    assert state.registration_id.endswith("/r2") and state.status == "valid"
    after_move = harness.events[harness.events.index(moved[0]) :]
    assert not any(e.registration_id.endswith("/r1") and e.status == "valid" for e in after_move)


def test_camera_silence_publishes_tracking_lost(harness):
    harness.spin(0.5)
    harness.publish(harness.scene.centred_pose())
    harness.spin(SYNTHETIC_RIG["thresholds"]["tracking_loss_timeout_s"] + 0.5)
    lost = [e for e in harness.events if e.failure_reason.startswith("tracking_lost")]
    assert lost and lost[0].registration_id.endswith("/r1")
    assert harness.get().status == "invalid"


def test_late_subscriber_gets_latest_state(harness):
    harness.spin(0.5)
    harness.publish(harness.scene.centred_pose())
    late = []
    harness.peer.create_subscription(
        RegistrationState, f"{harness.prefix}/events", late.append, EVENTS_QOS
    )
    harness.spin(0.5)
    assert len(late) == 1
    assert late[0].registration_id == harness.get().registration_id


def test_missing_extrinsic_gives_no_tf(tmp_path, ros):
    h = Harness(tmp_path, publish_extrinsic=False)
    try:
        h.spin(0.5)
        h.publish(h.scene.centred_pose())
        h.spin(0.5)
        state = h.get()
        assert state.tracking_valid
        assert not state.calibration_valid
        assert state.status == "invalid"
        assert "extrinsic_unavailable" in state.failure_reason
        assert h.board_frame not in h.tf_children
    finally:
        h.close()
