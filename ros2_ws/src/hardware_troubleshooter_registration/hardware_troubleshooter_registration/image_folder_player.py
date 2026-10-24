"""Publish a folder of images with a camera_calibration YAML as image_raw + camera_info.

For manual testing without a camera. Each image and its camera_info share one stamp
taken from this node's clock. For recorded topics use `ros2 bag play --clock` instead.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class ImageFolderPlayer(Node):
    def __init__(self, **kwargs):
        super().__init__("image_folder_player", **kwargs)
        folder = self.declare_parameter("folder", "").value
        info_file = self.declare_parameter("camera_info_file", "").value
        self.frame_id = self.declare_parameter("frame_id", "").value
        rate = self.declare_parameter("rate_hz", 5.0).value
        self.loop = self.declare_parameter("loop", False).value
        if not folder or not info_file or not self.frame_id:
            raise ValueError("folder, camera_info_file and frame_id parameters are required")
        self.files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in _SUFFIXES)
        if not self.files:
            raise ValueError(f"no images in {folder}")
        self.info = camera_info_from_yaml(yaml.safe_load(Path(info_file).read_text()))
        self.bridge = CvBridge()
        self.index = 0
        self.image_pub = self.create_publisher(Image, "image_raw", 10)
        self.info_pub = self.create_publisher(CameraInfo, "camera_info", 10)
        self.timer = self.create_timer(1.0 / rate, self.publish_next)
        self.get_logger().info(f"playing {len(self.files)} images from {folder}")

    def publish_next(self):
        if self.index >= len(self.files):
            if not self.loop:
                self.timer.cancel()
                self.get_logger().info("playback finished")
                return
            self.index = 0
        path = self.files[self.index]
        self.index += 1
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            self.get_logger().warning(f"skipping unreadable {path.name}")
            return
        msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.info.header = msg.header
        self.image_pub.publish(msg)
        self.info_pub.publish(self.info)


def camera_info_from_yaml(data: dict) -> CameraInfo:
    info = CameraInfo()
    info.width = int(data["image_width"])
    info.height = int(data["image_height"])
    info.distortion_model = data["distortion_model"]
    info.d = [float(v) for v in data["distortion_coefficients"]["data"]]
    info.k = [float(v) for v in data["camera_matrix"]["data"]]
    info.r = [float(v) for v in data["rectification_matrix"]["data"]]
    info.p = [float(v) for v in data["projection_matrix"]["data"]]
    return info


def main(args=None):
    rclpy.init(args=args)
    node = ImageFolderPlayer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
