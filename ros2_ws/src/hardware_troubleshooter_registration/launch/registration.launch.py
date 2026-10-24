"""Registration node, plus optional image-folder playback and extrinsic static TF."""

from pathlib import Path

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _nodes(context):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    use_sim_time = arg("use_sim_time").lower() == "true"
    remaps = [("image_raw", arg("image_topic")), ("camera_info", arg("camera_info_topic"))]
    nodes = [
        Node(
            package="hardware_troubleshooter_registration",
            executable="registration_node",
            name="board_registration",
            parameters=[
                {
                    "board_config": arg("board_config"),
                    "rig_config": arg("rig_config"),
                    "use_sim_time": use_sim_time,
                }
            ],
            remappings=remaps,
            output="screen",
        )
    ]

    if arg("image_folder"):
        nodes.append(
            Node(
                package="hardware_troubleshooter_registration",
                executable="image_folder_player",
                parameters=[
                    {
                        "folder": arg("image_folder"),
                        "camera_info_file": arg("camera_info_file"),
                        "frame_id": arg("playback_frame_id"),
                        "rate_hz": float(arg("playback_rate_hz")),
                        "loop": arg("playback_loop").lower() == "true",
                    }
                ],
                remappings=remaps,
                output="screen",
            )
        )

    if arg("publish_extrinsic").lower() == "true":
        rig = yaml.safe_load(Path(arg("rig_config")).read_text()) or {}
        extrinsic = rig.get("camera_extrinsic") or {}
        base_frame = rig.get("base_frame")
        translation = extrinsic.get("translation_m")
        rotation = extrinsic.get("rotation_xyzw")
        camera_frame = extrinsic.get("camera_frame")
        if not (base_frame and camera_frame and translation and rotation):
            raise RuntimeError(
                "publish_extrinsic:=true needs base_frame and camera_extrinsic in the rig config"
            )
        values = [str(float(v)) for v in list(translation) + list(rotation)]
        names = ["--x", "--y", "--z", "--qx", "--qy", "--qz", "--qw"]
        nodes.append(
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                arguments=[a for pair in zip(names, values) for a in pair]
                + ["--frame-id", base_frame, "--child-frame-id", camera_frame],
                output="screen",
            )
        )
    return nodes


def generate_launch_description():
    defaults = {
        "board_config": "",
        "rig_config": "",
        "image_topic": "image_raw",
        "camera_info_topic": "camera_info",
        "use_sim_time": "false",
        "publish_extrinsic": "false",
        "image_folder": "",
        "camera_info_file": "",
        "playback_frame_id": "",
        "playback_rate_hz": "5.0",
        "playback_loop": "false",
    }
    return LaunchDescription(
        [DeclareLaunchArgument(name, default_value=value) for name, value in defaults.items()]
        + [OpaqueFunction(function=_nodes)]
    )
