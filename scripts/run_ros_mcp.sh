#!/usr/bin/env bash
set -e
# Launch this script inside the bench's ROS environment (including WSL if used).
# TH_ROS_SETUP is the absolute path to that installation's setup.bash.
: "${TH_ROS_SETUP:?Set the absolute ROS setup.bash path}"
: "${TH_OUTER_TIMEOUT_S:?Agree the outer timeout with the orchestrator}"
: "${TH_CANCEL_GRACE_S:?Agree the cancellation grace with the orchestrator}"
source "$TH_ROS_SETUP"
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_dir/ros2_ws/install/setup.bash"
export TH_BACKEND=ros
exec "$project_dir/.venv-ros/bin/python" "$project_dir/run_mcp.py"
