"""Optional MCP -> rclpy -> synthetic ROS server tests. Never targets a live action."""

import asyncio
import os
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

pytest.importorskip("rclpy", reason="ROS isn't installed; real ROS transport is unverified here")
try:
    from hardware_troubleshooter_interfaces.action import MeasureVoltage  # noqa: F401
except ImportError:
    pytest.skip("Build and source the shared ROS interface package first", allow_module_level=True)

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        ("success", None),
        ("contact_failed", "contact_unverified"),
        ("board_moved", "registration_invalidated"),
        ("stage_timeout", "timeout"),
        ("stale_calibration", "precheck_failed:stale_calibration"),
        ("slow", "aborted"),
    ],
)
async def test_mcp_to_ros_action_stub(tmp_path, scenario, expected):
    prefix = f"/test_only_{uuid4().hex}"
    env = {key: value for key, value in os.environ.items() if not key.startswith("TH_")}
    env.update(
        TH_BACKEND="ros",
        TH_ACTION_NAME=f"{prefix}/measure",
        TH_REGISTRY_SERVICE=f"{prefix}/registry",
        TH_STATUS_SERVICE=f"{prefix}/status",
        TH_JOURNAL=str(tmp_path / "ros.sqlite3"),
        TH_OUTER_TIMEOUT_S="5",
        TH_CANCEL_GRACE_S="1",
        TH_AVAILABILITY_TIMEOUT_S="0.25",
    )
    params = StdioServerParameters(
        command=sys.executable, args=[str(ROOT / "run_mcp.py")], cwd=str(ROOT), env=env
    )
    with (tmp_path / "ros-stub.log").open("w") as log:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "troubleshooter.ros_stub",
            "--scenario",
            scenario,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=log,
        )
        try:
            async with stdio_client(params) as (read, write), ClientSession(read, write) as client:
                await client.initialize()
                for _ in range(30):
                    response = await client.call_tool("get_rig_status", {})
                    if response.structuredContent["status"] == "ok":
                        break
                    assert process.returncode is None, "ROS stub exited; inspect ros-stub.log"
                    await asyncio.sleep(0.1)
                else:
                    pytest.fail("Private ROS stub services didn't become available")
                pending = asyncio.create_task(
                    client.call_tool(
                        "measure_voltage", {"target": "TP_DEMO", "request_id": "ros-test"}
                    )
                )
                if scenario == "slow":
                    for _ in range(30):
                        status = await client.call_tool("get_rig_status", {})
                        if status.structuredContent.get("mcp_busy"):
                            break
                        await asyncio.sleep(0.02)
                    await client.call_tool("abort", {})
                response = await pending
                assert not response.isError
                result = response.structuredContent
                assert result["failure_reason"] == expected
                assert result["mode"] == "simulated"
                assert result["reading"] == (3.29 if expected is None else None)
                if scenario == "board_moved":
                    next_call = await client.call_tool("measure_voltage", {"target": "TP_DEMO"})
                    assert next_call.structuredContent["failure_reason"] == (
                        "precheck_failed:registration_invalidated"
                    )
        finally:
            if process.returncode is None:
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()
                await asyncio.wait_for(process.wait(), timeout=5)
