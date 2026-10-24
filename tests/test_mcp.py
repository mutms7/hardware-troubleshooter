import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from mcp import ClientSession, McpError, StdioServerParameters, types
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).parents[1]


@asynccontextmanager
async def connection(tmp_path, scenario="success", **settings):
    env = {key: value for key, value in os.environ.items() if not key.startswith("TH_")}
    env.update(
        TH_BACKEND="stub",
        TH_STUB_SCENARIO=scenario,
        TH_JOURNAL=str(tmp_path / "operations.sqlite3"),
    )
    env.update({f"TH_{key.upper()}": str(value) for key, value in settings.items()})
    params = StdioServerParameters(
        command=sys.executable, args=[str(ROOT / "run_mcp.py")], cwd=str(tmp_path), env=env
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as client:
        await client.initialize()
        yield client


async def call(client, tool, **arguments):
    response = await client.call_tool(tool, arguments)
    assert not response.isError, response
    return response.structuredContent


@pytest.mark.parametrize(
    ("scenario", "status", "reason"),
    [
        ("success", "ok", None),
        ("contact_failed", "inconclusive", "contact_unverified"),
        ("zero_unverified", "inconclusive", "protocol_violation"),
        ("board_moved", "inconclusive", "registration_invalidated"),
        ("stale_calibration", "failed", "precheck_failed:stale_calibration"),
        ("stage_timeout", "failed", "timeout"),
        ("unset_config", "failed", "precheck_failed:live_config_unset"),
        ("malformed", "inconclusive", "protocol_violation"),
        ("bad_position", "inconclusive", "protocol_violation"),
        ("bad_mode", "inconclusive", "protocol_violation"),
        ("unavailable", "failed", "orchestrator_unavailable"),
        ("lost_connection", "failed", "orchestrator_connection_lost"),
    ],
)
async def test_actual_mcp_measurement_scenarios(tmp_path, scenario, status, reason):
    async with connection(tmp_path, scenario) as client:
        stages = []

        async def progress(value, total, message):
            stages.append(message.split(" ")[0])

        response = await client.call_tool(
            "measure_voltage", {"target": "TP_DEMO"}, progress_callback=progress
        )
        assert not response.isError
        result = response.structuredContent
        assert result["status"] == status
        assert result["failure_reason"] == reason
        assert result["reading"] == (3.29 if status == "ok" else None)
        if scenario in {"contact_failed", "board_moved", "stale_calibration", "stage_timeout"}:
            assert "ACQUIRE" not in stages
        if status == "ok":
            assert "VERIFY_POSITION" in stages
            assert "VERIFY_CONTACT" in stages
            assert "ACQUIRE" in stages
            assert result["mode"] == "simulated"
            assert result["checks"]["move_completed"]
            assert result["checks"]["position_ok"]
            assert result["contact_verified"]
            assert result["checks"]["retraction_status"] == "completed"
        if scenario == "board_moved":
            result = await call(client, "measure_voltage", target="TP_DEMO")
            assert result["failure_reason"] == "precheck_failed:registration_invalidated"
            rig = await call(client, "get_rig_status")
            assert not rig["registration_valid"]


async def test_tool_contract_and_deduplication(tmp_path):
    async with connection(tmp_path) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        assert set(tools) == {
            "measure_voltage",
            "list_test_points",
            "get_rig_status",
            "abort",
            "get_operation",
            "capture_waveform",
            "read_digital",
            "capture_logic",
        }
        assert tools["measure_voltage"].annotations.readOnlyHint is False
        assert tools["list_test_points"].annotations.readOnlyHint is True
        assert "Jev" not in str(tools)
        points = await call(client, "list_test_points")
        assert points["targets"][0]["id"] == "TP_DEMO"
        result = await call(client, "measure_voltage", target="DEMO_RAIL", request_id="same")
        replay = await call(client, "measure_voltage", target="DEMO_RAIL", request_id="same")
        assert result == replay
        stored = await call(client, "get_operation", request_id="same")
        assert stored == result
        conflict = await call(client, "measure_voltage", target="TP_DEMO", request_id="same")
        assert conflict["failure_reason"] == "request_id_conflict"
        for target in ["", "x;do-something", "../../bad target", "x" * 97]:
            invalid = await call(client, "measure_voltage", target=target)
            assert invalid["failure_reason"] == "invalid_target"
        unknown = await call(client, "measure_voltage", target="UNREGISTERED")
        assert unknown["failure_reason"] == "unknown_target"
        for tool, args in [
            ("capture_waveform", {"target": "TP_DEMO", "profile_id": "scope"}),
            ("read_digital", {"group_id": "inputs"}),
            ("capture_logic", {"group_id": "inputs", "profile_id": "logic"}),
        ]:
            extension = await call(client, tool, **args)
            assert extension["failure_reason"] == "not_implemented"
            assert extension["reading"] is None
        prompt = await client.get_prompt("troubleshoot_hardware", {"request": "No output"})
        assert "registered" in prompt.messages[0].content.text


async def wait_until_busy(client):
    for _ in range(50):
        if (await call(client, "get_rig_status"))["mcp_busy"]:
            return
        await asyncio.sleep(0.02)
    pytest.fail("Measurement never became active")


async def test_abort_and_busy_through_actual_mcp(tmp_path):
    async with connection(tmp_path, "slow") as client:
        running = asyncio.create_task(call(client, "measure_voltage", target="TP_DEMO"))
        await wait_until_busy(client)
        busy = await call(client, "measure_voltage", target="TP_DEMO")
        assert busy["failure_reason"] == "busy"
        aborted = await call(client, "abort")
        result = await running
        assert aborted["failure_reason"] == "aborted"
        assert result["failure_reason"] == "aborted"
        assert result["reading"] is None
        assert result["checks"]["stop_confirmed"]
        assert not result["execution_uncertain"]


@pytest.mark.parametrize(("scenario", "uncertain"), [("slow", False), ("cancel_unconfirmed", True)])
async def test_outer_timeout_and_uncertain_restart(tmp_path, scenario, uncertain):
    async with connection(tmp_path, scenario, outer_timeout_s=0.1, cancel_grace_s=0.15) as client:
        result = await call(client, "measure_voltage", target="TP_DEMO", request_id="timed")
        assert result["failure_reason"] == "timeout"
        assert result["reading"] is None
        assert result["execution_uncertain"] == uncertain
        if uncertain:
            blocked = await call(client, "measure_voltage", target="TP_DEMO")
            assert blocked["failure_reason"] == "uncertain_motion_state"
    if uncertain:
        async with connection(tmp_path) as restarted:
            blocked = await call(restarted, "measure_voltage", target="TP_DEMO")
            assert blocked["failure_reason"] == "uncertain_motion_state"


async def test_missing_backend_never_falls_back_to_stub(tmp_path):
    async with connection(tmp_path, backend="none", outer_timeout_s=2, cancel_grace_s=1) as client:
        result = await call(client, "measure_voltage", target="TP_DEMO")
        assert result["failure_reason"] == "orchestrator_unavailable"
        assert result["reading"] is None
        assert result["mode"] is None


async def test_mcp_request_cancellation_reaches_backend(tmp_path):
    async with connection(tmp_path, "slow") as client:
        # Capture the protocol ID before dispatch; it differs from the measurement request_id.
        protocol_id = client._request_id
        running = asyncio.create_task(
            call(client, "measure_voltage", target="TP_DEMO", request_id="client-cancel")
        )
        await asyncio.sleep(0)  # Dispatch this protocol ID before polling with another request.
        await wait_until_busy(client)
        await client.send_notification(
            types.ClientNotification(
                types.CancelledNotification(
                    params=types.CancelledNotificationParams(
                        requestId=protocol_id, reason="User cancelled"
                    )
                )
            )
        )
        with pytest.raises(McpError, match="Request cancelled"):
            await running
        for _ in range(50):
            result = await call(client, "get_operation", request_id="client-cancel")
            if result["status"] != "executing":
                break
            await asyncio.sleep(0.02)
        assert result["failure_reason"] == "aborted"
        assert result["checks"]["stop_confirmed"]
        assert result["reading"] is None
