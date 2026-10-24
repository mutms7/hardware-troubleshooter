"""One-shot, software-only MCP client. Never connects to hardware."""

import json
import os
import sys
import tempfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run_smoke(scenario: str):
    with tempfile.TemporaryDirectory(prefix="hardware-mcp-simulation-") as folder:
        env = {key: value for key, value in os.environ.items() if not key.startswith("TH_")}
        env["TH_JOURNAL"] = str(Path(folder) / "operations.sqlite3")
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "troubleshooter.cli", "mcp", "--backend", "stub", "--scenario", scenario],
            env=env,
        )
        with (Path(folder) / "server.log").open("w", encoding="utf-8") as log:
            async with (
                stdio_client(params, errlog=log) as (read, write),
                ClientSession(read, write) as client,
            ):
                await client.initialize()
                targets = await client.call_tool("list_test_points", {})
                result = await client.call_tool("measure_voltage", {"target": "TP_DEMO"})
                if targets.isError or result.isError or result.structuredContent is None:
                    raise RuntimeError("MCP smoke test didn't return a valid tool result")
                print(
                    json.dumps(
                        {
                            "transport": "actual MCP stdio",
                            "hardware_accessed": False,
                            "scenario": scenario,
                            "registered_targets": targets.structuredContent,
                            "measurement": result.structuredContent,
                        },
                        indent=2,
                        allow_nan=False,
                    )
                )
