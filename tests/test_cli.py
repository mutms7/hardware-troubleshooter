import json
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("scenario", "status", "reading"),
    [
        ("success", "ok", 3.29),
        ("contact_failed", "inconclusive", None),
    ],
)
def test_smoke_command_runs_actual_mcp(scenario, status, reading):
    # Host backend settings can't turn the smoke command into a hardware operation.
    env = dict(os.environ, TH_BACKEND="none", TH_STUB_SCENARIO="unavailable")
    process = subprocess.run(
        [sys.executable, "-m", "troubleshooter.cli", "smoke", "--scenario", scenario],
        capture_output=True,
        text=True,
        timeout=15,
        env=env,
        check=True,
    )
    result = json.loads(process.stdout)
    assert result["hardware_accessed"] is False
    assert result["measurement"]["mode"] == "simulated"
    assert result["measurement"]["status"] == status
    assert result["measurement"]["reading"] == reading
