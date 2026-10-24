# Hardware troubleshooter MCP scaffold

Codex requests a measurement by registered pad, pin, or net identity. This Python
MCP server dispatches one task to a measurement orchestrator and returns validated
evidence. It never accesses a camera, robot, or instrument directly.

Start with [Simon's handoff](docs/SIMON_HANDOFF.md) and the
[measurement contract](docs/MEASUREMENT_CONTRACT.md).
[ARCHITECTURE.md](ARCHITECTURE.md) explains ownership and failure handling.
[registration/](registration/README.md) holds the CV board registration library.
[arm/](arm/README.md) builds each Maker Arm's URDF and SDK profile for the rotating toolhead.

## What works here

- Local MCP stdio transport, prompt, resource, and typed tool schemas.
- Voltage task dispatch through an explicit backend interface.
- Boundary checks separating motion, positioning, contact, calibration, and retraction.
- Busy rejection, bounded cancellation, request IDs, and persistent execution uncertainty.
- A software simulator that exercises the MCP flow without hardware.
- Board-edge CV registration as a plain Python library, tested on synthetic frames.

Jev, TypeSafe, RoboTwin, and Cartesia aren't active dependencies or execution paths.
Voice can later be a separate client using these measurement tools.

There is no real camera implementation, measurement orchestrator, or contact verifier
here. The AD3 capture tools and tool-head selection are in [docs/AD3.md](docs/AD3.md).
Passing tests here doesn't prove a physical measurement.

## Run the software simulator

Install uv, then clone and run the one-shot MCP test:

```powershell
git clone https://github.com/mutms7/hardware-troubleshooter.git
cd hardware-troubleshooter
uv sync --frozen --extra dev
uv run hardware-troubleshooter smoke
```

It starts an actual MCP server, performs the handshake, lists targets, measures
the synthetic point, prints the result, and exits. Expect status=ok, reading=3.29,
units=V, mode=simulated. No hardware is accessed. Each smoke run uses a
fresh, temporary simulated rig and journal.

Test a failed contact:

```powershell
uv run hardware-troubleshooter smoke --scenario contact_failed
```

Expect status=inconclusive, reading=null, failure_reason=contact_unverified.
Other smoke scenarios are board_moved and stage_timeout. Run the full automated
suite with `uv run pytest -q`.

To leave a server running for a desktop MCP client instead:

```powershell
uv sync --frozen --extra dev
uv run hardware-troubleshooter mcp --backend stub --scenario success
```

The process waits for an MCP client. It isn't an interactive terminal conversation.
All readings and hardware state from this backend are explicitly simulated.
Synthetic target IDs are `TP_DEMO` and `DEMO_RAIL`; neither describes an actual circuit.

Scenarios: success, contact_failed, zero_unverified, board_moved, stage_timeout,
slow, stale_calibration, unset_config, malformed, bad_mode, unavailable,
cancel_unconfirmed, lost_connection, bad_position.

Local `.env` files are ignored. Configuration comes from process environment
variables or CLI flags. No model-provider keys are needed.

## Connect Codex desktop

Add a local stdio MCP server in the desktop MCP settings. Use this checkout's
`.venv/Scripts/python.exe` as the command and the absolute path to `run_mcp.py`
as its argument on Windows. Set `TH_BACKEND=stub` for the simulator.

[config/codex-mcp.example.toml](config/codex-mcp.example.toml) contains a portable
example. Replace its paths before merging into a host configuration. No global
Codex settings were changed.

Ask Codex to list test points, read rig status, and measure `TP_DEMO`. A successful
simulation returns 3.29 V with `mode=simulated`. It demonstrates the interface only.

## Tools

| Tool | Current behavior |
| --- | --- |
| `measure_voltage(target, request_id?)` | Sends one task and waits for the validated final result. |
| `list_test_points()` | Reads the orchestrator registry or explicit stub registry. |
| `get_rig_status()` | Reads orchestrator state and reports the MCP busy/uncertainty latch. |
| `abort()` | Requests cancel and waits briefly for final stop/retraction checks. |
| `get_operation(request_id)` | Reads progress state or a saved result without repeating motion. |
| `get_instrument_status()` | Reads AD3 connection, each arm's current tool head, and the wiring. |
| `list_capture_profiles()` | Lists the configured scope profiles and logic types. |
| `select_tool(arm, head)` | Turns one arm's turret to the scope, logic, or multimeter head. |
| `capture_waveform(arm, profile_id)` | Selects the scope head and captures its AD3 channel. |
| `capture_logic(logic_type, lines)` | Selects logic heads and captures and decodes UART, PWM, I2C, or partial SPI. |
| `read_digital(group_id)` | Explicit `not_implemented` extension point. |

The AD3 tools run the WaveForms SDK on this host and capture wherever the probe already
is: they don't position it or verify contact. They need `TH_RIG_CONFIG`, and only
`TH_INSTRUMENTS=fake` simulates them. See [docs/AD3.md](docs/AD3.md) for wiring,
ground precautions, configuration, and outcomes. The multimeter head is a separate
device that isn't driven yet.

Each measurement returns status, nullable reading, units, requested/resolved target,
contact and stage checks, failure reason, request ID, mode, duration, and registration ID.
Only `status=ok` can carry a usable reading. An unverified contact isn't a zero reading.
An invalid result becomes `inconclusive: protocol_violation`.

Mode comes from the orchestrator's result. It is null for a boundary failure before
any trustworthy result arrives. A missing backend never activates the simulator.

Optional `request_id` makes an exact retry return the saved result. Reusing it for a
different target is rejected. Without it, each call is a new request, so the host
mustn't silently retry an interrupted movement.

## Verify

```powershell
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

Tests start a real MCP subprocess and exercise success, failed contact, unknown
target, moved board, stale calibration, malformed results, timeout, explicit abort,
MCP request cancellation, busy rejection, replay protection, and persistence after
uncertain execution. The registration library has its own suite, which needs numpy,
OpenCV and PyYAML: `python -m pytest -q registration/test`.

Unknown execution is latched in `data/mcp_operations.sqlite3`. No MCP reset tool
can dismiss it. The real orchestrator/operator recovery procedure remains a required
integration step; preserve the journal and inspect the physical bench. A process
restart mustn't be used as a motion reset.

## Next integration work

Simon supplies CV registration, board movement invalidation, and measured calibration
quality. This workspace owns the non-CV software: the measurement orchestrator, the
backend that connects MCP to it, arm/contact adapters, and AD3 acquisition remain to be
implemented against their hardware APIs. The registry needs the actual circuit's points, reference wiring,
and allowed operating conditions. Those values remain unset.

Share the [Simon handoff](docs/SIMON_HANDOFF.md) when coordinating the CV work.
