# Hardware troubleshooter MCP scaffold

Codex requests a measurement by registered pad, pin, or net identity. This Python
MCP server sends one ROS 2 action goal and returns validated evidence. It never
accesses a camera, TF, robot, or instrument directly.

Start with [Simon's handoff](docs/SIMON_HANDOFF.md) and the
[shared ROS contract](docs/ROS_CONTRACT.md).
[ARCHITECTURE.md](ARCHITECTURE.md) explains ownership and failure handling.

## What works here

- Local MCP stdio transport, prompt, resource, and typed tool schemas.
- Voltage task dispatch through an explicit backend interface.
- Boundary checks separating motion, positioning, contact, calibration, and retraction.
- Busy rejection, bounded cancellation, request IDs, and persistent execution uncertainty.
- A software simulator that exercises the MCP flow without ROS.
- Optional rclpy action/service client and a separate ROS action/service stub.
- Shared ROS action, service, and registration message source files for teammates.

Jev, TypeSafe, RoboTwin, and Cartesia aren't active dependencies or execution paths.
Voice can later be a separate client using these measurement tools.

There is no real camera implementation, arm controller, contact verifier, or AD3
driver here. The ROS client/stub source hasn't been run on this Windows environment
because ROS isn't installed. Passing tests here doesn't prove a physical measurement.

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
units=V, mode=simulated. No ROS or hardware is accessed. Each smoke run uses a
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
| `list_test_points()` | Reads the ROS registry or explicit stub registry. |
| `get_rig_status()` | Reads ROS state and reports the MCP busy/uncertainty latch. |
| `abort()` | Requests cancel and waits briefly for final stop/retraction checks. |
| `get_operation(request_id)` | Reads progress state or a saved result without repeating motion. |
| `capture_waveform(target, profile_id)` | Explicit `not_implemented` extension point. |
| `read_digital(group_id)` | Explicit `not_implemented` extension point. |
| `capture_logic(group_id, profile_id)` | Explicit `not_implemented` extension point. |

The AD3 extension tools never fabricate captures. Their future implementations use
ROS measurement interfaces; hardware access stays outside MCP. Digital input groups
will use fixed, configured wiring to minimize arm movement.

Each measurement returns status, nullable reading, units, requested/resolved target,
contact and stage checks, failure reason, request ID, mode, duration, and registration ID.
Only `status=ok` can carry a usable reading. An unverified contact isn't a zero reading.
An invalid result becomes `inconclusive: protocol_violation`.

Mode comes from the orchestrator's result. It is null for a boundary failure before
any trustworthy result arrives. Missing ROS never activates the simulator.

Optional `request_id` makes an exact retry return the saved result. Reusing it for a
different target is rejected. Without it, each call is a new request, so the host
mustn't silently retry an interrupted movement.

## Connect ROS on the bench computer

Use the ROS distribution and Python version supported by your teammates' setup.
The Windows simulator's Python installation isn't proof of rclpy compatibility.
No ROS installation or distribution was chosen here.

On the ROS machine, source its environment, then build the shared interfaces:

```bash
cd ros2_ws
colcon build --packages-select hardware_troubleshooter_interfaces
source install/setup.bash
cd ..
python3 -m venv --system-site-packages .venv-ros
.venv-ros/bin/python -m pip install -e .
```

Use the system Python belonging to that ROS installation. The environment needs
both rclpy and the generated interface package visible. Don't pip-install rclpy.

Start the synthetic ROS action server in one sourced terminal:

```bash
.venv-ros/bin/hardware-ros-stub --scenario success
```

In another sourced terminal, start MCP with `TH_BACKEND=ros`. Set
`TH_OUTER_TIMEOUT_S` longer than the orchestrator's agreed overall timeout, and
`TH_CANCEL_GRACE_S` to the agreed stop-confirmation window. Both default to unset
for ROS mode, so no goal is dispatched until configured. For the stub only,
the sample ROS integration test uses software test limits.

Default endpoints: action `/measure`, services `/registry/list` and `/rig/status`.
See `.env.example` for variable names (the file isn't loaded automatically).

For Codex on Windows and ROS inside WSL, launch MCP inside that sourced WSL
environment through a wrapper. [scripts/run_ros_mcp.sh](scripts/run_ros_mcp.sh)
is an example; set its required paths and timeout environment first.
Don't connect the Windows simulator environment to ROS by guessing Python paths.

## Verify

```powershell
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

Tests start a real MCP subprocess and exercise success, failed contact, unknown
target, moved board, stale calibration, malformed results, timeout, explicit abort,
MCP request cancellation, busy rejection, replay protection, and persistence after
uncertain execution. ROS future tests cover thread delivery and late goal acceptance.

Optional ROS smoke tests in `tests/test_ros_integration.py` run only when rclpy and
the generated interfaces are available. They launch the synthetic ROS server on
unique test endpoints and exercise MCP -> rclpy -> ROS stub. They never use the
production action name.

Unknown execution is latched in `data/mcp_operations.sqlite3`. No MCP reset tool
can dismiss it. The real orchestrator/operator recovery procedure remains a required
integration step; preserve the journal and inspect the physical bench. A process
restart mustn't be used as a motion reset.

## Next integration work

Simon supplies CV registration, board movement invalidation, and measured calibration
quality. This workspace owns the non-CV software: ROS orchestration, arm/contact
adapters, and AD3 acquisition integration remain to be implemented against their
hardware APIs. The registry needs the actual circuit's points, reference wiring,
and allowed operating conditions. Those values remain unset.

Share the [Simon handoff](docs/SIMON_HANDOFF.md) when coordinating the CV/ROS work.
