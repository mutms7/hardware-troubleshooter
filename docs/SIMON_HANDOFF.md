# Simon: CV and ROS integration handoff

Repo: https://github.com/mutms7/hardware-troubleshooter/

This handoff follows simoncontext3.txt where it moves the execution stages into
a ROS orchestrator. The user's current decisions override its remaining Jev and
Claude-specific references: Codex desktop is the host, and Jev, RoboTwin, and
Cartesia are out of the MVP.

## Ownership

Simon owns CV/registration ROS nodes. William is building the MCP and non-CV
software here, including the integration scaffolding. The real measurement
orchestrator, instrument driver, and arm/contact adapters remain implementation work;
this scaffold doesn't assign those implementations to Simon.

MCP is a thin task boundary: validate target -> ROS action goal -> wait for result ->
validate evidence -> reply. It doesn't access hardware, instruments, TF, coordinates,
or IK. The ROS measurement orchestrator owns the complete execution sequence.

## Files to share first

- `ros2_ws/src/hardware_troubleshooter_interfaces/action/MeasureVoltage.action`
- `ros2_ws/src/hardware_troubleshooter_interfaces/msg/RegistrationState.msg`
- `ros2_ws/src/hardware_troubleshooter_interfaces/srv/GetRegistration.srv`
- `ros2_ws/src/hardware_troubleshooter_interfaces/srv/ListTestPoints.srv`
- `ros2_ws/src/hardware_troubleshooter_interfaces/srv/GetRigStatus.srv`
- `docs/ROS_CONTRACT.md`

Build and source this same interface package on both sides before connecting.
Pull the same commit from the repository before building so both sides use the
same interface definitions.

## ROS endpoints

| Endpoint | Kind | Owner |
| --- | --- | --- |
| `/measure` | MeasureVoltage action | Non-CV measurement orchestrator |
| `/registry/list` | ListTestPoints service | Registry/orchestrator |
| `/rig/status` | GetRigStatus service | Orchestrator |
| `/registration/get` | GetRegistration service | Simon's registration node |
| `/registration/events` | RegistrationState topic | Simon's registration node |

All types belong to `hardware_troubleshooter_interfaces`. The MCP endpoint names
are configurable. The ROS orchestrator, not MCP, consumes the registration endpoints.

## What the registration node provides

Return board identity, registration ID/revision, camera observation time, camera
frame, board frame, board pose in the camera frame when known, calibration revision,
tracking status, board movement status, and measured registration quality.

Unknown pose, height, and reprojection error are represented by false presence flags.
A zero valid_until means unset, never valid forever. Calibration limits and transforms
must be supplied and validated by the hardware team. The scaffold supplies none.

Publish invalid state promptly when the board moves, tracking is lost, or calibration
expires. Each new accepted registration gets an identifiable revision. Keep the latest
state available; we propose reliable, transient-local QoS with depth 1 for the event
topic, provided both nodes agree on it. Topic delivery supplements fresh checks,
rather than making old observations valid.

Your proposed board-edge fitting method is fine. No AprilTag implementation is
assumed. Board-relative pad identity comes from a PCB export or an explicitly
measured registry for the homemade circuit. Don't infer electrical identities from
an unlabeled photo. A board or fixture that can shift needs observable movement
detection; table-fixed visual features alone don't establish that the board stayed put.

The arm team needs a board-to-base transform, probe offset, and a known approach
height. A planar registration and target overlay don't establish those values or
prove that the probe touches the intended conductor.

## Measurement result agreement

The action matches your draft with these explicit additions:

- request_id echoed in the final result, for correlation.
- calibration_valid and registration_id, so registration evidence is explicit.
- retraction_status, separate from approach completion.
- stop_confirmed, which refers to final physical state, not cancellation acceptance.

Successful results need all independent checks to pass. Non-success readings are
NaN on ROS and null in MCP. A malformed or contradictory success is rejected.
No result from the orchestrator means MCP mode is null rather than invented.
There is no model recovery node or Jev fallback test anymore.

One measurement runs at a time. The orchestrator must reject overlapping goals
from all clients. If a cancel arrives, it owns guarded stop and any safe retraction,
then returns failed:aborted with truthful checks. A cancel acknowledgement alone
doesn't establish stopped hardware. See ROS_CONTRACT.md for failure reasons.

## Hardware assumptions still open

We have two arms but want minimal movement. Prefer a fixed reference connection and
one moving probe where the circuit permits it. Don't make both arms move for every
reading. The probe/contact setup isn't finalized. A rigid probe was proposed in your
notes; no force sensor, spring mechanism, descent limit, or arm accuracy is assumed.

AD3 voltage, oscilloscope, and digital inputs are in scope. Only voltage has a
complete task contract in this scaffold. The other MCP tools explicitly return
not_implemented. Fixed digital wiring should avoid arm movements for logic capture.

## First CV milestone

Provide registration and a target overlay for one known point. Move the board and
show that the state becomes invalid. Record the actual quality/error and which
calibrations exist. Connect that to a simulated orchestrator before live movement.

The Python package includes an explicit in-process simulator and an optional
hardware-ros-stub server. Neither accesses hardware. They let us settle MCP/ROS
contracts independently of CV progress.

Before live integration, agree on: the ROS distribution and matching Python, board
and frame names, measured registration acceptance limits, orchestrator timeout and
stop-confirmation window, and the reference/contact verification method.
