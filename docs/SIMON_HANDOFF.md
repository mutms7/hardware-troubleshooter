# Simon: CV integration handoff

**Current localization update (2026-10-24):** runtime board pose now comes from two
settled wrist-camera views and measured camera motion, followed by joint pose fitting
and a triangulation consistency check. Homing is repeated for each requested pad.
The fixed depth-capable camera is optional RGB overview; its depth and fixed-camera
PnP aren't used. The fixed-camera registration description below is historical and
remains only as the explicit legacy mode. See [vision/README.md](../vision/README.md)
and its YAML for the current API, reference features and required hand-eye calibration.

Repo: https://github.com/mutms7/hardware-troubleshooter/

This handoff follows simoncontext3.txt where it moves the execution stages into
a measurement orchestrator. ROS is no longer part of the stack. The user's current
decisions override its remaining Jev and Claude-specific references: Codex desktop
is the host, and Jev, RoboTwin, and Cartesia are out of the MVP.

## Ownership

Simon owns CV/registration, in `registration/`. William is building the MCP and non-CV
software here, including the integration scaffolding. The real measurement
orchestrator, instrument driver, and arm/contact adapters remain implementation work;
this scaffold doesn't assign those implementations to Simon.

MCP is a thin task boundary: validate target -> orchestrator goal -> wait for result ->
validate evidence -> reply. It doesn't access hardware, instruments, frame transforms,
coordinates, or IK. The measurement orchestrator owns the complete execution sequence.

## Files to share first

- `docs/MEASUREMENT_CONTRACT.md`
- `registration/README.md`
- `src/troubleshooter/contracts.py` (`WireResult`)

The orchestrator transport isn't chosen yet. Agree on it before connecting.

## What registration provides

Return board identity, registration ID/revision, camera observation time, camera
frame, board frame, board pose in the camera frame when known, calibration revision,
tracking status, board movement status, and measured registration quality.

Unknown pose and fit residual are represented by a false flag or None. An unset
valid_until is never valid forever. Calibration limits and transforms must be
supplied and validated by the hardware team. The scaffold supplies none.

Report invalid state promptly when the board moves, tracking is lost, or calibration
expires. Each new accepted registration gets an identifiable revision. Keep the latest
state available. Change events supplement fresh checks, rather than making old
observations valid.

Your proposed board-edge fitting method is fine. No AprilTag implementation is
assumed. Board-relative pad identity comes from a PCB export or an explicitly
measured registry for the homemade circuit. Don't infer electrical identities from
an unlabeled photo. A board or fixture that can shift needs observable movement
detection; table-fixed visual features alone don't establish that the board stayed put.

The arm team needs a board-to-base transform, probe offset, and a known approach
height. A planar registration and target overlay don't establish those values or
prove that the probe touches the intended conductor.

## Measurement result agreement

The result matches your draft with these explicit additions:

- request_id echoed in the final result, for correlation.
- calibration_valid and registration_id, so registration evidence is explicit.
- retraction_status, separate from approach completion.
- stop_confirmed, which refers to final physical state, not cancellation acceptance.

Successful results need all independent checks to pass. Non-success readings are
NaN from the orchestrator and null in MCP. A malformed or contradictory success is
rejected. No result from the orchestrator means MCP mode is null rather than invented.
There is no model recovery node or Jev fallback test anymore.

One measurement runs at a time. The orchestrator must reject overlapping goals
from all clients. If a cancel arrives, it owns guarded stop and any safe retraction,
then returns failed:aborted with truthful checks. A cancel acknowledgement alone
doesn't establish stopped hardware. See MEASUREMENT_CONTRACT.md for failure reasons.

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

The Python package includes an explicit in-process simulator. It doesn't access
hardware, and lets us settle the MCP contract independently of CV progress.

Before live integration, agree on: the orchestrator transport, board and frame names,
measured registration acceptance limits, orchestrator timeout and stop-confirmation
window, and the reference/contact verification method.
