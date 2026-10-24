# Shared ROS contract, scaffold version 0.1

The source of truth is `ros2_ws/src/hardware_troubleshooter_interfaces`. Field additions
change the generated types, so rebuild and source the package on every ROS participant.
No message file contains physical pin positions, calibration matrices, or arm limits.

## Measurement action

`/measure`: `hardware_troubleshooter_interfaces/action/MeasureVoltage`.

Goal: target (registered test point, component-pin identity, or net), request_id.
Target validation permits 1 to 96 ASCII letters, digits, underscore, plus, minus,
period, slash, colon, hash, and at-sign. Whitespace and executable expressions are
rejected. Validation doesn't establish that the target exists; the registry does.

The orchestrator resolves aliases. Ambiguous nets need a configured preferred pad
or an explicit ambiguity failure; don't silently choose a point. It must implement
bench-wide exclusive ownership, including clients other than MCP.

The final result contains the fields in the action file. Valid enums:

- status: ok, inconclusive, failed.
- mode: simulated, live.
- contact: verified, not_verified, unknown.
- retraction_status: completed, failed, unknown, not_attempted.

An ok result requires a finite reading, V units, the matching request and target,
resolved test_point, nonempty registration_id, verified electrical contact,
move_completed, position_ok, board_unmoved, calibration_valid, retraction completed,
and stop_confirmed. failure_reason must be empty.

Every non-ok result requires NaN reading and a nonempty failure_reason.
MCP converts NaN to JSON null. A numeric value on a failed result is a protocol
violation, including zero with unverified contact. Raw diagnostic traces may exist
elsewhere but aren't usable circuit readings.

For failures before motion, stop_confirmed can be true when the orchestrator
independently knows the rig is stopped. retraction_status is not_attempted in that
case. Unknown stopping or failed/unknown retraction blocks later MCP dispatch.
The boolean checks must describe what actually happened, even on an inconclusive result.

Return the same request ID. Log deterministic recovery steps as JSON strings with
stage, failure, action, and source=deterministic or operator. There is no Jev service.
MCP preserves unparseable log entries as raw strings for debugging.

## Stages and progress

Feedback contains stage and attempt. Use descriptive stages such as PRECHECK,
RESOLVE, PLAN, APPROACH, VERIFY_POSITION, VERIFY_CONTACT, ACQUIRE, STOP, RETRACT.
MCP forwards changes as progress messages when a client supplies a progress token.
The stage count isn't a time estimate.

The orchestrator owns per-stage and overall deadlines. Before motion and before
acquisition, verify registration, reference state, limits, position, and contact as
appropriate. Never acquire usable evidence after a failed required check.

The orchestrator must enforce its deadlines even if the MCP process or network
disappears. ROS client disconnection mustn't leave an unbounded movement running.

## Cancellation and timeout

Accepting cancellation only means the request was received. Stop safely, retract
only if the physical state permits it, then return the final outcome and checks.
Use ROS canceled status with failed:aborted for a settled cancellation.

MCP's outer timeout must exceed the orchestrator's overall timeout. The outer
timeout and cancellation grace are unset by default in ROS mode. No action goal
is sent without them. The desktop host's tool timeout must exceed the outer
timeout plus cancellation grace and transport overhead.

After an outer timeout, MCP requests cancellation and waits for a final result
within its grace. It returns failed:timeout with no reading even if a late capture
arrives. If stopping can't be confirmed, execution is latched uncertain across
restarts. No automatic retry, homing, or reset is permitted.

## Boundary failures

MCP uses invalid_target, invalid_request_id, request_id_conflict, busy,
configuration_required:timeouts, orchestrator_unavailable, goal_rejected,
orchestrator_connection_lost, timeout, aborted, protocol_violation,
uncertain_motion_state, unknown_request, and no_active_operation.

Orchestrator failure reasons pass through unchanged when the wire result is valid.
Proposed shared reasons include unknown_target, ambiguous_target,
precheck_failed:<detail>, needs_registration, registration_invalidated,
uncertain_motion_state, contact_unverified, timeout, and aborted.

A pre-dispatch boundary failure has mode=null because no orchestrator result
establishes live/simulated mode. Missing or invalid mode in an action result is a
protocol violation; MCP doesn't infer it from its backend setting.

## Registry and rig status

`/registry/list` is ListTestPoints with no request fields. Each target has id, net,
and notes. Only ROS owns the real registry; MCP doesn't load a parallel YAML map.

`/rig/status` is GetRigStatus with no request fields. Return actual mode, board ID,
registration validity/age/ID, calibration validity, arm state, instrument connectivity,
busy, capabilities, and failure_reason. Unknown age is NaN and becomes null in MCP.
Advertise only implemented capabilities. MCP adds its own busy and execution-blocked
fields rather than overwriting ROS state.

## Registration, for Simon

`/registration/get` is GetRegistration: request board_id, response RegistrationState.
`/registration/events` publishes the same message when state changes or is invalidated.

header.stamp is the image observation timestamp, not time of service response.
header.frame_id is the camera optical frame. board_in_camera is the board pose
expressed in that frame, and board_frame identifies the registered board frame.
has_pose must be false when pose is unknown. An identity/default pose isn't a
calibrated pose.

status is valid, invalid, or unconfigured. calibration_valid, tracking_valid, and
board_unmoved are independent assertions. valid_until zero means unset. Presence
flags similarly distinguish unknown reprojection error and board height from zero.
The orchestrator must reject missing required calibration, stale observations,
out-of-limit quality, absent transforms, and invalid state. Acceptance limits remain
operator configuration based on actual measurements.

Registration is an input to motion planning, not permission to move by itself.
The motion adapter owns board-to-base calibration, probe tip offset, approach
height/path, positioning verification, and stop behavior. The contact verifier
owns electrical evidence. Neither is implemented by this message package.

## Software-only verification

`TH_BACKEND=stub` uses no ROS. `hardware-ros-stub` uses the real action/service types
but has no hardware. It always reports mode=simulated, including synthetic failure
scenarios. A scenario named unset_config simulates rejection of missing live setup;
it doesn't enable live mode.

The ROS integration tests use unique action/service names so they can't dispatch
to the production /measure endpoint. Physical verification is a separate milestone.
