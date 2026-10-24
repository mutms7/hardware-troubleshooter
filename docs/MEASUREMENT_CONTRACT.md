# Measurement contract, scaffold version 0.2

The source of truth for the result format is `WireResult` in
`src/troubleshooter/contracts.py`. No transport is chosen yet; a future backend
implements the `Backend` protocol in `src/troubleshooter/backend.py`. No field
carries physical pin positions, calibration matrices, or arm limits.

## Measurement task

Goal: target (registered test point, component-pin identity, or net), request_id.
Target validation permits 1 to 96 ASCII letters, digits, underscore, plus, minus,
period, slash, colon, hash, and at-sign. Whitespace and executable expressions are
rejected. Validation doesn't establish that the target exists; the registry does.

The orchestrator resolves aliases. Ambiguous nets need a configured preferred pad
or an explicit ambiguity failure; don't silently choose a point. It must implement
bench-wide exclusive ownership, including clients other than MCP.

The final result contains the `WireResult` fields. Valid enums:

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
stage, failure, action, and source=deterministic or operator.
MCP preserves unparseable log entries as raw strings for debugging.

## Stages and progress

Feedback contains stage and attempt. Use descriptive stages such as PRECHECK,
RESOLVE, PLAN, APPROACH, VERIFY_POSITION, VERIFY_CONTACT, ACQUIRE, STOP, RETRACT.
MCP forwards changes as progress messages when a client supplies a progress token.
The stage count isn't a time estimate.

The orchestrator owns per-stage and overall deadlines. Before motion and before
acquisition, verify registration, reference state, limits, position, and contact as
appropriate. Never acquire usable evidence after a failed required check.

The orchestrator must enforce its deadlines even if the MCP process or connection
disappears. Client disconnection mustn't leave an unbounded movement running.

## Cancellation and timeout

Accepting cancellation only means the request was received. Stop safely, retract
only if the physical state permits it, then return the final outcome and checks,
with failed:aborted for a settled cancellation.

MCP's outer timeout must exceed the orchestrator's overall timeout. The outer
timeout and cancellation grace are unset by default outside stub mode. No goal is
sent without them. The desktop host's tool timeout must exceed the outer timeout
plus cancellation grace and transport overhead.

After an outer timeout, MCP requests cancellation and waits for a final result
within its grace. It returns failed:timeout with no reading even if a late capture
arrives. If stopping can't be confirmed, execution is latched uncertain across
restarts. No automatic retry, homing, or reset is permitted.

## Boundary failures

MCP uses invalid_target, invalid_request_id, request_id_conflict, busy,
configuration_required:timeouts, orchestrator_unavailable, goal_rejected,
orchestrator_connection_lost, timeout, aborted, protocol_violation,
uncertain_motion_state, unknown_request, and no_active_operation.

Orchestrator failure reasons pass through unchanged when the result is valid.
Proposed shared reasons include unknown_target, ambiguous_target,
precheck_failed:<detail>, needs_registration, registration_invalidated,
uncertain_motion_state, contact_unverified, timeout, and aborted.

A pre-dispatch boundary failure has mode=null because no orchestrator result
establishes live/simulated mode. Missing or invalid mode in a result is a
protocol violation; MCP doesn't infer it from its backend setting.

## Registry and rig status

The registry returns status, mode, and targets, each with id, net, and notes.
Only the orchestrator owns the real registry; MCP doesn't load a parallel YAML map.

Rig status returns actual mode, board ID, registration validity/age/ID, calibration
validity, arm state, instrument connectivity, busy, capabilities, and failure_reason.
Unknown age is NaN and becomes null in MCP. Advertise only implemented capabilities.
MCP adds its own busy and execution-blocked fields rather than overwriting
orchestrator state.

## Registration

The orchestrator consumes registration state from Simon's engine; see
[registration/README.md](../registration/README.md) for its fields and semantics.
It must reject missing required calibration, stale observations, out-of-limit
quality, a missing board transform, and invalid state. Acceptance limits remain
operator configuration based on actual measurements.

Registration is an input to motion planning, not permission to move by itself.
The motion adapter owns board-to-base calibration, probe tip offset, approach
height/path, positioning verification, and stop behavior. The contact verifier
owns electrical evidence. Neither is implemented here.

## Software-only verification

`TH_BACKEND=stub` runs an in-process scripted simulation with no hardware. It always
reports mode=simulated, including synthetic failure scenarios. A scenario named
unset_config simulates rejection of missing live setup; it doesn't enable live mode.
Physical verification is a separate milestone.
