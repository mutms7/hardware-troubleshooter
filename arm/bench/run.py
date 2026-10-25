"""Powered runner: move between the points recorded by teach.py.

Usage: python -X utf8 run.py <workdir> [--max-velocity 0.5]
Soft limits are the hand-swept envelope from <workdir>/teach.yaml (plus the zero pose);
gains come from the factory Maker Arm v1 profile.
Commands (python send.py <workdir> "<command>"):
  goto <name>   move to a recorded point (joint-space, rate-limited) and hold; a partial
                point (e.g. a J6 knob angle or J7 jaw angle) moves only its joints
  seq <a> <b> ...   visit points in order, each after the previous arrives (arm joints only)
  home          open the jaws, climb to hover, "above", then the zero pose, and hold
  approach <pose>    above -> hover (wrist at <pose>'s angle, jaws open) -> <pose> with jaws open
  turn <from> <to>   above -> hover -> <from> jaws open -> grip -> <to> gripping -> open -> hover -> above
                (from/to are full knob poses, e.g. knob_closed knob_resistance; jaws use
                the taught jaw_open / jaw_closed angles throughout)
  jog J<n> <rad>   nudge one joint's goal (|rad| <= 0.1, clamped to the envelope)
  save <name> [J7 ...]   store the current goal as a point in teach.yaml (only those joints if given)
  hold          freeze at the current measured pose
  release       drop torque - refused unless the arm is at home (use "release force" to override)
  quit          go home, release, exit
"""

import argparse
import math
import time
from pathlib import Path

import yaml

from bench_common import Inbox, fmt, log, write_state
from gsusb_backend import GsUsbBackend
from maker_arm.arm import Arm, ArmState
from maker_arm.config import ArmConfig, JointConfig
from maker_arm.profiles import DEFAULT_ARM_CONFIG

HOME_TOL = 0.08    # rad, every joint within this of zero counts as "at home"
ARRIVE_TOL = 0.03  # rad, reported "arrived" once every joint is within this of its target
JOG_MAX = 0.1      # rad, largest single jog
JAW_SETTLED_VEL = 0.05  # rad/s, jaw counts as finished below this once its command is done
# Gravity-sag correction: the SDK's PD loop has no gravity feedforward, so loaded joints sit
# short of their goal (J3 ~0.035 rad at "above"). An outer integral nudges arm-joint commands
# (not the jaw, which stalls on the knob by design) until the measured pose matches.
SAG_GAIN = 1.5     # 1/s
SAG_MAX = 0.06     # rad, largest correction per joint
SAG_FREEZE = 0.1   # rad, don't integrate while a joint is this far off (moving or blocked)
LOOP_DT = 0.05     # s, runner loop period
# Contact guard, active only at the meter (goal is a knob pose). Free moves measured <= 0.7 Nm
# plus ~1-2 Nm of damping while moving on J1, J4-J6; J2/J3 carry up to 3.7 Nm of gravity.
# J1 (base) tripped overcurrent pushing sideways into a moved meter on 2026-10-25.
TAU_GUARD = [2.5, 6.0, 6.0, 2.5, 2.5, 2.5]  # Nm, J1..J6
GUARD_TICKS = 2    # consecutive loop ticks over the limit (~100 ms)
FREE_SPACE = {"current", "held", "guard-hold"}  # holds at a measured pose; others decided by pose

ap = argparse.ArgumentParser()
ap.add_argument("workdir")
ap.add_argument("--max-velocity", type=float, default=0.5)
a = ap.parse_args()
workdir = Path(a.workdir).resolve()

teach = yaml.safe_load((workdir / "teach.yaml").read_text(encoding="utf-8"))
# Full points are 7-lists; partial points ({"J6": x}) only move the joints they name.
points = {k: (list(v) if isinstance(v, list) else {int(j[1:]) - 1: x for j, x in v.items()})
          for k, v in teach["points"].items()}
points["home"] = [0.0] * 7
env_lo = [min(x, 0.0) for x in teach["envelope"]["min"]]
env_hi = [max(x, 0.0) for x in teach["envelope"]["max"]]

factory = ArmConfig.from_yaml(str(DEFAULT_ARM_CONFIG))
joints = []
for f, lo, hi in zip(factory.joints, env_lo, env_hi):
    if hi - lo < 0.02:  # joint never moved while teaching: pin it with a tiny band
        lo, hi = lo - 0.01, hi + 0.01
    joints.append(JointConfig(motor_id=f.motor_id, direction=1, offset=0.0, lo=round(lo, 4),
                              hi=round(hi, 4), kp=f.kp, kd=f.kd, model=f.model))
# Targets stay inside the swept envelope (`joints`); the SDK's command limits are widened by
# SAG_MAX on the arm joints so the sag correction can still act on a point at the envelope edge.
sdk_joints = [JointConfig(motor_id=j.motor_id, direction=1, offset=0.0,
                          lo=j.lo - (SAG_MAX if i < 6 else 0.0), hi=j.hi + (SAG_MAX if i < 6 else 0.0),
                          kp=j.kp, kd=j.kd, model=j.model) for i, j in enumerate(joints)]
cfg = ArmConfig(control_rate_hz=factory.control_rate_hz, max_velocity=a.max_velocity,
                limit_margin=0.0, feedback_timeout=factory.feedback_timeout,
                motor_can_timeout_ms=factory.motor_can_timeout_ms, joints=sdk_joints)
cfg._validate()
for name, p in points.items():
    items = enumerate(p) if isinstance(p, list) else p.items()
    bad = [i + 1 for i, x in items if not joints[i].lo <= x <= joints[i].hi]
    if bad:
        raise SystemExit(f"point {name} outside the envelope on joints {bad}")

arm = Arm(cfg, GsUsbBackend())
inbox = Inbox(workdir)
target_name, target = None, None
queue: list[str] = []  # waypoints still to visit, in order
bias = [0.0] * 7       # sag correction added to the arm-joint commands; bias[6] (jaw) stays 0
guard_count = 0        # consecutive loop ticks with an arm joint over TAU_GUARD


def at(q, t, tol):
    return all(abs(x - y) <= tol for x, y in zip(q, t))


def goal(t):
    """Command for target t: the target clamped to the envelope, plus the sag correction."""
    return [min(j.hi, max(j.lo, x)) + b for x, b, j in zip(t, bias, joints)]


def in_contact_zone():
    """True when the goal is at the meter (a knob pose), not in open air above it or at home.

    Decided by where the arm joints are going, not by the point's name: a jaw-only move
    (jaw_open) at "above" is still open air. Hover points share J1-J5 with "above".
    """
    if target is None or (target_name or "").split("+")[0] in FREE_SPACE:
        return False
    if "above" in points and at(target[:5], points["above"][:5], 1e-3):
        return False
    return not at(target[:6], points["home"][:6], 1e-3)


def guard_tripped(tau):
    """Contact guard: any arm joint over its torque limit for GUARD_TICKS loop ticks in a row."""
    global guard_count
    over = [i + 1 for i in range(6) if abs(tau[i]) > TAU_GUARD[i]]
    guard_count = guard_count + 1 if over else 0
    return over if guard_count >= GUARD_TICKS else None


def correct_sag(q):
    """One integral step per arm joint, only once the rate limiter has delivered the goal.

    Open air only: at the meter a stalled joint looks like sag, and integrating it would
    push into whatever is blocking it, so the correction is dropped there.
    """
    if in_contact_zone():
        if any(bias):
            bias[:] = [0.0] * 7
            arm.set_joint_targets(goal(target))
        return
    cmd = arm.get_commanded_positions()
    g = goal(target)
    for i in range(6):
        err = target[i] - q[i]
        if abs(cmd[i] - g[i]) < 1e-3 and abs(err) < SAG_FREEZE:
            b = bias[i] + SAG_GAIN * LOOP_DT * err
            bias[i] = max(-SAG_MAX, min(SAG_MAX, b))
    arm.set_joint_targets(goal(target))


def go(name):
    global target_name, target
    p = points[name]
    if isinstance(p, dict):  # partial: keep the other joints at their current goal
        base = list(target) if target else arm.get_commanded_positions()
        for i, x in p.items():
            base[i] = x
        p = base
    if arm.state is ArmState.ENABLED and arm.set_joint_targets(goal(p)):
        target_name, target = name, list(p)
        log(workdir, f"goto {name}: {fmt(p)}")
    else:
        log(workdir, f"goto {name}: REJECTED (state {arm.state.name})")


def hover(j6, label):
    """"above" with the wrist already at the knob angle and the jaws open, registered as above@<label>.

    Turning the wrist and opening the jaws happen up here, so the descent to the knob (and the
    climb back) moves the arm only - otherwise the slow wrist would still be turning at the knob.
    """
    q = list(points["above"])
    q[5] = j6
    if "jaw_open" in points:
        q[6] = points["jaw_open"][6]
    points[f"above@{label}"] = q
    return f"above@{label}"


def go_home():
    """Open the jaws, climb to hover, turn the wrist back at "above", then home."""
    q = arm.get_joint_positions()
    if at(q[:6], points["home"][:6], HOME_TOL):
        route = ["home"]
    elif "above" not in points:
        route = [n for n in ("jaw_open",) if n in points] + ["home"]
    else:
        j6 = (target or q)[5]
        route = [n for n in ("jaw_open",) if n in points] + [hover(j6, "here"), "above", "home"]
    queue[:] = route[1:]
    go(route[0])


def approach(name):
    """above -> hover at <name>'s wrist angle with jaws open -> descend to <name> with jaws open."""
    if "above" not in points or not isinstance(points.get(name), list):
        log(workdir, f"approach REFUSED: need 'above' and a full pose {name!r}")
        return
    route = ["above", hover(points[name][5], name), with_jaw(name, "jaw_open")]
    queue[:] = route[1:]
    go(route[0])


def with_jaw(name, jaw):
    """Full point <name> with J7 replaced by the taught jaw angle; registered as <name>@<jaw>."""
    q = list(points[name])
    q[6] = points[jaw][6]
    points[f"{name}@{jaw}"] = q
    return f"{name}@{jaw}"


def turn(frm, to):
    """above -> hover -> frm (open) -> grip -> to (gripping) -> release -> hover -> above."""
    missing = [n for n in ("above", "jaw_open", "jaw_closed") if n not in points]
    bad = [n for n in (frm, to) if not isinstance(points.get(n), list)]
    if missing or bad:
        log(workdir, f"turn REFUSED: missing points {missing}, not full poses {bad}")
        return
    route = ["above", hover(points[frm][5], frm), with_jaw(frm, "jaw_open"), with_jaw(frm, "jaw_closed"),
             with_jaw(to, "jaw_closed"), with_jaw(to, "jaw_open"), hover(points[to][5], to), "above"]
    queue[:] = route[1:]
    go(route[0])


def joint_index(s):
    i = int(s.upper().lstrip("J")) - 1
    if not 0 <= i < 7:
        raise ValueError(s)
    return i


def jog(joint, delta):
    global target_name, target
    try:
        i, d = joint_index(joint), float(delta)
    except ValueError:
        log(workdir, f"jog: bad arguments {joint!r} {delta!r} (use: jog J7 0.02)")
        return
    if abs(d) > JOG_MAX:
        log(workdir, f"jog REFUSED: |{d}| > {JOG_MAX} rad per jog")
        return
    base = list(target) if target else arm.get_commanded_positions()
    base[i] = min(joints[i].hi, max(joints[i].lo, base[i] + d))
    if arm.state is ArmState.ENABLED and arm.set_joint_targets(goal(base)):
        target_name, target = f"{target_name}+jog", base
        log(workdir, f"jog J{i + 1} {d:+.3f} -> {base[i]:+.4f}")
    else:
        log(workdir, f"jog REJECTED (state {arm.state.name})")


def save(name, joint_args):
    """save <name> [J7 ...]: store the current goal (not the measured pose) as a point in teach.yaml."""
    if target is None:
        log(workdir, "save: no current goal")
        return
    try:
        idx = [joint_index(j) for j in joint_args]
    except ValueError:
        log(workdir, f"save: bad joint list {joint_args}")
        return
    point = {i: round(target[i], 4) for i in idx} if idx else [round(x, 4) for x in target]
    points[name] = point
    path = workdir / "teach.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["points"][name] = {f"J{i + 1}": x for i, x in point.items()} if idx else point
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    log(workdir, f"saved {name}: {data['points'][name]}")


arm.connect()
log(workdir, "run: connected. envelope " + ", ".join(f"J{j.motor_id}[{j.lo:+.3f},{j.hi:+.3f}]" for j in joints)
    + f" (commands may exceed it by <= {SAG_MAX} rad of sag correction on J1-J6)")
try:
    arm.enable()
    target_name, target = "current", arm.get_joint_positions()
    log(workdir, "run: ENABLED, holding current pose")
    running, quitting, last_print, fault_logged = True, False, 0.0, False
    while running:
        q = arm.get_joint_positions()
        if arm.state is ArmState.FAULT and not fault_logged:
            # The SDK locks and holds what it can; stay up so the operator decides when to release.
            log(workdir, f"FAULT (holding what it can, goto disabled): {arm.fault_reason}")
            fault_logged = True
        for cmd in inbox.poll():
            parts = cmd.split()
            if parts[0] == "goto" and len(parts) == 2 and parts[1] in points:
                queue.clear()
                go(parts[1])
            elif parts[0] == "seq" and len(parts) >= 2 and all(n in points for n in parts[1:]):
                queue[:] = parts[2:]
                go(parts[1])
            elif parts[0] == "home":
                go_home()
            elif parts[0] == "turn" and len(parts) == 3:
                turn(parts[1], parts[2])
            elif parts[0] == "approach" and len(parts) == 2:
                approach(parts[1])
            elif parts[0] == "jog" and len(parts) == 3:
                queue.clear()
                jog(parts[1], parts[2])
            elif parts[0] == "save" and len(parts) >= 2:
                save(parts[1], parts[2:])
            elif parts[0] == "hold":
                queue.clear()
                arm.hold_current_position()
                bias[:] = [0.0] * 7
                target_name, target = "held", list(q)
                log(workdir, f"hold at {fmt(q)}")
            elif parts[0] == "release":
                if at(q, points["home"], HOME_TOL) or parts[1:] == ["force"]:
                    running = False
                else:
                    log(workdir, "release REFUSED: not at home (send 'home' first, or 'release force')")
            elif parts[0] == "quit":
                quitting = True
                go_home()
            else:
                log(workdir, f"unknown command: {cmd!r} (points: {sorted(points)})")
        if target is not None and arm.state is ArmState.ENABLED:
            tau = arm.get_joint_torques()
            over = guard_tripped(tau) if in_contact_zone() else None
            if over:
                # Freeze at the measured pose: the PD spring then stops pushing. No automatic
                # retreat - with the jaws gripping, pulling away would drag the knob.
                queue.clear()
                arm.hold_current_position()
                bias[:] = [0.0] * 7
                log(workdir, f"GUARD: J{over} torque {[round(t, 2) for t in tau]} over {TAU_GUARD} at "
                    f"{target_name}; holding at {fmt(q)}")
                target_name, target = "guard-hold", list(q)
                guard_count = 0
            correct_sag(q)
        # Advance a waypoint once the arm joints arrive and the jaw has finished moving: its
        # command reached the goal and it has stopped (closed jaws stall short on the knob).
        if queue and target is not None and at(q[:6], target[:6], ARRIVE_TOL):
            jaw_cmd = arm.get_commanded_positions()[6]
            jaw_vel = arm.get_joint_velocities()[6]
            if abs(jaw_cmd - target[6]) < 1e-3 and abs(jaw_vel) < JAW_SETTLED_VEL:
                go(queue.pop(0))
        # After the queue step, so a just-dispatched waypoint never reports the previous arrival.
        arrived = target is not None and at(q, target, ARRIVE_TOL)
        if quitting and not queue and at(q, points["home"], HOME_TOL):
            running = False
        write_state(workdir, {
            "mode": "run", "t": time.time(), "state": arm.state.name, "target": target_name,
            "arrived": arrived, "queue": list(queue), "bias": [round(x, 4) for x in bias],
            "q": [round(x, 4) for x in q],
            "commanded": [round(x, 4) for x in arm.get_commanded_positions()],
            "error": [round(x - t, 4) for x, t in zip(q, target)] if target else None,
            "torque": [round(x, 3) for x in arm.get_joint_torques()],
            "temps": arm.get_temperatures(), "faults": arm.get_faults(),
            "points": points})
        if time.monotonic() - last_print > 2.0:
            print(f"[{arm.state.name} -> {target_name}{' arrived' if arrived else ''}] {fmt(q)}", flush=True)
            last_print = time.monotonic()
        time.sleep(LOOP_DT)
finally:
    was = arm.state.name
    arm.disable()
    arm.disconnect()
    log(workdir, f"run: torque released (was {was}), exiting")
