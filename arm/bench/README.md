# Arm bench tools (Maker Arm over USB-CAN on Windows)

Hand-teach poses on the Maker Arm with torque off, then replay them under power to turn a
multimeter knob and look at the display from above. Built on the Maker SDK's `Arm` class;
nothing here changes the SDK. This bench arm still has the stock jaws on motor 7, not the
rotating toolhead described in [../README.md](../README.md).

## Hardware state this assumes

| What | State (2026-10-25) |
| --- | --- |
| Link | candleLight USB-CAN adapter (gs_usb, VID 1D50 PID 606F), WinUSB driver, 1 Mbit/s |
| Motor protocol | RobStride private protocol. The motors had been in MIT mode (LeRobot); [switch_protocol_gsusb.py](switch_protocol_gsusb.py) switched them back. |
| Zeros | All seven motors zeroed and saved at the arm's resting pose, so home is all zeros. |

The factory profile's limits don't fit these zeros (J5 and J7 sit outside them). Soft
limits come from the range swept by hand while teaching instead. **Don't re-zero** (`zero`
via [sdk_cli_gsusb.py](sdk_cli_gsusb.py)): it invalidates every taught point.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install --no-deps "maker-arm @ git+https://github.com/makermods-robotics/maker-arm-sdk@b30d05a23d72e8c155a8e00f807aba8e8c705f68"
.venv\Scripts\pip install -r arm\bench\requirements.txt
```

Run every script with `python -X utf8` from this folder. The SDK opens its YAML profile
without an encoding, which fails under Windows' default code page.

## Tools

| Script | Torque | What it does |
| --- | --- | --- |
| [scan.py](scan.py) | off | Probe motor IDs; print position, temperature, fault. Start here. |
| [teach.py](teach.py) `work` | off | Stream positions while you move the arm by hand; record points; track the swept range. |
| [run.py](run.py) `work` | **on** | Move between taught points with the guards below. |
| [send.py](send.py) `work "<cmd>"` | – | Send one command to the running teach.py / run.py. |
| [step.py](step.py) `work <timeout> <cmd>` | – | Send a run.py command, wait for arrival, print error/torque per joint. |
| [sim_check.py](sim_check.py) | – | Run teach.py and run.py against simulated motors. Prints `SIM OK`. |
| [sdk_cli_gsusb.py](sdk_cli_gsusb.py) | varies | Run an SDK CLI module (`maker_arm.cli.zero`, ...) over this adapter. |
| [switch_protocol_gsusb.py](switch_protocol_gsusb.py) | off | Run the SDK's MIT ↔ private protocol switch over this adapter. Power-cycle after. |

Only one process can hold the adapter. teach.py and run.py run in the background, read
commands from `work/cmd/` and publish `work/state.json` and `work/log.txt` (all ignored by
git). Taught points live in [work/teach.yaml](work/teach.yaml), which is tracked.

## Teaching

Run `python -X utf8 teach.py work` with the arm powered, then move it by hand. Torque is off,
so support it.

| Command | Saves |
| --- | --- |
| `record <name>` | the whole pose |
| `record <name> J7` | one joint only (a partial point, e.g. `jaw_open`) |
| `delete <name>` / `reset` | forget a point / forget the swept range |
| `quit` | write teach.yaml and exit |

The swept range becomes run.py's soft limits, so sweep everything the moves will pass
through. Teach full poses for each knob setting. Turning the knob by hand also moves J1 and
J5, so a wrist-only point twists the knob off-axis under power.

## Running

Start with the arm at home: `python -X utf8 run.py work`. It enables torque and holds where
the arm is.

| Command | Does |
| --- | --- |
| `turn <from> <to>` | `above` → hover → `<from>` with jaws open → grip → twist to `<to>` while gripping → open → hover → `above` |
| `approach <pose>` | `above` → hover → `<pose>` with jaws open |
| `goto <point>` / `seq <a> <b> ...` | one point / several in order |
| `jog J<n> <rad>` | nudge one joint, at most 0.1 rad |
| `save <name> [J7]` | store the current goal in teach.yaml |
| `home` | open jaws → hover → `above` → home |
| `hold` | freeze at the measured pose |
| `release` | drop torque. Refused away from home unless you send `release force`. |
| `quit` | `home`, then release |

Current points for the meter at its 2026-10-25 position: `knob_closed` (off), `knob_resistance`,
`knob` (off, jaws open), `above` (camera pose and climb-out point), `jaw_open`, `jaw_closed`.
For example, `turn knob_closed knob_resistance` switches to resistance and `turn knob_resistance knob_closed`
switches back, about 10 s each.

### Why the moves look like this

- **Hover.** "Hover" is `above`'s arm pose with the wrist already at the knob's angle and the
  jaws open. The wrist turns and the jaws open up there, so the descent moves the arm only.
  Each joint is rate-limited on its own (0.5 rad/s), so a direct move would leave the slow
  wrist still turning at the knob.
- **Waypoints wait.** A queued step starts only after J1–J6 arrive (0.03 rad) and the jaw
  has finished. Closed jaws stall short on the knob by design, so for the jaw "finished"
  means its command arrived and it stopped moving.
- **Sag correction, open air only.** The SDK's PD loop has no gravity feedforward. At
  `above`, J2 and J3 sit 0.02–0.04 rad low under 3.5 Nm. An outer integral, capped at
  0.06 rad, nudges the commands until the arm arrives. At the meter it's off: a joint that
  is blocked looks like sag, and integrating it pushes into the obstacle.
- **Contact guard.** At the meter, J1 or J4–J6 over 2.5 Nm, or J2/J3 over 6 Nm, for 100 ms
  freezes the arm at its measured pose and clears the queue. It doesn't retreat on its own:
  with the jaws gripping, pulling away drags the knob.
- **Grip.** The jaw gain is the factory 20 Nm/rad, so grip force is 20 × how far the jaw stops
  short of its target. Tighten under power with `jog J7 +0.02` and `save jaw_closed J7`.

## Known issues

- **Fixture the meter.** On 2026-10-25 the meter had moved. The knob pose pushed into its
  body, the sag correction (then active at the meter) integrated against it, J1 tripped
  overcurrent and the e-stop was used. Taught poses only hold if the meter stays put.
- After a power cycle J6 can read one full turn off (6.25 instead of −0.03). The SDK takes
  that turn back out on connect while the reading is within 0.35 rad of the soft limits.
- The home-route fix (open-air vs at-the-meter decided by the arm pose, not the point's name)
  has passed sim_check.py but hasn't run on the arm yet.
