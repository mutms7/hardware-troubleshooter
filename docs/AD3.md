# AD3 instrument tools

The MCP server drives one Analog Discovery 3 through the WaveForms SDK on the Windows host,
and each arm's tool turret through that arm's `sim_server.py`. The MCP doesn't position the
probe or verify contact: it captures wherever the probe already is, and every result says
`contact_verified: false, placement: unmanaged`.

## Hardware

Two arms. Each end effector has a three-head turret:

| Head | Wired to | Notes |
| --- | --- | --- |
| `scope` | AD3 scope input: arm A on CH1, arm B on CH2 | ±25 V input. Tie that channel's negative input (1− / 2−) to ground. |
| `logic` | AD3 DIO pin: arm A on DIO 0, arm B on DIO 1 | 3.3 V LVCMOS input, 5 V tolerant. Not for 1.8 V logic or anything above 5 V. |
| `dmm` | Separate multimeter | Not the AD3 and not driven by these tools yet. `select_tool` can still turn to it. |

The two scope probes use both AD3 scope inputs, so the logic probes go on DIO pins. Two-line
protocols use both arms' logic probes, one line each.

Ground is one clip from the AD3's ground to the board's ground, made by hand. The arm never
probes ground.

### Ground precautions

The AD3 ground is the PC's USB ground. When the board and the PC are both on earthed outlets of
the same building wiring, the clip makes a second ground path. That usually only adds mains hum
and a few mV of offset. It becomes dangerous when:

- the board isn't low voltage and isolated (an offline supply's primary side, transformerless
  circuits). Never clip to these.
- a supply ties one output terminal to earth and that terminal isn't the board's ground.
- the outlets are far apart or on different circuits.

Before every new setup:

1. Measure AC and DC volts between the board ground and the AD3 ground (or the USB shield).
   Both should read near 0 V before you clip.
2. Clip ground first and remove it last.
3. Tie the scope negative inputs (1−, 2−) to ground.
4. Keep idle tips away from the board: only the selected head should be able to reach it.
   `sim_server.py` refuses to turn the turret below 20 mm (`TOOL_CHANGE_CLEARANCE_MM`).
5. Remember the other ground paths: a bench DMM's COM may be earthed, and so is the servo
   bus's USB adapter.

## Configure

1. Install WaveForms; it provides `dwf.dll`. Close the WaveForms app while the MCP runs,
   because only one process can own the AD3.
2. Check the AD3 with no arms involved. This only reads, and never enables an output:

   ```powershell
   .\.venv\Scripts\python.exe -m troubleshooter.cli ad3-check
   ```

   It prints the device serial, min/max/frequency on both scope inputs, and the level and edge
   count of every DIO pin.
3. Copy [config/rig.example.json](../config/rig.example.json), fix the arm URLs, channels,
   and pins to match the wiring, and set `TH_RIG_CONFIG` to it. Nothing has a default: with
   `TH_RIG_CONFIG` unset the tools return `configuration_required:rig`.
4. In `sim_server.py`, set `TOOL_HEADS` to the head in each turret slot, then start one server
   per arm:

   ```powershell
   python sim_server.py --hardware COM10 --port 8765 --arm A
   python sim_server.py --hardware COM11 --port 8766 --arm B
   ```

   `--arm` turns on collision checks between the two arms; see [WORKCELL.md](WORKCELL.md).

Optional: `TH_AD3_SERIAL` picks a device when several are connected, and `TH_CAPTURES_DIR`
(default `data/captures`) holds one JSON file per capture with the raw samples.

### Profiles and logic types

The model picks these by name and never sends raw instrument settings. Add entries to the rig
file for other signals.

- `scope_profiles`: range, offset, sample rate, sample count, optional trigger. With
  `require_trigger`, an auto-triggered capture is `inconclusive: no_trigger`.
- `logic_types`: `uart` (baud, data bits, parity, stop bits), `pwm`, `i2c`, or `spi`
  (cpol, cpha, bit order), each with a sample rate and sample count.

## Tools

| Tool | Behaviour |
| --- | --- |
| `get_instrument_status()` | AD3 connection and serial, each arm's reachability and current head, the wiring. |
| `list_capture_profiles()` | Configured scope profiles and logic types, with each logic type's signal roles. |
| `select_tool(arm, head)` | Turns one turret to `scope`, `logic`, or `dmm`. The arm itself doesn't move. |
| `capture_waveform(arm, profile_id)` | Selects the scope head and captures its AD3 channel. Returns Vpp, min/max, mean, RMS, frequency, and duty cycle. |
| `capture_logic(logic_type, lines)` | Selects the logic head on each arm in `lines` and captures their DIO pins, then decodes. |
| `read_tof(arm)` | The arm's ToF range and, once zeroed, the probe tip's height above the board. No movement. See [Sensor head](#sensor-head). |
| `read_thermal(arm)` | The arm's AMG8833 8x8 thermal grid in °C, with min/max/mean and the hottest cell. No movement. |

`lines` maps each role of the logic type to an arm:

| Protocol | Roles | Example `lines` |
| --- | --- | --- |
| UART | `rx` | `{"rx": "A"}` |
| PWM | `signal` | `{"signal": "B"}` |
| I2C | `scl`, `sda` | `{"scl": "A", "sda": "B"}` |
| SPI | `sck`, `data` | `{"sck": "A", "data": "B"}` |

SPI is partial: SCK plus one data line (MOSI or MISO), no chip select. Its clock measurements
are reliable. Its bytes assume each clock burst starts on a word boundary, so treat them as
tentative.

### Outcomes

Only `status=ok` carries `measurements`. Other outcomes carry `diagnostics`, which aren't
evidence.

| Status | Reasons |
| --- | --- |
| `inconclusive` | `clipped` (signal hit the input range), `no_trigger`, `no_activity` (no edges on any line), `decode_failed` |
| `failed` | `configuration_required:rig`, `configuration_invalid:rig`, `unknown_arm`, `unknown_profile`, `unknown_logic_type`, `invalid_lines`, `busy`, `arm_unavailable`, `tool_change_failed:<arm message>`, `tool_not_confirmed`, `instrument_unavailable`, `instrument_error` |

`mode` is `live` for the real AD3 and `simulated` for the fake. It's null when nothing was
captured. Each arm's entry in `arms` reports `arm_server`: `hardware`, or `sim` when that arm
server runs without `--hardware`, in which case no turret actually turned.

A flat waveform or `no_activity` can mean the probe isn't touching anything.

## Sensor head

An ESP32-S3 carries a VL53L0X time-of-flight sensor and an AMG8833 thermal camera. Its
firmware, wiring and host driver are in
[firmware/amg8833_usb](../firmware/amg8833_usb/README.md). Only one process can open its
port, so the arm server owns it:

```powershell
python sim_server.py --hardware COM10 --port 8765 --arm A --sensors COM5
```

The arm server then uses the ToF as a second height estimate. The step governor, the floor
stop, the turret clearance and the near-board servo speed all use the **lower** of the model
height and the ToF height. A bad ToF reading can only make the arm more cautious, never
less. The ToF height is used only when its reading is valid, less than 0.3 s old, and taken
with the probe tilted no more than 30°. Its beam is assumed parallel to the probe:
`height = (range - zero) * cos(tilt)`.

**Zero it** with the tip touching the board: `touch_off` records the current range as that
head's zero. Do this once per head if the heads have different lengths. Or give one zero for
all heads with `--tof-offset-mm`. Until it's zeroed, the ToF reports a range but no height.

`/api/state` has a `tof` block, and `height_source` says which estimate is in use.
`/api/sensors` has both sensors plus the implied probe height. From scripts, use
`Arm.tof_height()` and `Arm.sensors()` in `arm_client.py`.

The ToF measures to whatever its beam hits. A tall part under the beam reads closer than
the board, which only slows the arm. A beam that misses the board edge reads farther, and
the model height then still limits the move. The ToF gives proximity, not contact.

| Tool | `status=ok` | Otherwise |
| --- | --- | --- |
| `read_tof` | `measurements`: `range_mm`, `height_above_board_mm` (null with `height_note` until zeroed), `zero_mm`, `tool`, `model` | `inconclusive: no_target`; `failed: stale_reading`, `sensor_head_disconnected`, `no_reading`, `sensors_not_enabled`, `arm_unavailable`, `unknown_arm` |
| `read_thermal` | `measurements`: `celsius` (8x8, row-major), `min_c`, `max_c`, `mean_c`, `hottest_cell` [row, col], `thermistor_c` | the same `failed` reasons |

Thermal cells span the sensor's ~60° view, so a hot cell points at a part that runs hot. It
isn't that part's temperature. `mode` is `live` from an arm server and `simulated` from the
fake.

## Software simulator

`TH_INSTRUMENTS=fake` replaces the AD3 and both arms. Any setting of `TH_INSTRUMENTS` other
than `fake` never falls back to it. The fake bench always has the requested signal:

- a 1 kHz 0 to 3.3 V square wave on each scope input
- `OK\r\n` over UART
- an I2C write of `01 60` to address 0x48
- SPI bytes `9f`, then `ef 40`
- 1 kHz PWM at 25 % duty

`TH_FAKE_SCENARIO` injects failures: `clipped`, `no_trigger`, `no_activity`,
`instrument_missing`, `capture_error`, `tool_change_blocked`, `tool_not_confirmed`, and
`arm_unreachable`.

```powershell
$env:TH_INSTRUMENTS = "fake"; $env:TH_RIG_CONFIG = "config\rig.example.json"
.\.venv\Scripts\python.exe run_mcp.py
```

## Not in this branch

- Probe positioning and contact verification (the measurement orchestrator).
- The multimeter driver.
- `read_digital`.
- The wavegen and power supplies, which stay off.
