---
name: embedded-debug
description: Debug embedded firmware against real hardware: an MCU that won't boot, a dead or corrupted UART/I2C/SPI bus, wrong GPIO behaviour, clock or reset problems, peripherals that don't respond. Uses the board's KiCad design, datasheets, and the rig's AD3 scope and logic probes. Use when firmware and hardware might both be at fault; for purely electrical faults use pcb-debug.
---

# Embedded debugging

Split the problem into hardware state you can measure and firmware behaviour you can read.
Measure before changing code, and change one thing at a time.

## 1. Establish the hardware ground truth

1. Make sure the board's design is active (`list_designs`, `ingest_design`). Resolve every
   signal through `query_design`: the MCU pin (`query_design(pad="U1.23")`), its net, and
   what else is on that net (pull-ups, series resistors, other devices). Firmware pin names
   (PA9, GPIO17) must be mapped to package pins from the MCU datasheet's pinout table, not
   from memory. Record it with `record_pinout` once found, citing the page.
2. Check that the MCU is alive before debugging firmware. Follow the pcb-debug skill for
   the power and probe-point steps:
   - VDD at the MCU's supply pins, and VDDA/VREF if analog matters.
   - Reset or NRST is high (released) and not toggling. Toggling usually means a watchdog
     or brown-out loop.
   - The clock: `capture_waveform` on the crystal or oscillator output, if it can be probed
     without loading it. Crystals are easily stopped by a probe, so prefer the MCU's clock-out
     pin if the firmware can enable it.
   - Boot-mode pins are at the levels the datasheet requires for normal boot.

## 2. Buses

Use `list_capture_profiles` and `get_instrument_status` first. Each arm's logic probe takes
one line, so two-line buses use both arms. `capture_logic` maps each line role to the arm
whose logic probe sits on it, e.g. `capture_logic("i2c", {"scl": "A", "sda": "B"})`.

| Bus | Capture | Look for |
| --- | --- | --- |
| UART | `capture_logic("uart", {"rx": <arm>})`, with that arm's logic probe on the MCU's TX | Baud mismatch (wrong clock config), idle-low line (inverted or not initialised), framing errors |
| I2C | SCL plus SDA, one per arm | No ACK (wrong address: 7-bit versus 8-bit), stuck-low SDA (a device holding the bus; clock out 9 pulses), slow edges (pull-ups too weak for the bus capacitance) |
| SPI | SCK plus one data line (no CS, so the bytes are tentative) | Mode (CPOL/CPHA) mismatch, MISO floating (device not selected or not powered) |

A flat line or `no_activity` can mean the probe isn't touching anything. Captures never
verify contact. Re-check the probe point with `locate_on_image` before concluding a bus is
dead.

## 3. Firmware side

- Read the init code for the peripheral involved: clock tree, pin mux and alternate
  function, pull configuration, peripheral clock enable. Compare each against the
  datasheet or reference manual (`search_datasheets`, with citations).
- Common culprits: the peripheral clock isn't enabled; the wrong alternate-function number;
  a GPIO left as input; an interrupt priority or vector not set; DMA pointing at the wrong
  peripheral; the watchdog not fed; a low-power mode gating the clock.
- When a fix is proposed, predict what the next capture should show, then capture to confirm.

## Reporting

Pair each finding with its evidence: capture or request IDs, design lookups, datasheet
pages, and the code lines involved. State what's confirmed and what's still a hypothesis.
