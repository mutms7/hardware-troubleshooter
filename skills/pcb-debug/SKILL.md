---
name: pcb-debug
description: Debug a physical PCB on the bench using its KiCad design, datasheets and the hardware-troubleshooter rig. Use when the user reports a board that's dead, hot, browning out, shorted, has a wrong rail voltage, or asks where or what to probe, or which pad is on a net. Not for firmware-only bugs; use embedded-debug for those.
---

# PCB debugging

The hardware-troubleshooter MCP server knows the board from its KiCad files and can find,
see and measure pads on the rig. Work from evidence: every claim about the board names
the design lookup or measurement it came from.

## 1. Load the design

1. Call `list_designs`. If the board under test isn't the active design, call `ingest_design`
   with the `.kicad_pcb` path, or with the project folder so its schematics come too.
   Re-ingest after any board edit. The `design_id` changes with the file, and stale geometry
   puts the probe on the wrong pad.
2. Confirm the board revision with the user if anything looks off: missing parts, a
   different connector, rework. A design that doesn't match the board gives confident
   wrong answers.
3. For each IC that matters (regulators, the MCU, anything suspect), check `query_design(mpn=...)`
   for a recorded pinout. If there isn't one and a datasheet is available, `ingest_datasheet` it,
   find the pin table with `search_datasheets`, and save it with `record_pinout`, citing
   the page.

## 2. Plan from the schematic, not from guesses

- Use `query_design(ref=...)`, `query_design(net=...)` and `query_design(pad=...)` for nets and
  pins. Never infer a pin's net or number from a datasheet passage or from memory. The design
  store is the only source of connectivity and geometry.
- Use `search_datasheets` for electrical limits, enable thresholds, sequencing, typical
  application circuits and pin descriptions. Cite the source and page.
- Write down the power tree: input, each regulator, enables, power-good signals and loads.
  Debug in power-tree order. A rail is only meaningful once its parent rail is good.

## 3. Pick where to probe

1. `find_probe_points(net=..., tip_diameter_mm=..., position_uncertainty_mm=...)`. Use the real
   tip size. Use a larger uncertainty when only the fixed camera has the board (about 0.3-0.5 mm),
   and a smaller one after wrist alignment (about 0.1 mm).
2. Take the first unrefused candidate. Test points beat passives, which beat vias and
   connectors, which beat IC pins. Never override a `short_risk` refusal. If every candidate
   is refused, tell the user and suggest a smaller tip, a test point to add in the next
   revision, or a manual probe.
3. `locate_on_image(pad_ids=[...], camera="fixed")` shows where the pad should appear, with a
   1-sigma pixel spread and an overlay when a frame is available. Look at the overlay: if the
   pad is occluded, out of view, or the outline clearly misses the copper, stop and report.
   Don't measure. This is a prediction, not proof of position or contact.

## 4. Measure

- `get_rig_status` first. Registration and calibration must be valid.
- `measure_voltage` for DC rails. Only `status=ok` with a reading is evidence. Inconclusive
  contact is not 0 V. A real 0 V with verified contact is.
- `capture_waveform` for ripple, switching nodes, ringing and startup. `capture_logic` for
  buses. These capture wherever the probe already is and never verify contact.
- One measurement at a time. Read each result before deciding the next. Never retry a
  movement silently. Stop on uncertain motion, timeouts or protocol violations.

## 5. Common patterns

| Symptom | First checks |
| --- | --- |
| Dead board | Input rail at the connector, then after the reverse-polarity, fuse and ferrite stage, then each regulator's VIN, EN and VOUT |
| Rail low | Load current (is something hot?), feedback divider values versus the datasheet formula, EN threshold, UVLO |
| Rail oscillates or droops | `capture_waveform` on the output and switch node; output capacitance and ESR against the datasheet's stability rules |
| Shorted rail | Low resistance to GND. Look for the part that heats under limited current; check decoupling caps and ESD diodes on that net (`query_design(net=...)`) |
| One IC dead, rails fine | Its VDD pins at the pin (not just the rail), reset/EN, clock, and the exposed pad's ground |

## Reporting

Summarise what was measured (request IDs), what the design and datasheets say it should be
(with citations), the conclusion, and the next check. Keep `mode` visible: simulated results
demonstrate software only.
