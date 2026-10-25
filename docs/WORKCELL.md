# Two arms on one bench

Both arms reach into the same space over the board. Each one's `sim_server.py` checks every
move against where the other arm is and refuses anything that would bring them too close.
`goto` plans a path around the other arm.

## Setup

1. Copy [config/workcell.example.json](../config/workcell.example.json) to
   `config/workcell.json`.
2. Measure the probe tool heads and set `probe` for each arm: `needle_mm` and `needle_radius_mm`
   for the thin tip, `head_mm` and `head_radius_mm` for the body behind it.
3. Start both servers with `--arm`:

   ```powershell
   python sim_server.py --hardware COM10 --port 8765 --arm A
   python sim_server.py --hardware COM11 --port 8766 --arm B
   ```

   Leave out `--hardware` to try it in simulation. Each viewer shows both arms. The other arm
   is drawn in translucent blue, **B** shows the collision boxes, and the HUD shows the
   clearance left and which parts are closest.
4. Register the bases: mark 3 or more points on the bench, spread out and not in a line, then
   run `python register_arms.py` and touch each arm's tip to each mark. It writes arm B's base
   pose into `config/workcell.json`. Restart both servers afterwards. Until you do this, the
   example layout (bases 0.40 m apart, facing each other) is a guess, and the collision checks
   are only as good as that guess.

## How it works

- **Model** (`collision.py`): every URDF link mesh is wrapped in oriented boxes that contain
  every triangle. Each turret slot also gets a needle box and a head box, because the probes
  aren't in the meshes. Boxes are fitted once and cached in `data/collision/`.
- **Clearance**: arm bodies keep `clearance_mm` (15) apart and probes keep `probe_clearance_mm`
  (4) apart, so both arms can still probe neighbouring pins. Distances are lower bounds, so
  errors fall on the safe side.
- **Sharing**: each server writes its commanded joints to `data/workcell/<arm>.json`, along
  with the goals from the last `settle_s` seconds that a real arm may still be moving from.
  A move is accepted only under a shared file lock, after its whole sweep has been checked
  against the other arm's recent path. That stops two servers from both accepting moves into
  the same space. Both servers must run on the same machine.
- **Unknown other arm**: if the other arm has never reported its pose, every move is blocked.
- **Blocked jog**: if the other arm is in the way of a move or tilt, the server first tries
  tilting the probe up to 20° either way, keeping the tip on the same point. `state["note"]`
  says when it did.
- **`goto`** (`arm.goto([x, y, z])`, add `frame="world"` for bench coordinates): lifts straight
  up to 30 mm above the board, crosses in joint space (straight if that's clear, otherwise
  RRT-Connect), then comes straight down. Each waypoint is checked again just before it's sent.
  If the other arm has moved into the way, the move stops with `goto stopped...`; call `goto`
  again to replan. On hardware it waits for each waypoint to be reached before sending the next.

## Limits

- The other arm is treated as fixed while planning; only the per-waypoint check reacts to it
  moving. Neither arm reserves its future path.
- Servos don't move in exact straight lines between goals. Waypoints are kept at most 5° apart
  to keep that error small.
- Arm-to-table and self-collisions aren't checked; only the probe tip's height is.
- With the probe tilt at 0°, the turret axis is vertical. In the URDF the probes splay about
  24° off that axis, so all three tips sit at the same height. Check this against the real
  turret: if the working probe should point straight down, the tilt convention in
  `sim_server.py` needs changing.
