"""Measure where the arms' bases are relative to each other, by touching the same marks.

Mark 3 or more points on the bench, spread out and not in a line. For each mark, jog arm A's
probe tip onto it and press Enter, then do the same with arm B. The tip positions each arm
reports (in its own frame) are fitted with a rotation about Z plus a shift, and B's base pose
is written to the workcell file with A's kept as the reference. Both arms must be on the same
flat table; the residual tells you how well the marks agree.

Run both servers in hardware mode, disarmed (the viewer then mirrors the real arm), and move
each arm by hand or armed with the keys.

  python register_arms.py --a http://127.0.0.1:8765 --b http://127.0.0.1:8766 --points 3
"""
import argparse
import json
import math

import numpy as np

import collision
from arm_client import Arm


def fit_yaw_shift(src, dst):
    """Yaw (rad) and shift t so that Rz(yaw) @ src + t ~= dst. Points are (n, 3)."""
    sc, dc = src.mean(axis=0), dst.mean(axis=0)
    s, d = src[:, :2] - sc[:2], dst[:, :2] - dc[:2]
    yaw = math.atan2(np.sum(s[:, 0] * d[:, 1] - s[:, 1] * d[:, 0]), np.sum(s[:, 0] * d[:, 0] + s[:, 1] * d[:, 1]))
    rot = collision.base_pose([0, 0, 0], math.degrees(yaw))[:3, :3]
    return yaw, dc - rot @ sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="http://127.0.0.1:8765", help="reference arm's server")
    ap.add_argument("--b", default="http://127.0.0.1:8766")
    ap.add_argument("--names", nargs=2, default=["A", "B"], help="the two arms' names in the workcell")
    ap.add_argument("--points", type=int, default=3)
    ap.add_argument("--workcell", default=str(collision.WORKCELL))
    args = ap.parse_args()
    if args.points < 2:
        raise SystemExit("need at least 2 marks (3 or more to see the error)")

    arms = {args.names[0]: Arm(args.a), args.names[1]: Arm(args.b)}
    tips = {name: [] for name in arms}
    for k in range(args.points):
        for name, arm in arms.items():
            input(f"Mark {k + 1}: put arm {name}'s probe tip on it, then press Enter ")
            tips[name].append(arm.state()["tip"])
    a, b = (np.array(tips[n]) for n in args.names)

    yaw, shift = fit_yaw_shift(b, a)  # arm B's frame -> arm A's frame
    b_in_a = collision.base_pose(shift, math.degrees(yaw))
    resid = np.linalg.norm((b_in_a[:3, :3] @ b.T).T + b_in_a[:3, 3] - a, axis=1) * 1000
    print("per-mark error (mm):", " ".join(f"{r:.1f}" for r in resid))
    if resid.max() > 5:
        print("Over 5 mm: a tip was off its mark, a mark moved, or the model's tip is off. Redo it.")

    with open(args.workcell) as f:
        cfg = json.load(f)
    ref = cfg["arms"][args.names[0]]
    a_base = collision.base_pose(ref.get("base_xyz_m", [0, 0, 0]), ref.get("base_yaw_deg", 0.0))
    b_base = a_base @ b_in_a
    other = cfg["arms"][args.names[1]]
    other["base_xyz_m"] = [round(float(v), 5) for v in b_base[:3, 3]]
    other["base_yaw_deg"] = round(math.degrees(math.atan2(b_base[1, 0], b_base[0, 0])), 3)
    with open(args.workcell, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    print(f"arm {args.names[1]}: base_xyz_m {other['base_xyz_m']}, base_yaw_deg {other['base_yaw_deg']}"
          f"  -> {args.workcell}. Restart both servers to use it.")


if __name__ == "__main__":
    main()
