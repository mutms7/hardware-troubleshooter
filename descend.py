"""Move the probe straight down along its own axis, in small steps, and stop safely.

  python descend.py                      dry run: prints the plan, moves nothing
  python descend.py --manual --send      you drive: Space = one step down, u = up, t = touch off, q = quit
  python descend.py --send --to-mm 3     automatic: go down until 3 mm above the board (by the model)

This is the depth half of the approach. servo.py lines the probe up over the target in x and y at a
fixed hover height; this then goes straight down. It does NOT run the xy loop while descending: the
camera sits beside the probe, so the target appears to shift as the camera drops and an xy loop would
chase a false error. Straight down along the axis lands where it was aligned at hover.

Automatic mode stops on whichever happens first:
  * the height target is reached,
  * the server says the probe is at the floor,
  * contact: the real tip lags the commanded tip (following error) by more than --contact-mm,
  * no progress: several steps in a row where the commanded height didn't change,
  * the arm refuses a jog.
On contact it backs the command off so the arm stops pressing.
Run it only on the real, armed arm. Heights are the model's estimate (with its safety margin) until
you press t at the moment the tip actually touches the board.
"""
import argparse
import time

from arm_client import Arm


def baseline_error(arm, n=5):
    """The following error with the arm sitting still: servo backlash and encoder noise, not contact."""
    vals = []
    for _ in range(n):
        s = arm.state(measure=True)
        if "follow_err_z_mm" in s:
            vals.append(s["follow_err_z_mm"])
        time.sleep(0.1)
    return sum(vals) / len(vals) if vals else 0.0


def descend(arm, args):
    base = baseline_error(arm) if args.send else 0.0
    s = arm.state()
    print(f"start: {s['height_mm']:.1f} mm above the board (model), baseline following error {base:+.2f} mm; "
          f"{'SENDING' if args.send else 'dry run'}; step {args.step_mm} mm, target {args.to_mm} mm, "
          f"contact threshold {args.contact_mm} mm")
    over, stalled, steps = 0, 0, 0
    h = s["height_mm"]
    while steps < args.max_steps:
        if args.send:
            h = arm.state()["height_mm"]
        if h <= args.to_mm + 1e-6:
            print(f"Reached the target height: {h:.1f} mm.")
            return "reached"
        step = min(args.step_mm, h - args.to_mm)
        if not args.send:
            print(f"  would go down {step:.2f} mm from {h:.1f} mm")
            h -= step
            steps += 1
            continue
        r = arm.jog(in_=step)
        steps += 1
        if r["msg"] != "ok":
            print(f"Stopped: the arm refused the step ({r['msg']}).")
            return "refused"
        note = r.get("note", "")
        if "floor" in note:
            print(f"Stopped at the floor ({note}).")
            return "floor"
        time.sleep(args.settle)
        s2 = arm.state(measure=True)
        gap = s2.get("follow_err_z_mm", 0.0) - base
        moved = h - s2["height_mm"]
        stalled = stalled + 1 if moved < 0.25 * step else 0
        over = over + 1 if gap > args.contact_mm else 0
        print(f"  height {s2['height_mm']:5.2f} mm  (moved {moved:.2f})  following error {gap:+.2f} mm"
              + (f"  [{note}]" if note else ""))
        if over >= args.confirm:
            back = gap + 0.2
            arm.jog(in_=-back)
            print(f"Contact suspected: the real tip lags the command by {gap:.2f} mm. "
                  f"Backed the command off {back:.2f} mm.")
            return "contact"
        if stalled >= 3:
            print("Stopped: three steps in a row without the commanded height changing.")
            return "stalled"
    print("Stopped: step limit reached." if args.send else "(dry run done)")
    return "limit"


def manual(arm, args):
    import msvcrt
    base = baseline_error(arm)
    print("Space/Enter = down one step, u = up one step, U = up 3 mm, X = up 10 mm, t = touch off here, q = quit")
    print(f"(the arm sags a little under its own weight: resting following error is {base:+.2f} mm. "
          "The number below is the CHANGE from that, which is what shows contact.)")
    while True:
        s = arm.state(measure=True)
        print(f"\r  height {s['height_mm']:5.2f} mm   change in following error {s.get('follow_err_z_mm', 0.0) - base:+.2f} mm   "
              f"{'(touched off)' if s.get('board_calibrated') else '(model estimate)'}   ", end="", flush=True)
        k = msvcrt.getwch()
        if k in (" ", "\r"):
            r = arm.jog(in_=args.step_mm)
        elif k == "u":
            r = arm.jog(in_=-args.step_mm)
        elif k == "U":
            r = arm.jog(in_=-3.0)
        elif k == "X":
            r = arm.jog(in_=-10.0)
        elif k == "t":
            r = arm.touch_off()
            print("\n" + r["msg"])
            continue
        elif k in ("q", "\x1b"):
            print()
            return
        else:
            continue
        if r["msg"] != "ok":
            print(f"\n  {r['msg']}")
        time.sleep(args.settle)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--send", action="store_true", help="actually move the arm (default: dry run)")
    ap.add_argument("--manual", action="store_true", help="drive it with the keyboard, one step per key")
    ap.add_argument("--step-mm", type=float, default=0.25, help="small steps: contact is only noticed after a step")
    ap.add_argument("--to-mm", type=float, default=0.0, help="stop this many mm above the board (model height)")
    ap.add_argument("--contact-mm", type=float, default=0.5, help="following error that counts as contact; tune on your arm")
    ap.add_argument("--confirm", type=int, default=2, help="readings in a row over the threshold to call it contact")
    ap.add_argument("--settle", type=float, default=0.4)
    ap.add_argument("--max-steps", type=int, default=200)
    args = ap.parse_args()
    arm = Arm(args.url)
    if (args.send or args.manual) and not arm.state()["hw"]["armed"]:
        raise SystemExit("The arm isn't armed. Press G in the viewer first.")
    if args.manual:
        if not args.send:
            raise SystemExit("Manual mode moves the arm. Add --send.")
        manual(arm, args)
    else:
        descend(arm, args)


if __name__ == "__main__":
    main()
