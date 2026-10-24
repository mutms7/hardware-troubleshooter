"""Ask Jev to find something in one camera frame, and draw its answer.

  python jev_probe.py                                  default frame (selection/selection.png), asks for the 3V3 pin
  python jev_probe.py --find "the pin labelled GND"
  python jev_probe.py --image some_frame.png --find "the pad next to the USB connector" --repeat 3
  python jev_probe.py --refine                         zoom in on the first answer and ask again (for small targets)

Needs a TypeSafe API key: set TYPESAFE_API_KEY in your environment, or put a line
  TYPESAFE_API_KEY=your-key
in a .env file in this folder (.env is gitignored). The key is never printed.

This sends the image to TypeSafe's servers. Jev's image questions are experimental: the request
needs the "Danger-Zone-Images" header, and a Box question always returns a box even when the thing
isn't there, so every box is paired with a yes/no "is it visible" question.
"""
import argparse
import os
import tempfile
import time
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
MAX_SIDE, MIN_SIDE, MAX_PIXELS = 1568, 280, 2_458_624   # Jev's image limits


def load_key():
    if os.environ.get("TYPESAFE_API_KEY"):
        return
    env = HERE / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            if line.startswith("TYPESAFE_API_KEY=") and line.split("=", 1)[1].strip():
                os.environ["TYPESAFE_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
                return
    raise SystemExit(
        "No API key. Get one at https://console.typesafe.ai/keys, then either run\n"
        "    setx TYPESAFE_API_KEY your-key      (and open a new terminal)\n"
        "or add the line  TYPESAFE_API_KEY=your-key  to a file named .env in this folder.\n"
        "Don't paste the key into chat.")


def fit_for_jev(img):
    """Scale down so each side is at most 1568 px and the pixel count is within Jev's limits."""
    h, w = img.shape[:2]
    s = min(1.0, MAX_SIDE / max(h, w), (MAX_PIXELS / (h * w)) ** 0.5)
    out = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1.0 else img
    if min(out.shape[:2]) < MIN_SIDE:
        raise SystemExit(f"Image is too small for Jev ({out.shape[1]}x{out.shape[0]}; each side must be at least {MIN_SIDE}).")
    return out


def ask(client, img, text, label="a"):
    """One request: where is `text` in `img`, and is it visible. Returns (box in img pixels, confidence, visible 0-1, seconds)."""
    from typesafe_sdk import Box, Image, Noul
    small = fit_for_jev(img)
    path = Path(tempfile.mkdtemp()) / f"{label}.png"
    cv2.imwrite(str(path), small)
    t0 = time.time()
    r = client.system_one(
        state={"photo": Image(path)},
        questions={"where": Box(instructions=f"{text} in `photo`"),
                   "visible": Noul(instructions=f"Is {text} clearly visible in `photo`?")},
        extra_headers={"TypeSafe-Danger-Zone-Images": "true"},
        model="jev-latest")
    dt = time.time() - t0
    b = r.boxes["where"]
    h, w = img.shape[:2]                      # fractions are of the image sent, so they apply to the original too
    box = (b.box.left * w, b.box.top * h, b.box.right * w, b.box.bottom * h)
    return box, b.confidence, r.nouls["visible"].noul, dt


def refine(client, img, text, box, margin=3.0):
    """Zoom in: crop around the first answer, ask again on the crop, map the new box back to full-frame pixels."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max((x1 - x0), (y1 - y0)) * margin
    side = max(side, 360)                      # keep the crop big enough for Jev's minimum size
    h, w = img.shape[:2]
    left, top = int(max(0, min(cx - side / 2, w - side))), int(max(0, min(cy - side / 2, h - side)))
    right, bottom = int(min(w, left + side)), int(min(h, top + side))
    crop = img[top:bottom, left:right]
    scale = max(1.0, MIN_SIDE * 1.5 / min(crop.shape[:2]))
    if scale > 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    cb, conf, vis, dt = ask(client, crop, text, "crop")
    ch, cw = crop.shape[:2]
    sx, sy = (right - left) / cw, (bottom - top) / ch
    return (left + cb[0] * sx, top + cb[1] * sy, left + cb[2] * sx, top + cb[3] * sy), conf, vis, dt


def annotate(img, box, text, conf, vis, out_path):
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    out = img.copy()
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 220, 0), 2)
    cv2.circle(out, ((x0 + x1) // 2, y1), 5, (0, 165, 255), -1)       # bottom-centre: the point the arm would aim at
    label = f"{text}  box conf {conf:.2f}  visible {vis:.2f}"
    cv2.putText(out, label, (max(4, x0), max(18, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(out, label, (max(4, x0), max(18, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imwrite(str(out_path), out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(HERE / "selection" / "selection.png"))
    ap.add_argument("--find", default="the pin labelled 3V3", help="what to look for, in words")
    ap.add_argument("--repeat", type=int, default=1, help="ask this many times to see how consistent the answers are")
    ap.add_argument("--refine", action="store_true", help="after the first answer, zoom in on it and ask again")
    ap.add_argument("--out", default=str(HERE / "jev_result.png"))
    ap.add_argument("--draw", metavar="LEFT,TOP,RIGHT,BOTTOM",
                    help="no API call: draw a box you got from the Playground (the four 0-1 fractions) on the image")
    ap.add_argument("--region", metavar="X0,Y0,X1,Y1",
                    help="with --draw: the part of --image, in pixels, that the picture you uploaded covered "
                         "(for a crop); default is the whole image")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read {args.image}")
    if args.draw:
        l, t, r, b = (float(v) for v in args.draw.split(","))
        x0, y0, x1, y1 = (float(v) for v in args.region.split(",")) if args.region else (0, 0, img.shape[1], img.shape[0])
        box = (x0 + l * (x1 - x0), y0 + t * (y1 - y0), x0 + r * (x1 - x0), y0 + b * (y1 - y0))
        annotate(img, box, args.find, 0.0, 0.0, args.out)
        print(f"box in {Path(args.image).name} pixels (x0,y0,x1,y1): {tuple(round(v) for v in box)}; "
              f"centre {round((box[0] + box[2]) / 2)},{round((box[1] + box[3]) / 2)}; drawn on {args.out}")
        return
    load_key()
    from typesafe_sdk import TypeSafeClient
    print(f"image {img.shape[1]}x{img.shape[0]} -> sending at {fit_for_jev(img).shape[1]}x{fit_for_jev(img).shape[0]}; "
          f"asking: {args.find!r}")
    results = []
    with TypeSafeClient() as client:
        for i in range(args.repeat):
            box, conf, vis, dt = ask(client, img, args.find)
            if args.refine:
                box2, conf2, vis2, dt2 = refine(client, img, args.find, box)
                print(f"  #{i + 1} coarse box {tuple(round(v) for v in box)} -> refined {tuple(round(v) for v in box2)} "
                      f"| box conf {conf2:.2f}, visible {vis2:.2f} | {dt + dt2:.1f} s")
                box, conf, vis = box2, conf2, vis2
            else:
                print(f"  #{i + 1} box (x0,y0,x1,y1) {tuple(round(v) for v in box)} px | box conf {conf:.2f} | "
                      f"visible {vis:.2f} | {dt:.1f} s")
            results.append(box)
    if len(results) > 1:
        cs = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in results]
        spread = max(max(c[0] for c in cs) - min(c[0] for c in cs), max(c[1] for c in cs) - min(c[1] for c in cs))
        print(f"consistency: the box centres differ by up to {spread:.0f} px across {len(results)} asks")
    annotate(img, results[-1], args.find, conf, vis, args.out)
    print(f"drawn on the image: {args.out}")


if __name__ == "__main__":
    main()
