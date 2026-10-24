"""Draw a target box and the fixed probe point on the wrist-camera feed, for Jev to track.

  python target_tracker.py --camera 1            draw only; writes the box + probe point for Jev
  python target_tracker.py --camera 1 --track    also follows the box itself (OpenCV CSRT)

  left-drag   draw the target box (with --track, tracking starts when you release)
  right-click set the probe point (where the probe tip appears in the image; saved to disk,
              redrawn at the same pixel every frame, never moves)
  c           clear the target        q / Esc   quit

Every frame the latest result is written to tracker_state.json, so a control loop (Jev or
anything else) can read it: the target box and centre, the probe point, and error_px =
target centre - probe point. error_px[0] > 0 means the target is right of the probe;
error_px[1] > 0 means it is below it (image y points down).
"""
import argparse
import json
import os
import time
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
LOST_AFTER = 8  # consecutive failed updates before the target is called lost


class Probe:
    """The probe point in image pixels. Stored with the frame size so a resolution change rescales it."""

    def __init__(self, path):
        self.path, self.x, self.y, self.size, self.is_set = Path(path), None, None, None, False
        if self.path.exists():
            d = json.loads(self.path.read_text())
            self.x, self.y, self.size, self.is_set = d["x"], d["y"], (d["width"], d["height"]), True

    def set(self, x, y, size):
        self.x, self.y, self.size, self.is_set = float(x), float(y), size, True
        self.path.write_text(json.dumps({"x": self.x, "y": self.y, "width": size[0], "height": size[1]}, indent=2))

    def at(self, size):
        if self.x is None:
            return size[0] / 2, size[1] / 2
        return self.x * size[0] / self.size[0], self.y * size[1] / self.size[1]


class TargetTracker:
    """Holds the target box. With use_tracker it also follows it with OpenCV's CSRT; without,
    the box is just the one you drew (tracking is left to Jev)."""

    def __init__(self, use_tracker=False, external=False):
        self.use_tracker = use_tracker
        self.external = external  # the box comes from Jev (read from a file), not from this program
        self.tracker, self.bbox, self.misses = None, None, 0

    def start(self, frame, bbox):
        if self.use_tracker:
            self.tracker = cv2.TrackerCSRT_create()
            self.tracker.init(frame, tuple(int(v) for v in bbox))
        self.bbox, self.misses = tuple(float(v) for v in bbox), 0

    def clear(self):
        self.tracker, self.bbox, self.misses = None, None, 0

    @property
    def active(self):
        return self.tracker is not None or (self.external and self.bbox is not None)

    @property
    def lost(self):
        return self.active and self.misses >= LOST_AFTER

    def set_external(self, data):
        """Take Jev's latest result: {"bbox": [x, y, w, h], "ok": true}. Missing file/ok=false = a miss."""
        if data and data.get("ok", True) and data.get("bbox"):
            self.bbox, self.misses = tuple(float(v) for v in data["bbox"]), 0
        elif self.bbox is not None:
            self.misses += 1

    def update(self, frame):
        if self.tracker is None:
            return
        ok, box = self.tracker.update(frame)
        if ok:
            self.bbox, self.misses = tuple(float(v) for v in box), 0
        else:
            self.misses += 1

    def centre(self):
        if self.bbox is None:
            return None
        x, y, w, h = self.bbox
        return x + w / 2, y + h / 2


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def save_selection(directory, frame, bbox, probe, size):
    """Save the frame you drew on and the box, so Jev can start tracking from exactly that image."""
    directory.mkdir(exist_ok=True)
    cv2.imwrite(str(directory / "selection.png"), frame)
    px, py = probe.at(size)
    (directory / "selection.json").write_text(json.dumps({
        "time": time.time(), "image": "selection.png",
        "frame": {"width": size[0], "height": size[1]},
        "bbox": list(bbox), "bbox_format": "x, y, width, height in pixels, origin top-left",
        "probe_px": [px, py],
    }, indent=2))


def write_state(path, size, probe, trk):
    px, py = probe.at(size)
    c = trk.centre()
    state = {
        "time": time.time(),
        "frame": {"width": size[0], "height": size[1]},
        "probe_px": [px, py],
        "probe_is_set": probe.is_set,
        "tracking": trk.active and not trk.lost,
        "source": "jev" if trk.external else ("opencv" if trk.use_tracker else "drawn"),
        "target_bbox": list(trk.bbox) if trk.bbox else None,
        "target_centre_px": list(c) if c else None,
        "error_px": [c[0] - px, c[1] - py] if c and not trk.lost else None,
    }
    tmp = str(path) + ".tmp"
    try:
        Path(tmp).write_text(json.dumps(state))
        os.replace(tmp, path)
    except OSError:
        pass  # a reader had the file open; the next frame will overwrite it


def draw(frame, probe, trk, drag):
    h, w = frame.shape[:2]
    px, py = probe.at((w, h))
    px, py = int(round(px)), int(round(py))
    colour = (255, 0, 255) if probe.is_set else (0, 165, 255)
    cv2.line(frame, (px - 14, py), (px + 14, py), colour, 1)
    cv2.line(frame, (px, py - 14), (px, py + 14), colour, 1)
    cv2.circle(frame, (px, py), 5, colour, 1)
    lines = ["probe point " + ("set" if probe.is_set else "NOT SET: right-click where the tip appears")]

    if drag:
        cv2.rectangle(frame, drag[0], drag[1], (255, 255, 0), 1)
    c = trk.centre()
    if c is not None:
        x, y, bw, bh = (int(v) for v in trk.bbox)
        ok = not trk.lost
        col = (0, 220, 0) if ok else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), col, 2)
        cx, cy = int(c[0]), int(c[1])
        cv2.circle(frame, (cx, cy), 3, col, -1)
        if ok:
            cv2.line(frame, (px, py), (cx, cy), (0, 255, 255), 1)
            dx, dy = c[0] - px, c[1] - py
            lines.append(f"error  dx {dx:+.0f}px  dy {dy:+.0f}px   (move probe {'right' if dx > 0 else 'left'}, "
                         f"{'down' if dy > 0 else 'up'} in the image)")
        else:
            lines.append("TARGET LOST: draw it again")
    else:
        lines.append("left-drag to draw the target box")
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(frame, t, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=1)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--track", action="store_true",
                    help="also follow the box with OpenCV's CSRT (off by default: Jev does the tracking)")
    ap.add_argument("--jev-file", metavar="PATH",
                    help="JSON that Jev writes every frame, {\"bbox\": [x, y, w, h], \"ok\": true}; shown as the target")
    ap.add_argument("--selection-dir", default=str(HERE / "selection"),
                    help="where the drawn box + the exact frame are saved for Jev to start from")
    ap.add_argument("--probe-file", default=str(HERE / "probe_point.json"))
    ap.add_argument("--state-file", default=str(HERE / "tracker_state.json"))
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"Could not read from camera {args.camera}. Try --camera 0, 1, 2.")

    probe = Probe(args.probe_file)
    trk = TargetTracker(use_tracker=args.track, external=bool(args.jev_file))
    sel_dir = Path(args.selection_dir)
    st = {"drag": None, "start": None, "pending": None, "probe": None}

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            st["start"], st["drag"] = (x, y), ((x, y), (x, y))
        elif event == cv2.EVENT_MOUSEMOVE and st["start"]:
            st["drag"] = (st["start"], (x, y))
        elif event == cv2.EVENT_LBUTTONUP and st["start"]:
            (x0, y0), st["start"], st["drag"] = st["start"], None, None
            box = (min(x0, x), min(y0, y), abs(x - x0), abs(y - y0))
            if box[2] >= 8 and box[3] >= 8:
                st["pending"] = box
        elif event == cv2.EVENT_RBUTTONDOWN:
            st["probe"] = (x, y)

    cv2.namedWindow("target tracker")
    cv2.setMouseCallback("target tracker", on_mouse)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        size = (frame.shape[1], frame.shape[0])
        if st["probe"]:
            probe.set(*st["probe"], size)
            st["probe"] = None
        if st["pending"]:
            save_selection(sel_dir, frame, st["pending"], probe, size)
            if trk.external:  # show your box until Jev's first result replaces it
                trk.bbox, trk.misses = tuple(float(v) for v in st["pending"]), 0
            else:
                trk.start(frame, st["pending"])
            st["pending"] = None
        elif trk.external:
            trk.set_external(read_json(args.jev_file))
        else:
            trk.update(frame)
        write_state(args.state_file, size, probe, trk)
        shown = frame.copy()
        draw(shown, probe, trk, st["drag"])
        cv2.imshow("target tracker", shown)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("c"):
            trk.clear()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
