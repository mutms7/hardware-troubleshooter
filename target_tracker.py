"""Draw a target and the fixed probe point on the wrist-camera feed, and find the target again
in every frame as you move the camera.

  python target_tracker.py --camera 1 --tracker homography    feature matching + homography (recommended)
  python target_tracker.py --camera 1 --tracker vit           ViT box tracker (csrt also works)
  python target_tracker.py --camera 1                         draw only; writes the box + probe point for Jev

  left-drag   draw the target box; its centre is the target point A
  right-click set the probe point (see below; saved to disk, redrawn at the same pixel, never moves)
  c           clear the target        q / Esc   quit

Homography mode: the frame you drew on is the reference. Every new frame it matches SIFT features
against the reference, fits one homography to the matches (RANSAC drops the wrong pairs), and maps
point A through it. Nothing is followed frame to frame, so it can't drift, and it handles the scale
change as the camera moves. It needs a textured board (about 15-20 good matches around the target).

Probe point: this should be where the probe LANDS at your fixed hover height, not just where the
tip appears. The camera sits beside the probe, so the two differ. Touch the tip down on paper at
that height, lift back up, and right-click the mark.

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
import numpy as np

HERE = Path(__file__).resolve().parent
LOST_AFTER = 8  # consecutive failed updates before the target is called lost
VIT_MODEL = HERE / "models" / "object_tracking_vittrack_2023sep.onnx"
VIT_MIN_SCORE = 0.5  # ViT tracking score below this counts as a miss; tune on your own footage
MIN_INLIERS = 20      # homography needs at least this many agreeing matches, or it's a miss
EDGE_MARGIN = 0.05    # the mapped point must be inside the frame (plus this fraction); beyond that it's extrapolating
RATIO = 0.75          # Lowe ratio test: a match must be clearly better than the runner-up
PROBE_MASK_FRAC = 0.06  # ignore features within this fraction of the frame height around the probe (--probe-radius)
FEW_FEATURES = 60     # warn when the reference frame has fewer features than this
FEATURE_WIDTH = 640   # features are detected on a copy this wide: faster, at a small cost in precision
MAX_FEATURES = 800    # strongest features kept per frame; fewer is faster, but needs a textured board
# Fixes for shiny boards. Each can be switched off to compare.
USE_GLARE_MASK = True   # ignore blown-out pixels: glare moves with the light, not with the board
GLARE_LEVEL = 245       # a pixel with any channel at or above this counts as glare
USE_LOCAL_FIT = True    # refit the homography using only matches near the target
LOCAL_RADIUS_FRAC = 0.3 # "near" = within this fraction of the frame height, in reference pixels
MIN_LOCAL = 12          # need at least this many local matches, otherwise keep the global fit
USE_ECC = True          # refine the point with a small image alignment around it (sub-pixel)
ECC_PATCH = 96          # side of the patch used for that alignment, in reference pixels
ECC_MAX_SHIFT = 4.0     # ignore a refinement that wants to move the point further than this (reference px)
ECC_MIN_CC = 0.85       # ...or whose alignment correlation is weaker than this
# Keeping up when the view changes (the camera moving in as the arm descends):
USE_KEYFRAMES = True    # also match against a recent frame, not only the one you drew on
KEYFRAME_BELOW = 80     # take a new keyframe when fewer than this many features still agree...
KEYFRAME_AREA = (0.6, 1.7)  # ...or when the box has grown or shrunk past this ratio since the keyframe
ANCHOR_EVERY = 10       # every this many frames, try the original again to cancel any accumulated drift


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


class Ref:
    """One reference view of the board: its features and where point A and the box are in it."""

    def __init__(self, gray, pts, desc, corners, centre):
        self.gray, self.pts, self.desc = gray, pts, desc
        self.corners = np.float32(corners).reshape(-1, 1, 2)
        self.centre = np.float32(centre).reshape(1, 1, 2)
        self.area0 = float(max(cv2.contourArea(self.corners.reshape(4, 2)), 1.0))
        self.h = gray.shape[0]


class FeatureLocator:
    """Finds a point on a flat board in each frame by matching features against reference views.

    There are two references. The original (the frame you drew on) never changes, so matching
    against it can't drift. As the camera moves in or turns, fewer features still match it, so the
    locator also keeps a recent keyframe: it matches that one first, re-anchors to the original
    whenever the original still matches, and takes a fresh keyframe when the view has changed a lot.
    """

    def __init__(self):
        self.sift = cv2.SIFT_create(nfeatures=MAX_FEATURES)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.orig = self.key = None
        self.probe_xy = (0, 0)
        self.frames = 0
        self.last_H = self.last_shift = None
        self.using = "original"
        self.ecc_state = "off"
        self.last_local = 0     # matches found near the target; few means it is leaning on far-away features

    @staticmethod
    def probe_radius(frame_height):
        """Radius, in full-resolution pixels, of the area around the probe that is ignored."""
        return PROBE_MASK_FRAC * frame_height

    def _mask(self, shape, scale):
        # The probe is bolted to the camera, so it sits at the same pixels in every frame and would
        # match itself at "no movement". Leave that area out of both the reference and every frame.
        mask = np.full(shape[:2], 255, np.uint8)
        centre = (int(self.probe_xy[0] * scale), int(self.probe_xy[1] * scale))
        cv2.circle(mask, centre, int(PROBE_MASK_FRAC * shape[0]), 0, -1)
        return mask

    def _features(self, frame):
        # Detect on a downscaled copy (SIFT cost grows with pixel count), then map the points back
        # to full-resolution coordinates so the result is still in the camera's own pixels.
        scale = min(1.0, FEATURE_WIDTH / frame.shape[1])
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mask = self._mask(gray.shape, scale)
        if USE_GLARE_MASK:
            glare = (small.max(axis=2) >= GLARE_LEVEL).astype(np.uint8) * 255
            mask[cv2.dilate(glare, np.ones((9, 9), np.uint8)) > 0] = 0   # a margin, so glare edges go too
        kps, desc = self.sift.detectAndCompute(gray, mask)
        pts = (np.float32([k.pt for k in kps]).reshape(-1, 1, 2) / scale if kps
               else np.zeros((0, 1, 2), np.float32))
        return pts, desc

    def _gray(self, frame):
        return cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (0, 0), 1.0)

    def _refine(self, ref, gray, H):
        """Sub-pixel correction of where point A really is, from the pixels around it.

        Warps the current frame into the reference's geometry using H, aligns that patch with the
        reference patch (ECC, which ignores brightness and contrast changes), and returns the
        leftover shift in reference pixels. A weak alignment or a big shift is rejected: it means
        ECC locked onto the wrong thing (glare, repeated texture)."""
        P, ax, ay = ECC_PATCH, ref.centre[0, 0, 0], ref.centre[0, 0, 1]
        rh, rw = ref.gray.shape
        if rw < P or rh < P:
            return None
        # The probe is bolted to the camera, so it sits at the same pixels in every frame. A patch that
        # overlaps it would be aligned on the probe, not on the board, and the point would get dragged
        # toward the probe. So don't refine while the target (in this frame or in the reference) is near it.
        clear = self.probe_radius(rh) + 0.7 * P
        pred = cv2.perspectiveTransform(ref.centre, H).reshape(2)
        if (np.hypot(*(pred - self.probe_xy)) < clear
                or np.hypot(ax - self.probe_xy[0], ay - self.probe_xy[1]) < clear):
            self.ecc_state = "skipped: the probe is over the target"
            return None
        # Centre the patch on A, but slide it inside the frame when A is near an edge (only the
        # shift between the two patches matters, not where A sits inside them).
        x0 = int(np.clip(round(ax) - P // 2, 0, rw - P))
        y0 = int(np.clip(round(ay) - P // 2, 0, rh - P))
        ref_patch = ref.gray[y0:y0 + P, x0:x0 + P]
        shift = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
        cur_patch = cv2.warpPerspective(gray, shift @ np.linalg.inv(H), (P, P), flags=cv2.INTER_LINEAR)
        ok = ((ref_patch < GLARE_LEVEL) & (cur_patch < GLARE_LEVEL) & (cur_patch > 0)).astype(np.uint8)
        if ok.mean() < 0.5:
            return None
        warp = np.eye(2, 3, dtype=np.float32)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-5)
        try:
            cc, warp = cv2.findTransformECC(ref_patch, cur_patch, warp, cv2.MOTION_TRANSLATION, crit, ok, 5)
        except cv2.error:
            return None
        t = warp[:, 2].astype(np.float64)
        return t if cc >= ECC_MIN_CC and np.hypot(*t) <= ECC_MAX_SHIFT else None

    def start(self, frame, bbox, probe_xy, anchor=1.0):
        self.probe_xy = probe_xy
        pts, desc = self._features(frame)
        x, y, w, h = bbox
        corners = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        # Point A: the middle of the box's bottom edge by default (anchor 1.0; 0.5 is the centre), so
        # the probe, which ends up on A, doesn't sit on top of the box and hide what is being matched.
        self.orig = self.key = Ref(self._gray(frame), pts, desc, corners, [x + w / 2, y + h * anchor])
        self.frames, self.using = 0, "original"
        return len(pts)

    def _fit(self, ref, pts, desc):
        """Match this frame against one reference and fit a homography (reference -> frame).
        Returns (H, inlier_count, src, dst, inlier_mask) or (None, matches_found, ...)."""
        if ref.desc is None or len(ref.pts) < 8 or desc is None or len(pts) < 8:
            return None, 0, None, None, None
        pairs = self.matcher.knnMatch(ref.desc, desc, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
        if len(good) < MIN_INLIERS:
            return None, len(good), None, None, None
        src = ref.pts[[m.queryIdx for m in good]]
        dst = pts[[m.trainIdx for m in good]]
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None:
            return None, 0, None, None, None
        n = int(inl.sum())
        if n < MIN_INLIERS:
            return None, n, None, None, None
        if USE_LOCAL_FIT:
            # Only the plane right around the target matters. Matches far away (tall parts, lens
            # distortion, glare) can pull one global fit off, so refit on the inliers near A.
            dist = np.linalg.norm(src.reshape(-1, 2) - ref.centre[0, 0], axis=1)
            near = (inl.ravel() == 1) & (dist < LOCAL_RADIUS_FRAC * ref.h)
            self.last_local = int(near.sum())
            if int(near.sum()) >= MIN_LOCAL:
                H_local, _ = cv2.findHomography(src[near], dst[near], 0)
                if H_local is not None:
                    H = H_local
        return H, n, src, dst, inl

    def update(self, frame):
        """Returns (ok, bbox, point, quad, inlier_count, inlier_points)."""
        self.frames += 1
        pts, desc = self._features(frame)
        H, n, src, dst, inl = self._fit(self.key, pts, desc)
        ref, using = self.key, ("original" if self.key is self.orig else "keyframe")
        if self.key is not self.orig and (H is None or self.frames % ANCHOR_EVERY == 0):
            H0, n0, src0, dst0, inl0 = self._fit(self.orig, pts, desc)   # try to re-anchor, drift-free
            if H0 is not None:
                H, n, src, dst, inl, ref, using = H0, n0, src0, dst0, inl0, self.orig, "original"
        if H is None:
            return False, None, None, None, n, []
        quad = cv2.perspectiveTransform(ref.corners, H).reshape(4, 2)
        area = cv2.contourArea(quad)
        if not cv2.isContourConvex(quad) or not 0.02 < area / ref.area0 < 50:
            return False, None, None, None, n, []   # a degenerate fit, not a real view of the board
        refined = ref.centre
        self.last_H, self.last_shift = H, None
        gray = self._gray(frame)
        self.ecc_state = "off"
        if USE_ECC:
            self.ecc_state = "rejected (weak alignment or too big a shift)"
            t = self._refine(ref, gray, H)
            if t is not None:
                self.last_shift = t
                self.ecc_state = "applied"
                refined = ref.centre + t.astype(np.float32).reshape(1, 1, 2)
        point = cv2.perspectiveTransform(refined, H).reshape(2)
        fh, fw = frame.shape[:2]
        mx, my = EDGE_MARGIN * fw, EDGE_MARGIN * fh
        if not (-mx <= point[0] <= fw + mx and -my <= point[1] <= fh + my):
            return False, None, None, None, n, []   # the target is off screen; the fit is extrapolating
        # The view has changed a lot, or few features still agree: remember this frame as the new
        # keyframe, with A and the box where we just found them.
        # Not while the target is under the probe: that frame would show the probe over the target, and
        # every later match against it would then be matching the probe.
        probe_over = np.hypot(*(point - self.probe_xy)) < self.probe_radius(fh) + 0.7 * ECC_PATCH
        if (USE_KEYFRAMES and not probe_over
                and (n < KEYFRAME_BELOW or not KEYFRAME_AREA[0] < area / ref.area0 < KEYFRAME_AREA[1])):
            self.key = Ref(gray, pts, desc, quad, point)
            using = "keyframe"
        self.using = using
        x, y, w, h = cv2.boundingRect(quad.astype(np.int32))
        inlier_pts = dst[inl.ravel() == 1].reshape(-1, 2)[:150]
        return True, (float(x), float(y), float(w), float(h)), (float(point[0]), float(point[1])), quad, n, inlier_pts


class TargetTracker:
    """Holds the target. kind: None (just the drawn box), "csrt", "vit", or "homography".
    With external=True the box comes from Jev (read from a file) instead."""

    def __init__(self, kind=None, external=False, anchor=1.0):
        self.anchor = anchor   # where in the box the target point is: 1.0 = middle of the bottom edge, 0.5 = centre
        self.kind = kind
        self.use_tracker = kind is not None
        self.external = external
        self.tracker, self.bbox, self.misses, self.score = None, None, 0, None
        self.point = self.quad = None   # homography mode: the mapped target point and box outline
        self.points = []                # homography mode: matched feature points, for drawing
        self.ref_features = None

    def _create(self):
        if self.kind == "vit":
            params = cv2.TrackerVit_Params()
            params.net = str(VIT_MODEL)
            return cv2.TrackerVit_create(params)
        return cv2.TrackerCSRT_create()

    def start(self, frame, bbox, probe_xy=None):
        self.bbox, self.misses, self.score, self.quad, self.points = tuple(float(v) for v in bbox), 0, None, None, []
        x, y, w, h = self.bbox
        self.point = (x + w / 2, y + h * self.anchor) if self.kind == "homography" else None
        if self.kind == "homography":
            self.tracker = FeatureLocator()
            self.ref_features = self.tracker.start(frame, bbox, probe_xy, self.anchor)
        elif self.use_tracker:
            self.tracker = self._create()
            self.tracker.init(frame, tuple(int(v) for v in bbox))

    def clear(self):
        self.tracker, self.bbox, self.misses, self.score = None, None, 0, None
        self.point = self.quad = None
        self.points = []

    @property
    def active(self):
        return self.tracker is not None or (self.external and self.bbox is not None)

    @property
    def lost(self):
        return self.active and self.misses >= LOST_AFTER

    @property
    def fresh(self):
        """True only when this frame produced the position. After a miss the position is held at its
        last good value, so a control loop must not steer on it."""
        return self.active and self.misses == 0

    def set_external(self, data):
        """Take Jev's latest result: {"bbox": [x, y, w, h], "ok": true}. Missing file/ok=false = a miss."""
        if data and data.get("ok", True) and data.get("bbox"):
            self.bbox, self.misses = tuple(float(v) for v in data["bbox"]), 0
        elif self.bbox is not None:
            self.misses += 1

    def update(self, frame):
        if self.tracker is None:
            return
        if self.kind == "homography":
            ok, box, point, quad, n, pts = self.tracker.update(frame)
            self.score = float(n)
            if ok:
                self.bbox, self.point, self.quad, self.points, self.misses = box, point, quad, pts, 0
            else:
                self.misses += 1
            return
        try:
            ok, box = self.tracker.update(frame)
            if self.kind == "vit":
                self.score = float(self.tracker.getTrackingScore())
                ok = ok and self.score >= VIT_MIN_SCORE   # ViT keeps returning a box when it has lost the target
        except cv2.error:
            # After the target is lost OpenCV's trackers can end up with a huge or empty box and
            # raise (ViT once tried to allocate 20 GB). Count it as a miss; the user redraws.
            ok, box = False, None
        if ok:
            self.bbox, self.misses = tuple(float(v) for v in box), 0
        else:
            self.misses += 1

    def centre(self):
        if self.point is not None:
            return self.point
        if self.bbox is None:
            return None
        x, y, w, h = self.bbox
        return x + w / 2, y + h * self.anchor


def size_str(frame):
    return f"{frame.shape[1]}x{frame.shape[0]}"


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def save_selection(directory, frame, bbox, probe, size, anchor=1.0):
    """Save the frame you drew on and the box, so Jev can start tracking from exactly that image."""
    directory.mkdir(exist_ok=True)
    cv2.imwrite(str(directory / "selection.png"), frame)
    px, py = probe.at(size)
    (directory / "selection.json").write_text(json.dumps({
        "time": time.time(), "image": "selection.png",
        "frame": {"width": size[0], "height": size[1]},
        "bbox": list(bbox), "bbox_format": "x, y, width, height in pixels, origin top-left",
        "target_px": [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] * anchor],
        "target_point": "middle of the bottom edge of the box" if anchor == 1.0 else "centre of the box",
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
        "fresh": trk.fresh,
        "source": "jev" if trk.external else (f"opencv-{trk.kind}" if trk.use_tracker else "drawn"),
        "tracker_score": trk.score,
        "target_bbox": list(trk.bbox) if trk.bbox else None,
        "target_centre_px": list(c) if c else None,
        "error_px": [c[0] - px, c[1] - py] if c and trk.fresh else None,
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
    if trk.kind == "homography":  # the probe area that is ignored when matching
        cv2.circle(frame, (px, py), int(PROBE_MASK_FRAC * h), (120, 120, 120), 1)
    lines = ["probe point " + ("set" if probe.is_set else "NOT SET: right-click where the probe lands")]

    if drag:
        cv2.rectangle(frame, drag[0], drag[1], (255, 255, 0), 1)
    c = trk.centre()
    if c is not None:
        ok = not trk.lost
        col = (0, 220, 0) if trk.fresh else ((0, 165, 255) if ok else (0, 0, 255))
        if trk.kind == "homography" and ok and trk.quad is not None:
            cv2.polylines(frame, [trk.quad.astype(np.int32)], True, col, 2)
            for p in trk.points:
                cv2.circle(frame, (int(p[0]), int(p[1])), 2, (255, 200, 0), -1)
        elif trk.bbox is not None:
            x, y, bw, bh = (int(v) for v in trk.bbox)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), col, 2)
        cx, cy = int(c[0]), int(c[1])
        cv2.circle(frame, (cx, cy), 4, col, -1)
        if ok and not trk.fresh:
            lines.append(f"no fix this frame ({trk.misses}/{LOST_AFTER} misses): holding the last position, don't steer on it")
        elif ok:
            cv2.line(frame, (px, py), (cx, cy), (0, 255, 255), 1)
            dx, dy = c[0] - px, c[1] - py
            lines.append(f"error  dx {dx:+.0f}px  dy {dy:+.0f}px   (move probe {'right' if dx > 0 else 'left'}, "
                         f"{'down' if dy > 0 else 'up'} in the image)")
            if trk.kind == "homography":
                using = getattr(trk.tracker, "using", "original")
                lines.append(f"{int(trk.score or 0)} matched features vs the {using}  (lost below {MIN_INLIERS})")
                loc = trk.tracker
                lines.append(f"{loc.last_local} of them near the target | sub-pixel step: {loc.ecc_state}")
            elif trk.score is not None:
                lines.append(f"{trk.kind} tracking score {trk.score:.2f}  (lost below {VIT_MIN_SCORE})")
        else:
            lines.append("TARGET LOST: draw it again")
            if trk.kind == "homography":
                lines.append(f"only {int(trk.score or 0)} matched features: textureless area, glare, or probe in the way?")
        if trk.kind == "homography" and trk.ref_features is not None and trk.ref_features < FEW_FEATURES:
            lines.append(f"reference frame has only {trk.ref_features} features: redraw on a more textured view")
    else:
        lines.append("left-drag to draw the target box")
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(frame, t, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=1)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--tracker", choices=["none", "csrt", "vit", "homography"], default="none",
                    help="homography: match features against the reference frame (best); vit/csrt: follow "
                         "the box; none: just the drawn box")
    ap.add_argument("--probe-radius", type=float, default=None, metavar="PX",
                    help="radius in camera pixels of the area around the probe point that is ignored when matching "
                         "(the grey circle). Make it just big enough to cover the probe you see. Default: 6%% of the frame height")
    ap.add_argument("--anchor", choices=["bottom", "center"], default="bottom",
                    help="which point of the drawn box is the target: the middle of its bottom edge (default, so "
                         "the probe doesn't cover the box) or its centre")
    ap.add_argument("--jev-file", metavar="PATH",
                    help="JSON that Jev writes every frame, {\"bbox\": [x, y, w, h], \"ok\": true}; shown as the target")
    ap.add_argument("--selection-dir", default=str(HERE / "selection"),
                    help="where the drawn box + the exact frame are saved for Jev to start from")
    ap.add_argument("--probe-file", default=str(HERE / "probe_point.json"))
    ap.add_argument("--state-file", default=str(HERE / "tracker_state.json"))
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # needed for HD over USB 2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"Could not read from camera {args.camera}. Try --camera 0, 1, 2.")

    global PROBE_MASK_FRAC
    if args.probe_radius:
        PROBE_MASK_FRAC = args.probe_radius / frame.shape[0]
    probe = Probe(args.probe_file)
    if args.tracker == "vit" and not VIT_MODEL.exists():
        raise SystemExit(f"Missing {VIT_MODEL}. Use --tracker csrt, or download the ViT model there.")
    anchor = 1.0 if args.anchor == "bottom" else 0.5
    trk = TargetTracker(kind=None if args.tracker == "none" else args.tracker, external=bool(args.jev_file),
                        anchor=anchor)
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

    cv2.namedWindow("target tracker", cv2.WINDOW_NORMAL)  # resizable; drawing still uses frame pixels
    print(f"camera {args.camera}: {size_str(frame)} MJPG")
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
            save_selection(sel_dir, frame, st["pending"], probe, size, anchor)
            if trk.external:  # show your box until Jev's first result replaces it
                trk.bbox, trk.misses = tuple(float(v) for v in st["pending"]), 0
            else:
                trk.start(frame, st["pending"], probe.at(size))
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
