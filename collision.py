"""Collision model for two arms sharing one bench.

Every link with a mesh in the URDF is wrapped in a few oriented boxes, fitted once from the STL
triangles and cached in data/. The boxes contain every triangle, so they are never smaller than
the part. The probes aren't in the meshes, so each turret slot gets two boxes of its own: a thin
needle at the tip and a thicker head behind it (sizes in config/workcell.json; measure them).

Each box carries a pad. Two boxes must stay pad_a + pad_b apart: 15 mm between arm bodies,
4 mm between two probes (so both arms can still probe neighbouring pins), 9.5 mm in between.
Distances are lower bounds (separating-axis gaps), so the check errs toward "too close".

Frames: each arm's URDF root sits at its base pose in a shared world frame (Z up, meters), set
in config/workcell.json. register_arms.py measures it by touching both tips to the same marks.

Both arm servers run on one host. Each publishes its committed joint angles, plus the last
SETTLE_S seconds of them (the real servos lag their goals), to data/workcell/<arm>.json. A move
is committed only under a shared file lock, after checking its whole joint-space sweep against
every other arm's recent poses, so two servers can never both accept moves into the same space.

  python collision.py                     fit the boxes and print them
"""
import contextlib
import json
import math
import os
import struct
import time
import xml.etree.ElementTree as ET
from itertools import pairwise
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
URDF = ROOT / "urdf" / "debugger_arm.urdf"
WORKCELL = ROOT / "config" / "workcell.json"
CACHE_DIR = ROOT / "data" / "collision"

CLEARANCE_MM = 15.0        # arm body to arm body
PROBE_CLEARANCE_MM = 4.0   # probe to probe; probe to body is the average of the two
SEGMENT_MM = 40.0          # a mesh is cut into pieces about this long, one box each
# A move is checked at samples where no point of the arm has moved more than SWEEP_MM, and each
# sample must clear by SWEEP_MM / 2 more, so the arm can't get too close between two samples.
SWEEP_MM = 4.0
REACH_M = 0.35             # no point of the arm is further than this from any of its joint axes
SETTLE_S = 1.5             # how long a commanded pose still counts as maybe-occupied
LOCK_TIMEOUT_S = 5.0
# Probe defaults, mm. The needle is the thin tip, the head is the body of the tool behind it.
PROBE = {"needle_mm": 15.0, "needle_radius_mm": 1.5, "head_mm": 55.0, "head_radius_mm": 7.0}


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def transform(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    t = np.eye(4)
    t[:3, :3] = rpy_matrix(*rpy)
    t[:3, 3] = xyz
    return t


def axis_angle(axis, angle):
    x, y, z = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    c, s, v = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    t = np.eye(4)
    t[:3, :3] = [
        [c + x * x * v, x * y * v - z * s, x * z * v + y * s],
        [y * x * v + z * s, c + y * y * v, y * z * v - x * s],
        [z * x * v - y * s, z * y * v + x * s, c + z * z * v],
    ]
    return t


def _origin(el):
    o = el.find("origin")
    if o is None:
        return np.eye(4)
    xyz = [float(v) for v in o.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in o.get("rpy", "0 0 0").split()]
    return transform(xyz, rpy)


def read_stl(path):
    """Triangles (n, 3, 3) of an STL file, binary or ASCII."""
    data = Path(path).read_bytes()
    n = struct.unpack_from("<I", data, 80)[0] if len(data) >= 84 else 0
    if len(data) == 84 + 50 * n:
        rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
                            count=n, offset=84)
        return rec["v"].astype(float)
    pts = [[float(v) for v in line.split()[1:4]]
           for line in data.decode(errors="ignore").splitlines() if line.strip().startswith("vertex")]
    return np.array(pts, dtype=float).reshape(-1, 3, 3)


class Urdf:
    """Forward kinematics for every link of a URDF, not just the IK chain."""

    def __init__(self, path=URDF):
        self.path = Path(path)
        robot = ET.parse(self.path).getroot()
        self.joints = {}   # child link -> (joint name, type, parent link, origin, axis)
        for j in robot.findall("joint"):
            axis = j.find("axis")
            self.joints[j.find("child").get("link")] = (
                j.get("name"), j.get("type"), j.find("parent").get("link"), _origin(j),
                [float(v) for v in axis.get("xyz").split()] if axis is not None else [0, 0, 1])
        self.links = [link.get("name") for link in robot.findall("link")]
        self.root = next(link for link in self.links if link not in self.joints)
        self.visuals = {}  # link -> [(mesh path, origin, scale)]
        for link in robot.findall("link"):
            for v in link.findall("visual"):
                m = v.find("geometry/mesh")
                if m is None:
                    continue
                scale = np.array([float(s) for s in m.get("scale", "1 1 1").split()])
                path = self.path.parent / "meshes" / m.get("filename").split("/")[-1]
                self.visuals.setdefault(link.get("name"), []).append((path, _origin(v), scale))

    def link_poses(self, q, base=None):
        """World pose (4x4) of every link. q maps joint name -> radians; missing joints are 0."""
        poses = {self.root: np.eye(4) if base is None else base}

        def pose(link):
            if link not in poses:
                name, kind, parent, origin, axis = self.joints[link]
                t = pose(parent) @ origin
                if kind in ("revolute", "continuous"):
                    t = t @ axis_angle(axis, q.get(name, 0.0))
                poses[link] = t
            return poses[link]

        for link in self.links:
            pose(link)
        return poses


def obb(points):
    """Tightest box along the points' principal axes: (centre, axes as columns, half sizes)."""
    c = points.mean(axis=0)
    axes = np.linalg.svd(points - c, full_matrices=False)[2].T
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1
    proj = (points - c) @ axes
    lo, hi = proj.min(axis=0), proj.max(axis=0)
    return c + axes @ ((lo + hi) / 2), axes, (hi - lo) / 2


def fit_boxes(tris):
    """Boxes that together contain every triangle. Long parts are cut into pieces along their
    main axis; a triangle goes into every piece it touches, so none falls between two boxes."""
    pts = tris.reshape(-1, 3)
    c = pts.mean(axis=0)
    axis = np.linalg.svd(pts - c, full_matrices=False)[2][0]
    s = (tris - c) @ axis
    s_lo, s_hi = s.min(axis=1), s.max(axis=1)
    pieces = max(1, math.ceil((s_hi.max() - s_lo.min()) * 1000.0 / SEGMENT_MM))
    edges = np.linspace(s_lo.min(), s_hi.max(), pieces + 1)
    out = []
    for a, b in pairwise(edges):
        chunk = tris[(s_hi >= a) & (s_lo <= b)].reshape(-1, 3)
        if len(chunk) >= 3:
            out.append(obb(np.unique(chunk, axis=0)))
    return out


def mesh_boxes(urdf):
    """{link: [(centre, axes, half) in the link frame]}, cached until the URDF or a mesh changes."""
    stamp = max([urdf.path.stat().st_mtime] +
                [p.stat().st_mtime for vs in urdf.visuals.values() for p, _, _ in vs])
    cache = CACHE_DIR / f"{urdf.path.stem}.json"
    if cache.exists():
        d = json.loads(cache.read_text())
        if d.get("stamp") == stamp and d.get("segment_mm") == SEGMENT_MM:
            return {link: [tuple(np.array(v) for v in box) for box in boxes]
                    for link, boxes in d["links"].items()}
    out = {}
    for link, visuals in urdf.visuals.items():
        tris = [read_stl(path) * scale @ origin[:3, :3].T + origin[:3, 3] for path, origin, scale in visuals]
        out[link] = fit_boxes(np.concatenate(tris))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({
        "stamp": stamp, "segment_mm": SEGMENT_MM,
        "links": {link: [[v.tolist() for v in box] for box in boxes] for link, boxes in out.items()},
    }))
    return out


def rod_box(p_from, p_to, radius):
    """Box around the segment p_from-p_to with a square cross-section of half size radius."""
    d = p_to - p_from
    z = d / np.linalg.norm(d)
    x = np.cross(z, [1.0, 0.0, 0.0] if abs(z[0]) < 0.9 else [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    return (p_from + p_to) / 2, np.column_stack([x, np.cross(z, x), z]), np.array(
        [radius, radius, np.linalg.norm(d) / 2])


class ArmModel:
    """One arm's boxes and where its base sits in the world.

    probe_angles are the turret angles (deg) that bring each slot to the working position, as in
    sim_server.PROBE_ANGLES. The URDF's probe_tip is where the working probe's tip is.
    """

    def __init__(self, base=None, urdf_path=URDF, probe_angles=(0.0,), probe=None,
                 clearance_mm=CLEARANCE_MM, probe_clearance_mm=PROBE_CLEARANCE_MM, tip_link="probe_tip"):
        self.urdf = Urdf(urdf_path)
        self.base = np.eye(4) if base is None else base
        boxes = [(link, *box, clearance_mm / 2000.0)
                 for link, bs in mesh_boxes(self.urdf).items() for box in bs]
        p = dict(PROBE, **(probe or {}))
        _, _, turret_link, tip_origin, _ = self.urdf.joints[tip_link]
        turret_axis = self.urdf.joints[turret_link][4]
        for angle in probe_angles:
            # Turning the turret by `angle` brings this slot to where probe_tip is at angle 0.
            t = axis_angle(turret_axis, -math.radians(angle)) @ tip_origin
            tip, back = t[:3, 3], -t[:3, 2]  # the probe runs back from the tip along -z
            needle_end = tip + back * p["needle_mm"] / 1000.0
            head_end = needle_end + back * p["head_mm"] / 1000.0
            pad = probe_clearance_mm / 2000.0
            boxes.append((turret_link, *rod_box(tip, needle_end, p["needle_radius_mm"] / 1000.0), pad))
            boxes.append((turret_link, *rod_box(needle_end, head_end, p["head_radius_mm"] / 1000.0), pad))
        self.owner = [b[0] for b in boxes]
        self.links = sorted(set(self.owner))
        self.index = np.array([self.links.index(o) for o in self.owner])
        self.c = np.array([b[1] for b in boxes])
        self.axes = np.array([b[2] for b in boxes])
        self.half = np.array([b[3] for b in boxes])
        self.pad = np.array([b[4] for b in boxes])

    def world_boxes(self, q):
        """(centres, axes, halves, pads) of every box in the world for joint angles q {name: rad}."""
        poses = self.urdf.link_poses(q, self.base)
        stack = np.array([poses[link] for link in self.links])[self.index]
        rot, off = stack[:, :3, :3], stack[:, :3, 3]
        return (np.einsum("nij,nj->ni", rot, self.c) + off, rot @ self.axes, self.half, self.pad)

    def describe(self):
        """Boxes in their link frames, for the viewer to hang on the URDF links."""
        return [{"link": o, "center": c.tolist(), "axes": a.tolist(), "half": h.tolist()}
                for o, c, a, h in zip(self.owner, self.c, self.axes, self.half)]


def _stack(boxes_list):
    return tuple(np.concatenate(parts) for parts in zip(*boxes_list))


def box_gaps(a, b):
    """Lower bound on the gap (m) between box i of a and box j of b, by separating axes. <= 0 on
    overlap. a and b are (centres, axes, halves) with the same leading length."""
    (ca, ra, ha), (cb, rb, hb) = a, b
    ua, ub = np.swapaxes(ra, 1, 2), np.swapaxes(rb, 1, 2)    # rows are the box axes
    cross = np.cross(ua[:, :, None, :], ub[:, None, :, :]).reshape(-1, 9, 3)
    axes = np.concatenate([ua, ub, cross], axis=1)             # (n, 15, 3)
    norm = np.linalg.norm(axes, axis=2)
    ok = norm > 1e-6
    axes = axes / np.where(ok, norm, 1.0)[..., None]
    proj_a = np.abs(np.einsum("nlx,nkx->nlk", axes, ua)) @ ha[:, :, None]
    proj_b = np.abs(np.einsum("nlx,nkx->nlk", axes, ub)) @ hb[:, :, None]
    gap = np.abs(np.einsum("nlx,nx->nl", axes, cb - ca)) - proj_a[..., 0] - proj_b[..., 0]
    return np.where(ok, gap, -np.inf).max(axis=1)


def margins(mine, theirs):
    """For each sample in `mine` (a list of world-box tuples), the smallest of gap - required pad
    against every box in `theirs` (also a list). Negative means too close. Also returns, for the
    worst pair, the indices (my sample, my box, their box)."""
    n_mine, n_theirs = len(mine[0][0]), len(theirs[0][0])
    a, b = _stack(mine), _stack(theirs)
    rad_a, rad_b = np.linalg.norm(a[2], axis=1), np.linalg.norm(b[2], axis=1)
    per_box, which = np.full(len(a[0]), np.inf), np.zeros(len(a[0]), dtype=int)
    chunk = max(1, 2_000_000 // len(a[0]))                 # bound the pairwise arrays' memory
    for lo in range(0, len(b[0]), chunk):
        sl = slice(lo, lo + chunk)
        need = a[3][:, None] + b[3][None, sl]
        best = (np.linalg.norm(a[0][:, None] - b[0][None, sl], axis=2)
                - rad_a[:, None] - rad_b[None, sl] - need)
        i, j = np.nonzero(best < 0)                        # bounding spheres too close: look closer
        if len(i):
            gap = box_gaps((a[0][i], a[1][i], a[2][i]),
                           (b[0][sl][j], b[1][sl][j], b[2][sl][j])) - need[i, j]
            best[i, j] = np.maximum(best[i, j], gap)
        k = best.argmin(axis=1)
        m = best[np.arange(len(k)), k]
        closer = m < per_box
        per_box[closer], which[closer] = m[closer], k[closer] + lo
    per_sample = per_box.reshape(len(mine), n_mine).min(axis=1)
    ia = int(np.argmin(per_box))
    return per_sample, (ia // n_mine, ia % n_mine, int(which[ia]) % n_theirs)


def sample_count(dq, step_mm=SWEEP_MM):
    """Pieces to cut a joint move dq (radians, any shape) into so that no point of the arm moves
    more than step_mm in one piece. A point r from a joint axis moves r * angle."""
    return max(1, math.ceil(REACH_M * 1000.0 * float(np.sum(np.abs(dq))) / step_mm))


def interpolate(path, step_mm=SWEEP_MM):
    """Joint-space samples along a list of {name: rad} poses, at most step_mm of motion apart."""
    out = [path[0]]
    for q0, q1 in pairwise(path):
        names = sorted(set(q0) | set(q1))
        n = sample_count([q1.get(k, 0.0) - q0.get(k, 0.0) for k in names], step_mm)
        out += [{k: q0.get(k, 0.0) + (q1.get(k, 0.0) - q0.get(k, 0.0)) * i / n for k in names}
                for i in range(1, n + 1)]
    return out


def base_pose(xyz_m, yaw_deg):
    return transform(xyz_m, (0.0, 0.0, math.radians(yaw_deg)))


def load_workcell(path=WORKCELL):
    cfg = json.loads(Path(path).read_text())
    for a in cfg["arms"].values():
        a["base"] = base_pose(a.get("base_xyz_m", [0, 0, 0]), a.get("base_yaw_deg", 0.0))
        a["urdf"] = ROOT / a.get("urdf", "urdf/debugger_arm.urdf")
    return cfg


@contextlib.contextmanager
def file_lock(path, timeout_s=LOCK_TIMEOUT_S):
    """Exclusive lock shared by every arm server on this host."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as f:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"workcell lock {path} held for over {timeout_s:.0f} s")
                time.sleep(0.005)
        try:
            yield
        finally:
            if os.name == "nt":
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


class Workcell:
    """This arm's view of the shared bench: every arm's model and every other arm's last poses."""

    def __init__(self, me, path=WORKCELL, probe_angles=(0.0,), state_dir=None):
        cfg = load_workcell(path)
        if me not in cfg["arms"]:
            raise SystemExit(f"arm {me!r} isn't in {path} (arms: {', '.join(cfg['arms'])})")
        self.me = me
        self.dir = Path(state_dir or ROOT / cfg.get("state_dir", "data/workcell"))
        self.settle_s = cfg.get("settle_s", SETTLE_S)
        self.models = {
            name: ArmModel(a["base"], a["urdf"], a.get("probe_angles", probe_angles), a.get("probe"),
                           cfg.get("clearance_mm", CLEARANCE_MM),
                           cfg.get("probe_clearance_mm", PROBE_CLEARANCE_MM))
            for name, a in cfg["arms"].items()}
        self.others = [n for n in self.models if n != me]
        self.msg = ""

    def lock(self):
        return file_lock(self.dir / "lock")

    def read(self, name):
        try:
            return json.loads((self.dir / f"{name}.json").read_text())
        except (OSError, ValueError):
            return None

    @staticmethod
    def recent(record, now=None):
        """The goals an arm may still be travelling between: the latest, and those commanded
        within its settle time (0 for a simulated arm, which is always at its goal)."""
        now = time.time() if now is None else now
        poses = [q for t, q in record["history"] if now - t <= record.get("settle_s", SETTLE_S)]
        return poses + [record["joints"]]

    def publish(self, joints, now=None):
        """Record this arm's new goal. Call with the lock held."""
        now = time.time() if now is None else now
        old = self.read(self.me)
        hist = [h for h in (old or {}).get("history", []) if now - h[0] <= self.settle_s]
        record = {"arm": self.me, "time": now, "settle_s": self.settle_s, "joints": joints,
                  "history": hist + [[now, joints]]}
        tmp = self.dir / f"{self.me}.json.tmp"
        tmp.write_text(json.dumps(record))
        for attempt in range(20):  # Windows refuses to replace a file another process has open
            try:
                os.replace(tmp, self.dir / f"{self.me}.json")
                return
            except PermissionError:
                time.sleep(0.01)
        raise PermissionError(f"could not write {self.dir / self.me}.json")

    def obstacles(self):
        """{arm: world boxes along the path through its recent goals} for every other arm, or
        None and a reason when an arm's pose is unknown."""
        out = {}
        for name in self.others:
            record = self.read(name)
            if record is None:
                return None, f"arm {name} hasn't reported its pose (start its server)"
            out[name] = [self.models[name].world_boxes(q) for q in interpolate(self.recent(record))]
        return out, ""

    def sample_margins(self, configs, obstacles):
        """Margin (m) at each of my configs against every other arm; and what is closest."""
        mine = [self.models[self.me].world_boxes(q) for q in configs]
        worst, where = np.full(len(configs), np.inf), ""
        for name, theirs in obstacles.items():
            m, (_, ia, ib) = margins(mine, theirs)
            if m.min() < worst.min():
                where = f"{self.models[self.me].owner[ia]} vs arm {name} {self.models[name].owner[ib]}"
            worst = np.minimum(worst, m)
        return worst, where

    def check(self, path, obstacles):
        """Is it safe to sweep through the poses in path (first = where the arm is now)?

        Allowed when the sweep keeps the clearance everywhere, or, if the arm already starts too
        close (e.g. both servers started in overlapping poses), when it never gets any closer.
        """
        m, where = self.sample_margins(interpolate(path), obstacles)
        m -= SWEEP_MM / 2000.0
        if m.min() >= 0 or (m[0] < 0 and m.min() >= m[0] - 1e-4 and m[-1] > m[0]):
            return True, float(m.min()), where
        return False, float(m.min()), where

    def commit(self, current, target):
        """Check the sweep from where this arm may be to target and, if clear, claim target."""
        try:
            with self.lock():
                obstacles, why = self.obstacles()
                if obstacles is None:
                    self.msg = f"blocked: {why}"
                    return False
                # The servos head straight for the new goal from wherever they are, which is
                # somewhere between the oldest goal they may still be leaving and the latest.
                record = self.read(self.me)
                recent = self.recent(record) if record else [current]
                ok, margin, where = self.check([recent[-1], target], obstacles)
                if ok and len(recent) > 1:
                    ok, margin, where = self.check([recent[0], target], obstacles)
                if not ok:
                    self.msg = f"blocked: {-margin * 1000:.0f} mm too close to the other arm ({where})"
                    return False
                self.publish(target)
                return True
        except TimeoutError as e:
            self.msg = f"blocked: {e}"
            return False

    def status(self, joints):
        """Current margin to the other arms, for the HUD. None when an arm hasn't reported."""
        with self.lock():
            obstacles, why = self.obstacles()
        if obstacles is None:
            return {"margin_mm": None, "nearest": why}
        m, where = self.sample_margins([joints], obstacles)
        return {"margin_mm": float(m[0]) * 1000.0, "nearest": where}

    def snapshot(self):
        """Every arm's base, boxes and latest joints, for the viewer."""
        arms = {}
        with self.lock():
            records = {name: self.read(name) for name in self.models}
        for name, model in self.models.items():
            r = records[name]
            arms[name] = {"base": model.base.tolist(), "boxes": model.describe(),
                          "joints": r["joints"] if r else None, "age_s": time.time() - r["time"] if r else None}
        return {"me": self.me, "arms": arms}


def rrt_connect(start, goal, lo, hi, edge_ok, rng, step=0.2, iters=1500):
    """Joint-space path start -> goal (arrays) where edge_ok(a, b) holds on every edge, or None."""
    trees = [([start], [-1]), ([goal], [-1])]

    def extend(tree, x):
        nodes, parents = tree
        i = int(np.argmin([np.linalg.norm(n - x) for n in nodes]))
        d = x - nodes[i]
        dist = np.linalg.norm(d)
        new = x if dist <= step else nodes[i] + d * (step / dist)
        if not edge_ok(nodes[i], new):
            return None
        nodes.append(new)
        parents.append(i)
        return len(nodes) - 1

    def trace(tree, i):
        nodes, parents = tree
        out = []
        while i != -1:
            out.append(nodes[i])
            i = parents[i]
        return out

    for it in range(iters):
        a, b = trees if it % 2 == 0 else trees[::-1]
        i = extend(a, rng.uniform(lo, hi))
        if i is None:
            continue
        target = a[0][i]
        while True:  # pull the other tree straight toward the new node
            j = extend(b, target)
            if j is None:
                break
            if np.linalg.norm(b[0][j] - target) < 1e-9:
                path = trace(a, i)[::-1] + trace(b, j)[1:]
                return path if np.allclose(path[0], start) else path[::-1]
    return None


def shortcut(path, edge_ok, rng, tries=60):
    path = list(path)
    for _ in range(tries):
        if len(path) < 3:
            break
        i, j = sorted(rng.choice(len(path), 2, replace=False))
        if j - i > 1 and edge_ok(path[i], path[j]):
            path = path[:i + 1] + path[j:]
    return path


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    model = ArmModel(probe_angles=(-120.0, 0.0, 120.0))
    vol = 8 * np.prod(model.half, axis=1).sum()
    print(f"{len(model.owner)} boxes over {len(model.links)} links, {vol * 1e6:.0f} cm^3 in all")
    for o, h, p in zip(model.owner, model.half, model.pad):
        print(f"  {o[:36]:36s} {' x '.join(f'{v * 2000:5.1f}' for v in h)} mm  pad {p * 1000:.1f} mm")
