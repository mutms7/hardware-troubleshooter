import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "config" / "workcell.example.json"
sys.path.insert(0, str(ROOT))

collision = pytest.importorskip("collision")


def random_box(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return rng.uniform(-0.1, 0.1, 3), q, rng.uniform(0.005, 0.04, 3)


def box_corners(c, axes, half):
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    return c + (signs * half) @ axes.T


def test_box_gap_never_exceeds_true_distance():
    rng = np.random.default_rng(0)
    for _ in range(200):
        a, b = random_box(rng), random_box(rng)
        gap = collision.box_gaps(*[tuple(np.array([v]) for v in box) for box in (a, b)])[0]
        # Points on b's surface: the true distance can be no more than the nearest one.
        u = rng.uniform(-1, 1, (3000, 3))
        face = rng.integers(0, 3, 3000)
        u[np.arange(3000), face] = np.sign(u[np.arange(3000), face])
        surface = b[0] + (u * b[2]) @ b[1].T
        inside_a = np.abs((surface - a[0]) @ a[1]) - a[2]
        nearest = np.linalg.norm(np.maximum(inside_a, 0), axis=1).min()
        assert gap <= nearest + 1e-9


def test_overlapping_boxes_have_no_gap():
    c, axes, half = np.zeros(3), np.eye(3), np.full(3, 0.01)
    gap = collision.box_gaps((c[None], axes[None], half[None]), (c[None] + 0.015, axes[None], half[None]))
    assert gap[0] <= 0


def test_mesh_boxes_contain_every_vertex():
    model = collision.ArmModel()
    for link, visuals in model.urdf.visuals.items():
        pts = np.vstack([(collision.read_stl(p) * s @ o[:3, :3].T + o[:3, 3]).reshape(-1, 3)
                         for p, o, s in visuals])
        idx = [i for i, owner in enumerate(model.owner) if owner == link]
        outside = np.min([np.max(np.abs((pts - model.c[i]) @ model.axes[i]) - model.half[i], axis=1)
                          for i in idx], axis=0)
        assert outside.max() < 1e-9, link


def test_rrt_connect_goes_around_a_wall():
    # 2-D: a wall at x = 0.5 with a gap only at y > 0.8.
    def ok(p):
        return not (0.45 < p[0] < 0.55 and p[1] < 0.8)

    def edge_ok(a, b):
        return all(ok(a + (b - a) * t) for t in np.linspace(0, 1, 30))

    rng = np.random.default_rng(0)
    path = collision.rrt_connect(np.array([0.1, 0.1]), np.array([0.9, 0.1]), np.zeros(2), np.ones(2),
                                 edge_ok, rng, step=0.1)
    assert path is not None
    assert np.allclose(path[0], [0.1, 0.1]) and np.allclose(path[-1], [0.9, 0.1])
    assert all(edge_ok(a, b) for a, b in pairwise(path))


# --- two arms ----------------------------------------------------------------------------------


@pytest.fixture
def bench(monkeypatch, tmp_path):
    pytest.importorskip("ikpy")
    pytest.importorskip("scservo_sdk")
    import sim_server

    def workcell(name):
        wc = collision.Workcell(name, EXAMPLE, sim_server.PROBE_ANGLES, state_dir=tmp_path)
        wc.settle_s = 0.0
        return wc

    a, b = workcell("A"), workcell("B")
    monkeypatch.setattr(sim_server, "WORKCELL", a)
    sim, other = sim_server.Sim(), sim_server.Sim()

    def park_b(tip):
        q, err, _ = other.solve(np.array(tip), 0.0, other.q)
        assert err < 0.5
        with b.lock():
            b.publish(other.joints(q))

    return sim_server, sim, a, park_b


def margin_along(wc, joints_path, step_mm=0.5):
    obstacles, _ = wc.obstacles()
    m, _ = wc.sample_margins(collision.interpolate(joints_path, step_mm), obstacles)
    return m.min()


def test_fk_matches_ikpy(bench):
    _, sim, wc, _ = bench
    poses = wc.models["A"].urdf.link_poses(sim.joints())
    assert np.allclose(poses["probe_tip"][:3, 3], sim.pose(sim.q)[0], atol=1e-9)


def test_moves_are_blocked_until_the_other_arm_reports(bench):
    _, sim, _, park_b = bench
    sim.move([0, -0.002, 0])
    assert "hasn't reported" in sim.msg
    park_b([-0.0787, -0.14, -0.02])  # B at home, well clear
    sim.move([0, -0.002, 0])
    assert sim.msg == "ok"


def test_jogging_into_the_other_arm_stops_short(bench):
    _, sim, wc, park_b = bench
    park_b([0.03, -0.20, 0.0])  # B's tip at about (-0.08, -0.20) in A's frame
    sim.announce()
    path = [sim.joints()]
    for _ in range(60):
        sim.move([0, -0.003, 0])
        if sim.msg != "ok":
            break
        path.append(sim.joints())
    assert "too close to the other arm" in sim.msg
    assert margin_along(wc, path) >= 0


def test_goto_plans_around_the_other_arm(bench):
    _, sim, wc, park_b = bench
    park_b([0.069, -0.179, 0.018])
    start = sim.plan_to(np.array([-0.181, -0.129, -0.03]))
    assert start is not None, sim.msg
    sim.q = start[-1]
    sim.announce()
    steps = sim.plan_to(np.array([0.058, -0.202, -0.03]))
    assert steps is not None, sim.msg
    assert np.linalg.norm(sim.pose(steps[-1])[0] - [0.058, -0.202, -0.03]) < 1e-3
    path = [sim.joints(q) for q in [sim.q] + steps]
    assert margin_along(wc, path) >= collision.SWEEP_MM / 2000.0
    # The straight joint-space line between the same poses would hit B: it really went around.
    assert margin_along(wc, [path[0], path[-1]]) < 0
    for q in steps:  # each waypoint passes the live check the executor makes
        assert sim.claim(q), sim.msg
        sim.q = q


def test_goto_refuses_a_goal_inside_the_other_arm(bench):
    _, sim, wc, park_b = bench
    park_b([0.03, -0.20, 0.0])
    sim.announce()
    b_tip = np.linalg.inv(wc.models["A"].base) @ wc.models["B"].base @ [0.03, -0.20, 0.0, 1.0]
    assert sim.plan_to(b_tip[:3]) is None
    assert "blocked" in sim.msg


def test_register_arms_recovers_the_base_pose():
    import register_arms

    rng = np.random.default_rng(1)
    truth = collision.base_pose([-0.11, -0.40, 0.003], 177.0)
    in_b = rng.uniform(-0.2, 0.2, (4, 3))
    in_a = (truth[:3, :3] @ in_b.T).T + truth[:3, 3] + rng.normal(0, 0.0003, (4, 3))
    yaw, shift = register_arms.fit_yaw_shift(in_b, in_a)
    assert abs(np.degrees(yaw) - 177.0) < 0.3
    assert np.allclose(shift, truth[:3, 3], atol=0.002)
