import cv2
import numpy as np
import pytest
from hardware_troubleshooter_vision import transforms
from hardware_troubleshooter_vision.capture import JointSample
from hardware_troubleshooter_vision.config import ContactConfig, SceneConfig, ThermalConfig
from hardware_troubleshooter_vision.contact import TouchdownDetector
from hardware_troubleshooter_vision.pads import Pad, rect_outline
from hardware_troubleshooter_vision.scene import SceneGuard
from hardware_troubleshooter_vision.thermal import cells_on_board, hot_spot, thermal_homography

# --- touchdown ---------------------------------------------------------------------------

CONTACT = ContactConfig(joints=(2, 3), torque_step_nm=0.4, persistence_s=0.03, baseline_s=0.2)


def sample(stamp, j2=1.0, j3=-0.5):
    return JointSample(stamp, (0.0,) * 7, (0.0,) * 7, (0.1, j2, j3, 0.0, 0.0, 0.0, 0.0))


def descent(detector, profile):
    """Feed samples every 5 ms from t=1.0; profile(t) -> (j2, j3). Returns the touchdown."""
    for i in range(200):
        t = 1.0 + 0.005 * i
        found = detector.update(sample(t, *profile(t)))
        if found:
            return found
    return None


@pytest.fixture
def detector():
    detector = TouchdownDetector(CONTACT)
    history = [sample(0.7 + 0.005 * i, 1.0 + 0.02 * (-1) ** i) for i in range(60)]
    assert detector.start(history, 1.0) is None
    return detector


def test_sustained_step_is_touchdown(detector):
    found = descent(detector, lambda t: (1.0 + (0.6 if t >= 1.3 else 0.0), -0.5))
    assert found.joint == 2
    assert found.stamp == pytest.approx(1.3, abs=0.006)
    assert found.delta_nm == pytest.approx(0.6, abs=0.03)


def test_a_spike_is_not_touchdown(detector):
    assert descent(detector, lambda t: (1.0 + (0.8 if 1.3 <= t < 1.31 else 0.0), -0.5)) is None


def test_either_watched_joint_counts(detector):
    found = descent(detector, lambda t: (1.0, -0.5 - (0.5 if t >= 1.2 else 0.0)))
    assert found.joint == 3 and found.delta_nm < 0


def test_touchdown_needs_a_baseline():
    detector = TouchdownDetector(CONTACT)
    assert detector.start([], 1.0).startswith("no_torque_baseline")
    no_torque = [JointSample(0.9, (0.0,) * 7, (0.0,) * 7)]
    assert "carry no torque" in detector.start(no_torque, 1.0)
    assert TouchdownDetector(ContactConfig()).start([], 1.0).startswith("unconfigured:contact.")
    with pytest.raises(RuntimeError):
        detector.update(sample(1.0))


# --- scene guard -------------------------------------------------------------------------

SCENE = SceneConfig(diff_threshold=25, min_blob_area_px=150, board_margin_px=10)


def bench(seed=0):
    rng = np.random.default_rng(seed)
    image = cv2.GaussianBlur(rng.integers(60, 200, (480, 640)).astype(np.uint8), (0, 0), 6)
    return cv2.normalize(image, None, 70, 190, cv2.NORM_MINMAX)


@pytest.fixture
def guard():
    guard = SceneGuard(SCENE)
    guard.set_reference(bench())
    return guard


def test_unchanged_scene_is_clear_even_with_a_lighting_change(guard):
    assert guard.check(bench()).clear
    brighter = np.clip(bench().astype(float) * 1.15, 0, 255).astype(np.uint8)
    assert guard.check(brighter).clear


def test_new_object_is_an_obstruction(guard):
    image = bench()
    cv2.circle(image, (500, 100), 15, 20, -1)
    result = guard.check(image)
    assert not result.clear
    assert result.reasons[0].startswith("scene_obstructed:")
    assert result.blobs[0][0] > SCENE.min_blob_area_px


def test_small_change_is_ignored(guard):
    image = bench()
    cv2.circle(image, (500, 100), 3, 20, -1)
    assert guard.check(image).clear


def test_arm_and_board_regions_are_left_out(guard):
    image = bench()
    cv2.circle(image, (500, 100), 15, 20, -1)  # under the arm
    cv2.rectangle(image, (100, 300), (200, 380), 250, -1)  # the board's LEDs changed
    arm = np.zeros((480, 640), np.uint8)
    cv2.circle(arm, (500, 100), 30, 255, -1)
    board = [[105, 305], [195, 305], [195, 375], [105, 375]]
    assert guard.check(image, arm_mask=arm, board_polygon_px=board).clear


def test_guard_refusals(guard):
    assert SceneGuard(SCENE).check(bench()).reasons == ["no_reference:capture the empty rig first"]
    assert SceneGuard(SceneConfig()).check(bench()).reasons[0].startswith("unconfigured:scene.")
    assert guard.check(np.zeros((10, 10), np.uint8)).reasons[0].startswith("invalid:")
    everything = np.full((480, 640), 255, np.uint8)
    assert guard.check(bench(), arm_mask=everything).reasons[0].startswith("scene_unobservable")


# --- thermal -----------------------------------------------------------------------------

THERMAL = ThermalConfig(homography=np.eye(3), min_delta_c=3.0)
BOARD_OUTLINE = np.array([[0.0, 0.0], [0.1, 0.0], [0.1, 0.06], [0.0, 0.06]])


def thermal_scene(camera):
    """Fixed camera 0.4 m above the board; thermal cell (c, r) sees board (0.012c+0.008, 0.052-0.007r)."""
    board_from_camera = transforms.transform(np.diag([1.0, -1.0, -1.0]), [0.05, 0.03, 0.4])
    camera_from_board = transforms.invert(board_from_camera)

    def board_of(cells):
        cells = np.asarray(cells, float)
        return np.column_stack([0.008 + 0.012 * cells[:, 0], 0.052 - 0.007 * cells[:, 1]])

    corners = np.array([[0, 0], [7, 0], [7, 7], [0, 7], [3, 4]], float)
    points = board_of(corners)
    ideal = camera.project_ideal(
        transforms.apply(camera_from_board, np.column_stack([points, np.zeros(5)]))
    )
    return camera_from_board, thermal_homography(corners, ideal), board_of


def test_cells_land_where_the_homography_says(camera):
    camera_from_board, homography, board_of = thermal_scene(camera)
    positions = cells_on_board(homography, camera, camera_from_board)
    rows, cols = np.mgrid[0:8, 0:8]
    expected = board_of(np.column_stack([cols.ravel(), rows.ravel()])).reshape(8, 8, 2)
    assert positions == pytest.approx(expected, abs=1e-6)


def test_hot_spot_names_nearby_pads(camera):
    camera_from_board, homography, board_of = thermal_scene(camera)
    positions = cells_on_board(homography, camera, camera_from_board)
    celsius = np.full((8, 8), 30.0)
    celsius[2, 5] = 41.0  # row 2, column 5
    hot = board_of([[5, 2]])[0]
    pads = [
        Pad("U3.1", hot + [0.001, 0.0], rect_outline(hot + [0.001, 0.0], (0.001, 0.001), 0.0)),
        Pad("R9.2", hot + [0.03, 0.0], rect_outline(hot + [0.03, 0.0], (0.001, 0.001), 0.0)),
        Pad("C2.1", hot + [0.0, -0.004], rect_outline(hot + [0.0, -0.004], (0.001, 0.001), 0.0)),
    ]
    spot, reason = hot_spot(celsius, positions, BOARD_OUTLINE, pads, THERMAL, nearby=2)
    assert reason is None
    assert spot.position_m == pytest.approx(hot)
    assert spot.celsius == 41.0 and spot.delta_c == pytest.approx(11.0)
    assert spot.nearby_pads == ("U3.1", "C2.1")


def test_flat_board_has_no_hot_spot(camera):
    camera_from_board, homography, _ = thermal_scene(camera)
    positions = cells_on_board(homography, camera, camera_from_board)
    assert hot_spot(np.full((8, 8), 30.0), positions, BOARD_OUTLINE, [], THERMAL) == (None, None)


def test_hot_cell_off_the_board_is_ignored(camera):
    camera_from_board, homography, _ = thermal_scene(camera)
    positions = cells_on_board(homography, camera, camera_from_board)
    small = np.array([[0.0, 0.0], [0.03, 0.0], [0.03, 0.03], [0.0, 0.03]])  # only the lower left
    celsius = np.full((8, 8), 30.0)
    celsius[0, 7] = 60.0  # far upper right, off this outline
    spot, reason = hot_spot(celsius, positions, small, [], THERMAL)
    assert spot is None and reason is None


def test_thermal_refusals(camera):
    with pytest.raises(ValueError, match="at least four"):
        thermal_homography([[0, 0]], [[0, 0]])
    spot, reason = hot_spot(
        np.zeros((8, 8)), np.zeros((8, 8, 2)), BOARD_OUTLINE, [], ThermalConfig()
    )
    assert spot is None and reason.startswith("unconfigured:thermal.")
    far = np.full((8, 8, 2), 5.0)
    spot, reason = hot_spot(np.zeros((8, 8)), far, BOARD_OUTLINE, [], THERMAL)
    assert reason.startswith("thermal_off_board")
