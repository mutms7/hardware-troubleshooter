import dataclasses
import math

import numpy as np
import pytest
from hardware_troubleshooter_arm.config import TipPixel
from hardware_troubleshooter_vision import transforms
from hardware_troubleshooter_vision.alignment import align, correction_in_base
from hardware_troubleshooter_vision.config import AlignmentConfig
from hardware_troubleshooter_vision.pads import pads_from_kicad, pads_from_specs
from wrist_synthetic import (
    ALIGNMENT,
    TARGET,
    TARGET_CENTER,
    board_from_camera,
    cluster_pads,
    project,
    render,
    row_pads,
    shifted,
)

STANDOFF = 0.03
TIP_OFFSET = np.array([0.0010, -0.0006])  # where the tip would land, relative to the target


class Hover:
    """True and predicted wrist poses at the hover, plus the calibrated tip pixel."""

    def __init__(self, camera, pads, tip_offset=TIP_OFFSET, board_error=(0.0007, 0.0004, 0.005)):
        self.camera, self.pads = camera, pads
        landing = TARGET_CENTER + tip_offset
        self.true = board_from_camera(landing[0] + 0.004, landing[1] - 0.002, 0.06, math.radians(5))
        self.predicted = shifted(self.true, *board_error)  # registration was slightly off
        camera_from_board = transforms.invert(self.true)
        self.tip = TipPixel(tuple(project(camera, camera_from_board, landing)), STANDOFF)
        self.landing = landing

    def measure(self, config=ALIGNMENT, standoff=STANDOFF, target=TARGET, tip="calibrated"):
        image = render(self.camera, transforms.invert(self.true), self.pads)
        return align(
            image,
            self.camera,
            transforms.invert(self.predicted),
            self.pads,
            target,
            self.tip if tip == "calibrated" else tip,
            standoff,
            config,
        )

    def move_tool(self, board_delta):
        """The arm moves the tool (and the wrist camera with it) by board_delta; FK sees it."""
        move = transforms.transform(np.eye(3), [*board_delta, 0.0])
        self.true = move @ self.true
        self.predicted = move @ self.predicted
        self.landing = self.landing + board_delta


def test_error_is_target_minus_landing_despite_registration_error(camera):
    hover = Hover(camera, cluster_pads())
    result = hover.measure()
    assert result.ok, result.reasons
    assert result.score > 0.9 and result.margin > ALIGNMENT.min_uniqueness_margin
    assert result.error_board_m == pytest.approx(-TIP_OFFSET, abs=5e-5)
    assert not result.aligned


def test_correction_closes_the_loop(camera):
    hover = Hover(camera, cluster_pads())
    for _ in range(3):
        result = hover.measure()
        assert result.ok, result.reasons
        if result.aligned:
            break
        hover.move_tool(result.error_board_m)
    assert result.aligned
    assert np.linalg.norm(hover.landing - TARGET_CENTER) < ALIGNMENT.converged_m


def test_correction_is_expressed_in_the_arm_base(camera):
    hover = Hover(camera, cluster_pads())
    result = hover.measure()
    base_from_board = transforms.transform(
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]]), [1, 2, 3]
    )
    correction = correction_in_base(result, base_from_board)
    assert correction == pytest.approx([-result.error_board_m[1], result.error_board_m[0], 0.0])


def test_periodic_pads_are_ambiguous(camera):
    result = Hover(camera, row_pads()).measure()
    assert not result.ok
    assert result.reasons[0].startswith("match_ambiguous")
    with pytest.raises(ValueError):
        correction_in_base(result, np.eye(4))


def test_wrong_polarity_is_weak(camera):
    config = dataclasses.replace(ALIGNMENT, pads_are_brighter=False)
    result = Hover(camera, cluster_pads()).measure(config)
    assert not result.ok and result.reasons[0].startswith("match_weak")


def test_board_further_off_than_the_search(camera):
    hover = Hover(camera, cluster_pads(), board_error=(0.004, 0.0, 0.0))
    result = hover.measure()
    assert not result.ok
    assert result.reasons[0].split(":")[0] in (
        "match_at_search_edge",
        "match_weak",
        "match_ambiguous",
    )


def test_standoff_must_match_the_tip_pixel(camera):
    result = Hover(camera, cluster_pads()).measure(standoff=STANDOFF + 0.01)
    assert not result.ok and result.reasons[0].startswith("standoff_mismatch")


def test_refusals(camera):
    hover = Hover(camera, cluster_pads())
    assert hover.measure(target="Q7.1").reasons == ["unknown_target:Q7.1"]
    assert hover.measure(tip=None).reasons == ["tip_pixel_uncalibrated"]
    assert hover.measure(config=AlignmentConfig()).reasons[0].startswith("unconfigured:alignment.")


def test_target_out_of_view(camera):
    hover = Hover(camera, cluster_pads())
    hover.predicted = shifted(hover.predicted, 0.05, 0.0)
    result = hover.measure()
    assert result.reasons[0].startswith("target_not_in_view")


def test_flat_copper_has_no_contrast(camera):
    from hardware_troubleshooter_vision.pads import Pad, rect_outline

    big = rect_outline(TARGET_CENTER, (0.02, 0.02), 0.0)
    result = Hover(camera, [Pad(TARGET, TARGET_CENTER, big)]).measure()
    assert result.reasons[0].startswith("no_pad_contrast")


KICAD = """(kicad_pcb (version 20221018)
  (footprint "R_0603" (layer "F.Cu") (at 100 50 90)
    (property "Reference" "R1")
    (pad "1" smd roundrect (at -0.8 0 90) (size 0.8 0.9) (layers "F.Cu" "F.Paste" "F.Mask"))
    (pad "2" smd roundrect (at 0.8 0 90) (size 0.8 0.9) (layers "F.Cu" "F.Paste" "F.Mask")))
  (footprint "TP" (layer "B.Cu") (at 10 10)
    (fp_text reference "TP1" (at 0 0))
    (pad "1" smd circle (at 0 0) (size 1 1) (layers "B.Cu")))
  (module "Hole" (layer "F.Cu") (at 20 20)
    (fp_text reference "H1" (at 0 0))
    (pad "" np_thru_hole circle (at 0 0) (size 3 3) (layers "*.Cu" "*.Mask"))
    (pad "1" thru_hole oval (at 2 0) (size 1.2 2) (layers "*.Cu" "*.Mask"))))
"""


def test_kicad_pads():
    pads = {pad.name: pad for pad in pads_from_kicad(KICAD)}
    assert set(pads) == {"R1.1", "R1.2", "H1.1"}  # back copper and the bare hole are skipped
    # Footprint turned 90 deg counter-clockwise: local -x moves to board -y.
    assert pads["R1.1"].center == pytest.approx([0.1, -0.05 - 0.0008])
    assert pads["R1.2"].center == pytest.approx([0.1, -0.05 + 0.0008])
    extent = np.ptp(pads["R1.1"].outline, axis=0)
    assert extent == pytest.approx([0.0009, 0.0008])  # the pad's 90 deg swaps width and height
    assert pads["H1.1"].center == pytest.approx([0.022, -0.02])
    assert np.ptp(pads["H1.1"].outline, axis=0) == pytest.approx([0.0012, 0.002], abs=1e-5)


def test_measured_pads():
    pads, problems = pads_from_specs(
        [
            {"name": "TP1", "shape": "rect", "center_m": [0.01, 0.02], "size_m": [0.001, 0.002]},
            {"name": "TP2", "shape": "circle", "center_m": [0.03, 0.02], "radius_m": 0.0005},
            {"name": "TP1", "shape": "circle", "center_m": [0, 0], "radius_m": 0.001},
            {"name": "TP3", "shape": "star", "center_m": [0, 0]},
            {"name": "TP4", "shape": "rect", "center_m": [0, 0], "size_m": [0, 0.001]},
        ]
    )
    assert [p.name for p in pads] == ["TP1", "TP2"]
    assert len(problems) == 3
