"""Board and rig config loading, including KiCad Edge.Cuts outlines."""

from pathlib import Path

import numpy as np
import pytest
from hardware_troubleshooter_registration.board_model import outline_from_kicad
from hardware_troubleshooter_registration.config import (
    board_from_dict,
    load_board_config,
    load_rig_config,
    rig_from_dict,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "config"

# 60 x 40 mm board with 3 mm corner radii, KiCad 7 syntax, origin at (100, 50) mm.
ROUNDED_PCB = """(kicad_pcb (version 20221018) (generator pcbnew)
  (gr_line (start 103 50) (end 157 50) (stroke (width 0.1) (type default)) (layer "Edge.Cuts"))
  (gr_arc (start 157 50) (mid 159.1213 50.8787) (end 160 53) (layer "Edge.Cuts"))
  (gr_line (start 160 53) (end 160 87) (layer "Edge.Cuts"))
  (gr_arc (start 160 87) (mid 159.1213 89.1213) (end 157 90) (layer "Edge.Cuts"))
  (gr_line (start 157 90) (end 103 90) (layer "Edge.Cuts"))
  (gr_arc (start 103 90) (mid 100.8787 89.1213) (end 100 87) (layer "Edge.Cuts"))
  (gr_line (start 100 53) (end 100 87) (layer "Edge.Cuts"))
  (gr_arc (start 100 53) (mid 100.8787 50.8787) (end 103 50) (layer "Edge.Cuts"))
  (gr_circle (center 110 60) (end 111.6 60) (layer "Edge.Cuts"))
  (gr_line (start 0 0) (end 10 0) (layer "F.SilkS"))
  (footprint "MountingHole" (at 110 60) (fp_line (start 0 0) (end 1 1) (layer "Edge.Cuts")))
)
"""


def test_kicad_rounded_outline_gives_virtual_corners():
    outline = outline_from_kicad(ROUNDED_PCB)
    assert len(outline.edges) == 4
    expected = {(0.1, -0.05), (0.16, -0.05), (0.16, -0.09), (0.1, -0.09)}
    got = {tuple(np.round(c, 6)) for c in outline.corners}
    assert got == expected
    x, y = outline.corners[:, 0], outline.corners[:, 1]
    assert 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) > 0  # counter-clockwise
    assert 0.05 < outline.trim_fraction < 0.2


def test_kicad_rect_and_board_config(tmp_path):
    (tmp_path / "board.kicad_pcb").write_text(
        '(kicad_pcb (gr_rect (start 0 0) (end 50 30) (layer "Edge.Cuts")))'
    )
    data = {
        "board_id": "rect",
        "thickness_m": 0.0016,
        "outline_source": "kicad",
        "kicad_pcb": "board.kicad_pcb",
        "features": [
            {"shape": "circle", "center_m": [0.004, -0.004], "radius_m": 0.0015, "expect": "darker"}
        ],
    }
    board, problems = board_from_dict(data, tmp_path)
    assert problems == []
    assert board.frame_id == "board_rect"
    assert len(board.edges) == 4


def test_open_kicad_outline_is_rejected():
    with pytest.raises(ValueError, match="closed"):
        outline_from_kicad(
            '(kicad_pcb (gr_line (start 0 0) (end 10 0) (layer "Edge.Cuts")) (gr_line (start 10 0) (end 10 5) (layer "Edge.Cuts")))'
        )


def test_example_configs_are_entirely_unset():
    board, problems = load_board_config(str(EXAMPLES / "board.example.yaml"))
    assert board is None
    assert "unconfigured:board.thickness_m" in problems
    assert "unconfigured:board.outline_source" in problems
    assert "unconfigured:board.features" in problems
    rig, problems = load_rig_config(str(EXAMPLES / "rig.example.yaml"))
    assert problems == []
    assert rig.missing_for_calibration() == ["rig.bench_plane", "rig.base_frame"]
    assert len(rig.missing_for_tracking()) == 7


def test_missing_config_paths_are_unconfigured():
    assert load_board_config("")[1] == ["unconfigured:board_config"]
    assert load_rig_config(None)[1] == ["unconfigured:rig_config"]


def test_invalid_rig_values_are_rejected_not_coerced():
    rig, problems = rig_from_dict(
        {
            "bench_plane": {"normal": [0, 0, 0], "distance_m": -1},
            "segmentation": {"board_is_brighter": "yes"},
            "thresholds": {"move_translation_m": 0, "feature_min_contrast": 5},
        }
    )
    assert rig.bench_normal is None and rig.bench_distance_m is None
    assert rig.board_is_brighter is None
    assert rig.thresholds.move_translation_m is None
    assert rig.thresholds.feature_min_contrast is None
    assert len(problems) == 5
