import json
import shutil
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from test_dmm_knob import FakeRunner

from troubleshooter.instruments.dmm_display import FrameReading, Reader, load_layout
from troubleshooter.instruments.dmm_knob import DmmKnob
from troubleshooter.settings import Settings

ROOT = Path(__file__).parents[1]
LAYOUT = ROOT / "arm" / "bench" / "display" / "dm6000ar.json"
# Wrist camera at "above", meter on DC volts showing 0004 mV. Glare hides the decimal point
# and the last digit's bottom segment.
FRAME = ROOT / "tests" / "fixtures" / "dm6000ar_0004_mV.jpg"


@pytest.fixture(scope="module")
def reader():
    layout, reference = load_layout(LAYOUT)
    return Reader(layout, reference)


def test_reads_the_calibration_frame(reader):
    r = reader.read(cv2.imread(str(FRAME)))
    assert r.status == "ok", r.failure_reason
    assert r.text == "0004"
    assert r.value == pytest.approx(0.4)  # 600.0 mV range: one decimal, inferred under glare
    assert r.decimal_inferred and "dp3" in r.unknown_segments
    assert reader.unit(r.annunciators) == "mV"
    assert r.annunciators["AUTO"] is True


def test_follows_a_moved_meter_and_refuses_a_missing_one(reader):
    image = cv2.imread(str(FRAME))
    shifted = np.roll(image, (25, -30), axis=(0, 1))
    r = reader.read(shifted)
    assert r.status == "ok" and r.text == "0004"
    assert r.shift_px == (-30, 25)
    gone = reader.read(np.full_like(image, 128))
    assert gone.status == "unreadable" and gone.failure_reason == "display_not_found"


def test_glare_never_reads_as_off(reader):
    image = cv2.imread(str(FRAME))
    layout = reader.layout
    x = int(np.mean([c[0] for c in layout.corners]))
    image[100:330, x - 120 : x + 140] = 255  # blow out most of the LCD
    r = reader.read(image)
    assert r.status == "unreadable"  # unknown segments, not a confident wrong number


def test_vote_needs_agreement(reader):
    def frame(text, value):
        return FrameReading("ok", text=text, value=value, annunciators={"m": True, "V": True})

    same = [frame("0004", 0.4)] * 3 + [frame("0003", 0.3), FrameReading("unreadable")]
    voted = reader.vote(same, min_agree=3)
    assert voted["status"] == "ok" and voted["value"] == 0.4 and voted["unit"] == "mV"
    assert voted["agreeing_frames"] == 3 and voted["readable_frames"] == 4
    wandering = [frame("0004", 0.4), frame("0003", 0.3), frame("0002", 0.2)]
    assert reader.vote(wandering, min_agree=2)["failure_reason"] == "unstable_reading"
    assert reader.vote([FrameReading("unreadable")], 1)["failure_reason"] == "display_unreadable"


def make_knob(tmp_path, recorded=None, **display):
    work = tmp_path / "work"
    work.mkdir()
    frame = tmp_path / "live.jpg"
    shutil.copy(FRAME, frame)
    body = {
        "bench_dir": "work",
        "settings": {
            "off": {"pose": "knob_closed", "units": []},
            "resistance": {"pose": "knob_resistance", "units": ["Ω", "kΩ", "MΩ"]},
            "dc_voltage": {"pose": "knob_vdc", "units": ["V", "mV"]},
        },
        "display": {
            "layout": str(LAYOUT),
            "frame_file": "live.jpg",
            "frames": 1,
            "min_agree": 1,
            **display,
        },
    }
    path = tmp_path / "dmm_knob.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    if recorded:
        (work / "dmm_setting.json").write_text(json.dumps({"setting": recorded, "source": "t"}))
    settings = Settings(dmm_knob_config=str(path), captures_dir=str(tmp_path / "captures"))
    return DmmKnob(settings), work


@pytest.mark.parametrize(
    ("recorded", "check"),
    [("dc_voltage", "consistent"), ("resistance", "mismatch"), (None, "unknown")],
)
async def test_read_dmm_through_the_service(tmp_path, recorded, check):
    knob, work = make_knob(tmp_path, recorded)
    with FakeRunner(work):  # parked above the meter
        result, png = await knob.read_display()
    assert result["status"] == "ok", result
    assert result["value"] == pytest.approx(0.4) and result["unit"] == "mV"
    assert result["value_base"] == pytest.approx(0.0004) and result["unit_base"] == "V"
    assert result["setting_check"] == check
    assert png and png.startswith(b"\x89PNG")
    assert Path(result["image_file"]).exists()


async def test_read_dmm_refuses_off_the_view_pose(tmp_path):
    knob, work = make_knob(tmp_path)
    with FakeRunner(work, q=[0.0] * 7):  # parked at home: the camera can't see the display
        result, png = await knob.read_display()
    assert result["failure_reason"] == "arm_not_at_view_pose" and png is None
    (tmp_path / "b").mkdir()
    no_runner, _ = await make_knob(tmp_path / "b")[0].read_display()
    assert no_runner["failure_reason"] == "bench_runner_unavailable"


async def test_read_dmm_refuses_a_stale_frame_file(tmp_path):
    import os
    import time

    knob, work = make_knob(tmp_path)
    old = time.time() - 60
    os.utime(tmp_path / "live.jpg", (old, old))
    with FakeRunner(work):
        result, _ = await knob.read_display()
    assert result["failure_reason"] == "frame_file_stale"
