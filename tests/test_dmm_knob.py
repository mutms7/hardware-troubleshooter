import json
import threading
import time
from pathlib import Path

import pytest

from troubleshooter.instruments import dmm_knob
from troubleshooter.instruments.dmm_knob import DmmKnob
from troubleshooter.settings import Settings

ROOT = Path(__file__).parents[1]
POSES = {"knob_closed", "knob_resistance", "knob_vdc"}
ABOVE = [-0.16, -1.61, 1.52, -1.27, 0.10, -0.24, 0.0]


class FakeRunner:
    """Speaks arm/bench/run.py's work-folder protocol: cmd/*.cmd in, state.json and log.txt out.

    Like run.py, it logs a waypoint before the next state.json reflects it.
    """

    def __init__(self, work: Path, scenario: str = "success", q=None):
        self.work, self.scenario = work, scenario
        self.q = list(q or ABOVE)
        self.target, self.queue, self.arrived, self.state = "above", [], True, "ENABLED"
        self.commands: list[str] = []
        self.finished_turns = 0
        self.running = True
        (work / "cmd").mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self):
        self._write_state()
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.running = False
        self.thread.join(timeout=2)

    def _log(self, msg):
        with open(self.work / "log.txt", "a", encoding="utf-8") as log:
            log.write(f"{time.strftime('%H:%M:%S')}.000 {msg}\n")

    def _write_state(self):
        body = {
            "mode": "run",
            "t": time.time(),
            "state": self.state,
            "target": self.target,
            "arrived": self.arrived,
            "queue": list(self.queue),
            "q": self.q,
            "points": {"above": ABOVE, "home": [0.0] * 7},
        }
        tmp = self.work / "state.json.tmp"
        tmp.write_text(json.dumps(body), encoding="utf-8")
        for _ in range(5):  # like run.py: Windows refuses the replace while a reader has it open
            try:
                tmp.replace(self.work / "state.json")
                return
            except PermissionError:
                time.sleep(0.01)

    def _go(self, name):
        self._log(f"goto {name}: J1=+0.000")
        # The log line lands before the state file catches up. Lag longer than DmmKnob's
        # 100 ms poll, so a reader that trusts the log alone sees the previous turn's state.
        time.sleep(0.15)
        self.target, self.arrived = name, False

    def _loop(self):
        steps = 0
        while self.running:
            if self.scenario == "dead":
                time.sleep(0.02)
                continue
            for f in sorted((self.work / "cmd").glob("*.cmd")):
                cmd = f.read_text(encoding="utf-8")
                f.unlink()
                self.commands.append(cmd)
                parts = cmd.split()
                if parts[0] == "turn":
                    if self.scenario == "refuse" or not set(parts[1:]) <= POSES:
                        self._log("turn REFUSED: missing points")
                        continue
                    frm, to = parts[1:]
                    self.queue = [
                        f"above@{frm}",
                        f"{frm}@jaw_open",
                        f"{frm}@jaw_closed",
                        f"{to}@jaw_closed",
                        f"{to}@jaw_open",
                        f"above@{to}",
                        "above",
                    ]
                    self._go("above")
                    steps = 0
            if self.queue and not self.arrived:
                self.arrived = True  # every waypoint arrives one tick after it starts
            elif self.queue:
                steps += 1
                if self.scenario == "guard" and steps == 3:
                    self.queue.clear()
                    self._log("GUARD: J[1] torque [4.0] over [2.5] at knob_closed@jaw_closed")
                    self.target, self.arrived = "guard-hold", True
                elif self.scenario == "fault" and steps == 3:
                    self.state = "FAULT"
                elif self.scenario == "die" and steps == 3:
                    self.running = False
                    return
                else:
                    self._go(self.queue.pop(0))
                    if not self.queue:
                        self.finished_turns += 1
            elif self.target == "above" and not self.arrived:
                self.arrived, self.q = True, list(ABOVE)
            self._write_state()
            time.sleep(0.02)


def make_knob(tmp_path, **config) -> tuple[DmmKnob, Path]:
    work = tmp_path / "work"
    work.mkdir()
    body = {
        "bench_dir": "work",
        "settings": {
            "off": {"pose": "knob_closed", "description": "Meter off."},
            "resistance": {"pose": "knob_resistance"},
            "dc_voltage": {"pose": "knob_vdc"},
        },
        "move_timeout_s": 5,
        **config,
    }
    path = tmp_path / "dmm_knob.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return DmmKnob(Settings(dmm_knob_config=str(path))), work


def recorded(work: Path):
    return json.loads((work / "dmm_setting.json").read_text())["setting"]


def test_example_config_is_valid():
    config, bench = dmm_knob.load_knob(ROOT / "config" / "dmm_knob.example.json")
    assert set(config.settings) == {"off", "resistance", "dc_voltage"}
    assert bench == (ROOT / "arm" / "bench" / "work").resolve()
    taught = (ROOT / "arm" / "bench" / "work" / "teach.yaml").read_text(encoding="utf-8")
    for setting in config.settings.values():
        assert f"\n  {setting.pose}:" in taught, setting.pose  # every pose is actually taught


async def test_unconfigured_fails_closed():
    knob = DmmKnob(Settings())
    assert knob.status()["failure_reason"] == "configuration_required:dmm_knob"
    result = await knob.set_setting("off")
    assert result["failure_reason"] == "configuration_required:dmm_knob"
    assert result["moved"] is False


def test_invalid_config_fails_closed(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"bench_dir": "w", "settings": {"off": {"pose": "x"}}}))
    knob = DmmKnob(Settings(dmm_knob_config=str(path)))  # fewer than two settings
    assert knob.status()["failure_reason"] == "configuration_invalid:dmm_knob"


async def test_turns_from_operator_then_recorded_setting(tmp_path):
    knob, work = make_knob(tmp_path)
    with FakeRunner(work) as runner:
        status = knob.status()
        assert status["current_setting"] is None
        assert status["runner"] == {
            "running": True,
            "target": "above",
            "busy": False,
            "arm_parked": "above",
        }
        unknown = await knob.set_setting("resistance")
        assert unknown["failure_reason"] == "current_setting_unknown"
        assert unknown["moved"] is False and runner.commands == []

        first = await knob.set_setting("resistance", current_setting="off")
        assert first["status"] == "ok", first
        assert first["moved"] and first["setting_verified"] is False
        assert runner.commands == ["turn knob_closed knob_resistance"]
        assert runner.finished_turns == 1  # returned only after the whole route, not early
        assert recorded(work) == "resistance"

        second = await knob.set_setting("dc_voltage")  # from the recorded setting
        assert second["status"] == "ok", second
        assert second["previous_setting"] == "resistance"
        assert runner.commands[-1] == "turn knob_resistance knob_vdc"
        assert knob.status()["current_setting"] == "dc_voltage"


async def test_same_setting_does_not_move(tmp_path):
    knob, work = make_knob(tmp_path)
    with FakeRunner(work) as runner:
        result = await knob.set_setting("off", current_setting="off")
        assert result["status"] == "ok" and result["moved"] is False
        assert runner.commands == []


@pytest.mark.parametrize(
    ("scenario", "reason"),
    [("guard", "contact_guard"), ("fault", "arm_fault"), ("die", "bench_runner_lost")],
)
async def test_failure_after_moving_marks_knob_unknown(tmp_path, scenario, reason):
    knob, work = make_knob(tmp_path)
    with FakeRunner(work, scenario):
        result = await knob.set_setting("resistance", current_setting="off")
    assert result["status"] == "failed"
    assert result["failure_reason"] == reason
    assert result["moved"] is True and result["knob_state"] == "unknown"
    assert recorded(work) is None  # the next turn needs the operator's word


async def test_refused_turn_leaves_knob_unchanged(tmp_path):
    knob, work = make_knob(tmp_path)
    with FakeRunner(work, "refuse"):
        result = await knob.set_setting("resistance", current_setting="off")
    assert result["failure_reason"] == "bench_refused"
    assert result["moved"] is False and result["knob_state"] == "unchanged"
    assert recorded(work) == "off"


async def test_prechecks(tmp_path, monkeypatch):
    knob, work = make_knob(tmp_path)
    missing = await knob.set_setting("resistance", current_setting="off")
    assert missing["failure_reason"] == "bench_runner_unavailable"  # no state.json at all
    assert (await knob.set_setting("ohms", current_setting="off"))["failure_reason"] == (
        "unknown_setting"
    )
    with FakeRunner(work, q=[-0.27, -1.50, 1.09, -0.91, -0.01, 0.85, -0.17]) as runner:
        result = await knob.set_setting("resistance", current_setting="off")
        assert result["failure_reason"] == "arm_not_clear"  # gripping the knob, not parked
        assert runner.commands == []
    with FakeRunner(work, q=[0.0] * 7) as runner:  # parked at home is fine
        assert (await knob.set_setting("resistance", current_setting="off"))["status"] == "ok"
    monkeypatch.setattr(dmm_knob, "START_S", 0.3)
    with FakeRunner(work, "dead"):
        result = await knob.set_setting("off", current_setting="resistance")
    assert result["failure_reason"] in {"bench_runner_unavailable", "bench_no_response"}
    assert result["moved"] is False
