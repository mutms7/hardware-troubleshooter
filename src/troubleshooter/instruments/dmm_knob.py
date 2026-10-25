"""Turn the multimeter's rotary switch with the bench arm (arm/bench/run.py).

The runner owns the USB-CAN adapter and every motion safeguard (hover, waypoint waits, contact
guard). This module only talks to it through its work folder: it drops a `turn <from> <to>`
command in work/cmd/ and follows work/state.json and work/log.txt until the arm is back above
the meter. It never reads the display, so a finished turn isn't a verified setting.
"""

import asyncio
import json
import math
import os
import re
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..settings import Settings

NAME_PATTERN = re.compile(r"[A-Za-z0-9_\-]{1,32}\Z")
RECORD = "dmm_setting.json"  # last setting the knob was turned to, kept in the bench folder
STALE_S = 2.0  # the runner rewrites state.json every 50 ms
START_S = 3.0  # a turn the runner accepted shows up in its log within this
HOME_TOL = 0.08  # rad, same as run.py's "at home"
ABOVE_TOL = 0.06  # rad, arm joints within this of "above" count as parked above the meter
STATE_READ_TRIES = 10


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class KnobSetting(Strict):
    pose: str  # full knob pose taught in arm/bench/work/teach.yaml, e.g. knob_resistance
    description: str = ""


class KnobConfig(Strict):
    bench_dir: str  # run.py's work folder; relative paths are from this config file
    settings: dict[str, KnobSetting] = Field(min_length=2)
    move_timeout_s: float = Field(default=60.0, gt=0, le=300)

    @model_validator(mode="after")
    def check_names(self):
        for name, setting in self.settings.items():
            if not NAME_PATTERN.fullmatch(name) or not NAME_PATTERN.fullmatch(setting.pose):
                raise ValueError(f"invalid setting or pose name {name!r}")
        return self


class Refused(Exception):
    def __init__(self, reason: str, diagnostics: dict | None = None, moved: bool = False):
        super().__init__(reason)
        self.reason, self.diagnostics, self.moved = reason, diagnostics, moved


def load_knob(path: str | Path) -> tuple[KnobConfig, Path]:
    path = Path(path)
    config = KnobConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))
    bench = Path(config.bench_dir)
    return config, (bench if bench.is_absolute() else path.parent / bench).resolve()


def _near(q, target, tol) -> bool:
    return all(math.isfinite(x) and abs(x - t) <= tol for x, t in zip(q, target))


class DmmKnob:
    def __init__(self, settings: Settings):
        self.config: KnobConfig | None = None
        self.bench: Path | None = None
        self.config_error: str | None = None
        if not settings.dmm_knob_config:
            self.config_error = "configuration_required:dmm_knob"
        else:
            try:
                self.config, self.bench = load_knob(settings.dmm_knob_config)
            except (OSError, ValueError, ValidationError):
                self.config_error = "configuration_invalid:dmm_knob"
        self.lock = asyncio.Lock()

    # --- bench folder -------------------------------------------------------------------

    def _read_state(self) -> dict | None:
        # run.py replaces state.json atomically; Windows briefly refuses opens during that.
        for _ in range(STATE_READ_TRIES):
            try:
                return json.loads((self.bench / "state.json").read_text(encoding="utf-8"))
            except FileNotFoundError:
                return None
            except (OSError, ValueError):
                time.sleep(0.01)
        return None

    def _runner(self) -> dict:
        """The live runner state, or Refused when it isn't running and holding the arm."""
        state = self._read_state()
        if state is None or state.get("mode") != "run" or time.time() - state["t"] > STALE_S:
            raise Refused("bench_runner_unavailable")
        if state.get("state") != "ENABLED":
            raise Refused("bench_runner_not_enabled", {"state": state.get("state")})
        return state

    def _send(self, command: str) -> None:
        inbox = self.bench / "cmd"
        inbox.mkdir(exist_ok=True)
        tmp = inbox / f"{time.time_ns()}.tmp"
        tmp.write_text(command, encoding="utf-8")
        tmp.rename(tmp.with_suffix(".cmd"))

    def _log_size(self) -> int:
        try:
            return (self.bench / "log.txt").stat().st_size
        except FileNotFoundError:
            return 0

    def _log_since(self, offset: int) -> list[str]:
        try:
            with open(self.bench / "log.txt", "rb") as log:
                log.seek(offset)
                return log.read().decode("utf-8", "replace").splitlines()
        except FileNotFoundError:
            return []

    def _recorded(self) -> dict:
        try:
            return json.loads((self.bench / RECORD).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"setting": None, "source": None, "updated": None}

    def _record(self, setting: str | None, source: str) -> None:
        body = {"setting": setting, "source": source, "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
        tmp = self.bench / f"{RECORD}.tmp"
        tmp.write_text(json.dumps(body), encoding="utf-8")
        os.replace(tmp, self.bench / RECORD)

    def _parked(self, state: dict) -> str | None:
        """'above' or 'home' when the arm is clear of the meter, else None."""
        q = state.get("q") or []
        above = (state.get("points") or {}).get("above")
        if above and _near(q[:6], above[:6], ABOVE_TOL):
            return "above"
        if _near(q[:6], [0.0] * 6, HOME_TOL):
            return "home"
        return None

    # --- tools --------------------------------------------------------------------------

    def status(self) -> dict:
        if self.config is None:
            return {"status": "failed", "failure_reason": self.config_error}
        recorded = self._recorded()
        runner: dict = {"running": False}
        try:
            state = self._runner()
            runner = {
                "running": True,
                "target": state.get("target"),
                "busy": bool(state.get("queue")),
                "arm_parked": self._parked(state),
            }
        except Refused as refused:
            runner["detail"] = refused.reason
        return {
            "status": "ok",
            "failure_reason": None,
            "settings": {name: s.description for name, s in self.config.settings.items()},
            "current_setting": recorded["setting"],
            "current_setting_source": recorded["source"],
            "current_setting_updated": recorded["updated"],
            "runner": runner,
            "busy": self.lock.locked(),
            "setting_verified": False,
            "note": "current_setting is the last commanded setting, not a reading of the knob.",
        }

    async def set_setting(self, setting: str, current_setting: str | None = None) -> dict:
        started = time.monotonic()
        result = {
            "status": "failed",
            "failure_reason": None,
            "setting": setting,
            "previous_setting": current_setting,
            "moved": False,
            "setting_verified": False,
        }
        try:
            if self.config is None:
                raise Refused(self.config_error)
            if self.lock.locked():
                raise Refused("busy")
            async with self.lock:
                result.update(await asyncio.to_thread(self._turn, setting, current_setting))
            result["status"] = "ok"
            result["note"] = (
                "The arm turned the knob and is waiting above the meter. The display isn't "
                "read, so the setting isn't verified."
            )
        except Refused as refused:
            result["failure_reason"] = refused.reason
            result["moved"] = refused.moved
            result["diagnostics"] = refused.diagnostics
            result["knob_state"] = "unknown" if refused.moved else "unchanged"
        result["duration_ms"] = int((time.monotonic() - started) * 1000)
        return result

    def _turn(self, setting: str, current_setting: str | None) -> dict:
        choices = sorted(self.config.settings)
        if setting not in self.config.settings:
            raise Refused("unknown_setting", {"settings": choices})
        source = "operator"
        if current_setting is None:
            current_setting, source = self._recorded()["setting"], "recorded"
        if current_setting is None:
            raise Refused(
                "current_setting_unknown",
                {"note": "ask the operator where the knob is and pass it as current_setting"},
            )
        if current_setting not in self.config.settings:
            raise Refused("unknown_setting", {"settings": choices, "current": current_setting})
        if source == "operator":
            self._record(current_setting, "operator")
        if current_setting == setting:
            return {"previous_setting": current_setting, "moved": False}

        state = self._runner()
        if state.get("queue"):
            raise Refused("bench_busy", {"target": state.get("target")})
        if self._parked(state) is None:
            raise Refused("arm_not_clear", {"target": state.get("target"), "q": state.get("q")})
        frm = self.config.settings[current_setting].pose
        to = self.config.settings[setting].pose
        offset, sent = self._log_size(), time.time()
        self._send(f"turn {frm} {to}")

        # A turn is 8 waypoints. Completion needs a state that showed this turn's queue: the log
        # line for a new waypoint lands before the state file catches up, so the first state
        # read after it can still be the previous turn's "arrived above, queue empty".
        moved = queue_seen = False
        deadline = time.monotonic() + self.config.move_timeout_s
        while time.monotonic() < deadline:
            time.sleep(0.1)
            for line in self._log_since(offset):
                if "turn REFUSED" in line or "unknown command" in line:
                    raise Refused("bench_refused", {"log": line}, moved=moved)
                if "GUARD:" in line:
                    self._record(None, "contact_guard")
                    raise Refused("contact_guard", {"log": line}, moved=True)
                if " goto " in line:
                    moved = True
            state = self._read_state()
            if state is None or time.time() - state["t"] > STALE_S:
                self._record(None, "runner_lost")
                raise Refused("bench_runner_lost", moved=moved)
            if state.get("state") != "ENABLED":
                self._record(None, "fault")
                raise Refused(
                    "arm_fault", {"state": state.get("state"), "faults": state.get("faults")}, True
                )
            if not moved and time.time() - sent > START_S:
                raise Refused("bench_no_response")
            if state["t"] > sent and state.get("queue"):
                queue_seen = True
            done = state.get("target") == "above" and not state.get("queue") and state["arrived"]
            if queue_seen and done:
                self._record(setting, "set_dmm_setting")
                return {"previous_setting": current_setting, "moved": True, "arm": "above"}
        # Leave the runner holding where it stopped; a person decides what happens next.
        self._send("hold")
        self._record(None, "timeout")
        raise Refused("timeout", moved=moved)
