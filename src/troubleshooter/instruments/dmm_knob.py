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
    # Display units this setting can show, e.g. ["V", "mV"]. A reading in one of them
    # confirms the knob is on this setting; [] means the display goes blank (off).
    units: list[str] | None = None


class DisplayConfig(Strict):
    layout: str  # dmm_display layout JSON; relative paths are from the knob config file
    # Frames come from the camera (OpenCV index), or from a file another program keeps
    # rewriting, e.g. a live viewer holding the camera (Windows allows one owner).
    camera_index: int | None = Field(default=None, ge=0)
    frame_file: str | None = None
    width: int = Field(default=1920, gt=0)
    height: int = Field(default=1080, gt=0)
    frames: int = Field(default=5, ge=1, le=30)
    min_agree: int = Field(default=3, ge=1, le=30)
    frame_interval_s: float = Field(default=0.4, ge=0, le=5)  # the LCD updates ~3 times a second
    max_frame_age_s: float = Field(default=2.0, gt=0, le=30)  # frame_file only

    @model_validator(mode="after")
    def check_source(self):
        if (self.camera_index is None) == (self.frame_file is None):
            raise ValueError("set exactly one of camera_index and frame_file")
        if self.min_agree > self.frames:
            raise ValueError("min_agree can't exceed frames")
        return self


class KnobConfig(Strict):
    bench_dir: str  # run.py's work folder; relative paths are from this config file
    settings: dict[str, KnobSetting] = Field(min_length=2)
    move_timeout_s: float = Field(default=60.0, gt=0, le=300)
    display: DisplayConfig | None = None  # unset: read_dmm refuses

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


def _beside(config_file: Path, path: str) -> Path:
    p = Path(path)
    return (p if p.is_absolute() else config_file.parent / p).resolve()


def load_knob(path: str | Path) -> tuple[KnobConfig, Path]:
    path = Path(path)
    config = KnobConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))
    if config.display is not None:
        config.display.layout = str(_beside(path, config.display.layout))
        if config.display.frame_file:
            config.display.frame_file = str(_beside(path, config.display.frame_file))
    return config, _beside(path, config.bench_dir)


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
        self._reader = None  # dmm_display.Reader, built on first read (needs OpenCV)
        self.captures_dir = Path(settings.captures_dir)

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

    # --- display --------------------------------------------------------------------------

    async def read_display(self) -> tuple[dict, bytes | None]:
        """Read the meter's display through the wrist camera. Returns (result, LCD PNG)."""
        started = time.monotonic()
        reading_id = f"dmm-{time.strftime('%Y%m%d-%H%M%S')}-{os.urandom(3).hex()}"
        result: dict = {"reading_id": reading_id, "status": "failed", "failure_reason": None}
        png = None
        try:
            if self.config is None:
                raise Refused(self.config_error)
            if self.config.display is None:
                raise Refused("configuration_required:dmm_display")
            if self.lock.locked():
                raise Refused("busy")
            async with self.lock:
                result, png = await asyncio.to_thread(self._read, reading_id)
        except Refused as refused:
            result["failure_reason"] = refused.reason
            result["diagnostics"] = refused.diagnostics
        result["duration_ms"] = int((time.monotonic() - started) * 1000)
        return result, png

    def _reader_for(self):
        if self._reader is None:
            try:
                from .dmm_display import Reader, load_layout
            except ImportError as exc:
                raise Refused("opencv_unavailable", {"detail": str(exc)}) from exc
            try:
                layout, reference = load_layout(self.config.display.layout)
                self._reader = Reader(layout, reference)
            except ImportError as exc:
                raise Refused("opencv_unavailable", {"detail": str(exc)}) from exc
            except (OSError, ValueError, ValidationError) as exc:
                raise Refused("configuration_invalid:dmm_display", {"detail": str(exc)}) from exc
        return self._reader

    def _frames(self) -> list:
        import cv2

        display = self.config.display
        frames = []
        if display.frame_file:
            path, last = Path(display.frame_file), None
            deadline = time.monotonic() + display.frames * max(display.frame_interval_s, 0.5) * 3
            while len(frames) < display.frames and time.monotonic() < deadline:
                try:
                    mtime = path.stat().st_mtime
                except FileNotFoundError as exc:
                    raise Refused("frame_file_missing") from exc
                if time.time() - mtime > display.max_frame_age_s:
                    raise Refused("frame_file_stale", {"age_s": round(time.time() - mtime, 1)})
                if mtime != last:  # only fresh frames count; the writer replaces the file
                    image = cv2.imread(str(path))
                    if image is not None:
                        frames.append(image)
                        last = mtime
                time.sleep(0.05)
            return frames
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        camera = cv2.VideoCapture(display.camera_index, backend)
        try:
            if not camera.isOpened():
                raise Refused("camera_unavailable", {"camera_index": display.camera_index})
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, display.width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, display.height)
            for _ in range(15):  # let auto exposure settle
                camera.read()
            for i in range(display.frames):
                if i:
                    time.sleep(display.frame_interval_s)
                ok, image = camera.read()
                if ok:
                    frames.append(image)
        finally:
            camera.release()
        return frames

    def _read(self, reading_id: str) -> tuple[dict, bytes | None]:
        import cv2

        state = self._runner()
        parked = self._parked(state)
        if parked != "above":
            raise Refused(
                "arm_not_at_view_pose", {"arm_parked": parked, "target": state.get("target")}
            )
        reader = self._reader_for()
        frames = self._frames()
        if not frames:
            raise Refused("no_frames")
        voted = reader.vote([reader.read(f) for f in frames], self.config.display.min_agree)
        canvas = voted.pop("canvas", None)
        png = None
        if canvas is not None:
            ok, buf = cv2.imencode(".png", canvas)
            png = buf.tobytes() if ok else None

        recorded = self._recorded()["setting"]
        expected = (
            self.config.settings[recorded].units if recorded in self.config.settings else None
        )
        result = {"reading_id": reading_id, "knob_setting_recorded": recorded, **voted}
        if voted["status"] == "ok":
            unit = voted["unit"]
            blank = voted["text"] == ""
            if not blank and not voted["overload"] and unit is None:
                result["status"], result["failure_reason"] = "inconclusive", "unit_unknown"
            elif voted["value"] is not None and unit:
                prefix = {"m": 1e-3, "u": 1e-6, "k": 1e3, "M": 1e6}.get(unit[0], 1.0)
                result["value_base"] = voted["value"] * (prefix if len(unit) > 1 else 1.0)
                result["unit_base"] = unit[1:] if len(unit) > 1 and unit[0] in "mukM" else unit
            if expected is None:
                result["setting_check"] = "unknown"
            elif (blank and expected == []) or (unit in expected):
                result["setting_check"] = "consistent"
            else:
                result["setting_check"] = "mismatch"
                result["note"] = (
                    f"The display doesn't match the recorded knob setting {recorded!r} "
                    f"(expects {expected or 'a blank display'}); the knob may be elsewhere."
                )
        result["image_file"] = self._save_reading(reading_id, result, png)
        return result, png

    def _save_reading(self, reading_id: str, result: dict, png: bytes | None) -> str | None:
        try:
            self.captures_dir.mkdir(parents=True, exist_ok=True)
            if png:
                (self.captures_dir / f"{reading_id}.png").write_bytes(png)
            path = self.captures_dir / f"{reading_id}.json"
            path.write_text(json.dumps(result, default=str), encoding="utf-8")
            return str((self.captures_dir / f"{reading_id}.png").resolve()) if png else None
        except OSError:
            return None
