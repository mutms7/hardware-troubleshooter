"""Shared bits for teach.py / run.py: command inbox, state file, joint formatting.

Only one process can hold the USB adapter, so the long-running tool reads commands from
files dropped in <workdir>/cmd/ (send.py writes them) and publishes its state to
<workdir>/state.json.
"""

import json
import os
import time
from pathlib import Path

N_JOINTS = 7


class Inbox:
    def __init__(self, workdir: Path):
        self.dir = workdir / "cmd"
        self.dir.mkdir(parents=True, exist_ok=True)
        for stale in self.dir.glob("*.cmd"):
            stale.unlink()

    def poll(self) -> list[str]:
        cmds = []
        for f in sorted(self.dir.glob("*.cmd")):
            try:
                cmds.append(f.read_text(encoding="utf-8").strip())
            finally:
                f.unlink(missing_ok=True)
        return [c for c in cmds if c]


def write_state(workdir: Path, state: dict) -> None:
    tmp = workdir / "state.json.tmp"
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    # Windows refuses the replace while a reader has state.json open; skip this update
    # rather than kill the process (the next tick writes a fresh one).
    for _ in range(5):
        try:
            os.replace(tmp, workdir / "state.json")
            return
        except PermissionError:
            time.sleep(0.01)


def log(workdir: Path, msg: str) -> None:
    now = time.time()
    line = f"{time.strftime('%H:%M:%S', time.localtime(now))}.{int(now % 1 * 1000):03d} {msg}"
    print(line, flush=True)
    with open(workdir / "log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def fmt(q) -> str:
    return " ".join(f"J{i + 1}={x:+.3f}" for i, x in enumerate(q))
