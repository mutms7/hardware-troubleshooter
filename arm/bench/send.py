"""Drop a command for teach.py / run.py: python send.py <workdir> "<command>"."""

import sys
import time
from pathlib import Path

inbox = Path(sys.argv[1]) / "cmd"
inbox.mkdir(parents=True, exist_ok=True)
tmp = inbox / f"{time.time_ns()}.tmp"
tmp.write_text(" ".join(sys.argv[2:]), encoding="utf-8")
tmp.rename(tmp.with_suffix(".cmd"))
