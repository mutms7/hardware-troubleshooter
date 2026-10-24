import sqlite3
from pathlib import Path

from .contracts import Measurement


class Journal:
    """Record dispatch before ROS; unresolved operations survive process restarts."""

    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS operations ("
            "request_id TEXT PRIMARY KEY, target TEXT NOT NULL, result TEXT, "
            "uncertain INTEGER NOT NULL DEFAULT 1)"
        )
        self.db.commit()

    def blocked(self) -> bool:
        return (
            self.db.execute("SELECT 1 FROM operations WHERE uncertain = 1 LIMIT 1").fetchone()
            is not None
        )

    def get(self, request_id: str):
        return self.db.execute(
            "SELECT target, result, uncertain FROM operations WHERE request_id = ?", (request_id,)
        ).fetchone()

    def begin(self, request_id: str, target: str):
        self.db.execute(
            "INSERT INTO operations(request_id, target) VALUES (?, ?)", (request_id, target)
        )
        self.db.commit()

    def finish(self, result: Measurement):
        self.db.execute(
            "UPDATE operations SET result = ?, uncertain = ? WHERE request_id = ?",
            (result.model_dump_json(), int(result.execution_uncertain), result.request_id),
        )
        self.db.commit()

    def close(self):
        self.db.close()
