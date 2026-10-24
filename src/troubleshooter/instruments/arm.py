"""HTTP link to one arm's sim_server.py: read state and turn the tool turret. No jogging."""

import json
import urllib.error
import urllib.request


class ArmLink:
    def __init__(self, url: str, timeout_s: float = 5.0):
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s

    def _open(self, request) -> dict:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.load(response)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise ConnectionError(str(exc)) from exc

    def state(self) -> dict:
        return self._open(f"{self.url}/api/state")

    def select_tool(self, head: str) -> dict:
        body = json.dumps({"cmd": "tool", "head": head}).encode()
        request = urllib.request.Request(
            f"{self.url}/api/cmd",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        return self._open(request)
