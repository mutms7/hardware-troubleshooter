"""Tiny client for the arm server, for Jev (or any script) to call.

Jogs are in millimetres in the wrist-camera frame: +right, +up (image up), +in (toward the
board along the probe). Each call is capped at 10 mm per axis; loop for bigger moves.

    from arm_client import Arm
    arm = Arm()
    arm.jog(right=2)            # nudge the probe right in the image
    arm.jog(up=-1, in_=0.5)
    arm.tilt(5)                 # tilt the probe 5 deg, tip stays put
    arm.stop()                  # hold position (real arm only)
"""
import json
import urllib.request


class Arm:
    def __init__(self, url="http://127.0.0.1:8765"):
        self.url = url.rstrip("/")

    def _post(self, body):
        req = urllib.request.Request(f"{self.url}/api/cmd", data=json.dumps(body).encode(),
                                     method="POST", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.load(r)

    def state(self, measure=False):
        """measure=True also reads the joint encoders (real arm, armed only) and adds
        follow_err_mm: how far the real tip lags the commanded one. A growing gap while the
        command keeps moving down means the probe is blocked, i.e. touching something."""
        with urllib.request.urlopen(f"{self.url}/api/state" + ("?measure=1" if measure else ""), timeout=10) as r:
            return json.load(r)

    def jog(self, right=0.0, up=0.0, in_=0.0, height_mm=None):
        """Returns the new state. Check state['msg'] == 'ok' to know the move happened.

        height_mm: height above the board as the camera sees it. When given it is trusted over
        the URDF model, so pass it whenever the tracker can estimate it. The server shortens big
        jogs near the board (see state['note']) and never goes below the floor.
        """
        body = {"cmd": "jog", "right_mm": right, "up_mm": up, "in_mm": in_}
        if height_mm is not None:
            body["height_mm"] = height_mm
        return self._post(body)

    def touch_off(self):
        """Call while the probe tip is touching the board: sets the board height from the tip."""
        return self._post({"cmd": "touch_off"})

    def tilt(self, deg):
        return self._post({"cmd": "tilt", "deg": deg})

    def stop(self):
        return self._post({"cmd": "stop"})
