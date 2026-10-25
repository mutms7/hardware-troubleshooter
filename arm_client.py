"""Tiny client for the arm server, for Jev (or any script) to call.

Jogs are in millimetres in the wrist-camera frame: +right, +up (image up), +in (toward the
board along the probe). Each call is capped at 10 mm per axis; loop for bigger moves.

    from arm_client import Arm
    arm = Arm()
    arm.jog(right=2)            # nudge the probe right in the image
    arm.jog(up=-1, in_=0.5)
    arm.tilt(5)                 # tilt the probe 5 deg, tip stays put
    arm.select_tool("logic")    # turn the turret to a tool head: scope, logic or dmm
    arm.goto([-0.08, -0.18, -0.03])   # planned move around the other arm (needs --arm)
    arm.move_z(-1)              # tip straight down 1 mm, tilt held
    arm.stop()                  # hold position, cancel a goto
    arm.tof_height()            # tip height above the board from the ToF, mm, or None
    arm.sensors()               # ToF and the 8x8 thermal frame (needs --sensors)
"""
import json
import time
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

    def move_z(self, dz_mm, contact_mm=None):
        """Move the tip straight up (+) or down (-) by dz_mm, keeping the tilt the server holds
        (set by the last goto that ended 'ok'). Shortened near the board and stopped at the
        floor; state['tip'] says where the command ended up. contact_mm: a deliberate touch, the
        command may go this far below the touched-off board (stop on state(measure=True))."""
        body = {"d": [0, 0, dz_mm / 1000.0]}
        if contact_mm is not None:
            body["contact_mm"] = contact_mm
        req = urllib.request.Request(f"{self.url}/api/move", data=json.dumps(body).encode(),
                                     method="POST", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.load(r)

    def touch_off(self, tip_z=None, tof_mm=None, zero_tof=True):
        """Call while the probe tip is touching the board: sets the board height from the tip
        (or from tip_z, meters, e.g. the measured tip) and zeroes the ToF (or to tof_mm; or
        not at all with zero_tof=False)."""
        body = {"cmd": "touch_off", "zero_tof": zero_tof}
        if tip_z is not None:
            body["tip_z"] = tip_z
        if tof_mm is not None:
            body["tof_mm"] = tof_mm
        return self._post(body)

    def clear_touch_off(self):
        """Forget the board height and ToF zeros from touch_off (back to the startup floors)."""
        return self._post({"cmd": "clear_touch_off"})

    def tilt(self, deg):
        return self._post({"cmd": "tilt", "deg": deg})

    def select_tool(self, head):
        """Turn the turret so this head is at the working position. Check state['tool'] == head.

        Blocked (state['msg'] starts with 'blocked') unless the tip is well above the board.
        """
        return self._post({"cmd": "tool", "head": head})

    def point_down(self, settle_s=3.0, max_calls=8):
        """Turn the probe to straight down (tilt 0) with the tip held in place. The server does it in safe increments
        (a joint swings more than the tilt does); this repeats until it says 'ok' or it is blocked. Check state['msg']."""
        import time
        s = {}
        for _ in range(max_calls):
            s = self._post({"cmd": "point_down"})
            if not str(s.get("msg", "")).startswith("ok: tilt now"):
                break
            time.sleep(settle_s)
        return s

    def goto(self, tip, phi=None, frame="arm", wait=True, timeout=60.0):
        """Move the tip to [x, y, z] meters (this arm's frame, or the shared bench frame with
        frame="world") along a path planned around the other arm: up, across, straight down.

        Returns the final state. Check state['msg']: 'ok' when it arrived; 'blocked...' or
        'unreachable...' if it never started; 'goto stopped...' if the other arm moved into the
        way part-way (re-plan from where it stopped).
        """
        body = {"cmd": "goto", "tip": list(tip), "frame": frame}
        if phi is not None:
            body["phi"] = phi
        s = self._post(body)
        deadline = time.monotonic() + timeout
        while wait and s["goto"]["active"] and time.monotonic() < deadline:
            time.sleep(0.1)
            s = self.state()
        return s

    def stop(self):
        return self._post({"cmd": "stop"})

    def sensors(self):
        """The sensor head: {"enabled", "tof", "thermal", "probe", ...}. thermal["celsius"] is
        8x8 degrees C. probe["height_mm"] is the tip height the ToF implies; probe["why"] says
        why it's None. enabled is False unless the server was started with --sensors."""
        with urllib.request.urlopen(f"{self.url}/api/sensors", timeout=10) as r:
            return json.load(r)

    def tof_height(self):
        """Tip height above the board from the ToF, in mm, or None (not zeroed, no target, stale,
        tilted, or no sensor head). Zero it with touch_off while the tip touches the board."""
        tof = self.state().get("tof")
        return tof and tof["height_mm"]
