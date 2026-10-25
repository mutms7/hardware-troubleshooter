"""Maker SDK CanBackend over a candleLight / gs_usb adapter (WinUSB) via python-can."""

import logging
import os
import threading
from typing import Callable, Optional

import libusb_package

# pyusb needs libusb-1.0.dll on the DLL search path on Windows.
_dll_dir = os.path.dirname(str(libusb_package.get_library_path()))
os.add_dll_directory(_dll_dir)
os.environ["PATH"] = _dll_dir + os.pathsep + os.environ.get("PATH", "")

import can  # noqa: E402

from maker_arm.transport.base import CanBackend  # noqa: E402

log = logging.getLogger("maker_arm.transport")


class GsUsbBackend(CanBackend):
    def __init__(self, index: int = 0, bitrate: int = 1_000_000):
        self._index = index
        self._bitrate = bitrate
        self._bus: Optional[can.BusABC] = None
        self._cb: Optional[Callable[[int, bytes], None]] = None
        self._tx_lock = threading.Lock()
        self._running = False
        self._rx_thread: Optional[threading.Thread] = None

    def open(self) -> None:
        self._bus = can.Bus(interface="gs_usb", channel=self._index, index=self._index,
                            bitrate=self._bitrate)
        self._running = True
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True, name="gsusb-rx")
        self._rx_thread.start()

    def close(self) -> None:
        self._running = False
        if self._rx_thread:
            self._rx_thread.join(timeout=1.0)
            self._rx_thread = None
        if self._bus:
            self._bus.shutdown()
            self._bus = None

    def send(self, can_id: int, data: bytes, extended: bool = True) -> None:
        msg = can.Message(arbitration_id=can_id, data=data, is_extended_id=extended)
        with self._tx_lock:
            self._bus.send(msg, timeout=0.1)

    def set_recv_callback(self, cb) -> None:
        self._cb = cb

    def _rx_loop(self) -> None:
        while self._running:
            try:
                msg = self._bus.recv(timeout=0.1)
            except Exception:
                if not self._running:
                    break
                log.exception("gs_usb recv failed")
                continue
            if msg is None or msg.is_error_frame or msg.is_rx is False:
                continue
            cb = self._cb
            if cb:
                try:
                    cb(msg.arbitration_id, bytes(msg.data))
                except Exception:
                    log.exception("gs_usb RX frame-processing exception (continuing)")
