"""Run the SDK's tools/experimental/lerobot/switch_protocol.py unchanged over the gs_usb adapter.

Usage: python switch_protocol_gsusb.py <path to switch_protocol.py> --ids 1,2,3 --to private
"""

import importlib.util
import sys

from gsusb_backend import GsUsbBackend

path = sys.argv.pop(1)
spec = importlib.util.spec_from_file_location("switch_protocol", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.create_backend = lambda *a, **k: GsUsbBackend()
mod.main()
