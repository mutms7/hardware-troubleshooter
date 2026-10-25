"""Run an SDK CLI module unchanged over the gs_usb adapter.

Usage: python -X utf8 sdk_cli_gsusb.py maker_arm.cli.zero --all [--yes]
--yes answers the CLI's ENTER confirmation prompt (only pass it when the operator already confirmed).
"""

import builtins
import importlib
import sys

import maker_arm.arm
import maker_arm.transport.base
from gsusb_backend import GsUsbBackend

module = sys.argv.pop(1)
if "--yes" in sys.argv:
    sys.argv.remove("--yes")
    builtins.input = lambda prompt="": print(prompt + "(confirmed by operator)") or ""

factory = lambda *a, **k: GsUsbBackend()
maker_arm.arm.create_backend = factory
maker_arm.transport.base.create_backend = factory
mod = importlib.import_module(module)
if hasattr(mod, "create_backend"):
    mod.create_backend = factory
sys.argv[0] = module
mod.main()
