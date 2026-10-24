"""Read-only bench check of the real AD3: no arms, no rig file, no outputs enabled."""

import json

from . import analysis
from .ad3 import AD3, AnalogRequest, DigitalRequest


def run_check(serial: str | None = None):
    device = AD3(serial=serial)
    try:
        report = {"device": device.info(), "scope": {}, "dio": {}}
        for channel in (1, 2):
            capture = device.capture_analog(
                AnalogRequest(
                    channel=channel,
                    range_v=50,
                    offset_v=0,
                    sample_rate_hz=100_000,
                    samples=8192,
                    trigger_level_v=None,
                    trigger_slope="rising",
                    auto_timeout_s=1.0,
                )
            )
            summary = analysis.waveform_summary(capture.samples, capture.sample_rate_hz, 0.1)
            report["scope"][f"ch{channel}"] = {
                key: summary[key] for key in ("min_v", "max_v", "mean_v", "frequency_hz", "flat")
            }
        capture = device.capture_digital(
            DigitalRequest(
                pins=tuple(range(16)),
                sample_rate_hz=1_000_000,
                samples=8192,
                trigger_pin=0,
                trigger_edge="rise",
                auto_timeout_s=0.2,
            )
        )
        for pin in range(16):
            stats = analysis.line_stats(analysis.bits(capture.words, pin), capture.sample_rate_hz)
            report["dio"][pin] = {
                key: stats[key] for key in ("idle_level", "edges", "frequency_hz")
            }
        print(json.dumps(report, indent=2))
    finally:
        device.close()
