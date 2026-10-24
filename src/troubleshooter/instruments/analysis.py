"""Waveform measurements and logic decoders over captured samples. Pure functions, no I/O."""

import math
from itertools import pairwise
from statistics import median

MAX_ITEMS = 128  # decoded frames returned to the model; the capture file keeps everything


def _rising_crossings(samples: list[float], mid: float, hysteresis: float) -> list[int]:
    crossings, armed = [], samples[0] < mid
    for i, v in enumerate(samples):
        if armed and v > mid + hysteresis:
            crossings.append(i)
            armed = False
        elif not armed and v < mid - hysteresis:
            armed = True
    return crossings


def waveform_summary(samples: list[float], rate: float, noise_floor_v: float) -> dict:
    vmin, vmax = min(samples), max(samples)
    vpp = vmax - vmin
    mean = sum(samples) / len(samples)
    result = {
        "min_v": vmin,
        "max_v": vmax,
        "vpp_v": vpp,
        "mean_v": mean,
        "rms_v": math.sqrt(sum(v * v for v in samples) / len(samples)),
        "frequency_hz": None,
        "period_s": None,
        "duty_cycle": None,
        "cycles": 0,
        "flat": vpp < noise_floor_v,
    }
    if result["flat"]:
        return result
    mid = (vmax + vmin) / 2
    crossings = _rising_crossings(samples, mid, 0.1 * vpp)
    if len(crossings) >= 2:
        first, last = crossings[0], crossings[-1]
        period = (last - first) / (len(crossings) - 1) / rate
        high = sum(1 for v in samples[first:last] if v > mid)
        result.update(
            frequency_hz=1 / period,
            period_s=period,
            duty_cycle=high / (last - first),
            cycles=len(crossings) - 1,
        )
    return result


def bits(words: list[int], pin: int) -> list[int]:
    return [(w >> pin) & 1 for w in words]


def _edges(line: list[int]) -> list[int]:
    return [i for i in range(1, len(line)) if line[i] != line[i - 1]]


def line_stats(line: list[int], rate: float) -> dict:
    edges = _edges(line)
    rises = [i for i in edges if line[i]]
    result = {
        "edges": len(edges),
        "idle_level": line[0],
        "high_fraction": sum(line) / len(line),
        "frequency_hz": None,
        "duty_cycle": None,
        "min_pulse_s": None,
    }
    if edges:
        widths = [b - a for a, b in pairwise(edges)]
        if widths:
            result["min_pulse_s"] = min(widths) / rate
    if len(rises) >= 2:
        first, last = rises[0], rises[-1]
        result["frequency_hz"] = (len(rises) - 1) * rate / (last - first)
        result["duty_cycle"] = sum(line[first:last]) / (last - first)
    return result


def decode_uart(line, rate, baud, data_bits=8, parity="none", stop_bits=1) -> dict:
    spb = rate / baud
    width = 1 + data_bits + (parity != "none") + stop_bits
    frames, i, n = [], 1, len(line)

    def at(bit):  # level at the centre of a frame bit, counted from the start bit
        return line[min(n - 1, int(start + (bit + 0.5) * spb))]

    while i < n:
        if not (line[i - 1] == 1 and line[i] == 0):
            i += 1
            continue
        start = i
        if start + width * spb > n:
            break  # frame runs past the capture
        if at(0) != 0:
            i += 1  # glitch, not a start bit
            continue
        data = [at(1 + k) for k in range(data_bits)]
        value = sum(b << k for k, b in enumerate(data))
        frame = {"value": value, "framing_error": False, "parity_error": False}
        stop_at = 1 + data_bits
        if parity != "none":
            expected = sum(data) % 2 if parity == "even" else 1 - sum(data) % 2
            frame["parity_error"] = at(stop_at) != expected
            stop_at += 1
        frame["framing_error"] = any(at(stop_at + k) != 1 for k in range(stop_bits))
        frames.append(frame)
        i = int(start + (stop_at + 0.5) * spb)
    good = [f["value"] for f in frames if not (f["framing_error"] or f["parity_error"])]
    return {
        "frames": len(frames),
        "framing_errors": sum(f["framing_error"] for f in frames),
        "parity_errors": sum(f["parity_error"] for f in frames),
        "bytes_hex": [f"{f['value']:02x}" for f in frames[:MAX_ITEMS]],
        "text": "".join(chr(v) if 32 <= v < 127 else "." for v in good[:MAX_ITEMS]),
        "valid_frames": len(good),
    }


def decode_i2c(scl: list[int], sda: list[int]) -> dict:
    transactions, current, pending = [], None, []
    starts = stops = 0

    def close(stop):
        if current is not None:
            current["stop"] = stop
            # The STOP's own SCL rise lands in pending as one partial bit.
            current["complete"] = stop and len(pending) <= 1
            transactions.append(current)

    for k in range(1, len(scl)):
        clock_high = scl[k] and scl[k - 1]
        if clock_high and sda[k - 1] and not sda[k]:
            starts += 1
            close(False)
            current = {"address": None, "rw": None, "address_ack": None, "data_hex": [], "acks": []}
            pending = []
        elif clock_high and not sda[k - 1] and sda[k]:
            stops += 1
            close(True)
            current, pending = None, []
        elif current is not None and scl[k] and not scl[k - 1]:
            pending.append(sda[k])
            if len(pending) == 9:
                byte = sum(b << (7 - j) for j, b in enumerate(pending[:8]))
                ack = pending[8] == 0
                if current["address"] is None:
                    current.update(
                        address=f"0x{byte >> 1:02x}",
                        rw="read" if byte & 1 else "write",
                        address_ack=ack,
                    )
                else:
                    current["data_hex"].append(f"{byte:02x}")
                    current["acks"].append(ack)
                pending = []
    close(False)
    transactions = [t for t in transactions if t["address"] is not None]
    return {
        "starts": starts,
        "stops": stops,
        "transactions": transactions[:MAX_ITEMS],
        "transaction_count": len(transactions),
    }


def decode_spi_partial(sck, data, rate, cpol=0, cpha=0, bit_order="msb") -> dict:
    """Two-line SPI: SCK plus one data line, with no chip select.

    Words are aligned to clock bursts (a gap of several clock periods starts a new word
    group), so treat bytes as tentative. Measurements on SCK are reliable.
    """
    sample_on_rise = cpol == cpha
    clocks = [
        k for k in range(1, len(sck)) if sck[k] != sck[k - 1] and bool(sck[k]) == sample_on_rise
    ]
    result = {
        "clock_edges": len(clocks),
        "sck_frequency_hz": None,
        "bursts": 0,
        "bits_per_burst": [],
        "bytes_hex": [],
        "note": "No chip select: words assume each clock burst starts on a word boundary.",
    }
    if len(clocks) < 2:
        return result
    gaps = [b - a for a, b in pairwise(clocks)]
    period = median(gaps)
    result["sck_frequency_hz"] = rate / period
    bursts, current = [], [clocks[0]]
    for prev, k in pairwise(clocks):
        if k - prev > 4 * period:
            bursts.append(current)
            current = []
        current.append(k)
    bursts.append(current)
    words = []
    for burst in bursts:
        sampled = [data[k] for k in burst]
        for j in range(0, len(sampled) - 7, 8):
            chunk = sampled[j : j + 8]
            if bit_order == "lsb":
                chunk = chunk[::-1]
            words.append(f"{sum(b << (7 - m) for m, b in enumerate(chunk)):02x}")
    result.update(
        bursts=len(bursts),
        bits_per_burst=[len(b) for b in bursts[:MAX_ITEMS]],
        bytes_hex=words[:MAX_ITEMS],
    )
    return result
