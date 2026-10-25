# Sensor head: AMG8833 thermal camera + VL53L0X ToF (ESP32-S3-DevKitC-1)

An ESP32-S3 reads two sensors, each on its own I2C bus, and streams both over the S3's
native USB port as a CDC serial device (built-in drivers on Windows, Linux and macOS):

- an AMG8833 8x8 thermal camera, at 10 fps (or 1 fps)
- a VL53L0X time-of-flight range sensor, at about 30 Hz. A VL53L1X also works: the
  firmware reads the model ID and drives whichever one answers.

On the rig, the arm server owns the port (`sim_server.py --sensors COM5`). It uses the ToF
for the probe's height above the board and serves both sensors to the MCP tools `read_tof`
and `read_thermal`. See [docs/AD3.md](../../docs/AD3.md#sensor-head).

## Wiring

Measured on the bench on 2026-10-24:

| Sensor | VIN | GND | SDA | SCL | Address |
|---|---|---|---|---|---|
| AMG8833 | 3V3 | GND | GPIO14 | GPIO13 | 0x69 (0x68 also tried) |
| VL53L0X | 3V3 | GND | GPIO47 | GPIO48 | 0x29 |

You don't have to get SDA and SCL right. At boot, and whenever a sensor goes missing, the
firmware tries both pin orders on that bus and keeps the one the sensor answers on. `info`
reports the order it found.

Both breakouts have their own I2C pull-ups. If either sensor is missing, the other keeps
streaming, and the missing one is retried once a second.

## Build and flash

Plug into the DevKitC-1 port labelled **USB** (native), not **UART**.

```powershell
cd firmware/amg8833_usb
pio run -t upload
```

## Use from the host

The driver is [sensor_head.py](../../sensor_head.py) at the repo root. It needs pyserial and
numpy, plus opencv-python for `--view`. Only one process can open the port, so stop the arm
server first.

```powershell
python sensor_head.py --scan     # which pins each sensor answers on
python sensor_head.py --view     # live false-colour thermal window, with the ToF range
python sensor_head.py            # print both streams
```

```python
from sensor_head import SensorHead
with SensorHead() as head:            # or SensorHead("COM5")
    head.wait_thermal().celsius       # (8, 8) float32, degrees C
    head.tof()                        # RangeReading: range_mm, valid, raw_status
```

A background thread keeps reading, so `tof()` and `thermal()` return the latest reading
immediately. It reopens the port if the board resets or the cable is pulled.

Auto-detect expects exactly one Espressif USB device. With several attached, pass the port.
On Linux, use the stable `/dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_<MAC>-if00`
path. The MAC is also the `id=` field in the `info` reply.

## ToF readings

`valid` is false when nothing is in range (VL53L0X: about 1.2 m indoors), there's too much
ambient IR, or the return is weak. Those readings come as `range_mm=8191`. The VL53L0X gets
inaccurate below about 30 mm, so mount it so it still reads 30 mm or more with the probe tip
on the board.

## Protocol

See [src/protocol.h](src/protocol.h). It uses CRC-checked binary packets: thermal frames,
ranges, and text statuses. Send plain-text commands: `info`, `scan`, `scan tof`,
`fps 1|10`, `avg on|off`, `stream on|off` (thermal), `tof on|off`. If the host reads too
slowly, the device drops whole packets, and the gap shows up in `seq`.

`scan` (AMG bus) and `scan tof` check the wiring. Each reports whether the lines have a
pull-up and whether they're shorted, then scans the bus in both pin orders:

```
scan amg pullup gpio14=1 gpio13=1 short=0 | sda=14 scl=13: 0x69 | sda=13 scl=14: none
scan tof pullup gpio47=1 gpio48=1 short=0 | sda=47 scl=48: 0x29 | sda=48 scl=47: none
```

If both pull-ups read 1 but the scan finds nothing, the breakout is connected but the chip
has no power. Check VIN and GND.

## Why not a UVC webcam

UVC carries 8-bit images, which would lose the absolute temperatures. This stream carries
the full 0.25 C sensor data, with a sequence number and a device timestamp.
