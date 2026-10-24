# AMG8833 USB thermal camera (ESP32-S3-DevKitC-1)

An ESP32-S3 reads an Adafruit AMG8833 8x8 thermal sensor over I2C. It streams
each frame over the S3's native USB port as a CDC serial device. Windows, Linux
and macOS all have built-in drivers for it.

## Wiring

| AMG8833 | ESP32-S3 |
|---------|----------|
| VIN     | 3V3      |
| GND     | GND      |
| SDA     | GPIO13   |
| SCL     | GPIO14   |

The Adafruit breakout has its own I2C pull-ups and defaults to address 0x69.
The firmware also tries 0x68.

## Build and flash

Plug into the DevKitC-1 port labelled **USB** (native), not **UART**.

```powershell
cd firmware/amg8833_usb
pio run -t upload
```

## Use from the host

```powershell
pip install pyserial numpy opencv-python
python host/thermal_cam.py --view      # live false-colour window
python host/thermal_cam.py             # print stats per frame
```

```python
from thermal_cam import ThermalCamera
with ThermalCamera() as cam:            # or ThermalCamera("COM7") / "/dev/ttyACM0"
    frame = cam.read()
    frame.celsius                       # (8, 8) float32, degrees C
```

Auto-detect expects exactly one Espressif USB device. With several arms
attached, pass the port. On Linux, use the stable
`/dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_<MAC>-if00` path.
The MAC is also the `id=` field in the `info` reply.

## Why not a UVC webcam

The S3 could enumerate as a UVC webcam. However, UVC carries 8-bit images, so
the absolute temperatures would be lost. Hosts also expect far more than 8x8
pixels. The current stream carries the full 0.25 C sensor data, with a sequence
number and a device timestamp.

## Protocol

See [src/protocol.h](src/protocol.h). It uses CRC-checked binary packets at
10 fps (or 1 fps). Send plain-text commands: `info`, `scan`, `fps 1|10`,
`avg on|off`, `stream on|off`. If the sensor disconnects, the device reports
`err sensor_not_found` once per second and recovers on its own. If the host
reads too slowly, the device drops whole frames and the gap shows up in `seq`.

`scan` checks the wiring. It reports whether SDA and SCL have a pull-up and
whether they're shorted together. It then scans the bus on the configured pins
and on swapped pins:

```
scan pullup sda13=1 scl14=1 short=0 | sda=13 scl=14: 0x69 | swapped: none
```

If both pull-ups read 1 but the scan finds nothing, the breakout is connected
but the chip has no power. Check VIN and GND.
