// Wire format shared with sensor_head.py at the repo root. Keep the two in sync.
//
// Device -> host packet (all integers little-endian):
//   0xA5 0x5A | type u8 | len u16 | payload[len] | crc16 u16
//   crc16 is CRC-16/CCITT-FALSE over type, len and payload.
//
// FRAME payload (142 bytes), AMG8833:
//   seq u32 | t_us u64 (since boot) | thermistor i16 (0.0625 C/LSB) |
//   pixels 64 x i16 (0.25 C/LSB, sensor row-major)
//
// RANGE payload (16 bytes), VL53L0X/VL53L1X:
//   seq u32 | t_us u64 (since boot) | range_mm u16 | valid u8 | raw_status u8
//   valid is 1 only when the sensor reports a good range. raw_status is the model's own
//   code (VL53L0X device range status, VL53L1X RangeStatus), for diagnostics.
//
// STATUS payload: ASCII text, e.g. "ok fps=10" or "err sensor_not_found".
//
// Host -> device: ASCII command lines terminated by '\n':
//   info | scan | scan tof | fps 1 | fps 10 | avg on | avg off |
//   stream on | stream off | tof on | tof off
#pragma once

#include <stdint.h>

namespace protocol {

constexpr uint8_t kSync0 = 0xA5;
constexpr uint8_t kSync1 = 0x5A;
constexpr uint8_t kTypeFrame = 0x01;
constexpr uint8_t kTypeStatus = 0x02;
constexpr uint8_t kTypeRange = 0x03;
constexpr uint16_t kMaxPayload = 256;

inline uint16_t crc16(const uint8_t *data, size_t len, uint16_t crc = 0xFFFF) {
  for (size_t i = 0; i < len; ++i) {
    crc ^= static_cast<uint16_t>(data[i]) << 8;
    for (int b = 0; b < 8; ++b) crc = (crc & 0x8000) ? (crc << 1) ^ 0x1021 : crc << 1;
  }
  return crc;
}

}  // namespace protocol
