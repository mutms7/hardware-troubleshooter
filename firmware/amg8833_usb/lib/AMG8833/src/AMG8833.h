// Register-level driver for the Panasonic AMG8833 (Adafruit 3538 breakout).
// Raw counts are kept so callers can transmit them losslessly:
//   pixel      int16, 0.25 C per LSB (12-bit two's complement)
//   thermistor int16, 0.0625 C per LSB (12-bit sign-magnitude, converted here)
#pragma once

#include <Arduino.h>
#include <Wire.h>

namespace amg8833 {

constexpr uint8_t kAddrDefault = 0x69;  // Adafruit breakout (AD_SELECT high)
constexpr uint8_t kAddrAlt = 0x68;
constexpr int kPixels = 64;
constexpr float kPixelCelsiusPerLsb = 0.25f;
constexpr float kThermistorCelsiusPerLsb = 0.0625f;

enum class FrameRate : uint8_t { k10Fps = 0x00, k1Fps = 0x01 };

class AMG8833 {
 public:
  explicit AMG8833(TwoWire &wire = Wire) : wire_(wire) {}

  // Resets the sensor and selects normal mode at 10 fps. Returns false if the
  // device doesn't ACK at addr.
  bool begin(uint8_t addr = kAddrDefault);
  bool setFrameRate(FrameRate rate);
  bool setMovingAverage(bool enabled);

  // Reads all 64 pixels in sensor order (row-major, 8 per row).
  bool readPixelsRaw(int16_t out[kPixels]);
  bool readThermistorRaw(int16_t &out);

  uint8_t address() const { return addr_; }

 private:
  bool write8(uint8_t reg, uint8_t value);
  bool read(uint8_t reg, uint8_t *buf, size_t len);

  TwoWire &wire_;
  uint8_t addr_ = kAddrDefault;
};

}  // namespace amg8833
