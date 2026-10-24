#include "AMG8833.h"

namespace amg8833 {
namespace {

// Register map, AMG88xx datasheet section 9.
constexpr uint8_t kRegPctl = 0x00;
constexpr uint8_t kRegRst = 0x01;
constexpr uint8_t kRegFpsc = 0x02;
constexpr uint8_t kRegIntc = 0x03;
constexpr uint8_t kRegAve = 0x07;
constexpr uint8_t kRegTthl = 0x0E;
constexpr uint8_t kRegPixel0 = 0x80;

constexpr uint8_t kPctlNormal = 0x00;
constexpr uint8_t kRstInitial = 0x3F;
constexpr uint8_t kAveMovingAverage = 0x20;

// Arduino-ESP32's Wire buffer is 128 bytes; stay well inside it.
constexpr size_t kMaxChunk = 32;

}  // namespace

bool AMG8833::begin(uint8_t addr) {
  addr_ = addr;
  wire_.beginTransmission(addr_);
  if (wire_.endTransmission() != 0) return false;

  if (!write8(kRegPctl, kPctlNormal)) return false;
  if (!write8(kRegRst, kRstInitial)) return false;
  delay(2);  // Initial reset reloads trim data.
  if (!write8(kRegIntc, 0x00)) return false;
  if (!setFrameRate(FrameRate::k10Fps)) return false;
  // First valid frame arrives after one frame period plus settling.
  delay(100);
  return true;
}

bool AMG8833::setFrameRate(FrameRate rate) {
  return write8(kRegFpsc, static_cast<uint8_t>(rate));
}

bool AMG8833::setMovingAverage(bool enabled) {
  // The datasheet requires this unlock sequence around writes to AVE.
  return write8(0x1F, 0x50) && write8(0x1F, 0x45) && write8(0x1F, 0x57) &&
         write8(kRegAve, enabled ? kAveMovingAverage : 0x00) && write8(0x1F, 0x00);
}

bool AMG8833::readPixelsRaw(int16_t out[kPixels]) {
  uint8_t buf[kPixels * 2];
  if (!read(kRegPixel0, buf, sizeof(buf))) return false;
  for (int i = 0; i < kPixels; ++i) {
    uint16_t v = buf[2 * i] | (buf[2 * i + 1] << 8);
    // Sign-extend the 12-bit two's complement value.
    out[i] = static_cast<int16_t>(v << 4) >> 4;
  }
  return true;
}

bool AMG8833::readThermistorRaw(int16_t &out) {
  uint8_t buf[2];
  if (!read(kRegTthl, buf, sizeof(buf))) return false;
  uint16_t v = (buf[0] | (buf[1] << 8)) & 0x0FFF;
  // Sign-magnitude: bit 11 is the sign.
  out = (v & 0x0800) ? -static_cast<int16_t>(v & 0x07FF) : static_cast<int16_t>(v);
  return true;
}

bool AMG8833::write8(uint8_t reg, uint8_t value) {
  wire_.beginTransmission(addr_);
  wire_.write(reg);
  wire_.write(value);
  return wire_.endTransmission() == 0;
}

bool AMG8833::read(uint8_t reg, uint8_t *buf, size_t len) {
  // The register pointer auto-increments, so long reads are split into chunks
  // that each restart at their own offset.
  for (size_t off = 0; off < len; off += kMaxChunk) {
    size_t n = min(kMaxChunk, len - off);
    wire_.beginTransmission(addr_);
    wire_.write(static_cast<uint8_t>(reg + off));
    if (wire_.endTransmission(false) != 0) return false;
    if (wire_.requestFrom(addr_, n) != n) return false;
    for (size_t i = 0; i < n; ++i) buf[off + i] = wire_.read();
  }
  return true;
}

}  // namespace amg8833
