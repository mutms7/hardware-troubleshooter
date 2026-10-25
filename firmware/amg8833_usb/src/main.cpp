// Streams AMG8833 frames and VL53L0X/VL53L1X ranges over the ESP32-S3 native USB CDC port.
// Each sensor has its own I2C bus. See protocol.h for the wire format.

#include <Arduino.h>
#include <VL53L0X.h>
#include <VL53L1X.h>
#include <Wire.h>
#include <esp_timer.h>

#include "AMG8833.h"
#include "protocol.h"

// Each pair is one bus, SDA first as wired on the bench. initBus still tries both orders and
// keeps the one the sensor answers on.
#ifndef AMG_PIN_A
#define AMG_PIN_A 14
#endif
#ifndef AMG_PIN_B
#define AMG_PIN_B 13
#endif
#ifndef TOF_PIN_A
#define TOF_PIN_A 47
#endif
#ifndef TOF_PIN_B
#define TOF_PIN_B 48
#endif

namespace {

constexpr char kFirmwareVersion[] = "2.0.0";
constexpr uint32_t kI2cHz = 400000;
constexpr uint32_t kProbeHz = 100000;
constexpr uint32_t kRetryMs = 1000;
constexpr size_t kTxBufferBytes = 4096;
constexpr size_t kFramePayloadBytes = 4 + 8 + 2 + amg8833::kPixels * 2;
constexpr size_t kRangePayloadBytes = 4 + 8 + 2 + 1 + 1;
constexpr uint8_t kTofAddr = 0x29;
constexpr uint32_t kTofBudgetUs = 33000;
constexpr uint16_t kTofTimeoutMs = 100;
constexpr uint16_t kVL53L1XModelId = 0xEACC;
constexpr uint8_t kVL53L0XModelId = 0xEE;
constexpr uint8_t kVL53L0XRangeValid = 11;  // device range status meaning "range complete"

struct Bus {
  TwoWire &wire;
  int pinA, pinB;
  int sda, scl;  // -1 until a device has answered
};

enum class TofModel : uint8_t { kNone, kVL53L0X, kVL53L1X };

Bus amgBus{Wire, AMG_PIN_A, AMG_PIN_B, -1, -1};
Bus tofBus{Wire1, TOF_PIN_A, TOF_PIN_B, -1, -1};
amg8833::AMG8833 sensor(Wire);
VL53L0X tofL0;
VL53L1X tofL1;

bool sensorOk = false;
bool streaming = true;
bool movingAverage = false;
amg8833::FrameRate frameRate = amg8833::FrameRate::k10Fps;
uint32_t seq = 0;
uint32_t lastReadMs = 0;
uint32_t lastRetryMs = 0;
uint32_t droppedFrames = 0;

TofModel tofModel = TofModel::kNone;
bool tofStreaming = true;
uint16_t tofIdSeen = 0;  // last model ID read at 0x29, reported when it's not a known one
uint32_t tofSeq = 0;
uint32_t tofLastRetryMs = 0;
uint32_t droppedRanges = 0;

char deviceId[13];
char cmdBuf[64];
size_t cmdLen = 0;

uint32_t framePeriodMs() { return frameRate == amg8833::FrameRate::k1Fps ? 1000 : 100; }

const char *tofName() {
  switch (tofModel) {
    case TofModel::kVL53L0X: return "VL53L0X";
    case TofModel::kVL53L1X: return "VL53L1X";
    default: return "missing";
  }
}

// Writes a whole packet or nothing, so a slow host never sees a torn frame.
bool sendPacket(uint8_t type, const uint8_t *payload, uint16_t len) {
  if (!Serial || Serial.availableForWrite() < static_cast<int>(len) + 7) return false;
  uint8_t header[5] = {protocol::kSync0, protocol::kSync1, type, static_cast<uint8_t>(len),
                       static_cast<uint8_t>(len >> 8)};
  uint16_t crc = protocol::crc16(header + 2, 3);
  crc = protocol::crc16(payload, len, crc);
  uint8_t trailer[2] = {static_cast<uint8_t>(crc), static_cast<uint8_t>(crc >> 8)};
  Serial.write(header, sizeof(header));
  Serial.write(payload, len);
  Serial.write(trailer, sizeof(trailer));
  return true;
}

void sendStatus(const char *fmt, ...) {
  char text[protocol::kMaxPayload];
  va_list args;
  va_start(args, fmt);
  int n = vsnprintf(text, sizeof(text), fmt, args);
  va_end(args);
  if (n < 0) return;
  sendPacket(protocol::kTypeStatus, reinterpret_cast<uint8_t *>(text),
             min(static_cast<size_t>(n), sizeof(text) - 1));
}

void sendInfo() {
  sendStatus(
      "info fw=%s sensor=%s addr=0x%02X sda=%d scl=%d fps=%u avg=%u stream=%u dropped=%lu "
      "tof=%s tof_sda=%d tof_scl=%d tof_stream=%u tof_dropped=%lu id=%s",
      kFirmwareVersion, sensorOk ? "ok" : "missing", sensor.address(), amgBus.sda, amgBus.scl,
      frameRate == amg8833::FrameRate::k1Fps ? 1 : 10, movingAverage, streaming,
      static_cast<unsigned long>(droppedFrames), tofName(), tofBus.sda, tofBus.scl, tofStreaming,
      static_cast<unsigned long>(droppedRanges), deviceId);
}

// Tries both pin orders and keeps the first one where any of addrs ACKs. On success the bus
// is left running at kI2cHz on those pins and found holds the address that answered.
bool initBus(Bus &bus, const uint8_t *addrs, size_t count, uint8_t &found) {
  const int orders[2][2] = {{bus.pinA, bus.pinB}, {bus.pinB, bus.pinA}};
  for (const auto &order : orders) {
    bus.wire.end();
    bus.wire.begin(order[0], order[1], kProbeHz);
    for (size_t i = 0; i < count; ++i) {
      bus.wire.beginTransmission(addrs[i]);
      if (bus.wire.endTransmission() == 0) {
        bus.sda = order[0];
        bus.scl = order[1];
        found = addrs[i];
        bus.wire.end();
        bus.wire.begin(bus.sda, bus.scl, kI2cHz);
        return true;
      }
    }
  }
  bus.wire.end();
  bus.sda = bus.scl = -1;
  return false;
}

// Appends every ACKing address on the given pins to out.
void scanBus(TwoWire &wire, int sda, int scl, char *out, size_t cap) {
  wire.end();
  wire.begin(sda, scl, kProbeHz);
  size_t len = strlen(out);
  int found = 0;
  for (uint8_t addr = 1; addr < 127; ++addr) {
    wire.beginTransmission(addr);
    if (wire.endTransmission() == 0 && len + 6 < cap) {
      len += snprintf(out + len, cap - len, " 0x%02X", addr);
      ++found;
    }
  }
  if (!found && len + 6 < cap) snprintf(out + len, cap - len, " none");
}

// Diagnoses one bus's wiring: line levels with only internal pull-downs (HIGH means an
// external pull-up, i.e. a powered breakout), then a bus scan in both pin orders.
void sendScan(const char *name, Bus &bus) {
  int a = bus.pinA, b = bus.pinB;
  bus.wire.end();
  pinMode(a, INPUT_PULLDOWN);
  pinMode(b, INPUT_PULLDOWN);
  delay(5);
  int aLevel = digitalRead(a);
  int bLevel = digitalRead(b);
  // If b follows a being driven low, the two lines are shorted.
  pinMode(a, OUTPUT);
  digitalWrite(a, LOW);
  delay(1);
  int shorted = bLevel && !digitalRead(b);
  pinMode(a, INPUT);

  char result[220];
  snprintf(result, sizeof(result), "scan %s pullup gpio%d=%d gpio%d=%d short=%d | sda=%d scl=%d:",
           name, a, aLevel, b, bLevel, shorted, a, b);
  scanBus(bus.wire, a, b, result, sizeof(result));
  size_t len = strlen(result);
  snprintf(result + len, sizeof(result) - len, " | sda=%d scl=%d:", b, a);
  scanBus(bus.wire, b, a, result, sizeof(result));

  bus.wire.end();
  if (bus.sda >= 0) bus.wire.begin(bus.sda, bus.scl, kI2cHz);
  sendStatus("%s", result);
}

bool initSensor() {
  const uint8_t addrs[] = {amg8833::kAddrDefault, amg8833::kAddrAlt};
  uint8_t addr = 0;
  sensorOk = initBus(amgBus, addrs, sizeof(addrs), addr) && sensor.begin(addr) &&
             sensor.setFrameRate(frameRate) && sensor.setMovingAverage(movingAverage);
  return sensorOk;
}

// Both models answer at 0x29. The VL53L1X has 16-bit register indexes and reads 0xEACC at
// 0x010F. A VL53L0X has 8-bit indexes, so that probe writes 0x0F into its register 0x01, which
// init() overwrites; it then reads 0xEE at 0xC0.
TofModel detectTof(TwoWire &wire) {
  tofIdSeen = 0;
  wire.beginTransmission(kTofAddr);
  wire.write(0x01);
  wire.write(0x0F);
  if (wire.endTransmission() == 0 && wire.requestFrom(kTofAddr, static_cast<uint8_t>(2)) == 2) {
    tofIdSeen = static_cast<uint16_t>(wire.read()) << 8;
    tofIdSeen |= wire.read();
    if (tofIdSeen == kVL53L1XModelId) return TofModel::kVL53L1X;
  }
  wire.beginTransmission(kTofAddr);
  wire.write(0xC0);
  if (wire.endTransmission() == 0 && wire.requestFrom(kTofAddr, static_cast<uint8_t>(1)) == 1) {
    uint8_t id = wire.read();
    if (id == kVL53L0XModelId) return TofModel::kVL53L0X;
    tofIdSeen = id;
  }
  return TofModel::kNone;
}

// Returns an empty string on success, else why the ToF isn't running.
const char *initTof() {
  const uint8_t addrs[] = {kTofAddr};
  uint8_t addr = 0;
  tofModel = TofModel::kNone;
  if (!initBus(tofBus, addrs, sizeof(addrs), addr)) return "tof_not_found";
  TofModel model = detectTof(tofBus.wire);
  if (model == TofModel::kVL53L0X) {
    tofL0.setBus(&tofBus.wire);
    tofL0.setTimeout(kTofTimeoutMs);
    if (!tofL0.init()) return "tof_init_failed";
    tofL0.setMeasurementTimingBudget(kTofBudgetUs);
    tofL0.startContinuous();
  } else if (model == TofModel::kVL53L1X) {
    tofL1.setBus(&tofBus.wire);
    tofL1.setTimeout(kTofTimeoutMs);
    if (!tofL1.init()) return "tof_init_failed";
    tofL1.setDistanceMode(VL53L1X::Short);  // best ambient immunity; the probe works close in
    tofL1.setMeasurementTimingBudget(kTofBudgetUs);
    tofL1.startContinuous(kTofBudgetUs / 1000);
  } else {
    return "tof_unknown_model";
  }
  tofModel = model;
  return "";
}

void put16(uint8_t *&p, uint16_t v) {
  *p++ = v;
  *p++ = v >> 8;
}

void putHeader(uint8_t *&p, uint32_t s, uint64_t tUs) {
  for (int i = 0; i < 4; ++i) *p++ = s >> (8 * i);
  for (int i = 0; i < 8; ++i) *p++ = tUs >> (8 * i);
}

void readAndSendFrame() {
  int16_t pixels[amg8833::kPixels];
  int16_t thermistor;
  uint64_t tUs = esp_timer_get_time();
  if (!sensor.readPixelsRaw(pixels) || !sensor.readThermistorRaw(thermistor)) {
    sensorOk = false;
    sendStatus("err i2c_read");
    return;
  }

  uint8_t payload[kFramePayloadBytes];
  uint8_t *p = payload;
  putHeader(p, seq++, tUs);
  put16(p, thermistor);
  for (int i = 0; i < amg8833::kPixels; ++i) put16(p, pixels[i]);

  // seq still advances on a drop so the host can count gaps.
  if (!sendPacket(protocol::kTypeFrame, payload, sizeof(payload))) ++droppedFrames;
}

// Sends a range once the sensor has a new one; never waits for it.
void pollTof() {
  uint16_t mm;
  uint8_t raw;
  bool valid;
  uint64_t tUs = esp_timer_get_time();
  if (tofModel == TofModel::kVL53L0X) {
    uint8_t irq = tofL0.readReg(VL53L0X::RESULT_INTERRUPT_STATUS);
    if (tofL0.last_status != 0) {
      tofModel = TofModel::kNone;
      sendStatus("err tof_i2c_read");
      return;
    }
    if ((irq & 0x07) == 0) return;
    raw = (tofL0.readReg(VL53L0X::RESULT_RANGE_STATUS) >> 3) & 0x0F;
    mm = tofL0.readRangeContinuousMillimeters();  // ready, so this doesn't block
    if (tofL0.timeoutOccurred() || tofL0.last_status != 0) {
      tofModel = TofModel::kNone;
      sendStatus("err tof_i2c_read");
      return;
    }
    valid = raw == kVL53L0XRangeValid && mm < 8190;  // 8190/8191 mean out of range
  } else {
    if (!tofL1.dataReady()) return;
    tofL1.read(false);
    if (tofL1.last_status != 0) {
      tofModel = TofModel::kNone;
      sendStatus("err tof_i2c_read");
      return;
    }
    mm = tofL1.ranging_data.range_mm;
    raw = tofL1.ranging_data.range_status;
    valid = tofL1.ranging_data.range_status == VL53L1X::RangeValid;
  }

  uint8_t payload[kRangePayloadBytes];
  uint8_t *p = payload;
  putHeader(p, tofSeq++, tUs);
  put16(p, mm);
  *p++ = valid;
  *p++ = raw;
  if (tofStreaming && !sendPacket(protocol::kTypeRange, payload, sizeof(payload))) ++droppedRanges;
}

void handleCommand(char *line) {
  if (strcmp(line, "info") == 0) {
    sendInfo();
  } else if (strcmp(line, "scan") == 0) {
    sendScan("amg", amgBus);
  } else if (strcmp(line, "scan tof") == 0) {
    sendScan("tof", tofBus);
  } else if (strcmp(line, "fps 1") == 0 || strcmp(line, "fps 10") == 0) {
    frameRate = strcmp(line, "fps 1") == 0 ? amg8833::FrameRate::k1Fps
                                           : amg8833::FrameRate::k10Fps;
    bool ok = sensorOk && sensor.setFrameRate(frameRate);
    sendStatus(ok ? "ok fps=%s" : "err fps=%s not applied", line + 4);
  } else if (strcmp(line, "avg on") == 0 || strcmp(line, "avg off") == 0) {
    movingAverage = strcmp(line, "avg on") == 0;
    bool ok = sensorOk && sensor.setMovingAverage(movingAverage);
    sendStatus(ok ? "ok avg=%u" : "err avg=%u not applied", movingAverage);
  } else if (strcmp(line, "stream on") == 0 || strcmp(line, "stream off") == 0) {
    streaming = strcmp(line, "stream on") == 0;
    sendStatus("ok stream=%u", streaming);
  } else if (strcmp(line, "tof on") == 0 || strcmp(line, "tof off") == 0) {
    tofStreaming = strcmp(line, "tof on") == 0;
    sendStatus("ok tof_stream=%u", tofStreaming);
  } else {
    sendStatus("err unknown_command %s", line);
  }
}

void pollCommands() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      cmdBuf[cmdLen] = '\0';
      if (cmdLen) handleCommand(cmdBuf);
      cmdLen = 0;
    } else if (cmdLen < sizeof(cmdBuf) - 1) {
      cmdBuf[cmdLen++] = c;
    }
  }
}

}  // namespace

void setup() {
  Serial.setTxBufferSize(kTxBufferBytes);
  Serial.begin();
  // Never block the loop when no host is reading; sendPacket drops instead.
  Serial.setTxTimeoutMs(0);

  // Same MAC the USB descriptor reports as its serial number.
  uint64_t mac = ESP.getEfuseMac();
  for (int i = 0; i < 6; ++i) {
    snprintf(deviceId + 2 * i, 3, "%02X", static_cast<unsigned>((mac >> (8 * i)) & 0xFF));
  }

  initSensor();
  initTof();
  lastRetryMs = tofLastRetryMs = millis();
}

void loop() {
  pollCommands();
  uint32_t now = millis();

  // Each sensor recovers on its own; a missing one never stalls the other.
  if (!sensorOk) {
    if (now - lastRetryMs >= kRetryMs) {
      lastRetryMs = now;
      if (initSensor()) {
        sendStatus("ok sensor_found addr=0x%02X sda=%d scl=%d", sensor.address(), amgBus.sda,
                   amgBus.scl);
      } else {
        sendStatus("err sensor_not_found pins=%d,%d", AMG_PIN_A, AMG_PIN_B);
      }
    }
  } else if (streaming && now - lastReadMs >= framePeriodMs()) {
    lastReadMs = now;
    readAndSendFrame();
  }

  if (tofModel == TofModel::kNone) {
    if (now - tofLastRetryMs >= kRetryMs) {
      tofLastRetryMs = now;
      const char *why = initTof();
      if (!*why) {
        sendStatus("ok tof_found model=%s sda=%d scl=%d", tofName(), tofBus.sda, tofBus.scl);
      } else if (strcmp(why, "tof_unknown_model") == 0) {
        sendStatus("err tof_unknown_model id=0x%04X sda=%d scl=%d", tofIdSeen, tofBus.sda,
                   tofBus.scl);
      } else {
        sendStatus("err %s pins=%d,%d", why, TOF_PIN_A, TOF_PIN_B);
      }
    }
  } else {
    pollTof();
  }
  delay(1);
}
