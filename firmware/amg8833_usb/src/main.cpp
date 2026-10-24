// Streams AMG8833 frames over the ESP32-S3 native USB CDC port.
// See protocol.h for the wire format.

#include <Arduino.h>
#include <Wire.h>
#include <esp_timer.h>

#include "AMG8833.h"
#include "protocol.h"

#ifndef AMG_SDA_PIN
#define AMG_SDA_PIN 13
#endif
#ifndef AMG_SCL_PIN
#define AMG_SCL_PIN 14
#endif

namespace {

constexpr char kFirmwareVersion[] = "1.0.0";
constexpr uint32_t kI2cHz = 400000;
constexpr uint32_t kRetryMs = 1000;
constexpr size_t kTxBufferBytes = 2048;
constexpr size_t kFramePayloadBytes = 4 + 8 + 2 + amg8833::kPixels * 2;

amg8833::AMG8833 sensor(Wire);
bool sensorOk = false;
bool streaming = true;
bool movingAverage = false;
amg8833::FrameRate frameRate = amg8833::FrameRate::k10Fps;
uint32_t seq = 0;
uint32_t lastReadMs = 0;
uint32_t lastRetryMs = 0;
uint32_t droppedFrames = 0;
char deviceId[13];
char cmdBuf[64];
size_t cmdLen = 0;

uint32_t framePeriodMs() { return frameRate == amg8833::FrameRate::k1Fps ? 1000 : 100; }

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
  sendStatus("info fw=%s sensor=%s addr=0x%02X fps=%u avg=%u stream=%u dropped=%lu id=%s",
             kFirmwareVersion, sensorOk ? "ok" : "missing", sensor.address(),
             frameRate == amg8833::FrameRate::k1Fps ? 1 : 10, movingAverage, streaming,
             static_cast<unsigned long>(droppedFrames), deviceId);
}

// Appends every ACKing address on the given pins to out.
void scanBus(int sda, int scl, char *out, size_t cap) {
  Wire.end();
  Wire.begin(sda, scl, 100000);
  size_t len = strlen(out);
  int found = 0;
  for (uint8_t addr = 1; addr < 127; ++addr) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0 && len + 6 < cap) {
      len += snprintf(out + len, cap - len, " 0x%02X", addr);
      ++found;
    }
  }
  if (!found && len + 6 < cap) snprintf(out + len, cap - len, " none");
}

// Diagnoses wiring: line levels with only internal pull-downs (HIGH means an
// external pull-up, i.e. a powered breakout), then a bus scan on the
// configured pins and on swapped pins.
void sendScan() {
  Wire.end();
  pinMode(AMG_SDA_PIN, INPUT_PULLDOWN);
  pinMode(AMG_SCL_PIN, INPUT_PULLDOWN);
  delay(5);
  int sdaLevel = digitalRead(AMG_SDA_PIN);
  int sclLevel = digitalRead(AMG_SCL_PIN);
  // If SCL follows SDA being driven low, the two lines are shorted.
  pinMode(AMG_SDA_PIN, OUTPUT);
  digitalWrite(AMG_SDA_PIN, LOW);
  delay(1);
  int shorted = sclLevel && !digitalRead(AMG_SCL_PIN);
  pinMode(AMG_SDA_PIN, INPUT);

  char result[200];
  snprintf(result, sizeof(result), "scan pullup sda%d=%d scl%d=%d short=%d | sda=%d scl=%d:",
           AMG_SDA_PIN, sdaLevel, AMG_SCL_PIN, sclLevel, shorted, AMG_SDA_PIN, AMG_SCL_PIN);
  scanBus(AMG_SDA_PIN, AMG_SCL_PIN, result, sizeof(result));
  size_t len = strlen(result);
  snprintf(result + len, sizeof(result) - len, " | swapped:");
  scanBus(AMG_SCL_PIN, AMG_SDA_PIN, result, sizeof(result));

  Wire.end();
  Wire.begin(AMG_SDA_PIN, AMG_SCL_PIN, kI2cHz);
  sendStatus("%s", result);
}

bool initSensor() {
  sensorOk = sensor.begin(amg8833::kAddrDefault) || sensor.begin(amg8833::kAddrAlt);
  if (sensorOk) {
    sensorOk = sensor.setFrameRate(frameRate) && sensor.setMovingAverage(movingAverage);
  }
  return sensorOk;
}

void put16(uint8_t *&p, uint16_t v) {
  *p++ = v;
  *p++ = v >> 8;
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
  uint32_t s = seq++;
  for (int i = 0; i < 4; ++i) *p++ = s >> (8 * i);
  for (int i = 0; i < 8; ++i) *p++ = tUs >> (8 * i);
  put16(p, thermistor);
  for (int i = 0; i < amg8833::kPixels; ++i) put16(p, pixels[i]);

  // seq still advances on a drop so the host can count gaps.
  if (!sendPacket(protocol::kTypeFrame, payload, sizeof(payload))) ++droppedFrames;
}

void handleCommand(char *line) {
  if (strcmp(line, "info") == 0) {
    sendInfo();
  } else if (strcmp(line, "scan") == 0) {
    sendScan();
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

  Wire.begin(AMG_SDA_PIN, AMG_SCL_PIN, kI2cHz);
  initSensor();
  lastRetryMs = millis();
}

void loop() {
  pollCommands();
  uint32_t now = millis();

  if (!sensorOk) {
    if (now - lastRetryMs >= kRetryMs) {
      lastRetryMs = now;
      if (initSensor()) {
        sendStatus("ok sensor_found addr=0x%02X", sensor.address());
      } else {
        sendStatus("err sensor_not_found sda=%d scl=%d", AMG_SDA_PIN, AMG_SCL_PIN);
      }
    }
    delay(5);
    return;
  }

  if (streaming && now - lastReadMs >= framePeriodMs()) {
    lastReadMs = now;
    readAndSendFrame();
  }
  delay(1);
}
