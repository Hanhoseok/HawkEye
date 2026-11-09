/*
  shelf_sensor.ino  —  스프링 푸셔 레일 거리 측정 펌웨어 (ESP32 + VL53L0X)

  역할: "측정과 전송만" 담당한다.
    - VL53L0X 초기화, 레인 순차 측정(동시 발광 없음), 측정 완료/타임아웃 관리, 재초기화
    - 결과를 USB Serial로 JSON Lines(한 줄 = JSON 객체 하나) 전송
    - Wi-Fi 가 설정되어 있으면 같은 줄을 UDP 로도 전송 (한 데이터그램 = 한 줄)
      Wi-Fi 이름/비밀번호는 코드에 쓰지 않는다. USB 로 한 번 보내면 ESP32 내부 저장소(NVS)에 저장된다.
        python tools/wifi_setup.py --port COM7
  하지 않는 일: 상품 수량 계산, ADD/REMOVE 판단, 필터링 (모두 PC의 Python이 수행)

  라이브러리: Pololu "VL53L0X" (Arduino 라이브러리 매니저에서 "VL53L0X" by Pololu 설치)
    https://github.com/pololu/vl53l0x-arduino
    사용 API: setBus, setTimeout, init, setMeasurementTimingBudget,
             readReg / writeReg / readReg16Bit, last_status, regAddr 열거형.
    라이브러리의 readRangeSingleMillimeters()는 측정이 끝날 때까지 루프를 막으므로,
    같은 레지스터 순서를 비차단(state machine) 방식으로 직접 수행한다.

  출력 메시지 (모두 한 줄 JSON):
    MEASUREMENT : message_type, device_id, boot_id, sequence, uptime_ms,
                  lane_id, distance_mm(실패 시 null), valid, sensor_status
    HELLO / HEARTBEAT / CONFIG_ERROR / LOG / WIFI_STATUS : 보조 메시지 (공통 헤더 + 추가 필드)
  USB 로 받는 명령 (한 줄 JSON):
    {"cmd":"set_wifi","ssid":"...","password":"..."}   {"cmd":"clear_wifi"}   {"cmd":"wifi_status"}
    {"cmd":"set_lane","lane":"2"}   (BUS_DIRECT_SINGLE: 이 보드가 맡을 레인 번호. 저장 후 재시작)

  레인마다 ESP32 를 1대씩 쓸 때는 모든 보드에 이 펌웨어를 그대로 올리고, 레인 번호만
  python tools/wifi_setup.py --port COMx --lane N 으로 보드마다 지정한다.
  그 보드의 device_id 는 "shelf-lane-N" 이 된다 (지정 전 기본값: 레인 1).
  uptime_ms 는 ESP32 부팅 후 경과 ms(millis())이며 실제 날짜·시각이 아니다.
  약 49.7일마다 0으로 되돌아간다(Python 쪽에서 보정).
*/

#include <Arduino.h>
#include <stdarg.h>
#include <Wire.h>
#include <VL53L0X.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Preferences.h>
#include <esp_system.h>
#if __has_include(<esp_random.h>)
#include <esp_random.h>   // ESP-IDF 5 (arduino-esp32 3.x)
#endif

// ============================================================================
// 사용자 설정
// ============================================================================

enum BusMode { BUS_DIRECT_SINGLE, BUS_TCA9548A };

// 현재: 센서 1개를 ESP32 I2C에 직접 연결 → BUS_DIRECT_SINGLE
// 향후: TCA9548A + 센서 5개 → BUS_TCA9548A 로 바꾸고 아래 LANE_HW 의 enabled 를 켠다.
static const BusMode BUS_MODE = BUS_DIRECT_SINGLE;

static const int      PIN_SDA      = 21;
static const int      PIN_SCL      = 22;
static const uint32_t I2C_CLOCK_HZ = 100000;   // 배선이 짧으면 400000 가능
static const uint16_t I2C_TIMEOUT_MS = 20;     // Wire 트랜잭션 타임아웃
static const uint8_t  TCA_ADDRESS  = 0x70;     // A0~A2 = GND 일 때

static const char*    DEVICE_ID    = "shelf-esp32-01";   // BUS_TCA9548A 일 때의 이름. 영문/숫자/하이픈만
                                                          // (DIRECT_SINGLE 은 "shelf-lane-<레인 번호>")
static const uint32_t SERIAL_BAUD  = 115200;

// 측정 타이밍 (README "측정 예산과 지연" 참고)
static const uint32_t TIMING_BUDGET_US     = 33000; // VL53L0X 측정 예산 (기본값 약 33ms)
static const uint32_t MEAS_TIMEOUT_MS      = 120;   // 측정 1회 최대 대기. 넘으면 TIMEOUT
static const uint16_t LIB_IO_TIMEOUT_MS    = 100;   // init() 내부 대기 상한
static const uint32_t LANE_MIN_PERIOD_MS   = 100;   // 한 레인의 최소 재방문 주기
static const uint32_t INTER_MEAS_GAP_MS    = 2;     // 다음 센서 발광 전 휴지
static const uint32_t POLL_INTERVAL_MS     = 2;     // 측정 완료 폴링 간격
static const uint8_t  ERRORS_BEFORE_REINIT = 5;     // 연속 통신 오류 n회 → 재초기화
static const uint32_t REINIT_MIN_MS        = 1000;  // 재초기화 간격 (오류가 계속되면 2배씩)
static const uint32_t REINIT_MAX_MS        = 10000;
static const uint32_t ERROR_REPORT_MS      = 500;   // 오류 레인 상태 보고 주기 (센서 접근 없음)
static const uint32_t HEARTBEAT_MS         = 1000;
static const uint32_t HELLO_MS             = 30000;

// Wi-Fi (2.4GHz 만 가능). 이름/비밀번호는 tools/wifi_setup.py 로 설정한다.
static const bool         WIFI_ENABLED        = true;
static const uint16_t     UDP_PORT            = 4210;   // Python config.toml [network] udp_port 와 같게
static const char*        UDP_TARGET_IP       = "";     // "" = 같은 공유기 안 브로드캐스트, "192.168.0.10" = 그 PC 로만
static const wifi_power_t WIFI_TX_POWER       = WIFI_POWER_11dBm;  // 낮출수록 순간 전류가 줄어 USB 전원이 약해도 덜 리셋됨
static const uint32_t     WIFI_START_DELAY_MS = 3000;   // 부팅 후 Wi-Fi 시작까지 대기 (그 사이 clear_wifi 명령 가능)
static const uint32_t     WIFI_BROWNOUT_DELAY_MS = 30000; // 직전 리셋이 전압 부족이면 더 늦게 시작
static const uint32_t     WIFI_RETRY_MS       = 10000;  // 연결이 안 되면 다시 시도하는 간격

static const uint8_t MAX_LANES = 5;

struct LaneHwConfig {
  uint8_t lane_id;      // 1..5 (Python 설정의 lane_id 와 일치해야 함)
  bool    enabled;      // false 면 해당 센서에 전혀 접근하지 않음
  uint8_t tca_channel;  // BUS_TCA9548A 일 때 사용하는 채널 0..7
};

static LaneHwConfig LANE_HW[MAX_LANES] = {
  // lane_id, enabled, tca_channel
  { 1, true,  0 },
  { 2, false, 1 },
  { 3, false, 2 },
  { 4, false, 3 },
  { 5, false, 4 },
};

// ============================================================================
// 내부 구현
// ============================================================================

static const char* FW_VERSION = "1.2.0";
static const uint8_t VL53L0X_ADDR = 0x29;

enum SensorStatus : uint8_t {
  ST_OK,             // 정상 측정
  ST_NO_TARGET,      // 측정은 완료됐으나 표적 없음/범위 밖 (8190 이상 등)
  ST_RANGE_INVALID,  // 측정 완료, 장치 range status 가 유효하지 않음
  ST_TIMEOUT,        // 제한 시간 내 측정 완료 안 됨
  ST_I2C_ERROR,      // 측정 중 I2C 통신 실패
  ST_NOT_CONNECTED,  // 주소 0x29 응답 없음 또는 모델 ID 불일치
  ST_INIT_FAILED,    // 응답은 하나 init()/예산 설정 실패
  ST_MUX_ERROR,      // TCA9548A 채널 선택 실패
  ST_NOT_INITIALIZED // 아직 초기화 시도 전
};

static const char* statusName(SensorStatus s) {
  switch (s) {
    case ST_OK:              return "OK";
    case ST_NO_TARGET:       return "NO_TARGET";
    case ST_RANGE_INVALID:   return "RANGE_INVALID";
    case ST_TIMEOUT:         return "TIMEOUT";
    case ST_I2C_ERROR:       return "I2C_ERROR";
    case ST_NOT_CONNECTED:   return "NOT_CONNECTED";
    case ST_INIT_FAILED:     return "INIT_FAILED";
    case ST_MUX_ERROR:       return "MUX_ERROR";
    default:                 return "NOT_INITIALIZED";
  }
}

struct LaneRuntime {
  VL53L0X  sensor;
  bool     ready = false;            // 초기화 완료되어 측정 가능
  uint8_t  stopVariable = 0;
  uint8_t  consecutiveErrors = 0;
  uint32_t lastStartMs = 0;
  bool     everStarted = false;
  uint32_t nextReinitMs = 0;
  uint32_t reinitBackoffMs = REINIT_MIN_MS;
  SensorStatus lastError = ST_NOT_INITIALIZED;
  uint32_t lastErrorReportMs = 0;
};

static LaneRuntime lanes[MAX_LANES];

enum Phase { PH_IDLE, PH_WAIT_START, PH_WAIT_RESULT };
static Phase    phase = PH_IDLE;
static int      activeLane = -1;
static int      cursorLane = -1;
static uint32_t phaseStartMs = 0;
static uint32_t lastPollMs = 0;
static uint32_t lastMeasEndMs = 0;

static uint32_t sequenceNo = 0;
static char     bootId[9];
static char     deviceId[33];
static uint8_t  boardLane = 1;      // BUS_DIRECT_SINGLE 에서 이 보드의 레인 번호 (NVS 저장)
static bool     configOk = true;
static char     configError[96] = "";
static uint32_t lastHeartbeatMs = 0;
static uint32_t lastHelloMs = 0;

static Preferences prefs;
static WiFiUDP     udp;
static char        wifiSsid[33] = "";
static char        wifiPass[65] = "";
static bool        wifiHaveCreds = false;
static bool        wifiStarted = false;
static bool        wifiUp = false;
static uint32_t    wifiStartAtMs = 0;
static uint32_t    wifiLastAttemptMs = 0;
static IPAddress   udpTarget;
static char        cmdBuf[256];
static size_t      cmdLen = 0;
static bool        cmdOverflow = false;

// millis() 오버플로에 안전한 비교: "now 가 t 이후인가"
static inline bool timeReached(uint32_t now, uint32_t t) { return (int32_t)(now - t) >= 0; }
static inline bool elapsed(uint32_t now, uint32_t since, uint32_t interval) { return (uint32_t)(now - since) >= interval; }

// ---------------------------------------------------------------------------
// JSON 출력 (한 줄을 버퍼에 완성한 뒤 한 번에 write)
// ---------------------------------------------------------------------------
static char outBuf[384];

static int beginMessage(const char* type) {
  return snprintf(outBuf, sizeof(outBuf),
                  "{\"message_type\":\"%s\",\"device_id\":\"%s\",\"boot_id\":\"%s\","
                  "\"sequence\":%lu,\"uptime_ms\":%lu",
                  type, deviceId, bootId,
                  (unsigned long)(sequenceNo++), (unsigned long)millis());
}

static void endMessage(int len) {
  if (len < 0) return;
  if (len > (int)sizeof(outBuf) - 3) len = sizeof(outBuf) - 3;
  outBuf[len++] = '}';
  outBuf[len++] = '\n';
  Serial.write((const uint8_t*)outBuf, len);
  if (wifiUp) {   // UDP 는 막히지 않는다: 보내지 못한 줄은 Python 이 sequence 누락으로 처리
    if (udp.beginPacket(udpTarget, UDP_PORT)) {
      udp.write((const uint8_t*)outBuf, len);
      udp.endPacket();
    }
  }
}

static int appendf(int len, const char* fmt, ...) {
  if (len < 0 || len >= (int)sizeof(outBuf)) return len;
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(outBuf + len, sizeof(outBuf) - len, fmt, ap);
  va_end(ap);
  if (n < 0) return len;
  len += n;
  return len < (int)sizeof(outBuf) ? len : (int)sizeof(outBuf) - 1;
}

static void emitMeasurement(uint8_t laneId, bool valid, uint16_t mm, SensorStatus st) {
  int len = beginMessage("MEASUREMENT");
  len = appendf(len, ",\"lane_id\":%u", laneId);
  if (valid) len = appendf(len, ",\"distance_mm\":%u,\"valid\":true", mm);
  else       len = appendf(len, ",\"distance_mm\":null,\"valid\":false");
  len = appendf(len, ",\"sensor_status\":\"%s\"", statusName(st));
  endMessage(len);
}

static void emitLog(const char* level, uint8_t laneId, const char* text) {
  int len = beginMessage("LOG");
  len = appendf(len, ",\"level\":\"%s\"", level);
  if (laneId) len = appendf(len, ",\"lane_id\":%u", laneId);
  len = appendf(len, ",\"text\":\"%s\"", text);   // text 에는 따옴표/역슬래시를 쓰지 않는다
  endMessage(len);
}

static void emitEnabledLanes(int& len) {
  len = appendf(len, ",\"enabled_lanes\":[");
  bool first = true;
  for (uint8_t i = 0; i < MAX_LANES; i++) {
    if (!LANE_HW[i].enabled) continue;
    len = appendf(len, first ? "%u" : ",%u", LANE_HW[i].lane_id);
    first = false;
  }
  len = appendf(len, "]");
}

static void emitHello() {
  int len = beginMessage("HELLO");
  len = appendf(len, ",\"fw_version\":\"%s\",\"bus_mode\":\"%s\",\"timing_budget_us\":%lu,"
                     "\"lane_min_period_ms\":%lu,\"reset_reason\":%d",
                FW_VERSION, BUS_MODE == BUS_TCA9548A ? "TCA9548A" : "DIRECT_SINGLE",
                (unsigned long)TIMING_BUDGET_US, (unsigned long)LANE_MIN_PERIOD_MS,
                (int)esp_reset_reason());
  emitEnabledLanes(len);
  len = appendf(len, ",\"config_ok\":%s", configOk ? "true" : "false");
  endMessage(len);
}

static const char* wifiStateName() {
  if (!WIFI_ENABLED)  return "DISABLED";
  if (!wifiHaveCreds) return "NOT_CONFIGURED";
  if (wifiUp)         return "CONNECTED";
  return wifiStarted ? "CONNECTING" : "WAITING";
}

static void emitHeartbeat() {
  int len = beginMessage("HEARTBEAT");
  len = appendf(len, ",\"config_ok\":%s", configOk ? "true" : "false");
  emitEnabledLanes(len);
  len = appendf(len, ",\"wifi\":\"%s\"", wifiStateName());
  if (wifiUp) len = appendf(len, ",\"ip\":\"%s\",\"rssi\":%d", WiFi.localIP().toString().c_str(), (int)WiFi.RSSI());
  endMessage(len);
}

// 비밀번호는 어떤 메시지에도 넣지 않는다
static void emitWifiStatus(const char* note) {
  int len = beginMessage("WIFI_STATUS");
  len = appendf(len, ",\"state\":\"%s\",\"note\":\"%s\"", wifiStateName(), note);
  if (BUS_MODE == BUS_DIRECT_SINGLE) len = appendf(len, ",\"lane_id\":%u", boardLane);
  if (wifiHaveCreds) {
    len = appendf(len, ",\"ssid\":\"");
    for (const char* c = wifiSsid; *c && len < (int)sizeof(outBuf) - 8; c++) {
      if (*c == '"' || *c == '\\') outBuf[len++] = '\\';
      outBuf[len++] = *c;
    }
    len = appendf(len, "\"");
  }
  if (wifiUp) {
    len = appendf(len, ",\"ip\":\"%s\",\"rssi\":%d,\"udp_target\":\"%s\",\"udp_port\":%u",
                  WiFi.localIP().toString().c_str(), (int)WiFi.RSSI(),
                  udpTarget.toString().c_str(), UDP_PORT);
  }
  endMessage(len);
}

static void emitConfigError() {
  int len = beginMessage("CONFIG_ERROR");
  len = appendf(len, ",\"text\":\"%s\"", configError);
  endMessage(len);
}

// ---------------------------------------------------------------------------
// Wi-Fi (비차단: WiFi.begin 후 상태만 확인)
// ---------------------------------------------------------------------------
// 레인 번호 (보드마다 다름) → LANE_HW 와 device_id 에 반영. validateConfig 전에 호출
static void loadBoardConfig() {
  prefs.begin("board", true);
  uint8_t lane = prefs.getUChar("lane", 1);
  prefs.end();
  if (lane < 1 || lane > MAX_LANES) lane = 1;
  boardLane = lane;
  if (BUS_MODE == BUS_DIRECT_SINGLE) {
    for (uint8_t i = 0; i < MAX_LANES; i++) {
      if (LANE_HW[i].enabled) { LANE_HW[i].lane_id = lane; break; }
    }
    snprintf(deviceId, sizeof(deviceId), "shelf-lane-%u", lane);
  } else {
    strlcpy(deviceId, DEVICE_ID, sizeof(deviceId));
  }
}

static void wifiLoadCreds() {
  prefs.begin("wifi", true);
  String ssid = prefs.getString("ssid", "");
  String pass = prefs.getString("pass", "");
  prefs.end();
  strlcpy(wifiSsid, ssid.c_str(), sizeof(wifiSsid));
  strlcpy(wifiPass, pass.c_str(), sizeof(wifiPass));
  wifiHaveCreds = wifiSsid[0] != '\0';
}

static void wifiConnect(uint32_t now) {
  wifiLastAttemptMs = now;
  if (!wifiStarted) {
    WiFi.setHostname(deviceId);
    WiFi.mode(WIFI_STA);
    WiFi.setAutoReconnect(true);
    wifiStarted = true;
  } else {
    WiFi.disconnect();
  }
  WiFi.begin(wifiSsid, wifiPass);
  WiFi.setTxPower(WIFI_TX_POWER);
}

static void wifiStop() {
  if (wifiStarted) {
    udp.stop();
    WiFi.disconnect(true);
    WiFi.mode(WIFI_OFF);
  }
  wifiStarted = false;
  wifiUp = false;
}

static void wifiService(uint32_t now) {
  if (!WIFI_ENABLED || !wifiHaveCreds) return;
  if (!wifiStarted) {
    if (timeReached(now, wifiStartAtMs)) {
      wifiConnect(now);
      emitWifiStatus("connecting");
    }
    return;
  }
  bool up = WiFi.status() == WL_CONNECTED;
  if (up && !wifiUp) {
    if (UDP_TARGET_IP[0] == '\0' || !udpTarget.fromString(UDP_TARGET_IP)) {
      IPAddress ip = WiFi.localIP(), mask = WiFi.subnetMask();
      udpTarget = IPAddress(ip[0] | (uint8_t)~mask[0], ip[1] | (uint8_t)~mask[1],
                            ip[2] | (uint8_t)~mask[2], ip[3] | (uint8_t)~mask[3]);
    }
    udp.begin(UDP_PORT);
    wifiUp = true;
    emitWifiStatus("connected");
  } else if (!up && wifiUp) {
    wifiUp = false;
    udp.stop();
    emitWifiStatus("disconnected");
  }
  if (!up && elapsed(now, wifiLastAttemptMs, WIFI_RETRY_MS)) {
    wifiConnect(now);
  }
}

// ---------------------------------------------------------------------------
// USB 명령 (Wi-Fi 설정). JSON 문자열 값만 읽는 최소 파서
// ---------------------------------------------------------------------------
static bool jsonGetString(const char* s, const char* key, char* out, size_t outSize) {
  char pat[24];
  snprintf(pat, sizeof(pat), "\"%s\"", key);
  const char* p = strstr(s, pat);
  if (!p) return false;
  p += strlen(pat);
  while (*p == ' ') p++;
  if (*p++ != ':') return false;
  while (*p == ' ') p++;
  if (*p++ != '"') return false;
  size_t n = 0;
  while (*p && *p != '"') {
    char c = *p++;
    if (c == '\\') {
      c = *p++;
      if (c != '"' && c != '\\' && c != '/') return false;   // \uXXXX 는 쓰지 않는다 (UTF-8 그대로 전송)
    }
    if (n + 1 >= outSize) return false;
    out[n++] = c;
  }
  if (*p != '"') return false;
  out[n] = '\0';
  return true;
}

static void handleCommand(const char* line, uint32_t now) {
  char cmd[24];
  if (!jsonGetString(line, "cmd", cmd, sizeof(cmd))) {
    emitLog("WARN", 0, "unknown serial input ignored");
    return;
  }
  if (strcmp(cmd, "set_wifi") == 0) {
    char ssid[33], pass[65];
    if (!jsonGetString(line, "ssid", ssid, sizeof(ssid)) || ssid[0] == '\0' ||
        !jsonGetString(line, "password", pass, sizeof(pass))) {
      emitWifiStatus("rejected: ssid 1-32 bytes, password 0-64 bytes");
      return;
    }
    prefs.begin("wifi", false);
    prefs.putString("ssid", ssid);
    prefs.putString("pass", pass);
    prefs.end();
    memset(pass, 0, sizeof(pass));
    wifiStop();
    wifiLoadCreds();
    wifiStartAtMs = now;          // 바로 연결 시도
    emitWifiStatus("saved");
  } else if (strcmp(cmd, "clear_wifi") == 0) {
    prefs.begin("wifi", false);
    prefs.clear();
    prefs.end();
    wifiStop();
    wifiLoadCreds();
    emitWifiStatus("cleared");
  } else if (strcmp(cmd, "set_lane") == 0) {
    char laneText[4];
    int lane = jsonGetString(line, "lane", laneText, sizeof(laneText)) ? atoi(laneText) : 0;
    if (BUS_MODE != BUS_DIRECT_SINGLE || lane < 1 || lane > MAX_LANES) {
      emitWifiStatus("rejected: lane must be 1-5 and BUS_DIRECT_SINGLE");
      return;
    }
    prefs.begin("board", false);
    prefs.putUChar("lane", (uint8_t)lane);
    prefs.end();
    emitLog("INFO", (uint8_t)lane, "lane saved, restarting");
    Serial.flush();
    delay(50);
    ESP.restart();               // 새 레인 번호·device_id·boot_id 로 깨끗하게 다시 시작
  } else if (strcmp(cmd, "wifi_status") == 0) {
    emitWifiStatus("status");
  } else {
    emitLog("WARN", 0, "unknown cmd");
  }
}

static void serviceSerialInput(uint32_t now) {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      cmdBuf[cmdLen] = '\0';
      if (!cmdOverflow && cmdLen > 0) handleCommand(cmdBuf, now);
      memset(cmdBuf, 0, sizeof(cmdBuf));   // 비밀번호가 버퍼에 남지 않게
      cmdLen = 0;
      cmdOverflow = false;
    } else if (cmdLen + 1 < sizeof(cmdBuf)) {
      cmdBuf[cmdLen++] = c;
    } else {
      cmdOverflow = true;
    }
  }
}

// ---------------------------------------------------------------------------
// I2C / TCA9548A
// ---------------------------------------------------------------------------
static bool selectLaneBus(uint8_t i) {
  if (BUS_MODE != BUS_TCA9548A) return true;
  Wire.beginTransmission(TCA_ADDRESS);
  Wire.write((uint8_t)(1u << LANE_HW[i].tca_channel));
  return Wire.endTransmission() == 0;
}

static void validateConfig() {
  uint8_t enabledCount = 0;
  for (uint8_t i = 0; i < MAX_LANES; i++) {
    if (!LANE_HW[i].enabled) continue;
    enabledCount++;
    if (LANE_HW[i].lane_id < 1 || LANE_HW[i].lane_id > MAX_LANES) {
      configOk = false;
      snprintf(configError, sizeof(configError), "lane index %u has invalid lane_id %u", i, LANE_HW[i].lane_id);
      return;
    }
    if (BUS_MODE == BUS_TCA9548A) {
      if (LANE_HW[i].tca_channel > 7) {
        configOk = false;
        snprintf(configError, sizeof(configError), "lane %u tca_channel %u out of 0-7", LANE_HW[i].lane_id, LANE_HW[i].tca_channel);
        return;
      }
      for (uint8_t j = 0; j < i; j++) {
        if (LANE_HW[j].enabled && LANE_HW[j].tca_channel == LANE_HW[i].tca_channel) {
          configOk = false;
          snprintf(configError, sizeof(configError), "lanes %u and %u share tca_channel %u",
                   LANE_HW[j].lane_id, LANE_HW[i].lane_id, LANE_HW[i].tca_channel);
          return;
        }
      }
    }
  }
  if (BUS_MODE == BUS_DIRECT_SINGLE && enabledCount > 1) {
    configOk = false;
    snprintf(configError, sizeof(configError),
             "DIRECT_SINGLE allows 1 enabled lane but %u are enabled. Use BUS_TCA9548A", enabledCount);
  }
  if (enabledCount == 0) {
    configOk = false;
    snprintf(configError, sizeof(configError), "no lane enabled");
  }
}

// ---------------------------------------------------------------------------
// 센서 초기화 (제한 시간 있는 차단 호출: 센서가 없으면 1ms 이내, 있으면 수십 ms)
// ---------------------------------------------------------------------------
static void markLaneFailed(uint8_t i, SensorStatus st, uint32_t now) {
  LaneRuntime& L = lanes[i];
  L.ready = false;
  L.lastError = st;
  L.nextReinitMs = now + L.reinitBackoffMs;
  L.reinitBackoffMs = min(L.reinitBackoffMs * 2, REINIT_MAX_MS);
}

static void tryInitLane(uint8_t i) {
  LaneRuntime& L = lanes[i];
  const uint8_t id = LANE_HW[i].lane_id;
  uint32_t now = millis();

  if (!selectLaneBus(i)) {
    markLaneFailed(i, ST_MUX_ERROR, now);
    emitMeasurement(id, false, 0, ST_MUX_ERROR);
    return;
  }
  Wire.beginTransmission(VL53L0X_ADDR);
  if (Wire.endTransmission() != 0) {
    markLaneFailed(i, ST_NOT_CONNECTED, now);
    emitMeasurement(id, false, 0, ST_NOT_CONNECTED);
    return;
  }

  L.sensor.setBus(&Wire);
  L.sensor.setTimeout(LIB_IO_TIMEOUT_MS);
  if (L.sensor.readReg(VL53L0X::IDENTIFICATION_MODEL_ID) != 0xEE || L.sensor.last_status != 0) {
    markLaneFailed(i, ST_NOT_CONNECTED, millis());
    emitMeasurement(id, false, 0, ST_NOT_CONNECTED);
    emitLog("WARN", id, "device at 0x29 is not a VL53L0X (model id mismatch)");
    return;
  }
  if (!L.sensor.init() || !L.sensor.setMeasurementTimingBudget(TIMING_BUDGET_US)) {
    markLaneFailed(i, ST_INIT_FAILED, millis());
    emitMeasurement(id, false, 0, ST_INIT_FAILED);
    return;
  }
  // 라이브러리의 stop_variable 은 private 이므로 init() 과 같은 순서로 다시 읽는다.
  L.sensor.writeReg(0x80, 0x01);
  L.sensor.writeReg(0xFF, 0x01);
  L.sensor.writeReg(0x00, 0x00);
  L.stopVariable = L.sensor.readReg(0x91);
  L.sensor.writeReg(0x00, 0x01);
  L.sensor.writeReg(0xFF, 0x00);
  L.sensor.writeReg(0x80, 0x00);
  if (L.sensor.last_status != 0) {
    markLaneFailed(i, ST_INIT_FAILED, millis());
    emitMeasurement(id, false, 0, ST_INIT_FAILED);
    return;
  }

  L.ready = true;
  L.consecutiveErrors = 0;
  L.reinitBackoffMs = REINIT_MIN_MS;
  L.lastError = ST_OK;
  emitLog("INFO", id, "sensor initialized");
}

// ---------------------------------------------------------------------------
// 비차단 단발 측정 (Pololu readRangeSingleMillimeters 와 같은 레지스터 순서)
// ---------------------------------------------------------------------------
static bool startMeasurement(uint8_t i) {
  LaneRuntime& L = lanes[i];
  if (!selectLaneBus(i)) return false;
  VL53L0X& s = L.sensor;
  s.writeReg(0x80, 0x01);
  s.writeReg(0xFF, 0x01);
  s.writeReg(0x00, 0x00);
  s.writeReg(0x91, L.stopVariable);
  s.writeReg(0x00, 0x01);
  s.writeReg(0xFF, 0x00);
  s.writeReg(0x80, 0x00);
  s.writeReg(VL53L0X::SYSRANGE_START, 0x01);
  return s.last_status == 0;
}

static void finishWithError(uint8_t i, SensorStatus st, uint32_t now) {
  LaneRuntime& L = lanes[i];
  // 다음 측정을 깨끗하게 시작하도록 인터럽트를 정리(실패해도 무시)
  L.sensor.writeReg(VL53L0X::SYSTEM_INTERRUPT_CLEAR, 0x01);
  emitMeasurement(LANE_HW[i].lane_id, false, 0, st);
  L.lastError = st;
  if (++L.consecutiveErrors >= ERRORS_BEFORE_REINIT) {
    markLaneFailed(i, st, now);
    emitLog("WARN", LANE_HW[i].lane_id, "repeated errors, sensor will be re-initialized");
  }
  phase = PH_IDLE;
  lastMeasEndMs = now;
}

static void serviceMeasurement(uint32_t now) {
  if (!elapsed(now, lastPollMs, POLL_INTERVAL_MS)) return;
  lastPollMs = now;
  const uint8_t i = (uint8_t)activeLane;
  LaneRuntime& L = lanes[i];
  VL53L0X& s = L.sensor;

  if (elapsed(now, phaseStartMs, MEAS_TIMEOUT_MS)) {
    finishWithError(i, ST_TIMEOUT, now);
    return;
  }

  if (phase == PH_WAIT_START) {
    uint8_t v = s.readReg(VL53L0X::SYSRANGE_START);
    if (s.last_status != 0) { finishWithError(i, ST_I2C_ERROR, now); return; }
    if ((v & 0x01) == 0) phase = PH_WAIT_RESULT;
    return;
  }

  // PH_WAIT_RESULT
  uint8_t irq = s.readReg(VL53L0X::RESULT_INTERRUPT_STATUS);
  if (s.last_status != 0) { finishWithError(i, ST_I2C_ERROR, now); return; }
  if ((irq & 0x07) == 0) return;

  uint8_t  rs = s.readReg(VL53L0X::RESULT_RANGE_STATUS);
  uint16_t mm = s.readReg16Bit(VL53L0X::RESULT_RANGE_STATUS + 10);
  s.writeReg(VL53L0X::SYSTEM_INTERRUPT_CLEAR, 0x01);
  if (s.last_status != 0) { finishWithError(i, ST_I2C_ERROR, now); return; }

  // 장치 range status (ST API: (reg & 0x78) >> 3), 11 = Range Complete
  uint8_t devStatus = (rs & 0x78) >> 3;
  SensorStatus st;
  if (mm >= 8190 || devStatus == 4 /* MSRC no target */) st = ST_NO_TARGET;
  else if (devStatus != 11)                               st = ST_RANGE_INVALID;
  else                                                    st = ST_OK;

  // 센서 자체는 응답했으므로 통신 오류 카운터는 초기화
  L.consecutiveErrors = 0;
  L.lastError = st;
  emitMeasurement(LANE_HW[i].lane_id, st == ST_OK, mm, st);
  phase = PH_IDLE;
  lastMeasEndMs = now;
}

// 다음에 할 일을 고른다: 측정 시작 / 재초기화 / 오류 상태 보고
static void scheduleNext(uint32_t now) {
  if (!elapsed(now, lastMeasEndMs, INTER_MEAS_GAP_MS)) return;

  for (uint8_t k = 1; k <= MAX_LANES; k++) {
    uint8_t i = (uint8_t)((cursorLane + k + MAX_LANES) % MAX_LANES);
    if (!LANE_HW[i].enabled) continue;  // 비활성 레인은 절대 접근하지 않음
    LaneRuntime& L = lanes[i];

    if (L.ready) {
      if (L.everStarted && !elapsed(now, L.lastStartMs, LANE_MIN_PERIOD_MS)) continue;
      cursorLane = i;
      L.everStarted = true;
      L.lastStartMs = now;
      if (!startMeasurement(i)) {
        activeLane = i;
        finishWithError(i, BUS_MODE == BUS_TCA9548A ? ST_MUX_ERROR : ST_I2C_ERROR, now);
        return;
      }
      activeLane = i;
      phase = PH_WAIT_START;
      phaseStartMs = now;
      lastPollMs = now;
      return;
    }

    if (timeReached(now, L.nextReinitMs)) {
      cursorLane = i;
      tryInitLane(i);
      L.lastErrorReportMs = millis();
      lastMeasEndMs = millis();
      return;  // 한 번에 한 레인만 (다른 레인 지연 최소화)
    }

    if (elapsed(now, L.lastErrorReportMs, ERROR_REPORT_MS)) {
      L.lastErrorReportMs = now;
      emitMeasurement(LANE_HW[i].lane_id, false, 0, L.lastError);
    }
  }
}

// ============================================================================
void setup() {
  Serial.begin(SERIAL_BAUD);
  uint32_t r = esp_random() ^ (uint32_t)micros() ^ ((uint32_t)esp_reset_reason() << 24);
  snprintf(bootId, sizeof(bootId), "%08lx", (unsigned long)r);

  loadBoardConfig();
  Wire.begin(PIN_SDA, PIN_SCL, I2C_CLOCK_HZ);
  Wire.setTimeOut(I2C_TIMEOUT_MS);

  validateConfig();
  uint32_t now = millis();
  for (uint8_t i = 0; i < MAX_LANES; i++) {
    lanes[i].nextReinitMs = now;   // 첫 루프에서 순서대로 초기화
    lanes[i].lastErrorReportMs = now;
  }
  emitHello();
  if (!configOk) emitConfigError();
  lastHelloMs = lastHeartbeatMs = now;

  wifiLoadCreds();
  bool brownout = esp_reset_reason() == ESP_RST_BROWNOUT;
  wifiStartAtMs = now + (brownout ? WIFI_BROWNOUT_DELAY_MS : WIFI_START_DELAY_MS);
  if (brownout) emitLog("WARN", 0, "last reset was brownout (power too weak). Wi-Fi start delayed");
  emitWifiStatus("boot");
}

void loop() {
  uint32_t now = millis();
  serviceSerialInput(now);
  wifiService(now);

  if (elapsed(now, lastHeartbeatMs, HEARTBEAT_MS)) {
    lastHeartbeatMs = now;
    emitHeartbeat();
    if (!configOk) emitConfigError();   // 설정 오류는 계속 명확히 표시
  }
  if (elapsed(now, lastHelloMs, HELLO_MS)) {
    lastHelloMs = now;
    emitHello();
  }
  if (!configOk) return;   // 잘못된 설정에서는 센서에 접근하지 않는다

  if (phase == PH_IDLE) scheduleNext(now);
  else                  serviceMeasurement(now);
}
