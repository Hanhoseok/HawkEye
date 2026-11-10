# 스프링 푸셔 진열대 재고 모니터 (ESP32 + VL53L0X + Python)

레일 맨 뒤의 VL53L0X 가 움직이는 밀판 뒷면까지의 거리를 재고, PC의 Python이 그 거리로 레인별 수량과 입출고를 추정해 창에 레일 5개를 그려 보여 준다.

> 현재 설치(레인마다 ESP32 1대, Wi-Fi 수신)의 하드웨어·센서 동작, 데이터 파이프라인, Wi-Fi 연결, 레인별 보정 상태는 **[docs/SYSTEM_OVERVIEW.md](docs/SYSTEM_OVERVIEW.md)** 에 정리되어 있다.

```
거리 증가 = 밀판이 앞으로 감 = 수량 감소 (REMOVE)
거리 감소 = 밀판이 뒤로 감   = 수량 증가 (ADD)
```

| 구성 | 역할 |
|---|---|
| ESP32 펌웨어 `firmware/shelf_sensor/shelf_sensor.ino` | 센서 초기화, 순차 측정, 완료·타임아웃 관리, 재초기화, JSON Lines 전송. **수량 계산이나 ADD/REMOVE 판단은 하지 않음** |
| Python `python/` | 수신, 필터, 안정 판정, 히스테리시스, 절대 위치 기반 수량 추정, 이벤트 생성, 오류·복구·RESYNC, 재입고 모드, 화면 표시, JSONL 기록 |

```
firmware/shelf_sensor/shelf_sensor.ino   ESP32 펌웨어
python/run.py                            실행 파일
python/config.toml                       레인·상품·보정값 설정  ← 주로 여기를 수정
python/requirements.txt
python/shelf_monitor/
    config.py      설정 로드·검증
    estimator.py   레인 1개 수량 추정 상태 머신 (순수 로직, 전송 방식과 무관)
    link.py        boot_id / sequence / uptime wrap / 수신 백로그 추적
    app.py         메시지 → 추정기 → 이벤트 (시리얼·GUI와 분리)
    sinks.py       콘솔 출력, JSONL 기록
    sources.py     시리얼 수신 스레드, 모의 장치
    gui.py         tkinter 창 (레일 5개 그림)
python/tests/                            모의 거리 시퀀스 테스트
python/examples/min_reader.py            최소 수신 예제
```

---

## 1. 준비물과 설치

### Arduino IDE (펌웨어)
1. Arduino IDE 2.x 를 설치한다.
2. **ESP32 보드 패키지**: `파일 → 기본 설정 → 추가 보드 관리자 URL` 에
   `https://espressif.github.io/arduino-esp32/package_esp32_index.json` 를 넣고, `도구 → 보드 → 보드 매니저` 에서 **esp32 by Espressif Systems** 를 설치한다.
3. **라이브러리**: `스케치 → 라이브러리 포함하기 → 라이브러리 관리` 에서 **VL53L0X by Pololu** 를 설치한다.
   (Adafruit_VL53L0X 가 아님. 저장소: https://github.com/pololu/vl53l0x-arduino)
   TCA9548A 는 I2C 바이트 하나만 쓰면 되므로 별도 라이브러리가 필요 없다.

펌웨어가 쓰는 Pololu API: `setBus`, `setTimeout`, `init`, `setMeasurementTimingBudget`, `readReg`/`writeReg`/`readReg16Bit`, `last_status`(마지막 I2C 결과), `VL53L0X::SYSRANGE_START` 등 레지스터 열거형.
라이브러리의 `readRangeSingleMillimeters()` 는 측정이 끝날 때까지 루프를 막는다. 그래서 같은 레지스터 순서(stop variable 기록 → `SYSRANGE_START`=1 → 시작 비트 해제 대기 → `RESULT_INTERRUPT_STATUS` 확인 → `RESULT_RANGE_STATUS+10` 읽기 → 인터럽트 해제)를 `millis()` 기반 상태 머신으로 직접 수행한다. 라이브러리의 `stop_variable` 은 private 이라, `init()` 과 같은 순서로 레지스터 0x91 에서 다시 읽는다. 측정 상태는 ST API 와 같은 방식(`(0x14 레지스터 & 0x78) >> 3`, 11 = Range Complete)으로 판정한다.

### Python (PC)
- Python **3.11 이상**이 필요하다(설정 파일 파서 `tomllib`). 이 PC에는 3.11 과 3.13 이 설치되어 있다.
- tkinter 는 Windows용 Python 에 기본으로 들어 있다.

```bash
cd F:\arduino\python
py -3.11 -m pip install -r requirements.txt
```

---

## 2. 배선

### 2-1. 현재: 센서 1개 직접 연결 (`BUS_DIRECT_SINGLE`)

| VL53L0X 모듈 | ESP32-WROOM-32 개발보드 |
|---|---|
| VIN (또는 VCC) | 3V3 |
| GND | GND |
| SDA | GPIO21 |
| SCL | GPIO22 |
| XSHUT | 연결 안 함 (대부분의 모듈은 내부 풀업) |
| GPIO1 (INT) | 연결 안 함 |

- 핀은 펌웨어의 `PIN_SDA`, `PIN_SCL` 에서 바꿀 수 있다.
- 센서는 레일 맨 뒤에 고정하고 밀판 뒷면의 **불투명·무광 표적**을 정면으로 보게 한다. VL53L0X 시야각은 약 25° 이므로 레일 벽이나 롤러, 와이어가 빔 안에 들어오지 않도록 높이와 각도를 맞춘다. 표적이 작으면 측정이 바닥이나 벽 쪽으로 섞일 수 있다.
- 사진의 레일 길이(약 40~45cm)는 VL53L0X 기본 모드 범위 안이지만, 표적 반사율과 주변광에 따라 잡음이 늘 수 있다.

### 2-2. 향후: TCA9548A + 센서 5개 (`BUS_TCA9548A`)

| TCA9548A | 연결 |
|---|---|
| VIN | ESP32 3V3 |
| GND | GND |
| SDA / SCL | GPIO21 / GPIO22 |
| A0, A1, A2 | GND → 주소 0x70 |
| RESET | 3V3 (모듈에 풀업이 있으면 생략) |

| 레인 | TCA 채널 | 센서 SDA | 센서 SCL | 전원 |
|---|---|---|---|---|
| 1 | 0 | SD0 | SC0 | 3V3, GND 공통 |
| 2 | 1 | SD1 | SC1 | 3V3, GND 공통 |
| 3 | 2 | SD2 | SC2 | 3V3, GND 공통 |
| 4 | 3 | SD3 | SC3 | 3V3, GND 공통 |
| 5 | 4 | SD4 | SC4 | 3V3, GND 공통 |

모든 센서가 같은 주소 0x29 를 쓰지만 채널이 분리되어 있어 충돌하지 않는다. TCA9548A 는 **주소 충돌만 해결**할 뿐 빛 간섭은 막지 않는다. 그래서 펌웨어는 한 번에 한 센서만 측정을 시작한다(측정 완료 또는 타임아웃 → 다음 센서). 선이 길면 `I2C_CLOCK_HZ` 를 100kHz 로 유지한다.
한 센서 케이블이 SDA를 GND로 붙잡는 식으로 고장 나면 TCA9548A 상류 버스 전체가 막힐 수 있다. 이는 배선 문제라 소프트웨어로 완전히 막을 수 없다.

#### 5개로 바꿀 때 수정할 곳
1. 펌웨어 `shelf_sensor.ino`
   ```cpp
   static const BusMode BUS_MODE = BUS_TCA9548A;
   static LaneHwConfig LANE_HW[MAX_LANES] = {
     { 1, true, 0 }, { 2, true, 1 }, { 3, true, 2 }, { 4, true, 3 }, { 5, true, 4 },
   };
   ```
   `BUS_DIRECT_SINGLE` 인데 활성 레인이 2개 이상이면 펌웨어는 센서에 접근하지 않는다. 대신 `CONFIG_ERROR` 메시지를 1초마다 보내고, Python 창 상단에도 빨간 글씨로 표시된다.
2. Python `config.toml` 의 각 `[[lanes]]` 에서 `enabled = true`, `tca_channel` 을 펌웨어와 같게 맞춘다. 펌웨어와 Python 의 활성 레인이 다르면 `LANE_CONFIG_MISMATCH` 가 기록된다.

---

## 3. 펌웨어 업로드 (Arduino IDE)

1. `F:\arduino\firmware\shelf_sensor\shelf_sensor.ino` 를 연다. 폴더 이름과 파일 이름이 같아야 한다.
2. `도구 → 보드 → esp32 → ESP32 Dev Module` 을 고른다.
3. `도구 → 포트` 에서 ESP32 의 COM 포트를 고른다(아래 4장 참고).
4. 업로드한다. `Connecting...` 에서 멈추면 보드의 **BOOT** 버튼을 누른 채 기다렸다가 업로드가 시작되면 뗀다.
5. 확인: `도구 → 시리얼 모니터` 를 115200 baud 로 열면 다음과 같은 줄이 계속 나온다.
   ```json
   {"message_type":"MEASUREMENT","device_id":"shelf-esp32-01","boot_id":"3fa2c91b","sequence":57,"uptime_ms":6120,"lane_id":1,"distance_mm":312,"valid":true,"sensor_status":"OK"}
   {"message_type":"MEASUREMENT","device_id":"shelf-esp32-01","boot_id":"3fa2c91b","sequence":58,"uptime_ms":6224,"lane_id":1,"distance_mm":null,"valid":false,"sensor_status":"TIMEOUT"}
   ```
6. **확인이 끝나면 시리얼 모니터를 반드시 닫는다.** Windows에서는 한 COM 포트를 한 프로그램만 열 수 있다. 시리얼 모니터가 열려 있으면 Python 이 `COMx 사용 중 (Arduino 시리얼 모니터 등 다른 프로그램을 닫으세요)` 를 표시하고 2초마다 다시 시도한다. 반대로 Python 실행 중에는 업로드도 실패하므로 Python 을 먼저 끈다.

### 펌웨어 메시지 형식
| message_type | 필드 |
|---|---|
| `MEASUREMENT` | `message_type, device_id, boot_id, sequence, uptime_ms, lane_id, distance_mm, valid, sensor_status` |
| `HELLO` (부팅 시 + 30초마다) | 공통 헤더 + fw_version, bus_mode, timing_budget_us, enabled_lanes, reset_reason |
| `HEARTBEAT` (1초마다) | 공통 헤더 + config_ok, enabled_lanes |
| `CONFIG_ERROR` / `LOG` | 공통 헤더 + text (LOG 는 level, lane_id) |

- 디버그 문자열을 따로 찍지 않는다. 모든 출력이 한 줄 JSON 이다. 다만 전원을 켠 직후 ESP32 ROM 부트로더가 JSON이 아닌 줄을 내보낼 수 있는데, Python 은 이런 줄을 파싱 오류로 세고 버린다.
- `distance_mm` 는 측정 실패 시 `null` 이다. 0 으로 대체하지 않는다.
- `sensor_status`: `OK`, `NO_TARGET`(표적 없음/범위 밖), `RANGE_INVALID`, `TIMEOUT`, `I2C_ERROR`, `NOT_CONNECTED`(0x29 응답 없음/모델 ID 불일치), `INIT_FAILED`, `MUX_ERROR`(TCA 채널 선택 실패)
- 정상적으로 측정할 때마다 1줄씩 보낸다. 오류 레인은 센서에 접근하지 않고 500ms마다 상태를 보고하며, 재초기화는 1→2→4→…최대 10초 간격으로 시도한다. 그동안 다른 레인은 계속 측정한다.
- `uptime_ms` 는 **ESP32 부팅 후 경과 ms 로, 실제 날짜·시각이 아니다.** 약 49.7일마다 0으로 돌아간다. 펌웨어는 `(uint32_t)(now - since)` 형태로 비교하고, Python 은 32비트 차이로 이어 붙인다.

---

## 4. Windows 에서 COM 포트 확인과 실행

- **장치 관리자 → 포트(COM & LPT)** 에서 `Silicon Labs CP210x USB to UART Bridge (COM5)` 나 `USB-SERIAL CH340 (COM7)` 같은 항목을 찾는다. 안 보이면 해당 USB-UART 드라이버를 설치하고, 충전 전용이 아닌 데이터 케이블을 쓴다.
- 또는 다음 명령으로 확인한다.
  ```bash
  python run.py --list-ports
  ```
  CP210x/CH340/FTDI/Espressif 칩이면 두 번째 열에 칩 이름이 나온다. 지금 이 PC 에서는 블루투스 가상 포트(COM3~6)만 보이므로 ESP32 를 꽂은 뒤 다시 확인해야 한다.

실행 (`F:\arduino\python` 폴더에서):
```bash
python run.py --port COM5
```
- 창이 열리고 레일 5개가 보인다. 비활성 레인은 회색 "비활성"으로 표시된다. 콘솔에는 이벤트와 5초마다의 상태가 한국어로 나온다. 예: `1번 레인 / 콜라 / 2개 제거 / 현재 3개`
- 콘솔만 쓰려면 `python run.py --port COM5 --no-gui` 로 실행한다. `r 1` + Enter 로 1번 레인 재입고 모드를 켜고 끄며, `q` 로 종료한다.
- `config.toml` 의 `port = "auto"` 는 위 USB 칩을 가진 첫 포트를 고른다. 장치가 여러 개면 `--port` 로 지정한다.
- 포트를 열 때 DTR/RTS 를 끈 상태로 열어 ESP32 자동 리셋을 피하도록 했지만, 보드나 드라이버에 따라 리셋될 수 있다. 리셋되더라도 boot_id 변경으로 감지해 RESYNC 처리한다.

### 화면 보는 법
- 위쪽이 레일 뒤(센서), 아래쪽이 앞(고객 쪽 투명 스토퍼)이다. 사진처럼 검은 롤러 레일, 양옆 은색 와이어, 앞 투명 스토퍼를 그렸다.
- **밀판**(검은 판 + 흰 표적)은 실시간 필터 거리 위치에 그리고, 빨간 점선이 센서 빔이다.
- **상품**은 *확정 수량*만큼 앞에서부터 쌓아 그린다(`shape="can"` 은 위에서 본 원). 확정 전에 밀판이 이미 앞으로 온 경우, 밀판보다 뒤에 있어야 할 상품은 점선 윤곽으로만 표시한다(= "변화 확인 중").
- 오른쪽 숫자 눈금은 수량별 기준 거리, 왼쪽 녹색 막대는 허용 오차 구간이다. 확정 수량은 밝은 녹색이다.
- 센서 오류, USB 데이터 없음, stale, 범위 밖 상태는 레일 위에 배지로 표시된다. 이때도 수량을 0으로 바꾸지 않고 마지막 확정 수량을 회색으로 유지한다.
- 레인마다 **재입고 시작/종료** 버튼이 있다.
- 하단 목록은 최근 이벤트다. 확정 이벤트는 `python/logs/events.jsonl`, 주기 상태는 `python/logs/status.jsonl` 에 저장된다.

---

## 4-1. Wi-Fi 로 받기 (펌웨어 1.1.0 이상)

ESP32 는 USB 와 똑같은 JSON 줄을 **UDP** 로도 보낸다. 한 데이터그램이 JSON 한 줄이다. 기본값은 같은 공유기 안 브로드캐스트라서, PC 쪽에 ESP32 의 IP 를 적을 필요가 없다.

1. **Wi-Fi 설정 (USB 연결 상태에서 한 번만)**. run.py, sensor_check.py, Arduino 시리얼 모니터는 먼저 닫는다.
   ```bash
   python tools/wifi_setup.py --port COM7
   ```
   - Wi-Fi 이름과 비밀번호를 물어본다. 비밀번호는 화면에 보이지 않고 ESP32 내부 저장소(NVS)에만 저장된다. 코드나 PC 파일에는 남지 않는다.
   - ESP32 는 **2.4GHz** Wi-Fi 만 연결된다.
   - 연결되면 ESP32 의 IP 와 신호 세기가 표시된다.
   - 상태 확인은 `--status`, 지우기는 `--clear` 다.
2. **실행**: 이제 ESP32 는 USB 대신 일반 USB 충전기만 꽂아도 된다.
   ```bash
   python run.py --wifi
   ```
   매번 `--wifi` 를 쓰기 싫으면 `config.toml` 의 `[network] source = "wifi"` 로 바꾼다. 거리 확인 도구도 `python tools/sensor_check.py --wifi` 로 쓸 수 있다.
3. **Windows 방화벽**: 처음 실행할 때 "Python 이 네트워크에 액세스하도록 허용" 창이 뜨면 **개인 네트워크**를 허용한다. 이미 거부했다면 Windows 보안 → 방화벽 → 앱 허용에서 Python 을 허용한다. 허용하지 않으면 "데이터 없음(stale)" 으로만 나온다.

**동작과 한계**
- UDP 는 도착을 보장하지 않는다. 몇 줄 빠지는 것(5줄 이하)은 기록만 하고 수량 판단은 그대로 진행한다. 크게 빠지거나 끊기면 USB 끊김과 같이 stale → RESYNC 로 처리한다.
- 브로드캐스트를 막는 공유기(게스트망, AP 격리 등)에서는 받지 못한다. 이때는 펌웨어의 `UDP_TARGET_IP` 에 PC 의 IP 를 적어 그 PC 로만 보내게 한다. 이 방식이 손실도 더 적다.
- Wi-Fi 를 켜면 순간 전류가 커진다. 처음 받은 보드의 공장 펌웨어가 Wi-Fi 를 켤 때 전압 부족(brownout)으로 계속 재부팅했던 것과 같은 현상을 줄이려고, 송신 출력을 낮추고(`WIFI_TX_POWER`) 부팅 3초 뒤에 Wi-Fi 를 시작한다. 직전 리셋이 전압 부족이면 30초 늦게 시작한다. 그래도 재부팅이 반복되면 전원(포트, 케이블, 충전기)을 바꾸거나 `--clear` 로 Wi-Fi 를 끈다.
- Wi-Fi 가 연결되지 않아도 USB 출력과 측정은 그대로 동작한다. 연결이 끊기면 10초마다 다시 시도한다.

## 4-2. 레인마다 ESP32 를 1대씩 쓸 때 (펌웨어 1.2.0 이상)

모든 보드에 **같은 펌웨어**를 올리고, 보드마다 **레인 번호만 다르게** 지정한다. 레인 번호는 보드 내부 저장소에 저장되고, 보드 이름(device_id)은 `shelf-lane-<번호>` 가 된다. 지정하지 않은 보드는 1번 레인이다.

보드 하나씩 USB 로 연결해서:
1. 업로드: `arduino-cli upload -p COMx --fqbn esp32:esp32:esp32 firmware/shelf_sensor` (또는 Arduino IDE)
2. 레인 번호와 Wi-Fi 설정. 보드는 번호를 저장한 뒤 재시작한다.
   ```bash
   python tools/wifi_setup.py --port COMx --lane 2
   ```
   이미 Wi-Fi 가 저장된 보드라면 다시 입력할지 묻는다. 확인은 `--status` 로 한다.
3. 모든 보드를 마친 뒤, PC 에서 Wi-Fi 하나로 전부 받는다.
   ```bash
   python run.py --wifi
   ```

Python 은 보드별로 재부팅, 메시지 누락, 수신 지연, 데이터 끊김을 **따로** 추적한다. 한 보드가 꺼지거나 재부팅해도 그 보드의 레인만 stale 또는 RESYNC 가 되고 다른 레인은 계속 동작한다.
- 두 보드가 같은 레인 번호를 보내면 `LANE_CONFLICT` 경고를 내고, 나중에 나타난 보드의 값은 쓰지 않는다. 이때는 `--lane` 으로 번호를 고친다.
- 고장 난 보드를 같은 번호의 새 보드로 바꾸면 `LANE_OWNER_CHANGED` 후 RESYNC 로 이어받는다.
- 창 상단에 보드 목록과 IP 가 표시된다. ● 는 수신 중, ○ 는 끊김이다.

주의할 점:
- 보드마다 따로 측정하므로 **옆 레인 센서가 동시에 발광할 수 있다.** 레인 사이에 검은 무광 칸막이를 두고, 옆 레인을 움직일 때 내 레인 값이 흔들리지 않는지 `tools/sensor_check.py --wifi --lane N` 으로 확인한다.
- 보드 4대를 PC USB 에 한꺼번에 꽂으면 전원이 부족할 수 있다. 일반 USB 충전기나 전원 공급형 USB 허브를 쓴다.
- `config.toml` 에서 해당 레인의 `enabled = true` 와 상품, 보정값을 맞춘다. 2~4번 레인은 현재 예시값이다.

## 5. 설정 수정 위치 (`python/config.toml`)

| 항목 | 위치 | 설명 |
|---|---|---|
| 상품 ID·이름 | `[[lanes]] product_id, product_name` | 센서는 상품 종류를 모른다. 여기서 레인별로 등록한다 |
| 상품 두께 | `thickness_mm` | 진열(밀판 이동) 방향 길이. **세운 캔은 높이가 아니라 지름**. 레인마다 다르게 지정 가능 |
| 빈 레인 거리 | `empty_distance_mm` | 0개일 때 실측 거리 (아래 보정 참고) |
| 최대 수량 | `max_count` | |
| 실측 거리표 | `count_distance_mm = [0개, 1개, …]` | 있으면 두께 계산보다 우선. 길이는 max_count+1, 값은 감소해야 함 |
| 허용 오차 / 히스테리시스 | `tolerance_mm`, `hysteresis_mm` | 검증 규칙: `2*tolerance + hysteresis < 인접 수량 간격` |
| 유효 거리 | `valid_min_mm`, `valid_max_mm` | 범위 밖 측정은 무효 처리 |
| 안정 판정 | `stable_time_ms`, `min_stable_samples`, `stable_spread_mm` | |
| 필터 | `median_window`, `max_sample_gap_ms` | |
| 오류 판정 | `error_invalid_count`, `error_invalid_ms` | |
| TCA 채널 | `tca_channel` | 펌웨어 `LANE_HW` 와 동일하게 |

공통값은 `[lane_defaults]` 에 두고, 레인별로 다르게 하려면 해당 `[[lanes]]` 에 같은 키를 다시 적는다. 잘못된 설정(허용 구간 겹침, 표 길이·순서 오류, 빈 레인 거리가 유효 범위 밖 등)이 있으면 프로그램이 시작하지 않고 이유를 출력한다.

---

## 6. 최초 보정과 확인 순서

1. **배선과 펌웨어 확인**: 시리얼 모니터에서 `valid:true` 와 `sensor_status:"OK"` 가 나오는지 본다. 손을 센서 앞에서 움직여 거리가 바뀌는지 확인한 뒤 **시리얼 모니터를 닫는다**.
2. **빈 레인 거리**
   - 상품을 모두 빼고 밀판이 앞 스토퍼에 닿아 멈춘 상태로 둔다.
   - `python run.py --port COM5` 를 실행해 창의 `거리 ___ mm` 값을 10~20초 지켜본다. `logs/status.jsonl` 의 `distance_mm` 중앙값을 써도 된다.
   - 그 값을 `empty_distance_mm` 에 넣는다. 처음에는 설정이 틀려 `수량 범위 밖` 이 뜰 수 있는데, 정상이다.
   - 이 값은 **사람이 정하는 기준값이며 프로그램이 자동으로 바꾸지 않는다.** 정상적인 측정 흔들림을 따라 기준이 움직이면 수량 오차가 누적되기 때문이다.
3. **두께 확인**: 상품을 1개, 2개, … 넣으면서 거리값을 기록한다.
   - 각 수량에서 실측 거리와 `empty - n*thickness` 차이가 `tolerance_mm` 의 절반을 넘으면, 상품 사이 틈, 밀판 기울어짐, 스프링 힘 차이 때문일 수 있다. 이때는 **`count_distance_mm` 실측표**를 쓴다.
4. **허용 오차 정하기**: 정지 상태에서 거리값의 흔들림(최대−최소)을 본다. 측정값이 mm 단위로 나온다고 정확도가 1mm인 것은 아니다. VL53L0X 는 표적, 거리, 주변광에 따라 수 mm~수십 mm 오차가 생길 수 있다. `tolerance_mm` 를 이 흔들림과 보정 오차보다 넉넉하게 잡되, `2*tolerance + hysteresis < 두께` 를 지킨다. 얇은 상품(예: 30mm 이하)은 레인 간격이 작아 오판 위험이 커진다.
5. **동작 시험**: 1개 빼기 → `REMOVE 1`, 2개 동시에 빼기 → `REMOVE 2`, 넣기 → `ADD`. 손으로 센서 앞을 잠깐 가렸을 때 이벤트가 없어야 한다.
6. **상품 종류나 두께가 바뀌면** `thickness_mm`, `empty_distance_mm`(밀판이나 표적을 바꾼 경우), `count_distance_mm` 를 다시 보정한다.

---

## 7. 수량 추정과 상태 머신 (Python)

1. 유효 측정(`valid=true` 이면서 유효 거리 범위 안)만 **중앙값 필터**(기본 5개)에 넣는다. 짧은 필터라 실제 변화가 크게 늦어지지 않는다.
2. 필터값 `m` 으로 후보 수량을 구한다.
   - 기준거리 `ref(n) = count_distance_mm[n]` 또는 `empty_distance_mm − n*thickness_mm`
   - 현재 확정 수량 c 는 `|m − ref(c)| ≤ tolerance + hysteresis` 이면 유지된다(히스테리시스).
   - 다른 n 은 `|m − ref(n)| ≤ tolerance` 일 때만 후보가 된다.
   - 어느 구간에도 들지 않으면 `BETWEEN`(불확실), 빈 레인보다 멀면 `BEYOND_EMPTY`, max_count 보다 가까우면 `OVER_MAX` 다. 범위 밖 값은 0 이나 max_count 로 **잘라서 확정하지 않고** `OUT_OF_RANGE` 상태와 이벤트로 알린다.
3. **확정 조건** (장치 측정 시각 `uptime_ms` 기준)
   - 같은 후보가 `stable_time_ms` 이상 유지되고, 그 사이 연속 유효 샘플이 `min_stable_samples` 이상이며, 최근 구간의 필터값 폭이 `stable_spread_mm` 이하여야 한다.
   - 후보가 바뀌거나, 폭이 커지거나(스프링 반동), 무효 측정이 끼거나, 샘플 간격이 `max_sample_gap_ms` 를 넘거나, 수신 백로그·sequence 누락·재부팅·USB 끊김이 생기면 안정 타이머와 필터를 다시 시작한다.
4. 확정 수량은 매번 **절대 위치**에서 다시 계산하고, 이전 확정 수량과 다를 때만 이벤트를 **한 번** 낸다. 이동량을 누적하지 않으므로 오차가 쌓이지 않는다. 변화량이 커도 버리지 않으므로 여러 개를 한꺼번에 빼거나 넣어도 한 이벤트로 처리된다. 이동 중간값을 놓쳐도 이동 전후의 안정 위치만으로 판단한다.

상태: `INITIALIZING`, `STABLE`, `CHANGING`, `UNCERTAIN`, `OUT_OF_RANGE`, `SENSOR_ERROR`. 이와 별도로 `stale`, `restock`, `pending_resync` 플래그가 있다.

### 이벤트 (`logs/events.jsonl`, 한 줄 = JSON 하나)
| event | 의미 |
|---|---|
| `INIT` | Python 시작 후 레인의 첫 안정 수량. 이미 있던 상품을 ADD 로 기록하지 않는다 |
| `ADD` / `REMOVE` | 관찰 중 확정 수량 변화. `quantity` = 변화 개수 |
| `RESYNC` | 센서 오류, USB 끊김, 데이터 없음, ESP32 재부팅, 긴 공백, 재입고 종료 뒤 다시 동기화. `net_change`, `reason`, `observed:false` 를 함께 기록한다. **입출고로 단정하지 않는다** |
| `SENSOR_ERROR` / `RECOVERED` | 센서 오류 발생(종류 `error_kind`)과 복구. 오류 중에는 마지막 확정 수량을 stale 로 유지한다 |
| `OUT_OF_RANGE` | 안정적이지만 수량 범위 밖인 거리. 설치나 보정을 확인해야 한다 |
| `RESTOCK_START` / `RESTOCK_END` | 재입고 모드 |
| (SYSTEM) `LINK_UP`, `LINK_LOST`, `LINK_STALE`, `LINK_RESTORED`, `DEVICE_REBOOT`, `SEQUENCE_GAP`, `BACKLOG_START/END`, `DEVICE_CONFIG_ERROR`, `LANE_CONFIG_MISMATCH` | 통신 상태 |

이벤트 필드: `message_type, device_id, boot_id, sequence, uptime_ms, lane_id, product_id, product_name, event, previous_count, current_count, quantity, distance_mm, status` (+ `net_change`, `error_kind`, `reason` 등), `pc_time`(PC 기록 시각), `pc_event_seq`.
주기 상태는 `logs/status.jsonl` 에 `message_type:"STATUS"` 로 기록된다. 상태 보고는 이벤트를 만들지 않는다. 상태 보고에는 `count`(첫 확정 전에는 `null`), `stale`, `seconds_since_valid`, `last_valid_uptime_ms` 가 들어 있다.

### 통신과 시간 처리
- **센서 오류와 USB 끊김을 구분한다.**
  - 센서 오류는 `sensor_status` 가 `OK` 가 아닌 측정에서 온다. `NOT_CONNECTED`/`INIT_FAILED`/`MUX_ERROR` 는 즉시, 그 외는 연속 5회 또는 1.5초 동안 지속되면 SENSOR_ERROR 로 본다.
  - USB 끊김은 포트 예외(`LINK_LOST`)나 `link_stale_s` 동안 아무 메시지도 없는 경우(`LINK_STALE`, ESP32 는 1초마다 HEARTBEAT 전송)로 판단한다. 이때 수량을 0 으로 보지 않는다.
- **시계 분리**
  - 안정 판정 시간은 ESP32 `uptime_ms` 차이로만 계산하고, stale 판정은 PC `time.monotonic()` 차이로만 계산한다.
  - 두 시계 값을 서로 빼서 시각을 환산하지 않는다.
  - PC 수신 시각(`pc_time`)은 기록용이다.
- **버퍼 적체 판정**
  - 메시지마다 `(PC 수신 간격 − 장치 측정 간격)` 을 누적한 지연이 `backlog_limit_ms` 를 넘으면, 그 샘플들은 안정 판정에서 뺀다(`BACKLOG_START`). PC 가 멈췄다가 쌓인 데이터를 한꺼번에 읽어도 실시간 연속 측정으로 오인하지 않는다.
  - 포트를 열 때는 OS 수신 버퍼를 비운다.
- **sequence**
  - 같은 boot_id 안에서 중복이나 과거 번호는 버린다.
  - 번호가 건너뛰면 `SEQUENCE_GAP` 를 기록하고 모든 레인의 안정 구간을 새로 시작한다.
  - boot_id 가 바뀌면 재부팅으로 보고 필터와 시간 기준을 버린 뒤 RESYNC 를 기다린다.
- **재연결**: USB 가 다시 연결되면 필터와 안정 구간을 새로 시작한다. 다음 안정 수량은 RESYNC 로 처리한다.

### 측정 예산, 재방문 주기, 필터, 안정 시간의 관계와 예상 지연
| 항목 | 기본값 | 비고 |
|---|---|---|
| 측정 예산 `TIMING_BUDGET_US` | 33ms | VL53L0X 기본. 늘리면 잡음은 줄고 주기는 길어짐 |
| 측정 1회 소요 | 약 38~42ms | 예산 + 시작·읽기 I2C(100kHz) + 폴링 2ms + 휴지 2ms |
| 레인 재방문 주기 | 센서 1개: 100ms (`LANE_MIN_PERIOD_MS`) / 5개: 약 200ms (5×40ms) | 순차 측정이므로 센서 수에 비례 |
| 측정 타임아웃 | 120ms | 넘으면 TIMEOUT, 다음 레인으로 |
| 중앙값 필터 5개 | 1개: 0.5초 분량 / 5개: 1.0초 분량 | 새 위치가 다수가 되기까지 약 3샘플 = 0.3초 / 0.6초 |
| 안정 시간 1200ms + 최소 5샘플 | 1개: 약 12샘플 / 5개: 약 6샘플 | 5개일 때도 최소 샘플 수를 만족한다 |
| `max_sample_gap_ms` 600 | 5개 재방문 200ms + 타임아웃 1회 여유 | |

예상 감지 지연(밀판이 멈춘 뒤부터 확정까지)은 센서 1개에서 약 1.3~1.6초, 5개에서 약 1.6~2.0초다. 여기에 스프링 반동이 잦아드는 시간(보통 수백 ms, 레일마다 다름)이 더해진다. 실제 하드웨어로 측정한 값이 아니라 설정값에서 계산한 추정치다. 반응을 빠르게 하려면 `stable_time_ms` 를 줄이면 되지만, 반동이나 손 동작을 잘못 확정할 위험이 커진다.

### 재입고 모드
버튼(또는 콘솔 `r 1`)으로 켠다. 켜진 동안에는 사람이 밀판을 뒤로 당기거나 잡고 있어도 ADD/REMOVE 를 내지 않고, 안정된 수량은 "관측"으로 화면에만 표시한다. 끄면 필터를 새로 시작해 안정된 수량으로 `RESYNC`(reason `RESTOCK_END`, 재입고 전 수량 대비 net_change)를 기록한다. 켜둔 채 잊는 경우를 막기 위해 `restock_timeout_s`(기본 600초)가 지나면 자동으로 끈다.

---

## 8. JSON Lines 최소 수신 예제

`python/examples/min_reader.py`:
```python
import json, serial
ser = serial.Serial(); ser.port, ser.baudrate, ser.timeout = "COM5", 115200, 1
ser.dtr = ser.rts = False; ser.open(); ser.reset_input_buffer()
while True:
    raw = ser.readline()
    try: msg = json.loads(raw)
    except ValueError: continue
    if msg["message_type"] == "MEASUREMENT":
        print(msg["lane_id"], msg["distance_mm"], msg["valid"], msg["sensor_status"])
```
이벤트 파일 읽기: `[json.loads(l) for l in open("logs/events.jsonl", encoding="utf-8")]`

---

## 9. 하드웨어 없이 검증

### 9-1. 자동 테스트 (모의 거리 시퀀스)
```bash
cd F:\arduino\python
python -m unittest discover -s tests -t . -v
```
`tests/helpers.py` 의 `Feeder` 가 장치 시각을 진행시키며 거리값(잡음, 튐, 스프링 반동, 무효값 포함)을 추정기에 넣는다. `test_monitor.py` 는 JSON 줄을 직접 넣어 링크 처리부터 이벤트까지 전체 경로를 시험한다. 작성 시점에 34개 테스트가 모두 통과했다.

### 9-2. 화면 시연 (모의 장치)
```bash
python run.py --mock --enable-all
```
5개 레인에 무작위 동작(제거, 여러 개 제거, 추가, 손 가림, 밀판 당김, 3번 레인 센서 끊김)과 잡음, 반동을 넣은 모의 장치가 실제 펌웨어와 같은 JSON 을 만든다. `--seed 7` 로 재현할 수 있다.

### 9-3. 시나리오별 기대 결과
| 시나리오 | 기대 이벤트 | 나오면 안 되는 이벤트 | 테스트 |
|---|---|---|---|
| 수량 그대로, 거리만 흔들림(잡음·순간 튐) | 없음 | ADD/REMOVE | `test_noise_and_spikes_no_event`, `test_same_count_no_event` |
| 1개 제거 | `REMOVE quantity=1` 한 번 | 중복 REMOVE | `test_remove_one` |
| 여러 개 동시 제거 | `REMOVE quantity=n` 한 번 | 1개씩 여러 번 | `test_remove_many_at_once_with_fast_jump` |
| 1개 / 여러 개 추가 | `ADD 1`, `ADD 3` | | `test_add_one_and_many` |
| 밀판이 빠르게 이동해 중간값 없음 | 이동 후 위치로 한 번 확정 | 중간 수량 이벤트 | `..._fast_jump`, `test_spring_rebound_single_event` |
| 순간적인 손 가림 | 없음 | ADD/REMOVE | `test_brief_hand_cover` |
| 손이 오래 가림(max 보다 가까움) | `OUT_OF_RANGE` | max_count 로 잘라 ADD | `test_hand_held_long_...` |
| 수량 경계에서 반복 흔들림 | 없음 (UNCERTAIN) | 반복 ADD/REMOVE | `test_boundary_oscillation_hysteresis` |
| 센서 끊김 → 복구(그 사이 1개 빠짐) | `SENSOR_ERROR` → `RECOVERED` → `RESYNC net_change=-1` | REMOVE, 재고 0 | `test_disconnect_and_recover_gives_resync` |
| 상품이 있는 상태에서 부팅 | `INIT` | ADD | `test_boot_with_items_is_init_not_add` |
| 5개 중 1개만 오류 | 해당 레인만 SENSOR_ERROR, 나머지는 정상 REMOVE/ADD | 다른 레인 정지 | `test_one_of_five_lanes_fails` |
| 재입고 중 밀판 당겼다 놓음 | `RESTOCK_END` 후 `RESYNC 2→4` | 중간 ADD/REMOVE | `test_restock_pull_and_release` |
| ESP32 재부팅 / USB 재연결 | `DEVICE_REBOOT`/`LINK_LOST` → `RESYNC` | ADD/REMOVE | `test_device_reboot_resync`, `test_usb_disconnect_resync` |
| 데이터 끊김 | `LINK_STALE`, 화면 stale, 수량 유지 | 재고 0 | `test_no_data_marks_stale_not_zero` |
| 버퍼 적체 데이터 일괄 수신 | `BACKLOG_START`, 실시간 재개 후 확정 | 몰려온 데이터로 즉시 확정 | `test_backlog_burst_not_treated_as_realtime` |
| sequence 중복·부트로그 | 무시, 카운트 | | `test_duplicates_and_garbage_ignored` |
| uptime / sequence 32비트 wrap | 시간·누락 계산 정상 | | `test_uptime_wrap`, `test_sequence_gap_and_wrap` |

---

## 10. 남는 한계 (해결했다고 주장하지 않음)

- 거리만으로는 다음을 확실히 구분할 수 없다.
  - **손이나 물체가 표적을 오래 가리는 경우**: 가린 거리가 어떤 수량의 기준거리와 우연히 같고, 안정 시간 이상 가만히 있으면 잘못된 ADD/REMOVE 가 날 수 있다.
  - **사람이 밀판을 오래 잡고 있는 경우**: 재입고 모드를 켜지 않았다면 오판할 수 있다.
  - **상품이 걸려서 밀판이 덜 나온 경우**: 상품을 뺐는데 밀판이 그대로면 변화가 없다고 판단한다.
- 상품 사이 틈, 기울어짐, 찌그러진 포장처럼 실제 점유 길이가 다르면 두께 기반 계산이 어긋난다. 실측표로 줄일 수는 있지만 없앨 수는 없다.
- 오류나 끊김 동안 일어난 변화는 RESYNC 의 순변화량으로만 알 수 있다. 실제로 몇 번 빼고 넣었는지는 알 수 없다(예: 2개 빼고 2개 넣으면 0).
- VL53L0X 측정값의 정확도는 표적 재질, 각도, 주변광(햇빛), 거리에 따라 달라진다. 이 코드는 정확도를 보장하지 않는다.
- 같은 레일의 인접 센서끼리는 순차 측정으로 동시 발광을 피하지만, 다른 장치의 IR 이나 강한 햇빛에 의한 간섭까지 막지는 못한다.

## 11. 검증하지 않은 부분 (명확히 구분)

- **실제 하드웨어 시험을 하지 않았다.** ESP32, VL53L0X, TCA9548A 동작, 레지스터 순서, 측정 상태값 판정, 재초기화, 핫플러그 복구, 예상 지연은 실물로 확인해야 한다.
- **펌웨어는 이 PC 에서 컴파일하지 않았다.** Arduino 툴체인이 설치되어 있지 않다. Pololu 라이브러리의 공개 헤더와 소스를 확인해 API 이름과 공개 여부를 맞췄지만, Arduino IDE 에서 처음 빌드할 때 경고나 오류가 나면 알려 주면 된다.
- Python 쪽은 자동 테스트 34개와 모의 장치(콘솔·GUI)로만 확인했다. 실제 COM 포트 연결과 끊김, 재연결은 시험하지 않았다.
- 기본 설정값(허용 오차, 안정 시간 등)은 예시다. 실제 레일에서 6장의 순서대로 보정해야 한다.
