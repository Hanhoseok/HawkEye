"""ESP32 JSON Lines 를 받는 최소 예제 (수량 추정 없음, 원시 측정만 출력).

  python examples/min_reader.py COM5
Arduino IDE 시리얼 모니터가 같은 포트를 열고 있으면 실패하므로 먼저 닫는다.
"""
import json
import sys

import serial

port = sys.argv[1] if len(sys.argv) > 1 else "COM5"
ser = serial.Serial()
ser.port, ser.baudrate, ser.timeout = port, 115200, 1
ser.dtr = ser.rts = False          # 포트 열 때 ESP32 리셋 방지
ser.open()
ser.reset_input_buffer()
while True:
    raw = ser.readline()
    if not raw:
        continue
    try:
        msg = json.loads(raw)
    except ValueError:
        continue                    # 부트 로그·잘린 줄
    if msg["message_type"] == "MEASUREMENT":
        print(msg["lane_id"], msg["distance_mm"], msg["valid"], msg["sensor_status"], msg["uptime_ms"])
