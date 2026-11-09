"""ESP32 Wi-Fi 설정 (USB 로 한 번만). 비밀번호는 화면에 표시되지 않고, ESP32 내부 저장소(NVS)에만 저장된다.

  python tools/wifi_setup.py --port COM7            # Wi-Fi 이름/비밀번호 입력 → 저장 → 연결 확인
  python tools/wifi_setup.py --port COM8 --lane 2   # 이 보드를 2번 레인으로 지정(재시작) + Wi-Fi 설정
  python tools/wifi_setup.py --port COM7 --status   # 현재 상태 (레인 번호, Wi-Fi, IP)
  python tools/wifi_setup.py --port COM7 --clear    # 저장된 Wi-Fi 지우기 (USB 전용으로 돌아감)

레인마다 ESP32 를 1대씩 쓸 때: 모든 보드에 같은 펌웨어를 올리고, 보드마다 --lane 으로 번호를 다르게 준다.

ESP32 는 2.4GHz Wi-Fi 만 연결할 수 있다. run.py, sensor_check.py, Arduino 시리얼 모니터는 먼저 닫는다.
"""
from __future__ import annotations

import argparse
import getpass
import json
import sys
import time

import serial


def open_port(port: str) -> serial.Serial:
    ser = serial.Serial()
    ser.port, ser.baudrate, ser.timeout = port, 115200, 0.2
    ser.dtr = ser.rts = False          # 포트를 열 때 ESP32 리셋 방지
    ser.open()
    ser.reset_input_buffer()
    return ser


def read_messages(ser: serial.Serial, seconds: float):
    end = time.monotonic() + seconds
    buf = b""
    while time.monotonic() < end:
        buf += ser.read(512)
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            try:
                yield json.loads(line)
            except ValueError:
                continue


def wait_alive(ser: serial.Serial) -> bool:
    for m in read_messages(ser, 4):
        if m.get("message_type"):
            return True
    return False


def send(ser: serial.Serial, obj: dict) -> None:
    ser.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
    ser.flush()


def show(m: dict) -> None:
    state = m.get("state")
    text = {"CONNECTED": "연결됨", "CONNECTING": "연결 중", "WAITING": "시작 대기",
            "NOT_CONFIGURED": "설정 없음", "DISABLED": "펌웨어에서 꺼짐"}.get(state, state)
    line = f"  보드 {m.get('device_id')}"
    if m.get("lane_id") is not None:
        line += f" (레인 {m['lane_id']})"
    line += f" · Wi-Fi: {text}"
    if m.get("ssid"):
        line += f" · 이름 {m['ssid']}"
    if m.get("ip"):
        line += f" · ESP32 IP {m['ip']} · 신호 {m.get('rssi')} dBm · 전송 대상 {m.get('udp_target')}:{m.get('udp_port')}"
    if m.get("note", "").startswith("rejected"):
        line += f" · 거부됨 ({m['note']})"
    print(line, flush=True)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--clear", action="store_true")
    ap.add_argument("--lane", type=int, help="이 보드가 맡을 레인 번호 1~5 (보드마다 다르게)")
    args = ap.parse_args()
    if args.lane is not None and not 1 <= args.lane <= 5:
        ap.error("--lane 은 1~5")

    try:
        ser = open_port(args.port)
    except serial.SerialException as e:
        print(f"{args.port} 를 열 수 없음: {e}\n(run.py, sensor_check.py, Arduino 시리얼 모니터를 닫으세요)")
        return 1
    try:
        if not wait_alive(ser):
            print("ESP32 응답이 없습니다. 펌웨어(1.1.0 이상)가 올라가 있는지, 포트가 맞는지 확인하세요.")
            return 1

        if args.lane is not None:
            send(ser, {"cmd": "set_lane", "lane": str(args.lane)})
            current = None
            for m in read_messages(ser, 10):          # 저장 후 재시작 → 부팅 메시지에 새 레인 번호
                if m.get("message_type") == "WIFI_STATUS" and m.get("lane_id") == args.lane:
                    current = m
                    break
                if m.get("message_type") == "WIFI_STATUS" and m.get("note", "").startswith("rejected"):
                    show(m)
                    return 1
            if current is None:
                print("레인 번호 설정 응답이 없습니다. 펌웨어가 1.2.0 이상인지 확인하세요.")
                return 1
            print(f"레인 {args.lane} 으로 설정했습니다.")
            show(current)
            if args.status or args.clear:
                pass
            elif current.get("state") != "NOT_CONFIGURED":
                again = input("이 보드에는 Wi-Fi 가 이미 저장돼 있습니다. 새로 입력할까요? [y/N]: ").strip().lower()
                if again != "y":
                    return 0

        if args.status or args.clear:
            send(ser, {"cmd": "clear_wifi" if args.clear else "wifi_status"})
            for m in read_messages(ser, 3):
                if m.get("message_type") == "WIFI_STATUS":
                    show(m)
                    return 0
            print("응답 없음")
            return 1

        ssid = input("Wi-Fi 이름(SSID, 2.4GHz): ").strip()
        password = getpass.getpass("Wi-Fi 비밀번호 (입력해도 화면에 안 보임, 없으면 Enter): ")
        if not 1 <= len(ssid.encode("utf-8")) <= 32:
            print("Wi-Fi 이름은 1~32바이트여야 합니다.")
            return 1
        if len(password.encode("utf-8")) > 63 or 0 < len(password) < 8:
            print("비밀번호는 8~63자(또는 비밀번호 없는 공유기는 빈 값)여야 합니다.")
            return 1
        send(ser, {"cmd": "set_wifi", "ssid": ssid, "password": password})
        password = ""
        print("ESP32 에 저장했습니다. 연결을 기다리는 중 (최대 25초)...")
        for m in read_messages(ser, 25):
            if m.get("message_type") != "WIFI_STATUS":
                continue
            show(m)
            if m.get("note", "").startswith("rejected"):
                return 1
            if m.get("state") == "CONNECTED":
                print("\n완료. 이제 USB 없이 전원만 연결해도 됩니다. PC 에서 실행:\n  python run.py --wifi")
                return 0
        print("\n25초 안에 연결되지 않았습니다. 확인할 것:\n"
              "  - 2.4GHz Wi-Fi 인지 (5GHz 전용 이름은 연결 불가)\n"
              "  - 이름/비밀번호 오타 (다시 실행해 입력)\n"
              "  - 보드가 계속 재부팅하면 USB 전원 부족: 다른 포트·짧은 케이블 사용\n"
              "ESP32 는 10초마다 계속 다시 시도합니다. 상태 확인: python tools/wifi_setup.py --port "
              + args.port + " --status")
        return 1
    finally:
        ser.close()


if __name__ == "__main__":
    sys.exit(main())
