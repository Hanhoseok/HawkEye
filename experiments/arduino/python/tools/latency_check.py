"""Wi-Fi(UDP) 전송 지연 측정: 같은 메시지를 USB 와 Wi-Fi 로 동시에 받아 PC 도착 시각을 비교한다.

  python tools/latency_check.py --port COM7 --seconds 30

ESP32 는 한 줄을 USB 로 쓴 직후 같은 줄을 UDP 로 보낸다. 두 경로의 PC 도착 시각 차이
(Wi-Fi − USB)에 USB 경로 자체의 지연(115200bps 전송 시간 + USB 드라이버 지연)을 더하면
Wi-Fi 경로의 대략적인 지연이 된다. ESP32 와 PC 시계를 맞추지 않으므로 절대값은 추정치다.
보드는 USB 와 Wi-Fi 에 모두 연결돼 있어야 하고, run.py · Arduino 시리얼 모니터는 닫는다.
"""
from __future__ import annotations

import argparse
import json
import socket
import statistics as st
import sys
import threading
import time

import serial

BAUD = 115200


def percentile(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--udp-port", type=int, default=4210)
    ap.add_argument("--seconds", type=float, default=30)
    args = ap.parse_args()

    usb: dict[tuple, tuple[float, int]] = {}
    wifi: dict[tuple, float] = {}
    stop = threading.Event()

    ser = serial.Serial()
    ser.port, ser.baudrate, ser.timeout = args.port, BAUD, 0.005
    ser.dtr = ser.rts = False
    try:
        ser.open()
    except serial.SerialException as e:
        print(f"{args.port} 를 열 수 없음: {e}")
        return 1
    ser.reset_input_buffer()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", args.udp_port))
    sock.settimeout(0.2)

    def read_usb():
        buf = b""
        while not stop.is_set():
            chunk = ser.read(ser.in_waiting or 1)
            if not chunk:
                continue
            t = time.perf_counter()
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    m = json.loads(line)
                except ValueError:
                    continue
                usb[(m.get("boot_id"), m.get("sequence"))] = (t, len(line) + 1)

    def read_wifi():
        while not stop.is_set():
            try:
                data, _ = sock.recvfrom(4096)
            except (socket.timeout, ConnectionResetError):
                continue
            t = time.perf_counter()
            try:
                m = json.loads(data)
            except ValueError:
                continue
            wifi[(m.get("boot_id"), m.get("sequence"))] = t

    threads = [threading.Thread(target=read_usb, daemon=True), threading.Thread(target=read_wifi, daemon=True)]
    for th in threads:
        th.start()
    print(f"{args.seconds:.0f}초 동안 측정 중...", flush=True)
    time.sleep(args.seconds)
    stop.set()
    for th in threads:
        th.join(1)
    ser.close()
    sock.close()

    common = [k for k in usb if k in wifi]
    if not usb:
        print("USB 로 받은 메시지가 없습니다.")
        return 1
    if not common:
        print(f"USB {len(usb)}개는 받았지만 Wi-Fi 로 같은 메시지를 못 받았습니다. PC 와 ESP32 가 같은 네트워크인지 확인하세요.")
        return 1
    diff_ms = [(wifi[k] - usb[k][0]) * 1000 for k in common]
    line_ms = st.median(usb[k][1] * 10 / BAUD * 1000 for k in common)    # 1바이트 = 10비트
    wifi_only = sorted(wifi[k] for k in wifi)
    gaps = [(b - a) * 1000 for a, b in zip(wifi_only, wifi_only[1:])]

    print(f"\n받은 메시지: USB {len(usb)} · Wi-Fi {len(wifi)} · 양쪽 모두 {len(common)}")
    print(f"Wi-Fi 손실: {len(usb) - len(common)} / {len(usb)} ({100 * (len(usb) - len(common)) / len(usb):.1f}%)")
    print("\nWi-Fi 도착 − USB 도착 (ms, 음수 = Wi-Fi 가 먼저 도착)")
    print(f"  중앙값 {st.median(diff_ms):6.1f} · 평균 {st.mean(diff_ms):6.1f} · 최소 {min(diff_ms):6.1f}"
          f" · 95% {percentile(diff_ms, 95):6.1f} · 99% {percentile(diff_ms, 99):6.1f} · 최대 {max(diff_ms):6.1f}")
    print(f"  흔들림(표준편차) {st.pstdev(diff_ms):.1f}")
    print(f"\nUSB 한 줄 전송 시간 약 {line_ms:.1f} ms (115200bps) + USB 드라이버 지연 수 ms")
    est = [d + line_ms for d in diff_ms]
    print(f"Wi-Fi 경로 지연 추정 (USB 지연 보정, 드라이버 지연 제외): 중앙값 {st.median(est):.0f} ms"
          f" · 95% {percentile(est, 95):.0f} ms · 최대 {max(est):.0f} ms")
    if gaps:
        print(f"\nWi-Fi 도착 간격: 중앙값 {st.median(gaps):.0f} ms · 최대 {max(gaps):.0f} ms (전송 주기 약 90~100 ms)")
    buckets = [(0, 10), (10, 30), (30, 100), (100, 300), (300, 10 ** 9)]
    print("\n지연(추정) 분포")
    for lo, hi in buckets:
        n = sum(lo <= e < hi for e in est)
        label = f"{lo}~{hi} ms" if hi < 10 ** 9 else f"{lo} ms 이상"
        print(f"  {label:>12}: {n:4d} ({100 * n / len(est):5.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
