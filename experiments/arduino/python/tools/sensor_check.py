"""설치·간섭 확인용 실시간 거리 표시 (수량 추정 없음).

  python tools/sensor_check.py --port COM7
  python tools/sensor_check.py --wifi           # Wi-Fi(UDP)로 받기
  python tools/sensor_check.py --port COM7 --lane 1 --csv logs/check.csv

0.5초마다 레인별로 [현재 중앙값, 기준 대비 변화, 0.5초 구간 최소~최대, 무효 횟수]를 보여 준다.
- 시작 후 처음 3초의 중앙값이 "기준"이 된다.
- Enter 를 누르면 지금 값으로 기준을 다시 잡는다.
- q + Enter 로 종료하면 기준 대비 최대 변화와 무효 비율을 요약한다.
Arduino 시리얼 모니터나 run.py 가 같은 포트를 열고 있으면 먼저 닫는다.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import threading
import time
from pathlib import Path

import serial

BASELINE_S = 3.0
WINDOW_S = 0.5


class LaneStat:
    def __init__(self):
        self.window: list[int] = []
        self.invalid_window = 0
        self.baseline_samples: list[int] = []
        self.baseline: float | None = None
        self.max_dev = 0.0
        self.total = 0
        self.invalid = 0
        self.last_status = "-"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", help="USB 로 받을 때: 예 COM7")
    ap.add_argument("--wifi", action="store_true", help="Wi-Fi(UDP)로 받기")
    ap.add_argument("--udp-port", type=int, default=4210)
    ap.add_argument("--lane", type=int, help="이 레인만 표시")
    ap.add_argument("--csv", help="모든 측정을 CSV 로 저장")
    args = ap.parse_args()
    if not args.wifi and not args.port:
        ap.error("--port COM7 또는 --wifi 중 하나를 지정하세요")

    if args.wifi:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind(("", args.udp_port))
        except OSError as e:
            print(f"UDP {args.udp_port} 포트를 열 수 없음: {e}\n(run.py --wifi 가 실행 중이면 닫으세요)")
            return 1
        sock.settimeout(0.1)
        source_name = f"Wi-Fi UDP:{args.udp_port}"

        def read_chunk() -> bytes:
            try:
                data, _ = sock.recvfrom(4096)
            except (socket.timeout, ConnectionResetError):
                return b""
            return data if data.endswith(b"\n") else data + b"\n"
        close = sock.close
    else:
        ser = serial.Serial()
        ser.port, ser.baudrate, ser.timeout = args.port, 115200, 0.1
        ser.dtr = ser.rts = False
        try:
            ser.open()
        except serial.SerialException as e:
            print(f"{args.port} 를 열 수 없음: {e}\n(Arduino 시리얼 모니터나 run.py 가 열려 있으면 닫으세요)")
            return 1
        ser.reset_input_buffer()
        source_name = args.port

        def read_chunk() -> bytes:
            return ser.read(512)
        close = ser.close

    csv = None
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        csv = open(args.csv, "w", encoding="utf-8")
        csv.write("pc_time,uptime_ms,lane_id,distance_mm,valid,sensor_status\n")

    lanes: dict[int, LaneStat] = {}
    cmd = {"rebase": False, "quit": False}

    def stdin_loop():
        for line in sys.stdin:
            if line.strip().lower() == "q":
                cmd["quit"] = True
                return
            cmd["rebase"] = True

    threading.Thread(target=stdin_loop, daemon=True).start()
    print(f"{source_name} 수신 중. 처음 {BASELINE_S:.0f}초는 움직이지 마세요 (기준 측정). Enter=기준 재설정, q+Enter=종료")
    t0 = time.monotonic()
    next_print = t0 + WINDOW_S
    buf = b""
    try:
        while not cmd["quit"]:
            buf += read_chunk()
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    m = json.loads(line)
                except ValueError:
                    continue
                if m.get("message_type") != "MEASUREMENT":
                    continue
                lane_id = m.get("lane_id")
                if args.lane and lane_id != args.lane:
                    continue
                s = lanes.setdefault(lane_id, LaneStat())
                s.total += 1
                s.last_status = m.get("sensor_status", "?")
                if csv:
                    csv.write(f"{time.time():.3f},{m.get('uptime_ms')},{lane_id},"
                              f"{'' if m.get('distance_mm') is None else m['distance_mm']},"
                              f"{m.get('valid')},{s.last_status}\n")
                if m.get("valid") and m.get("distance_mm") is not None:
                    d = int(m["distance_mm"])
                    s.window.append(d)
                    if s.baseline is None:
                        s.baseline_samples.append(d)
                else:
                    s.invalid += 1
                    s.invalid_window += 1

            now = time.monotonic()
            if cmd["rebase"]:
                cmd["rebase"] = False
                for s in lanes.values():
                    s.baseline, s.baseline_samples, s.max_dev = None, [], 0.0
                t0 = now
                print("--- 기준 다시 측정 중 (3초간 움직이지 마세요) ---")
            if now >= next_print:
                next_print = now + WINDOW_S
                parts = []
                for lane_id in sorted(lanes):
                    s = lanes[lane_id]
                    if s.baseline is None and now - t0 >= BASELINE_S and s.baseline_samples:
                        s.baseline = st.median(s.baseline_samples)
                        print(f"[L{lane_id}] 기준 = {s.baseline:.0f} mm")
                    if s.window:
                        med = st.median(s.window)
                        rng = f"{min(s.window)}~{max(s.window)}"
                        if s.baseline is not None:
                            dev = med - s.baseline
                            s.max_dev = max(s.max_dev, abs(dev))
                            dev_txt = f"{dev:+6.0f}"
                        else:
                            dev_txt = "  기준중"
                        parts.append(f"L{lane_id} {med:5.0f}mm 변화{dev_txt} 범위 {rng:>9} 무효 {s.invalid_window}")
                    else:
                        parts.append(f"L{lane_id}  측정 없음 ({s.last_status}) 무효 {s.invalid_window}")
                    s.window, s.invalid_window = [], 0
                if parts:
                    print(time.strftime("%H:%M:%S") + "  " + " | ".join(parts), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        close()
        if csv:
            csv.close()
    print("\n요약")
    for lane_id in sorted(lanes):
        s = lanes[lane_id]
        rate = 100 * s.invalid / s.total if s.total else 0
        base = "-" if s.baseline is None else f"{s.baseline:.0f}mm"
        print(f"  L{lane_id}: 기준 {base}, 기준 대비 최대 변화 {s.max_dev:.0f}mm, 무효 {s.invalid}/{s.total} ({rate:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
