"""스프링 푸셔 진열대 모니터 실행 파일.

  python run.py                      # config.toml 의 포트로 연결, 창 표시
  python run.py --port COM5          # 포트 지정
  python run.py --wifi               # Wi-Fi(UDP)로 수신 (먼저 tools/wifi_setup.py 로 ESP32 Wi-Fi 설정)
  python run.py --list-ports         # COM 포트 목록
  python run.py --no-gui             # 콘솔만 (입력: "r 1" = 1번 레인 재입고 토글, "q" = 종료)
  python run.py --mock --enable-all  # 하드웨어 없이 5개 레인 모의 데이터로 시연
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

from shelf_monitor.app import Monitor
from shelf_monitor.config import ConfigError, load_config
from shelf_monitor.sinks import ConsoleSink, JsonlSink


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description="ESP32 + VL53L0X 스프링 푸셔 진열대 모니터")
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.toml")))
    ap.add_argument("--port", help="예: COM5 (config.toml 의 serial.port 보다 우선, USB 로 수신)")
    ap.add_argument("--wifi", action="store_true", help="Wi-Fi(UDP)로 수신 (config.toml [network] udp_port)")
    ap.add_argument("--udp-port", type=int, help="Wi-Fi 수신 포트 (기본 config.toml 값)")
    ap.add_argument("--list-ports", action="store_true")
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--mock", action="store_true", help="하드웨어 없이 모의 장치 데이터 사용")
    ap.add_argument("--enable-all", action="store_true", help="설정 파일의 5개 레인을 모두 활성화 (모의 시연용)")
    ap.add_argument("--seed", type=int, default=None, help="모의 장치 난수 시드")
    args = ap.parse_args()

    if args.list_ports:
        from shelf_monitor.sources import list_ports
        rows = list_ports()
        print("\n".join(rows) if rows else "COM 포트가 없습니다.")
        return 0

    try:
        cfg = load_config(args.config, enable_all=args.enable_all)
    except ConfigError as e:
        print(f"[설정 오류] {args.config}\n{e}", file=sys.stderr)
        return 2
    if args.port:
        cfg.port = args.port
        cfg.source = "serial"
    if args.wifi:
        cfg.source = "wifi"
    if args.udp_port:
        cfg.udp_port = args.udp_port

    jsonl = JsonlSink(cfg.resolve(cfg.event_log), cfg.resolve(cfg.status_log) if cfg.status_log else None)
    console = ConsoleSink(show_status=True)
    monitor = Monitor(cfg, [console, jsonl])

    q: queue.Queue = queue.Queue()
    stop = threading.Event()
    if args.mock:
        from shelf_monitor.sources import MockSource
        MockSource(cfg, q, stop, seed=args.seed).start()
    elif cfg.source == "wifi":
        from shelf_monitor.sources import UdpSource
        UdpSource(cfg.udp_port, q, stop).start()
    else:
        from shelf_monitor.sources import SerialSource
        SerialSource(cfg.port, cfg.baud, q, stop).start()

    lock = threading.Lock()

    def pump() -> None:
        with lock:
            for _ in range(2000):
                try:
                    kind, payload, mono, wall = q.get_nowait()
                except queue.Empty:
                    break
                if kind == "LINE":
                    monitor.handle_line(payload, mono, wall)
                elif kind == "UP":
                    monitor.handle_link_up(payload, mono, wall)
                elif kind == "DOWN":
                    monitor.handle_link_down(payload, mono, wall)
                    console.on_event({"message_type": "SYSTEM", "event": "PORT", "reason": payload})
            monitor.tick(time.monotonic(), time.time())

    print(f"이벤트 기록: {cfg.resolve(cfg.event_log)}")
    print("uptime_ms 는 ESP32 부팅 후 경과 ms 이며 날짜·시각이 아닙니다. pc_time 은 PC 기록 시각입니다.")
    try:
        if args.no_gui:
            def stdin_loop() -> None:
                for line in sys.stdin:
                    parts = line.split()
                    if parts and parts[0] == "q":
                        stop.set()
                        return
                    if len(parts) == 2 and parts[0] == "r" and parts[1].isdigit():
                        lane = monitor.lanes.get(int(parts[1]))
                        if lane is not None:
                            with lock:
                                monitor.set_restock(int(parts[1]), not lane.restock, time.monotonic(), time.time())
            threading.Thread(target=stdin_loop, daemon=True).start()
            while not stop.is_set():
                pump()
                stop.wait(0.05)
        else:
            from shelf_monitor.gui import RailWindow
            RailWindow(cfg, monitor, q, pump, on_close=stop.set).run()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        jsonl.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
