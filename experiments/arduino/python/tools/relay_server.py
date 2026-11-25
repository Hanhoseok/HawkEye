#!/usr/bin/env python3
"""서버(da-perfect57)에서 돌리는 센서 중계 수신기. 표준 라이브러리만 쓴다 (파일 하나만 복사하면 됨).

  노트북: python run.py --wifi            (config.toml [relay] target = "100.118.214.101:4220")
  서버  : python3 relay_server.py         (기본: UDP 4220, Tailscale 주소에서 온 것만 받음)

옵션
  --port 4220             수신 포트 (노트북 [relay] target 의 포트와 같게)
  --bind 100.118.214.101  이 주소로만 받기 (기본 0.0.0.0 = 모든 인터페이스, 출발지는 Tailscale 만 허용)
  --out shelf_relay       저장 폴더. 날짜별 raw/events/status/logs .jsonl + latest.json
  --feed-local 4210       받은 원본 측정을 127.0.0.1:4210 으로 다시 보냄 → 서버에서 run.py --wifi --no-gui 로
                          같은 추정기를 돌릴 수 있다 (선택)
  --allow-any             Tailscale 이 아닌 출발지도 받기 (기본은 거부)

저장 형식 (한 줄 = JSON 하나):
  {"recv":"서버 수신 시각", "sent":노트북 송신 epoch초, "run_id":"…", "rseq":n, "data":…}
latest.json: 레인별 마지막 상태(수량, 거리, 상태)와 마지막 이벤트 — 다른 프로그램(HawkEye 등)이 읽기 좋게.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

KINDS = ("raw", "event", "status", "log")


def is_tailscale_ip(ip: str) -> bool:
    try:
        a, b = (int(x) for x in ip.split(".")[:2])
    except ValueError:
        return False
    return a == 100 and 64 <= b <= 127


def iso(t: float) -> str:
    return datetime.fromtimestamp(t).astimezone().isoformat(timespec="milliseconds")


class RunStats:
    """노트북 실행(run_id) 하나의 유실·중복 계산."""

    def __init__(self):
        self.seen: set[int] = set()
        self.order: deque[int] = deque()
        self.lo: int | None = None
        self.hi = 0
        self.unique = 0

    def add(self, rseq: int) -> bool:
        """처음 보는 rseq 면 True (중복이면 False)."""
        if rseq in self.seen:
            return False
        self.seen.add(rseq)
        self.order.append(rseq)
        if len(self.order) > 20000:
            self.seen.discard(self.order.popleft())
        self.lo = rseq if self.lo is None else min(self.lo, rseq)
        self.hi = max(self.hi, rseq)
        self.unique += 1
        return True

    def loss_ratio(self) -> float:
        if self.lo is None:
            return 0.0
        expected = self.hi - self.lo + 1
        return max(0.0, 1 - self.unique / expected)


class RelayReceiver:
    def __init__(self, out_dir: Path, allow_any: bool = False, feed_local: int | None = None,
                 echo=print):
        self.out_dir = Path(out_dir)
        self.allow_any = allow_any
        self.echo = echo
        self.runs: dict[str, RunStats] = {}
        self.files: dict[str, object] = {}
        self.file_day = ""
        self.latest: dict = {"lanes": {}, "last_event": None, "updated": None}
        self.rejected = 0
        self.bad = 0
        self.dups = 0
        self.window: list[tuple[str, float]] = []          # (kind, 지연 초) — 통계 주기마다 비움
        self.feed_sock = None
        self.feed_addr = None
        if feed_local:
            self.feed_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.feed_addr = ("127.0.0.1", feed_local)

    # ------------------------------------------------------------ 파일
    def _file(self, kind: str, now: float):
        day = datetime.fromtimestamp(now).strftime("%Y-%m-%d")
        if day != self.file_day:
            for f in self.files.values():
                f.close()
            self.files = {}
            self.file_day = day
        if kind not in self.files:
            d = self.out_dir / day
            d.mkdir(parents=True, exist_ok=True)
            name = {"event": "events", "log": "logs"}.get(kind, kind)
            self.files[kind] = open(d / f"{name}.jsonl", "a", encoding="utf-8", buffering=1)
        return self.files[kind]

    def _write_latest(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.out_dir / "latest.json.tmp"
        tmp.write_text(json.dumps(self.latest, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.out_dir / "latest.json")

    # ------------------------------------------------------------ 처리
    def handle(self, datagram: bytes, src_ip: str, now: float | None = None) -> str | None:
        """데이터그램 1개 처리. 저장한 종류를 돌려준다 (거부·중복·깨짐이면 None)."""
        now = time.time() if now is None else now
        if not self.allow_any and not (is_tailscale_ip(src_ip) or src_ip.startswith("127.")):
            self.rejected += 1
            return None
        try:
            env = json.loads(datagram.decode("utf-8"))
            kind, run_id, rseq = env["kind"], str(env["run_id"]), int(env["rseq"])
            if env.get("relay_v") != 1 or kind not in KINDS:
                raise ValueError
        except (ValueError, KeyError, TypeError, UnicodeDecodeError):
            self.bad += 1
            return None
        if not self.runs.setdefault(run_id, RunStats()).add(rseq):
            self.dups += 1
            return None
        data = env.get("data")
        sent = env.get("sent")
        if kind == "raw" and isinstance(data, str):
            if self.feed_sock is not None:
                try:
                    self.feed_sock.sendto(data.encode("utf-8"), self.feed_addr)
                except OSError:
                    pass
            try:
                data = json.loads(data)
            except ValueError:
                pass                                   # 깨진 줄은 문자열 그대로 저장
        rec = {"recv": iso(now), "sent": sent, "run_id": run_id, "rseq": rseq, "src": src_ip, "data": data}
        if isinstance(env.get("pc_recv"), (int, float)):
            rec["pc_recv"] = iso(env["pc_recv"])        # 노트북이 보드에서 받은 시각 (raw)
        self._file(kind, now).write(json.dumps(rec, ensure_ascii=False) + "\n")
        if isinstance(sent, (int, float)):
            self.window.append((kind, now - sent))
        if kind == "event" and isinstance(data, dict):
            self.latest["last_event"] = rec
            if data.get("lane_id") is not None and data.get("current_count") is not None:
                lane = self.latest["lanes"].setdefault(str(data["lane_id"]), {})
                lane.update({"count": data["current_count"], "product_name": data.get("product_name"),
                             "event_time": rec["recv"]})
            self.latest["updated"] = rec["recv"]
            self._write_latest()
            self.echo(f"[{datetime.fromtimestamp(now):%H:%M:%S}] {describe_event(data)}")
        elif kind == "status" and isinstance(data, dict) and data.get("lane_id") is not None:
            lane = self.latest["lanes"].setdefault(str(data["lane_id"]), {})
            lane.update({k: data.get(k) for k in ("product_name", "count", "state", "distance_mm",
                                                   "stale", "link_ok", "device_id")})
            lane["status_time"] = rec["recv"]
            self.latest["updated"] = rec["recv"]
            self._write_latest()
        return kind

    def stats_line(self, period_s: float) -> str:
        n = len(self.window)
        delays = [d for _k, d in self.window]
        raw = sum(1 for k, _d in self.window if k == "raw")
        self.window = []
        loss = [r.loss_ratio() for r in self.runs.values()]
        parts = [f"수신 {n / period_s:.1f}개/s (원본 {raw / period_s:.1f})"]
        if delays:
            parts.append(f"지연 중앙 {statistics.median(delays) * 1000:.0f}ms·최대 {max(delays) * 1000:.0f}ms"
                         f" (두 PC 시계차 포함)")
        if loss:
            parts.append(f"유실 {max(loss) * 100:.1f}%")
        if self.rejected or self.bad:
            parts.append(f"거부 {self.rejected}·깨짐 {self.bad}")
        if n == 0:
            parts.append("— 데이터 없음: 노트북 모니터·Tailscale 확인")
        return "[통계] " + " · ".join(parts)

    def close(self) -> None:
        for f in self.files.values():
            f.close()
        if self.feed_sock is not None:
            self.feed_sock.close()


def describe_event(d: dict) -> str:
    ev = d.get("event")
    if d.get("message_type") == "SYSTEM":
        return f"[시스템] {ev} {d.get('device_id') or ''}".rstrip()
    head = f"{d.get('lane_id')}번 레인 / {d.get('product_name')}"
    cur = d.get("current_count")
    if ev in ("ADD", "REMOVE"):
        s = f"{head} / {d.get('quantity')}개 {'추가' if ev == 'ADD' else '제거'} / 현재 {cur}개"
        if d.get("change_start_time"):
            s += f" / 시작 {str(d['change_start_time'])[11:23]} (확정까지 {d.get('confirm_delay_ms', 0) / 1000:.1f}초)"
        return s
    if ev == "RESYNC":
        return f"{head} / 재동기화 {d.get('previous_count')} → {cur}개"
    return f"{head} / {ev} / 현재 {cur}"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description="스프링 푸셔 진열대 센서 중계 수신기 (서버용)")
    ap.add_argument("--port", type=int, default=4220)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--out", default="shelf_relay")
    ap.add_argument("--feed-local", type=int, default=None, metavar="PORT")
    ap.add_argument("--allow-any", action="store_true")
    ap.add_argument("--stats-every", type=float, default=10.0, help="통계 출력 주기(초)")
    args = ap.parse_args()

    rx = RelayReceiver(Path(args.out), allow_any=args.allow_any, feed_local=args.feed_local)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind, args.port))
    sock.settimeout(0.5)
    print(f"수신 대기: UDP {args.bind}:{args.port} → {Path(args.out).resolve()}"
          + ("" if args.allow_any else " (Tailscale 출발지만 허용)")
          + (f" · 원본을 127.0.0.1:{args.feed_local} 로 재전송" if args.feed_local else ""), flush=True)
    next_stats = time.monotonic() + args.stats_every
    try:
        while True:
            try:
                data, (ip, _port) = sock.recvfrom(65535)
                rx.handle(data, ip)
            except socket.timeout:
                pass
            if time.monotonic() >= next_stats:
                next_stats += args.stats_every
                print(rx.stats_line(args.stats_every), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        rx.close()
        sock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
