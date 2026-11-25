"""서버 중계: 노트북 모니터가 받은 센서 데이터를 Tailscale 너머 서버로 실시간 전달한다.

ESP32 는 Tailscale 을 돌릴 수 없으므로, 보드 → (핫스팟 UDP 브로드캐스트) → 노트북 모니터 →
(Tailscale UDP) → 서버(tools/relay_server.py) 순서로 보낸다.

데이터그램 1개 = JSON 1개 (봉투):
  {"relay_v":1, "run_id":"…", "rseq":n, "kind":"raw|event|status|log", "sent":epoch초, "data":…}
  - raw    : 보드가 보낸 원본 한 줄 (문자열 그대로) — 서버에서 같은 추정기를 다시 돌릴 수 있다
  - event  : 노트북이 확정한 이벤트 (ADD/REMOVE/RESYNC/...) — 유실에 대비해 2번 보낸다 (서버가 중복 제거)
  - status : 레인별 현재 상태 (status_interval_s 마다, 레인 하나당 1개)
  - log    : 보드 LOG 메시지
run_id 는 모니터를 켤 때마다 새로 정해지고, rseq 는 그 안에서 1씩 증가한다 → 서버가 유실·중복을 센다.

UDP 라서 서버가 꺼져 있거나 Tailscale 이 끊겨도 모니터는 멈추지 않는다 (보내기 실패만 센다).
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
from typing import Optional

RELAY_VERSION = 1
MAX_DATAGRAM = 1200          # Tailscale MTU(1280) 안에 들어가게. 넘으면 그래도 보내지만 경고를 센다


def is_tailscale_ip(ip: str) -> bool:
    """100.64.0.0/10 (Tailscale 이 쓰는 대역)."""
    try:
        a, b = (int(x) for x in ip.split(".")[:2])
    except ValueError:
        return False
    return a == 100 and 64 <= b <= 127


def local_ip_for(addr: tuple) -> Optional[str]:
    """그 주소로 보낼 때 OS 가 고르는 이 PC 쪽 주소 (패킷은 보내지 않는다)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(addr)
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def parse_target(text: str) -> tuple[str, int]:
    """"100.118.214.101:4220" / "da-perfect57:4220" → (host, port)."""
    host, sep, port = text.strip().rpartition(":")
    if not sep or not host or not port.isdigit() or not 1 <= int(port) <= 65535:
        raise ValueError(f'중계 대상은 "호스트:포트" 형식이어야 함 (받은 값: {text!r})')
    return host, int(port)


class RelaySender:
    """모니터 sink 이자 원본 줄 전달자. 모든 메서드는 막히지 않는다."""

    RESOLVE_RETRY_S = 10.0
    WARN_EVERY_S = 30.0
    EVENT_COPIES = 2

    def __init__(self, target: str, log=None):
        self.host, self.port = parse_target(target)
        self.target_text = f"{self.host}:{self.port}"
        self.run_id = os.urandom(4).hex()
        self.rseq = 0
        self.sent = 0
        self.failed = 0
        self.oversize = 0
        self._addr: Optional[tuple] = None
        self._next_resolve = 0.0
        self._last_warn = 0.0
        self._log = log or (lambda m: print(m, file=sys.stderr, flush=True))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)

    # ------------------------------------------------------------ 내부
    def _resolve(self) -> Optional[tuple]:
        """주소 확인 + 경로 확인 (RESOLVE_RETRY_S 마다 다시). 실패하면 None → 보내지 않음."""
        now = time.monotonic()
        if now < self._next_resolve:
            return self._addr
        self._next_resolve = now + self.RESOLVE_RETRY_S
        try:
            info = socket.getaddrinfo(self.host, self.port, socket.AF_INET, socket.SOCK_DGRAM)
            addr = info[0][4]
        except OSError as e:
            self._addr = None
            self._warn(f"[중계] {self.host} 주소를 찾지 못함 ({e}) — Tailscale 이 켜져 있는지 확인. "
                       f"{self.RESOLVE_RETRY_S:g}초 뒤 다시 시도")
            return None
        # Tailscale 주소(100.64.0.0/10)는 통신사 내부망(CGNAT)과 같은 대역이다. Tailscale 이 꺼져 있으면
        # 패킷이 핫스팟 → 통신사 쪽으로 나가 엉뚱한 기기에 닿을 수 있으므로, 경로가 Tailscale 일 때만 보낸다.
        if is_tailscale_ip(addr[0]):
            local = local_ip_for(addr)
            if local is None or not is_tailscale_ip(local):
                if self._addr is not None:          # 잘 보내다가 끊긴 순간은 바로 알린다
                    self._last_warn = 0.0
                self._addr = None
                self._warn(f"[중계] {addr[0]} 로 가는 경로가 Tailscale 이 아님 (노트북 쪽 주소 {local}) — "
                           f"Tailscale 을 켜고 로그인하세요. 그 전까지 보내지 않습니다")
                return None
        if self._addr is None:
            self._log(f"[중계] {self.target_text} 로 보내기 시작 (run_id {self.run_id})")
        self._addr = addr
        return addr

    def _warn(self, msg: str) -> None:
        now = time.monotonic()
        if now - self._last_warn >= self.WARN_EVERY_S:
            self._last_warn = now
            self._log(msg)

    def _send(self, kind: str, data, copies: int = 1, extra: Optional[dict] = None) -> None:
        self.rseq += 1
        env = {"relay_v": RELAY_VERSION, "run_id": self.run_id, "rseq": self.rseq, "kind": kind,
               "sent": round(time.time(), 3), **(extra or {}), "data": data}
        payload = json.dumps(env, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(payload) > MAX_DATAGRAM:
            self.oversize += 1
        addr = self._resolve()
        if addr is None:
            self.failed += 1
            return
        for _ in range(copies):
            try:
                self.sock.sendto(payload, addr)
                self.sent += 1
            except OSError as e:          # 경로 없음, 버퍼 가득 등 — 모니터는 계속 돈다
                self.failed += 1
                self._warn(f"[중계] {self.target_text} 로 보내기 실패: {e} (실패 누적 {self.failed})")
                if isinstance(e, socket.gaierror):
                    self._addr = None

    # ------------------------------------------------------------ 원본 줄
    def send_raw(self, line: bytes, pc_recv: Optional[float] = None) -> None:
        """pc_recv: 노트북이 이 줄을 받은 epoch 초 (수신 스레드에서 기록한 값)."""
        self._send("raw", line.decode("utf-8", errors="replace"),
                   extra=None if pc_recv is None else {"pc_recv": round(pc_recv, 3)})

    # ------------------------------------------------------------ Sink
    def on_event(self, rec: dict) -> None:
        self._send("event", rec, copies=self.EVENT_COPIES)

    def on_status(self, recs: list[dict]) -> None:
        for r in recs:                     # 레인 하나씩: 한 데이터그램이 MTU 를 넘지 않게
            self._send("status", r)

    def on_device_log(self, msg: dict) -> None:
        self._send("log", msg)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass
