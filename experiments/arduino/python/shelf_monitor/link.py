"""ESP32 메시지 흐름 추적: 재부팅(boot_id), sequence 중복·누락, uptime_ms wrap, 수신 백로그.

시간 원칙
- 장치 시각(uptime_ms)끼리만 빼고, PC 시각(time.monotonic)끼리만 뺀다.
  두 시계의 값을 서로 빼서 시각을 환산하지 않는다.
- 백로그 판정은 "메시지 간 PC 수신 간격 - 메시지 간 장치 측정 간격"을 누적해서 한다.
  (PC 가 잠깐 멈췄다 버퍼를 한꺼번에 읽으면 첫 메시지에서 지연이 크게 쌓이고,
   이후 몰려온 메시지들에서 서서히 줄어든다.)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

MASK32 = 0xFFFFFFFF
HALF32 = 0x80000000
DRIFT_ALLOWANCE = 0.001     # 두 시계의 속도 차이 허용 (1000 ppm)


@dataclass
class LinkResult:
    accept: bool
    reboot: bool = False
    gap: int = 0
    late: bool = False
    dev_ms: int = 0
    duplicate: bool = False


class LinkTracker:
    def __init__(self, device_id_filter: str = "", backlog_limit_ms: int = 1000):
        self.device_id_filter = device_id_filter
        self.backlog_limit_ms = backlog_limit_ms
        self.device_id: Optional[str] = None
        self.boot_id: Optional[str] = None
        self.last_seq: Optional[int] = None
        self.last_uptime: Optional[int] = None
        self.dev_ms = 0
        self.last_pc: Optional[float] = None
        self.lateness_ms = 0.0
        self.stats = {"accepted": 0, "duplicates": 0, "missing": 0, "foreign": 0, "reboots": 0}

    def reset_connection(self) -> None:
        """포트를 새로 열었을 때: 수신 간격 기준만 초기화 (boot_id/sequence 는 유지해 재부팅·누락 판정)."""
        self.last_pc = None
        self.lateness_ms = 0.0

    def process(self, msg: dict, pc_mono: float) -> LinkResult:
        if self.device_id_filter and msg.get("device_id") != self.device_id_filter:
            self.stats["foreign"] += 1
            return LinkResult(accept=False)
        seq = int(msg["sequence"]) & MASK32
        uptime = int(msg["uptime_ms"]) & MASK32

        reboot = False
        if msg["boot_id"] != self.boot_id:
            reboot = self.boot_id is not None
            if reboot:
                self.stats["reboots"] += 1
            self.boot_id = msg["boot_id"]
            self.device_id = msg.get("device_id")
            self.last_seq = None
            self.last_uptime = None
            self.dev_ms = 0
            self.last_pc = None
            self.lateness_ms = 0.0

        gap = 0
        if self.last_seq is not None:
            diff = (seq - self.last_seq) & MASK32
            if diff == 0 or diff >= HALF32:
                self.stats["duplicates"] += 1
                return LinkResult(accept=False, duplicate=True)
            gap = diff - 1
            self.stats["missing"] += gap
        self.last_seq = seq

        du = 0
        if self.last_uptime is None:
            self.dev_ms = uptime
        else:
            du = (uptime - self.last_uptime) & MASK32   # millis() 49.7일 wrap 보정
            if du >= HALF32:
                du = 0                                    # 시간 역행: 비정상, 진행 없음으로 처리
            self.dev_ms += du
        self.last_uptime = uptime

        if self.last_pc is not None:
            dpc_ms = (pc_mono - self.last_pc) * 1000.0
            self.lateness_ms = max(0.0, self.lateness_ms + dpc_ms - du * (1.0 + DRIFT_ALLOWANCE))
        self.last_pc = pc_mono

        self.stats["accepted"] += 1
        return LinkResult(accept=True, reboot=reboot, gap=gap,
                          late=self.lateness_ms > self.backlog_limit_ms, dev_ms=self.dev_ms)
