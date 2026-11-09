"""레인 1개의 수량 추정 상태 머신.

시리얼·화면·파일과 무관한 순수 로직이다. 입력은 Sample(장치 측정 시각 기준), 출력은 LaneEvent 목록.
USB 대신 HTTP/MQTT 로 거리값을 받더라도 이 모듈은 그대로 재사용할 수 있다.

핵심 규칙
- 유효 측정만 중앙값 필터에 넣는다. 무효값·긴 공백·백로그 샘플은 연속 구간을 끊는다.
- 수량은 매번 절대 거리에서 다시 추정한다 (이동량 누적 없음).
- 후보 수량 n 의 기준거리 ref(n) 과 필터값의 잔차가 tolerance 이내일 때만 n 이 후보가 된다.
  이미 확정된 수량은 tolerance + hysteresis 까지 유지된다.
- 같은 후보가 stable_time_ms(장치 시각) 이상, min_stable_samples 개 이상 연속 유지되고
  그 구간의 필터값 폭이 stable_spread_mm 이하일 때만 확정한다.
- 확정 수량이 바뀔 때만 이벤트를 한 번 만든다.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import median
from typing import Optional, Union

from .config import LaneConfig

INITIALIZING = "INITIALIZING"
STABLE = "STABLE"
CHANGING = "CHANGING"
UNCERTAIN = "UNCERTAIN"
OUT_OF_RANGE = "OUT_OF_RANGE"
SENSOR_ERROR = "SENSOR_ERROR"

BEYOND_EMPTY = "BEYOND_EMPTY"   # 빈 레인 거리보다 멀다 → 보정/설치 오류 가능
OVER_MAX = "OVER_MAX"           # max_count 보다 가깝다 → 손 가림, 보정 오류, 이물질 가능
BETWEEN = "BETWEEN"             # 두 수량 사이 (어느 허용 구간에도 들지 않음)

# 펌웨어가 센서에 접근조차 못 한 상태 → 1회만 와도 즉시 SENSOR_ERROR
HARD_ERRORS = {"NOT_CONNECTED", "INIT_FAILED", "MUX_ERROR", "NOT_INITIALIZED"}

Candidate = Union[int, str]


@dataclass
class Sample:
    lane_id: int
    t_ms: int                       # 장치 측정 시각 (같은 부팅 안에서 단조 증가, wrap 보정됨)
    distance_mm: Optional[float]
    valid: bool
    sensor_status: str
    pc_mono: float = 0.0            # PC 수신 시각 (time.monotonic) — stale 판정에만 사용
    late: bool = False              # 수신 백로그로 실시간성이 의심되는 샘플
    meta: dict = field(default_factory=dict)   # device_id, boot_id, sequence, uptime_ms


@dataclass
class LaneEvent:
    event: str
    lane_id: int
    previous_count: Optional[int] = None
    current_count: Optional[int] = None
    quantity: Optional[int] = None
    distance_mm: Optional[float] = None
    status: str = ""
    detail: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


class LaneEstimator:
    def __init__(self, cfg: LaneConfig, resync_gap_ms: int = 5000):
        self.cfg = cfg
        self.resync_gap_ms = resync_gap_ms
        self.refs = [cfg.reference_mm(n) for n in range(cfg.max_count + 1)]

        self.confirmed: Optional[int] = None
        self.confirmed_distance: Optional[float] = None
        self.pending_resync = False
        self.resync_previous: Optional[int] = None
        self.resync_reason: Optional[str] = None

        self.restock = False
        self.restock_previous: Optional[int] = None
        self.restock_observed: Optional[int] = None

        self.error = False
        self.error_kind: Optional[str] = None
        self.error_since_t: Optional[int] = None
        self.consecutive_invalid = 0
        self.first_invalid_t: Optional[int] = None

        self.last_status = "NO_DATA"
        self.last_valid_t: Optional[int] = None
        self.last_valid_pc: Optional[float] = None
        self.last_valid_uptime: Optional[int] = None
        self.last_raw: Optional[float] = None
        self.display_mm: Optional[float] = None   # 화면 표시용 마지막 필터값 (구간이 끊겨도 유지, 판단에는 안 씀)
        self.state = INITIALIZING
        self.out_of_range_reported = False
        self._reset_segment()

    # ------------------------------------------------------------------ 내부
    def _reset_segment(self) -> None:
        """연속 측정 구간을 끊는다: 필터와 안정 타이머를 처음부터 다시 시작."""
        self._med: deque[float] = deque(maxlen=self.cfg.median_window)
        self._cand: Optional[Candidate] = None
        self._cand_start: Optional[int] = None
        self._cand_n = 0
        self._win: deque[tuple[int, float]] = deque()
        self.filtered: Optional[float] = None
        self.stable_elapsed_ms = 0

    def _classify(self, m: float) -> Candidate:
        c = self.cfg
        if (self.confirmed is not None
                and abs(m - self.refs[self.confirmed]) <= c.tolerance_for(self.confirmed) + c.hysteresis_mm):
            return self.confirmed
        for n, ref in enumerate(self.refs):
            if abs(m - ref) <= c.tolerance_for(n):
                return n
        if m > self.refs[0] + c.tolerance_for(0):
            return BEYOND_EMPTY
        if m < self.refs[-1] - c.tolerance_for(len(self.refs) - 1):
            return OVER_MAX
        return BETWEEN

    def continuous_count(self, m: Optional[float]) -> Optional[float]:
        """표시용 연속 추정값 (확정에는 쓰지 않음)."""
        if m is None:
            return None
        r = self.refs
        if m >= r[0]:
            return (r[0] - m) / (r[0] - r[1])
        for n in range(len(r) - 1):
            if r[n] >= m >= r[n + 1]:
                return n + (r[n] - m) / (r[n] - r[n + 1])
        return len(r) - 1 + (r[-1] - m) / (r[-2] - r[-1])

    def _mark_unobserved(self, reason: str) -> None:
        """관찰하지 못한 구간이 생겼다 → 다음 확정은 ADD/REMOVE 가 아니라 RESYNC."""
        if self.confirmed is not None and not self.pending_resync:
            self.pending_resync = True
            self.resync_previous = self.confirmed
            self.resync_reason = reason

    def _event(self, name: str, s: Optional[Sample], **kw) -> LaneEvent:
        ev = LaneEvent(event=name, lane_id=self.cfg.lane_id, **kw)
        if s is not None:
            ev.meta = dict(s.meta)
        return ev

    # ------------------------------------------------------------------ 입력
    def on_sample(self, s: Sample) -> list[LaneEvent]:
        if s.late:
            # 버퍼에 쌓였다 몰려온 샘플: 실시간 연속 측정으로 보지 않는다.
            self._reset_segment()
            return []
        self.last_status = s.sensor_status
        d = s.distance_mm
        if not s.valid or d is None:
            return self._on_invalid(s, s.sensor_status)
        if not self.cfg.valid_min_mm <= d <= self.cfg.valid_max_mm:
            return self._on_invalid(s, "OUT_OF_VALID_RANGE")
        return self._on_valid(s, float(d))

    def _on_invalid(self, s: Sample, kind: str) -> list[LaneEvent]:
        ev: list[LaneEvent] = []
        c = self.cfg
        self.consecutive_invalid += 1
        if self.first_invalid_t is None:
            self.first_invalid_t = s.t_ms
        self._reset_segment()      # 무효값을 사이에 둔 샘플은 연속 구간이 아님
        trip = (kind in HARD_ERRORS
                or self.consecutive_invalid >= c.error_invalid_count
                or s.t_ms - self.first_invalid_t >= c.error_invalid_ms)
        if not self.error and trip:
            self.error = True
            self.error_kind = kind
            self.error_since_t = s.t_ms
            self._mark_unobserved("SENSOR_ERROR")
            ev.append(self._event(
                "SENSOR_ERROR", s, previous_count=self.confirmed, current_count=self.confirmed,
                status=SENSOR_ERROR,
                detail={"error_kind": kind, "stale": True, "last_valid_uptime_ms": self.last_valid_uptime}))
        elif self.error:
            self.error_kind = kind
        if self.error:
            self.state = SENSOR_ERROR
        elif self.confirmed is None:
            self.state = INITIALIZING
        else:
            self.state = UNCERTAIN
        return ev

    def _on_valid(self, s: Sample, d: float) -> list[LaneEvent]:
        ev: list[LaneEvent] = []
        c = self.cfg
        if self.error:
            outage = s.t_ms - self.error_since_t if self.error_since_t is not None else None
            ev.append(self._event("RECOVERED", s, previous_count=self.confirmed, current_count=self.confirmed,
                                  distance_mm=d, status=CHANGING,
                                  detail={"error_kind": self.error_kind, "outage_ms": outage}))
            self.error = False
            self.error_kind = None
            self.error_since_t = None
        self.consecutive_invalid = 0
        self.first_invalid_t = None

        if self.last_valid_t is not None:
            gap = s.t_ms - self.last_valid_t
            if gap > self.resync_gap_ms:
                self._mark_unobserved("DATA_GAP")
            if gap > c.max_sample_gap_ms:
                self._reset_segment()
        self.last_valid_t = s.t_ms
        self.last_valid_pc = s.pc_mono
        self.last_valid_uptime = s.meta.get("uptime_ms")
        self.last_raw = d

        self._med.append(d)
        if len(self._med) < min(3, c.median_window):
            self.state = INITIALIZING if self.confirmed is None else CHANGING
            return ev
        m = median(self._med)
        self.filtered = m
        self.display_mm = m

        key = self._classify(m)
        if key != self._cand:
            # 후보가 바뀌면 안정 타이머 초기화
            self._cand, self._cand_start, self._cand_n = key, s.t_ms, 0
            self._win.clear()
        self._win.append((s.t_ms, m))
        self._cand_n += 1
        while self._win and s.t_ms - self._win[0][0] > c.stable_time_ms:
            self._win.popleft()
        vals = [v for _, v in self._win]
        if max(vals) - min(vals) > c.stable_spread_mm:
            # 같은 후보 안에서도 흔들림(반동 등)이 크면 안정 타이머 초기화
            self._cand_start, self._cand_n = s.t_ms, 1
            self._win = deque([(s.t_ms, m)])
        self.stable_elapsed_ms = s.t_ms - self._cand_start
        stable = self.stable_elapsed_ms >= c.stable_time_ms and self._cand_n >= c.min_stable_samples

        if stable and isinstance(key, int):
            ev.extend(self._confirm(key, m, s))
        elif stable:
            self.state = OUT_OF_RANGE if key in (BEYOND_EMPTY, OVER_MAX) else UNCERTAIN
            if self.state == OUT_OF_RANGE and not self.out_of_range_reported:
                self.out_of_range_reported = True
                ev.append(self._event("OUT_OF_RANGE", s, previous_count=self.confirmed,
                                      current_count=self.confirmed, distance_mm=m, status=OUT_OF_RANGE,
                                      detail={"reason": key,
                                              "estimated_count": round(self.continuous_count(m), 2)}))
        elif self.confirmed is None:
            self.state = INITIALIZING
        elif key == self.confirmed and not self.pending_resync:
            self.state = STABLE
        else:
            self.state = CHANGING
        return ev

    def _confirm(self, n: int, m: float, s: Sample) -> list[LaneEvent]:
        self.out_of_range_reported = False
        self.state = STABLE
        if self.restock:
            self.restock_observed = n       # 재입고 중: 이벤트 억제, 표시만
            return []
        prev = self.confirmed
        self.confirmed_distance = m
        if prev is None:
            self.confirmed = n
            self.pending_resync = False
            return [self._event("INIT", s, previous_count=None, current_count=n, quantity=None,
                                distance_mm=m, status=STABLE)]
        if self.pending_resync:
            before = self.resync_previous if self.resync_previous is not None else prev
            reason = self.resync_reason
            self.confirmed = n
            self.pending_resync = False
            self.resync_previous = self.resync_reason = None
            return [self._event("RESYNC", s, previous_count=before, current_count=n, quantity=None,
                                distance_mm=m, status=STABLE,
                                detail={"net_change": n - before, "reason": reason, "observed": False})]
        self.confirmed = n
        if n == prev:
            return []
        name = "ADD" if n > prev else "REMOVE"
        return [self._event(name, s, previous_count=prev, current_count=n, quantity=abs(n - prev),
                            distance_mm=m, status=STABLE)]

    # ------------------------------------------------------------------ 외부 제어
    def on_discontinuity(self) -> None:
        """시퀀스 누락 등: 안정 구간만 새로 시작 (수량 판단은 절대 위치라 유지)."""
        self._reset_segment()

    def on_link_lost(self, reason: str) -> None:
        """USB 끊김 / ESP32 재부팅 / 데이터 없음: 필터·시간 기준을 버리고 RESYNC 대기."""
        self._reset_segment()
        self._mark_unobserved(reason)
        self.consecutive_invalid = 0
        self.first_invalid_t = None
        self.last_valid_t = None          # 다른 부팅의 장치 시각과 비교하지 않는다
        self.error_since_t = None

    def set_restock(self, on: bool) -> list[LaneEvent]:
        if on and not self.restock:
            self.restock = True
            self.restock_previous = self.confirmed
            self.restock_observed = None
            self._reset_segment()
            return [self._event("RESTOCK_START", None, previous_count=self.confirmed,
                                current_count=self.confirmed, status=self.state)]
        if not on and self.restock:
            self.restock = False
            self._reset_segment()
            if self.confirmed is not None:
                if not self.pending_resync:
                    self.resync_previous = self.restock_previous if self.restock_previous is not None else self.confirmed
                self.pending_resync = True
                self.resync_reason = "RESTOCK_END"
            self.state = CHANGING if self.confirmed is not None else INITIALIZING
            return [self._event("RESTOCK_END", None, previous_count=self.restock_previous,
                                current_count=self.confirmed, status=self.state,
                                detail={"observed_count": self.restock_observed})]
        return []

    def snapshot(self, pc_now: float, lane_stale_s: float, link_ok: bool) -> dict:
        since = None if self.last_valid_pc is None else pc_now - self.last_valid_pc
        stale = (not link_ok) or self.error or since is None or since > lane_stale_s
        shown = next((v for v in (self.filtered, self.display_mm, self.last_raw) if v is not None), None)
        return {
            "lane_id": self.cfg.lane_id,
            "product_id": self.cfg.product_id,
            "product_name": self.cfg.product_name,
            "state": self.state,
            "count": self.confirmed,
            "distance_mm": None if self.filtered is None else round(self.filtered, 1),
            "raw_mm": self.last_raw,
            "display_mm": shown,
            "estimated_count": None if shown is None else round(self.continuous_count(shown), 2),
            "candidate": self._cand,
            "stable_elapsed_ms": self.stable_elapsed_ms,
            "pending_resync": self.pending_resync,
            "restock": self.restock,
            "restock_observed": self.restock_observed,
            "error": self.error,
            "error_kind": self.error_kind,
            "sensor_status": self.last_status,
            "stale": stale,
            "seconds_since_valid": None if since is None else round(since, 1),
            "last_valid_uptime_ms": self.last_valid_uptime,
        }
