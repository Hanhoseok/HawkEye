"""레인 짧은 보정: 빈 상태 → 상품 1개 → 상품 2개 위치를 재서 수량별 기준 거리와 허용 오차를 계산한다.

화면·파일과 무관한 순수 로직. GUI 는 LevelCapture 에 측정값을 넣고 진행 상태를 보여 주기만 한다.

- 각 단계는 "값이 안정될 때까지 기다림(settle) → 일정 시간 측정(measure)" 이다.
  측정 중에 값이 크게 바뀌면(손이 닿음 등) 다시 기다린다.
- 상품 지름(두께) = 1개 위치 − 2개 위치. 두 번째 상품부터는 밀판이 지름만큼 움직이기 때문이다.
- 0개·1개 간격은 레일마다 다르다(밀판 앞 고정 블록). 두 상태의 측정 범위를 보고 겹치지 않게 나눈다.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import median
from typing import Optional

WAIT_STABLE = "WAIT_STABLE"
MEASURING = "MEASURING"
DONE = "DONE"


@dataclass
class CaptureResult:
    median: float
    lo: float          # 필터값 하위 2%
    hi: float          # 필터값 상위 2%
    samples: int

    @property
    def center(self) -> float:
        return (self.lo + self.hi) / 2

    @property
    def half_range(self) -> float:
        return (self.hi - self.lo) / 2


class LevelCapture:
    """한 위치(빈 상태, 1개, 2개)의 거리를 안정된 구간에서만 잰다. 시각은 ESP32 측정 시각(ms)."""

    def __init__(self, settle_ms: int = 2500, measure_ms: int = 5000, spread_mm: float = 8.0,
                 median_window: int = 5, max_gap_ms: int = 1000):
        self.settle_ms, self.measure_ms, self.spread_mm = settle_ms, measure_ms, spread_mm
        self.median_window, self.max_gap_ms = median_window, max_gap_ms
        self.reset()

    def reset(self) -> None:
        self.state = WAIT_STABLE
        self._raw: deque[float] = deque(maxlen=self.median_window)
        self._win: deque[tuple[int, float]] = deque()
        self._measured: list[float] = []
        self._measure_start: Optional[int] = None
        self._last_t: Optional[int] = None
        self.result: Optional[CaptureResult] = None

    def progress(self) -> tuple[str, float]:
        """(상태, 0~1 진행률)"""
        if self.state == DONE:
            return DONE, 1.0
        if self.state == MEASURING and self._measure_start is not None and self._last_t is not None:
            return MEASURING, min(1.0, (self._last_t - self._measure_start) / self.measure_ms)
        if self._win:
            return WAIT_STABLE, min(1.0, (self._win[-1][0] - self._win[0][0]) / self.settle_ms)
        return WAIT_STABLE, 0.0

    def feed(self, t_ms: int, distance: float) -> None:
        if self.state == DONE:
            return
        if self._last_t is not None and (t_ms < self._last_t or t_ms - self._last_t > self.max_gap_ms):
            self.reset()                      # 재부팅·긴 공백: 처음부터
        self._last_t = t_ms
        self._raw.append(float(distance))
        if len(self._raw) < min(3, self.median_window):
            return
        m = median(self._raw)
        self._win.append((t_ms, m))
        while self._win and t_ms - self._win[0][0] > self.settle_ms:
            self._win.popleft()
        vals = [v for _, v in self._win]
        steady = max(vals) - min(vals) <= self.spread_mm

        if self.state == WAIT_STABLE:
            enough = self._win[-1][0] - self._win[0][0] >= self.settle_ms * 0.95 and len(self._win) >= 8
            if steady and enough:
                self.state = MEASURING
                self._measure_start = t_ms
                self._measured = [m]
            return

        # MEASURING: 측정 중 값이 크게 바뀌면 다시 기다림
        ref = median(self._measured)
        if abs(m - ref) > self.spread_mm * 1.5:
            self.state = WAIT_STABLE
            self._win = deque([(t_ms, m)])
            self._measured = []
            return
        self._measured.append(m)
        if t_ms - self._measure_start >= self.measure_ms and len(self._measured) >= 15:
            xs = sorted(self._measured)
            k = max(0, int(len(xs) * 0.02))
            self.result = CaptureResult(median(xs), xs[k], xs[-1 - k], len(xs))
            self.state = DONE


@dataclass
class CalibrationResult:
    count_distance_mm: list[float]
    count_tolerance_mm: list[float]
    hysteresis_mm: float
    median_window: int
    thickness_mm: float
    empty_distance_mm: float
    max_count_geometry: int          # 센서 최소 거리 기준으로 가능한 최대 수량
    gap01_mm: float                  # 0개·1개 기준 거리 간격
    warnings: list[str] = field(default_factory=list)

    def for_max_count(self, max_count: int) -> tuple[list[float], list[float]]:
        """사용자가 정한 최대 수량에 맞춰 표 길이를 자른다."""
        n = max_count + 1
        return self.count_distance_mm[:n], self.count_tolerance_mm[:n]


class CalibrationError(Exception):
    pass


def _r(x: float) -> float:
    return round(x, 1)


def compute_settings(empty: CaptureResult, one: CaptureResult, two: CaptureResult,
                     valid_min_mm: float = 30, max_items: int = 20) -> CalibrationResult:
    thickness = one.median - two.median
    if not 20 <= thickness <= 250:
        raise CalibrationError(
            f"1개({one.median:.0f}mm)와 2개({two.median:.0f}mm) 위치 차이가 {thickness:.0f}mm 입니다. "
            "두 번째 상품을 넣었는지, 밀판이 걸리지 않았는지 확인하세요.")
    gap_raw = empty.median - one.median
    if gap_raw < 3:
        raise CalibrationError(
            f"상품 1개를 넣어도 거리가 {gap_raw:.0f}mm 만 바뀌었습니다. 밀판이 움직이지 않아 0개와 1개를 구분할 수 없습니다. "
            "밀판 앞면에 스페이서를 붙이거나 센서 방향을 확인하세요.")

    warnings: list[str] = []
    # 기준은 측정 범위의 가운데 (빈 상태 값은 한쪽으로 치우쳐 퍼지는 경우가 있음)
    ref0, ref1, ref2 = empty.center, one.center, two.center
    gap = ref0 - ref1
    # 짧은 측정은 실제 흔들림을 덜 보므로 30% + 1mm 여유
    s0 = empty.half_range * 1.3 + 1
    s1 = one.half_range * 1.3 + 1
    h = min(4.0, max(0.0, round(gap * 0.15)))
    avail = gap - h - 0.5
    if s0 + s1 <= avail:
        extra = avail - s0 - s1
        tol0, tol1 = min(15.0, s0 + extra / 2), min(15.0, s1 + extra / 2)
    else:
        tol0 = avail * s0 / (s0 + s1)
        tol1 = avail - tol0
        warnings.append(
            f"빈 상태와 1개 상태의 측정 범위가 겹칩니다 (간격 {gap:.0f}mm). 가끔 0개·1개 판정이 늦거나 틀릴 수 있습니다.")
    if gap < 8:
        warnings.append(f"0개·1개 간격이 {gap:.0f}mm 로 매우 좁습니다. 밀판 앞면에 스페이서를 붙이면 안정적으로 구분됩니다.")
    elif gap < 12:
        warnings.append(f"0개·1개 간격이 {gap:.0f}mm 로 좁아 센서가 조금만 움직여도 다시 보정해야 합니다.")
    tol_n = max(3.0, min(15.0, (thickness - h) / 2 - 2))

    refs = [ref0, ref1, ref2]
    tols = [tol0, tol1, tol_n]
    while len(refs) < max_items + 1:
        nxt = refs[-1] - thickness
        if nxt - tol_n - h < valid_min_mm + 5:
            break
        refs.append(nxt)
        tols.append(tol_n)
    max_geo = len(refs) - 1
    if refs[-1] - tols[-1] - h < valid_min_mm:
        warnings.append("상품 2개 위치가 센서에 너무 가깝습니다. 센서를 레일 뒤로 더 빼야 합니다.")
    return CalibrationResult(
        count_distance_mm=[_r(x) for x in refs], count_tolerance_mm=[_r(x) for x in tols],
        hysteresis_mm=float(h), median_window=7 if gap < 12 else 5,
        thickness_mm=_r(thickness), empty_distance_mm=_r(empty.median),
        max_count_geometry=max_geo, gap01_mm=_r(gap), warnings=warnings)
