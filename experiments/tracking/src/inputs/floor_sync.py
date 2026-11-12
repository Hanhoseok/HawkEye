"""두 카메라의 발 위치(공통 바닥 좌표)로 시차를 찾는다 — 3차 촬영처럼 두 카메라 모두 바닥이 보일 때.

시차 정의는 src/inputs/sync.py 와 같다: 공통 시각 = angle0 재생 시각 + offset_ms (기준 angle1).

시차 후보마다 angle0 의 각 표본 시각 t 와 angle1 의 t + lag 표본을 짝지어, 두 발 집합의
양방향 최근접 거리(cm, 상한 CAP_CM)를 잰다. 같은 사람이 같은 바닥 자리에 서는 시차에서 비용이 가장 작다.
사람 수가 달라도 되고(한쪽만 잡힌 가짜 상자는 상한으로 영향 제한), 두 카메라가 함께 보는 바닥만 쓴다
(한 카메라 시야 밖의 사람은 늘 '짝 없음'이라 모든 시차에 같은 벌점을 줘 골을 흐리게 하므로 뺀다).

영상 라이브러리 없이 계산만 — 테스트 대상.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CAP_CM = 150.0
"""한 발의 최근접 거리 상한 — 한쪽에만 보이는 사람·가짜 상자 영향 제한."""
TOL_MS = 25.0
"""angle1 표본을 짝지을 때 허용 시각 차 (angle1 20Hz → 최근접 표본은 25ms 안)."""
STEP_MS = 20.0
MAX_LAG_MS = 3000.0
MIN_PAIRS = 20
"""한 시차 후보에서 짝지은 표본이 이보다 적으면 그 후보는 뺀다."""
EXCLUDE_MS = 500.0
"""뾰족함: (최소 지점 ±이 범위 밖의 최소 비용 + NOISE_CM) / (최소 비용 + NOISE_CM)."""
NOISE_CM = 5.0
"""비용이 0 에 가까울 때 뾰족함이 터지지 않게 더하는 바닥값 (발 점·보정 오차 수준)."""

# 믿을 만함 기준 (method=floor). 39쌍 결과를 보고 바꾸면 문서에 적는다.
OK_MIN_PAIRS = 30
OK_MIN_SHARPNESS = 1.5
OK_MAX_COST_CM = 60.0

Samples = list[tuple[float, np.ndarray]]
"""(시각 ms, (k, 2) 바닥 cm 발 위치) — 시각 순."""


def chamfer(a: np.ndarray, b: np.ndarray, cap: float = CAP_CM) -> float:
    """두 점 집합의 양방향 최근접 거리 평균 (각 거리 cap 으로 자름)."""
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
    d = np.minimum(d, cap)
    return float((d.min(axis=1).sum() + d.min(axis=0).sum()) / (len(a) + len(b)))


def lag_cost(a0: Samples, a1: Samples, lag_ms: float, tol_ms: float = TOL_MS) -> tuple[float, int]:
    """(평균 비용 cm, 짝지은 표본 수). 양쪽 다 사람이 있는 표본만."""
    if not a0 or not a1:
        return float("inf"), 0
    t1 = np.array([t for t, _ in a1])
    costs = []
    for t, pts in a0:
        if len(pts) == 0:
            continue
        target = t + lag_ms
        i = int(np.searchsorted(t1, target))
        best = min((k for k in (i - 1, i) if 0 <= k < len(t1)), key=lambda k: abs(t1[k] - target))
        if abs(t1[best] - target) > tol_ms or len(a1[best][1]) == 0:
            continue
        costs.append(chamfer(pts, a1[best][1]))
    return (float(np.mean(costs)) if costs else float("inf")), len(costs)


@dataclass
class FloorSyncResult:
    offset_ms: float | None
    cost_cm: float
    pairs: int
    sharpness: float
    second_ms: float | None
    curve: list[tuple[float, float, int]] = field(default_factory=list)
    """(lag, 비용, 표본) — 그림용."""

    def judge(self) -> tuple[bool, list[str]]:
        """믿을 만한가와 아니면 이유."""
        why = []
        if self.offset_ms is None:
            return False, ["두 카메라에 함께 보이는 사람 표본이 부족"]
        if self.pairs < OK_MIN_PAIRS:
            why.append(f"표본 {self.pairs} < {OK_MIN_PAIRS}")
        if self.sharpness < OK_MIN_SHARPNESS:
            why.append(f"뾰족함 {self.sharpness:.2f} < {OK_MIN_SHARPNESS}")
        if self.cost_cm > OK_MAX_COST_CM:
            why.append(f"비용 {self.cost_cm:.0f}cm > {OK_MAX_COST_CM:.0f}")
        return not why, why


def search(a0: Samples, a1: Samples, max_lag_ms: float = MAX_LAG_MS, step_ms: float = STEP_MS,
           min_pairs: int = MIN_PAIRS, center_ms: float = 0.0) -> FloorSyncResult:
    """center ± max_lag 를 step 간격으로 훑어 비용이 가장 작은 시차. 3점 포물선으로 다듬는다."""
    lags = np.arange(center_ms - max_lag_ms, center_ms + max_lag_ms + step_ms / 2, step_ms)
    curve = []
    for lag in lags:
        c, n = lag_cost(a0, a1, float(lag))
        curve.append((float(lag), c, n))
    valid = [(lag, c, n) for lag, c, n in curve if n >= min_pairs and np.isfinite(c)]
    if not valid:
        return FloorSyncResult(None, float("inf"), 0, 0.0, None, curve)
    k = min(range(len(valid)), key=lambda i: valid[i][1])
    lag, c, n = valid[k]
    refined = lag
    if 0 < k < len(valid) - 1 and valid[k - 1][0] == lag - step_ms and valid[k + 1][0] == lag + step_ms:
        y0, y1, y2 = valid[k - 1][1], c, valid[k + 1][1]
        den = y0 - 2 * y1 + y2
        if den > 0:
            refined = lag + 0.5 * (y0 - y2) / den * step_ms
    far = [(l2, c2) for l2, c2, _ in valid if abs(l2 - lag) > EXCLUDE_MS]
    if far:
        second_ms, second_cost = min(far, key=lambda x: x[1])
        sharp = (second_cost + NOISE_CM) / (c + NOISE_CM)
    else:
        second_ms, sharp = None, 0.0
    return FloorSyncResult(float(refined), float(c), int(n), float(sharp), second_ms, curve)


def median_distance(a0: Samples, a1: Samples, lag_ms: float, tol_ms: float = TOL_MS) -> float:
    """맞춘 시차에서 한 명씩만 보일 때 두 카메라 발 거리의 중앙값 (cm) — 보정 품질 확인용."""
    t1 = np.array([t for t, _ in a1]) if a1 else np.array([])
    dists = []
    for t, pts in a0:
        if len(pts) != 1 or len(t1) == 0:
            continue
        i = int(np.argmin(np.abs(t1 - (t + lag_ms))))
        if abs(t1[i] - (t + lag_ms)) <= tol_ms and len(a1[i][1]) == 1:
            dists.append(float(np.hypot(*(pts[0] - a1[i][1][0]))))
    return float(np.median(dists)) if dists else float("nan")


AGREE_MS = 150.0
"""소리와 발 위치 최소가 이만큼 안이면 같은 답으로 본다."""
AGREE_COST_CM = 1.5
"""소리 시차에서의 발 위치 비용이 최소보다 이만큼 이하로만 크면 발 위치도 그 시차를 받아들인다."""
AGREE_MIN_SHARPNESS = 1.2
"""소리 시차 둘레(±EXCLUDE_MS 밖 최소 / 소리 시차 비용)가 이만큼은 돼야 발 위치가 그 시차를 '가려냈다'고 본다.
비용 곡선이 평평하면(사람이 내내 서 있음) 어느 시차든 '맞는다' — 그런 일치는 증거가 아니다 (C01·T09)."""


def sharpness_at(res: FloorSyncResult, lag_ms: float, cost: float, min_pairs: int = MIN_PAIRS) -> float:
    """주어진 시차에서의 뾰족함: (그 시차 ±EXCLUDE_MS 밖 최소 비용 + NOISE_CM) / (그 시차 비용 + NOISE_CM)."""
    far = [c for lag, c, n in res.curve if abs(lag - lag_ms) > EXCLUDE_MS and n >= min_pairs and np.isfinite(c)]
    if not far or not np.isfinite(cost):
        return 0.0
    return (min(far) + NOISE_CM) / (cost + NOISE_CM)


@dataclass
class Decision:
    offset_ms: float
    method: str
    """floor+audio (두 신호 일치, 믿음) / floor (소리 없음·무효, 발 위치만 통과) / floor-low (확인 필요)."""
    ok: bool
    why: list[str]


def combine(res: FloorSyncResult, audio_ms: float | None, audio_ok: bool,
            cost_at_audio: float, pairs_at_audio: int, fallback_ms: float = 0.0) -> Decision:
    """발 위치 결과와 소리 후보를 합친다 (기준값은 이번 39쌍을 보고 정함 — docs 에 적음).

    - 소리 유효 + 두 값 차 ≤ AGREE_MS + 소리 시차에서 비용 ≤ 최소 + AGREE_COST_CM + 표본·절대 비용 기준 통과
      → floor+audio, 시차 = 소리값 (ms 단위로 더 촘촘). 뾰족함만 면제.
    - 소리 유효, 발 위치와 모순은 없지만 발 위치 곡선이 그 시차를 가려내지 못함(뾰족함 < AGREE_MIN_SHARPNESS)
      → floor-low, 시작값 = 소리값 (발 위치가 반대하지 않으므로 소리가 더 나은 출발점).
    - 소리 유효인데 맞지 않음 → floor-low (발 위치 판정과 무관). 시작값(사람 확인의 출발점)은
      발 위치가 소리 시차를 반대하지 않으면(그 시차 비용 ≤ 최소 + AGREE_COST_CM — 곡선이 넓어 두 값을 못 가름) 소리값,
      반대하면 발 위치값. 어느 쪽이든 note 에 두 후보가 남는다.
    - 소리 없음·무효 → 발 위치 판정 (floor / floor-low).
    """
    floor_ok, floor_why = res.judge()
    base = res.offset_ms if res.offset_ms is not None else fallback_ms
    if res.offset_ms is None:
        floor_why = floor_why + [f"추정값 아님, 믿을 만한 짝 중앙값 {fallback_ms:+.0f}ms"]
    if not audio_ok or audio_ms is None:
        why = floor_why + (["소리 후보 무효(피크 약함·무음)"] if audio_ms is not None else ["소리 없음"])
        return Decision(base, "floor" if floor_ok else "floor-low", floor_ok, [] if floor_ok else why)
    why = []
    consistent = res.offset_ms is not None and (abs(res.offset_ms - audio_ms) <= AGREE_MS
                                                or cost_at_audio <= res.cost_cm + AGREE_COST_CM)
    if res.offset_ms is None or abs(res.offset_ms - audio_ms) > AGREE_MS:
        why.append(f"소리와 불일치 (소리 {audio_ms:+.0f}ms)")
    elif cost_at_audio > res.cost_cm + AGREE_COST_CM:
        why.append(f"소리 시차에서 발 위치 비용 {cost_at_audio:.1f}cm > 최소 {res.cost_cm:.1f}+{AGREE_COST_CM}")
    if pairs_at_audio < OK_MIN_PAIRS:
        why.append(f"소리 시차 표본 {pairs_at_audio} < {OK_MIN_PAIRS}")
    if not np.isfinite(cost_at_audio) or cost_at_audio > OK_MAX_COST_CM:
        why.append(f"소리 시차 비용 {cost_at_audio:.0f}cm > {OK_MAX_COST_CM:.0f}")
    sharp = sharpness_at(res, audio_ms, cost_at_audio)
    if not why and sharp < AGREE_MIN_SHARPNESS:
        why.append(f"발 위치가 소리 시차를 가려내지 못함 (그 둘레 뾰족함 {sharp:.2f} < {AGREE_MIN_SHARPNESS})")
    if why:
        start = float(audio_ms) if consistent else base
        return Decision(start, "floor-low", False, why + ([] if floor_ok else floor_why))
    return Decision(float(audio_ms), "floor+audio", True, [])
