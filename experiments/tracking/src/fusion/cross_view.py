"""⑦-a 교차 화면 연결 — 선반 카메라(angle0)의 사람이 전체 카메라(angle1)의 누구인가.

설계 M-4·M-5 (docs/multiview-design.md):
  - 매장 신원은 전체 카메라(angle1)의 화면 신원이 기준이고, 선반 카메라의 사람은 짝을 통해 그 번호를 받는다.
  - 두 화면을 잇는 1순위 근거는 **위치 관계**다 (Complementary-View MHT: 위치를 빼면 CVIDF1 84.0 → 15.5).
    외형은 가린 사진을 빼고 보조로만 쓴다 (VoxelTrack: 가림 처리한 외형 + 위치로 신원 바뀜 0).
  - 장면마다 결정하지 않고 짧은 조각(기본 0.5초) 단위로 잇고, 짝 없음(한쪽 화면에만 있는 사람)을 허용한다.

위치를 비교하는 공통 축 = '선반 앞 띠' 좌표 (0 = 선반의 카메라 쪽 끝, 1 = 계산대·문 쪽 끝).
  - 전체 카메라: 발의 바닥 좌표를 띠 축에 내린 값 (src/zones/floor.py)
  - 선반 카메라: 사람 상자의 가운데 x 를 띠 축으로 옮긴 값 (1차식, 정상 영상으로 맞춤 — scripts/dual_eval.py)

선반 카메라에도 발과 바닥이 보이면(3차 촬영) `fusion.position_mode: floor` — 두 카메라의 발을 같은 바닥 좌표(cm)로 옮겨
바닥 위 거리로 견준다. 한 줄 좌표로는 못 가르던 '같은 좌우 위치에 앞뒤로 선 두 사람'(B-14)을 가른다.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..core.config import FusionConfig


@dataclass
class _Sample:
    pts_ms: float
    band: float | None
    """선반 앞 띠 위치 0~1. 띠 축으로 옮길 수 없으면 None."""
    floor: tuple[float, float] | None = None
    """발의 바닥 좌표 (cm, 두 카메라 공통). 바닥 지도가 없거나 지평선 너머면 None."""
    ray: tuple[float, float] | None = None
    """발목이 안 보일 때(화면 아래로 잘림·가림): 실제 발은 floor 에서 이 방향(단위 벡터)으로 뻗는 반직선 위 어딘가. 발목이 보이면 None."""


def _point_to_ray(p: tuple[float, float], origin: tuple[float, float], direction: tuple[float, float]) -> float:
    vx, vy = p[0] - origin[0], p[1] - origin[1]
    t = vx * direction[0] + vy * direction[1]
    if t <= 0.0:
        return float(np.hypot(vx, vy))
    return float(abs(vx * direction[1] - vy * direction[0]))


def floor_distance(a: _Sample, b: _Sample) -> float:
    """두 표본의 바닥 위 거리 (cm). 발목이 안 보이는 쪽은 반직선 — 점과 반직선, 반직선과 반직선 사이의 가장 가까운 거리."""
    if a.ray is None and b.ray is None:
        return float(np.hypot(a.floor[0] - b.floor[0], a.floor[1] - b.floor[1]))
    if a.ray is None:
        return _point_to_ray(a.floor, b.floor, b.ray)
    if b.ray is None:
        return _point_to_ray(b.floor, a.floor, a.ray)
    # 둘 다 반직선: 만나면 0, 아니면 한쪽 시작점에서 다른 반직선까지 가까운 쪽
    (ax, ay), (adx, ady), (bx, by), (bdx, bdy) = a.floor, a.ray, b.floor, b.ray
    det = adx * (-bdy) - ady * (-bdx)
    if abs(det) > 1e-9:
        t = ((bx - ax) * (-bdy) - (by - ay) * (-bdx)) / det
        s = (adx * (by - ay) - ady * (bx - ax)) / det
        if t >= 0.0 and s >= 0.0:
            return 0.0
    return min(_point_to_ray(a.floor, b.floor, b.ray), _point_to_ray(b.floor, a.floor, a.ray))


@dataclass
class _Track:
    """한 화면의 한 사람: 최근 위치 표본과 외형."""

    samples: deque = field(default_factory=lambda: deque(maxlen=200))
    appearance: np.ndarray | None = None
    last_ms: float = -1e18
    candidate_ms: float = -1e18
    """짝 후보로 삼아도 된다고 표시된 마지막 표본의 시각 (전체 카메라 — observe 의 candidate)."""


@dataclass
class Link:
    """한 조각에서 정한 짝 하나."""

    start_ms: float
    end_ms: float
    detail_id: int
    """선반 카메라 화면 신원."""
    overview_id: int
    """전체 카메라 화면 신원 (= 매장 신원)."""
    score: float
    position: float | None
    appearance: float | None

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


class CrossViewLinker:
    def __init__(self, config: FusionConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._detail: dict[int, _Track] = {}
        self._overview: dict[int, _Track] = {}
        self.links: list[Link] = []
        self._current: dict[int, int] = {}
        """선반 카메라 신원 → 지금 짝인 전체 카메라 신원."""
        self._current_end: dict[int, float] = {}
        """선반 카메라 신원 → 그 짝이 마지막으로 맺힌 조각의 끝 시각 (keep_bonus_recent_only)."""
        self._history: dict[int, list[tuple[float, float, int]]] = {}
        """선반 카메라 신원 → [(시작, 끝, 전체 카메라 신원)]. 짝이 이어지면 구간을 늘린다."""
        self._last_window_ms: float | None = None
        self.windows: deque = deque(maxlen=8)
        """닫힌 조각들 (시작, 끝, {선반 카메라 신원: 전체 카메라 신원}) — 짝이 없던 조각은 빈 사전. 번호 맞바뀜 확인(swap_repair)이 본다."""
        self.closed = 0
        """지금까지 닫힌 조각 수."""

    # ------------------------------------------------------------------ 입력

    def observe(self, view: str, person_id: int, pts_ms: float, band: float | None,
                appearance: np.ndarray | None = None, floor: tuple[float, float] | None = None,
                ray: tuple[float, float] | None = None, candidate: bool = True) -> None:
        """view: 'detail'(선반 카메라) / 'overview'(전체 카메라). floor·ray = 발의 바닥 좌표(cm)와, 발목이 안 보일 때의 방향
        — position_mode "floor" 에서 쓴다.

        candidate=False 인 표본도 위치·외형에는 그대로 쓰지만, **그 조각에 candidate=True 표본이 하나도 없는 전체 카메라 신원은
        짝 후보에서 빠진다** (overview_candidates). 표본을 아예 안 넣으면 남은 몇 장만으로 위치 점수가 높게 나와
        오히려 짝을 가로챘다 (3차 C19 17.7초, 후속 43)."""
        if person_id <= 0:
            return
        tracks = self._detail if view == "detail" else self._overview
        track = tracks.setdefault(person_id, _Track())
        track.samples.append(_Sample(pts_ms, band, floor, ray))
        track.last_ms = pts_ms
        if candidate:
            track.candidate_ms = pts_ms
        if appearance is not None:
            track.appearance = appearance

    def forget(self, view: str, person_id: int) -> None:
        """신원이 합쳐져 없어진 번호."""
        (self._detail if view == "detail" else self._overview).pop(person_id, None)

    # ------------------------------------------------------------------ 짝짓기

    def update(self, now_ms: float) -> list[Link]:
        """조각(window_seconds)이 찰 때마다 짝을 다시 정한다. 이번에 정한 짝들을 돌려준다."""
        window = self.config.window_seconds * 1000.0
        if self._last_window_ms is None:
            self._last_window_ms = now_ms
            return []
        if now_ms - self._last_window_ms < window:
            return []
        start, end = self._last_window_ms, now_ms
        self._last_window_ms = now_ms
        pairs: dict[int, int] = {}
        self.windows.append((start, end, pairs))
        self.closed += 1

        detail = [pid for pid, t in self._detail.items() if t.last_ms >= start]
        overview = self.overview_candidates(start)
        if not detail or not overview:
            return []

        scores = np.full((len(detail), len(overview)), -1.0)
        parts: dict[tuple[int, int], tuple[float | None, float | None]] = {}
        for i, d in enumerate(detail):
            pos_of = {o: self._position_score(self._detail[d], self._overview[o], start, end) for o in overview}
            solo = sum(p is not None for p in pos_of.values()) == 1
            for j, o in enumerate(overview):
                pos = pos_of[o]
                app = self._appearance_score(self._detail[d], self._overview[o])
                score = self._combine(pos, app)
                if score is None:
                    continue
                if solo and self._solo_rescue(app):
                    score = max(score, self.config.min_score)   # 비교할 상대가 한 명뿐 — 외형이 높으면 위치가 어긋나도 잇는다
                if self._current.get(d) == o and (not self.config.keep_bonus_recent_only
                                                  or start - self._current_end.get(d, -1e18) <= window * 1.5):
                    score += self.config.keep_bonus   # 한 조각의 흔들림으로 짝을 바꾸지 않는다
                scores[i, j] = score
                parts[(d, o)] = (pos, app)

        rows, cols = linear_sum_assignment(-scores)
        made: list[Link] = []
        for i, j in zip(rows, cols):
            if scores[i, j] < self.config.min_score:
                continue
            d, o = detail[i], overview[j]
            pos, app = parts[(d, o)]
            link = Link(start, end, d, o, float(scores[i, j]), pos, app)
            made.append(link)
            pairs[d] = o
            self._remember(d, o, start, end)
        for d in detail:
            if d not in {l.detail_id for l in made}:
                self._current.pop(d, None)
        self.links.extend(made)
        return made

    def overview_candidates(self, start: float) -> list[int]:
        """이 조각(start 부터)에 보인 전체 카메라 신원 중 짝 후보 — 후보 표시(candidate)가 붙은 표본이 이 조각에 하나라도 있는 신원.
        표시를 안 쓰면(기본) 보인 신원 전부다."""
        return [pid for pid, t in self._overview.items() if t.last_ms >= start and t.candidate_ms >= start]

    def _combine(self, pos: float | None, app: float | None) -> float | None:
        if pos is None:
            return None   # 위치를 비교할 수 없으면(띠 밖, 같은 때 표본 없음) 외형만으로는 잇지 않는다
        if app is None:
            return pos
        w = self.config.position_weight
        return w * pos + (1.0 - w) * app

    def _solo_rescue(self, app: float | None) -> bool:
        floor = self.config.solo_appearance_min
        return floor > 0 and app is not None and app >= floor

    def _position_score(self, a: _Track, b: _Track, start: float, end: float) -> float | None:
        """같은 때(±pair_tolerance) 두 화면의 위치가 얼마나 가까운가. 0~1.

        position_mode "band": 띠 위치 차이. "floor": 바닥 위 거리(floor_distance) — 한쪽이라도 바닥 좌표가 없는 표본은 띠 위치로 대신한다.
        """
        tol = self.config.pair_tolerance_seconds * 1000.0
        use_floor = self.config.position_mode == "floor"
        bs = [s for s in b.samples if start - tol <= s.pts_ms <= end + tol
              and (s.band is not None or (use_floor and s.floor is not None))]
        if not bs:
            return None
        sims = []
        for s in a.samples:
            if not (start <= s.pts_ms <= end) or (s.band is None and not (use_floor and s.floor is not None)):
                continue
            nearest = min(bs, key=lambda x: abs(x.pts_ms - s.pts_ms))
            if abs(nearest.pts_ms - s.pts_ms) > tol:
                continue
            if use_floor and s.floor is not None and nearest.floor is not None:
                gap = floor_distance(s, nearest)
                if s.ray is not None or nearest.ray is not None:
                    gap = max(0.0, gap - self.config.ray_slack_cm)   # 가린 발은 이 정도 차이를 못 가른다
                sims.append(max(0.0, 1.0 - gap / self.config.max_floor_gap_cm))
            elif s.band is not None and nearest.band is not None:
                gap = abs(nearest.band - s.band)
                sims.append(max(0.0, 1.0 - gap / self.config.max_band_gap))
        return float(np.mean(sims)) if sims else None

    @staticmethod
    def _appearance_score(a: _Track, b: _Track) -> float | None:
        if a.appearance is None or b.appearance is None:
            return None
        return float(np.clip(a.appearance @ b.appearance, 0.0, 1.0))

    def _remember(self, d: int, o: int, start: float, end: float) -> None:
        self._current[d] = o
        self._current_end[d] = end
        spans = self._history.setdefault(d, [])
        if spans and spans[-1][2] == o and start - spans[-1][1] <= self.config.window_seconds * 1000.0 * 1.5:
            spans[-1] = (spans[-1][0], end, o)
        else:
            spans.append((start, end, o))

    # ------------------------------------------------------------------ 번호 맞바뀜

    def mutual_swaps(self) -> list[tuple[int, int, int, int]]:
        """방금 닫힌 조각에서 짝이 서로 맞바뀐 선반 카메라 두 사람: [(가, 나, A, B)].
        앞 두 조각은 (가⇄A, 나⇄B) 그대로였고 이번 조각은 (가⇄B, 나⇄A)."""
        if len(self.windows) < 3:
            return []
        (_, _, first), (_, _, prev), (_, _, now) = list(self.windows)[-3:]
        ids = sorted(set(first) & set(prev) & set(now))
        out = []
        for i, d1 in enumerate(ids):
            for d2 in ids[i + 1:]:
                a, b = prev[d1], prev[d2]
                if a != b and first[d1] == a and first[d2] == b and now[d1] == b and now[d2] == a:
                    out.append((d1, d2, a, b))
        return out

    def swap_overview(self, a: int, b: int, since_ms: float, seen_since_ms: float | None = None) -> None:
        """since_ms 부터의 기록에서 전체 카메라 신원 a·b 를 서로 바꾼다 — 그때부터 전체 카메라가 두 사람의 번호를 뒤바꿔 붙였다.
        짝 기록(links·구간), 지금 짝, 닫힌 조각, 두 신원의 위치 표본·외형.

        seen_since_ms (fusion.swap_since_confusion): 두 신원의 **위치 표본**만 이때부터 바꾼다 (since_ms 보다 이르다).
        번호는 그때 바뀌었지만 짝은 더 늦게 뒤집혔다 — 그 사이의 짝 기록은 선반 카메라 번호를 따라 이어진 것이라 안 바꾼다."""
        flip = {a: b, b: a}
        since = since_ms - 1e-6
        gap = self.config.window_seconds * 1000.0 * 1.5
        for link in self.links:
            if link.start_ms >= since and link.overview_id in flip:
                link.overview_id = flip[link.overview_id]
        for d, spans in self._history.items():
            fixed: list[tuple[float, float, int]] = []
            for s, e, o in spans:
                if s >= since and o in flip:
                    o = flip[o]
                if fixed and fixed[-1][2] == o and s - fixed[-1][1] <= gap:
                    fixed[-1] = (fixed[-1][0], e, o)
                else:
                    fixed.append((s, e, o))
            self._history[d] = fixed
        for d, o in list(self._current.items()):
            if o in flip and self._current_end.get(d, -1e18) > since:
                self._current[d] = flip[o]
        for s, _, pairs in self.windows:
            if s >= since:
                for d, o in list(pairs.items()):
                    pairs[d] = flip.get(o, o)
        ta, tb = self._overview.setdefault(a, _Track()), self._overview.setdefault(b, _Track())
        if seen_since_ms is not None:
            since = min(since, seen_since_ms - 1e-6)
        old_a, new_a = [x for x in ta.samples if x.pts_ms < since], [x for x in ta.samples if x.pts_ms >= since]
        old_b, new_b = [x for x in tb.samples if x.pts_ms < since], [x for x in tb.samples if x.pts_ms >= since]

        def marked(mine: float, other: float) -> float:
            return other if other >= since else (mine if mine < since else -1e18)

        ta.candidate_ms, tb.candidate_ms = marked(ta.candidate_ms, tb.candidate_ms), marked(tb.candidate_ms, ta.candidate_ms)
        for t, samples in ((ta, old_a + new_b), (tb, old_b + new_a)):
            t.samples = deque(samples, maxlen=t.samples.maxlen)
            t.last_ms = samples[-1].pts_ms if samples else -1e18
        if new_a or new_b:
            ta.appearance, tb.appearance = tb.appearance, ta.appearance
        for pid, t in ((a, ta), (b, tb)):
            if not t.samples:
                self._overview.pop(pid, None)

    # ------------------------------------------------------------------ 조회

    def owner(self, detail_id: int, start_ms: float, end_ms: float | None = None) -> int | None:
        """선반 카메라 신원이 그 시간 동안 누구였나 — 겹친 시간이 가장 긴 짝. 없으면 가장 가까운 짝(2초 안)."""
        end_ms = start_ms if end_ms is None else end_ms
        spans = self._history.get(detail_id, [])
        best, best_overlap = None, 0.0
        for s, e, o in spans:
            overlap = min(e, end_ms) - max(s, start_ms)
            if overlap > best_overlap:
                best, best_overlap = o, overlap
        if best is not None:
            return best
        near, near_gap = None, 2000.0
        for s, e, o in spans:
            gap = max(s - end_ms, start_ms - e, 0.0)
            if gap <= near_gap:
                near, near_gap = o, gap
        return near

    def current(self, detail_id: int) -> int | None:
        return self._current.get(detail_id)

    def detail_ids(self, since_ms: float) -> list[int]:
        """since_ms 뒤로 선반 카메라에 보인 신원."""
        return [pid for pid, t in self._detail.items() if t.last_ms >= since_ms]

    def detail_feet(self, detail_id: int, start_ms: float, end_ms: float) -> dict[float, tuple[float, float]]:
        """선반 카메라 신원의 그 시간 발 자리 {시각: 바닥 cm} — **발목이 보인 장면만** (가려서 짐작한 자리는 뺀다)."""
        track = self._detail.get(detail_id)
        if track is None:
            return {}
        return {s.pts_ms: s.floor for s in track.samples
                if start_ms <= s.pts_ms <= end_ms and s.floor is not None and s.ray is None}
