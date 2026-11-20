"""전체 카메라의 상품 개수로 선반 방문의 꺼냄 · 놓음을 센다 (fusion.overview_count, 기본 꺼짐).

선반 지도(src/shelf)는 선반 카메라가 앞에서 본 자리를 견준다. 푸셔 줄은 앞 상품을 빼면 뒤 상품이 앞으로 와서
자리로는 개수가 안 보인다 (docs/shelf-map.md). 전체 카메라는 그 줄을 옆에서 봐서 상품이 나란히 보인다 — 종류별 **개수**를 센다.

하는 일
  1. 선반 자리: 처음 start_ms 동안 본 상품 상자들을 감싼 네모 + 여유. 그 뒤로는 안 바뀐다.
  2. 장면마다 선반 자리 안의 상품을 종류별로 센다 (같은 종류가 많이 겹친 상자는 하나로).
  3. 사람 상자가 선반 자리와 겹친 장면은 '가림' — 그때의 개수는 안 믿는다.
  4. 가리지 않은 채 개수가 stable_ms 넘게 그대로인 구간 = '본 개수'.
  5. 선반 방문(선반 카메라가 정한 시작 · 끝)마다 **방문 전에 본 개수 - 방문 뒤에 본 개수** = 꺼냄 · 놓음.
     전 = 방문 시작 전에 시작한 마지막 구간, 뒤 = 방문이 끝난 때까지 이어진(또는 그 뒤 처음 나온) 구간.
     '전' 구간은 **앞 방문이 끝난 뒤에도 보고 있던** 것이어야 한다 — 앞 방문보다 먼저 끝난 구간은 앞 방문의 변화를 모른다
     (3차 N11: 손님이 선반 앞에 서서 18초 동안 가렸다. 세 방문 중 마지막만 전 · 뒤가 보였는데, 그 '전'이 첫 방문 전의 개수였다).
     그런 구간이 없으면 앞 방문의 '뒤' 구간을 쓴다. 그것도 없으면 못 센다 — 그 방문은 선반 지도 개수 그대로.
     방문의 **끝**(선반 카메라의 가림이 풀린 때)과 전체 카메라 구간의 끝은 edge_ms 만큼 어긋나도 같은 때로 본다 — 두 카메라가 보는 '가림'이 다르다
     (3차 N08-1: 선반 카메라는 방문이 18.8초에 끝났다고 봤는데 전체 카메라는 18.7초부터 가렸다. 안 봐주면 다음 방문의 꺼냄이 이 방문에 붙는다).
     '뒤' 구간은 방문이 끝나고 late_ms 안에 시작한 것이어야 한다 — 더 늦게 보인 개수는 그 사이 다른 방문의 변화가 섞였을 수 있다.
     (언제 묻든 답이 같게 하려는 규칙이기도 하다: 판정은 이 시간만큼만 기다렸다 묻는다.)
     방문의 **시작**은 안 봐준다 — 시작 뒤에 시작한 구간을 '전'으로 쓰면 그 사이의 꺼냄을 놓친다 (3차 39편: 1.5초 봐주면 5편이 틀려진다).

방문이 없는 때의 개수 변화는 세지 않는다 (전체 카메라만으로는 꺼냄을 만들지 않는다).
지금까지 들어온 장면만 쓴다 — 뒤 개수가 아직 없으면 change() 는 None 이고, 판정은 fusion.overview_count_wait_seconds(= late_ms) 까지 기다린다.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

SCORE = 0.25
MERGE = 0.4
STABLE_MS = 600.0
EDGE_MS = 1000.0
"""방문의 끝과 본 개수 구간의 끝이 이만큼 어긋나도 같은 때로 본다 (판정은 fusion.event_tolerance_seconds 를 넣는다)."""
LATE_MS = 3000.0
"""'뒤 개수'는 방문이 끝나고 이 시간 안에 보이기 시작한 것만 쓴다 (판정은 fusion.overview_count_wait_seconds 를 넣는다)."""
START_MS = 1000.0
PAD = 40.0
NEAR = 200.0
"""처음 start_ms 의 상자 가운데, 가운데값에서 이 거리 안에 있는 것만 선반의 상자로 본다 (엉뚱한 상자 하나가 선반 자리를 키우지 않게)."""


def center(item: list) -> tuple[float, float]:
    return (item[1] + item[3]) / 2, (item[2] + item[4]) / 2


def iou(a: list, b: list) -> float:
    w = min(a[3], b[3]) - max(a[1], b[1])
    h = min(a[4], b[4]) - max(a[2], b[2])
    inter = max(0.0, w) * max(0.0, h)
    union = (a[3] - a[1]) * (a[4] - a[2]) + (b[3] - b[1]) * (b[4] - b[2]) - inter
    return inter / union if union > 0 else 0.0


def merge(items: list, limit: float = MERGE) -> list:
    """같은 종류끼리 limit 이상 겹친 상자는 점수 높은 것 하나만 남긴다. items = [[종류, x1, y1, x2, y2, 점수], …]"""
    keep: list = []
    for it in sorted(items, key=lambda i: -i[5]):
        if all(k[0] != it[0] or iou(k, it) < limit for k in keep):
            keep.append(it)
    return keep


def region_of(boxes: list, pad: float = PAD, near: float = NEAR) -> tuple[float, float, float, float] | None:
    """상품 상자들을 감싼 네모 + 여유 = 선반 자리. 상자가 없으면 None."""
    if not boxes:
        return None
    xs, ys = sorted(center(b)[0] for b in boxes), sorted(center(b)[1] for b in boxes)
    mx, my = xs[len(xs) // 2], ys[len(ys) // 2]
    boxes = [b for b in boxes if abs(center(b)[0] - mx) <= near and abs(center(b)[1] - my) <= near]
    return (min(b[1] for b in boxes) - pad, min(b[2] for b in boxes) - pad,
            max(b[3] for b in boxes) + pad, max(b[4] for b in boxes) + pad)


def inside(items: list, region, score: float = SCORE, limit: float = MERGE) -> list:
    """선반 자리 안의 상품 상자 (가운데 점으로 본다, 점수 score 이상, 겹친 것은 하나로)."""
    if region is None:
        return []
    x1, y1, x2, y2 = region
    return merge([it for it in items if it[5] >= score and x1 <= center(it)[0] <= x2 and y1 <= center(it)[1] <= y2], limit)


def blocked(people: list, region) -> bool:
    """사람 상자가 선반 자리와 겹쳤나. people = [[x1, y1, x2, y2], …]"""
    x1, y1, x2, y2 = region
    return any(min(b[2], x2) > max(b[0], x1) and min(b[3], y2) > max(b[1], y1) for b in people)


@dataclass(frozen=True)
class Seen:
    """가리지 않은 채 개수가 그대로였던 구간."""

    start_ms: float
    end_ms: float
    counts: tuple
    """((종류, 개수), …) — 종류 이름 순, 0개인 종류는 없다."""

    def text(self) -> str:
        return " ".join(f"{name} {n}" for name, n in self.counts) or "없음"


@dataclass(frozen=True)
class CountChange:
    """선반 방문 하나 사이의 개수 변화."""

    removed: int
    added: int
    before: Seen
    after: Seen

    def text(self) -> str:
        return f"{self.before.text()} → {self.after.text()}"


class OverviewCounter:
    def __init__(self, score: float = SCORE, merge_iou: float = MERGE, stable_ms: float = STABLE_MS,
                 start_ms: float = START_MS, region: tuple | None = None, edge_ms: float = EDGE_MS,
                 late_ms: float = LATE_MS) -> None:
        self.score, self.merge_iou, self.stable_ms, self.start_ms = score, merge_iou, stable_ms, start_ms
        self.edge_ms, self.late_ms = edge_ms, late_ms
        self._given = region
        """선반 자리를 밖에서 정해 줄 때 (테스트 · 구역 파일). 없으면 처음 start_ms 의 상자로 정한다."""
        self.reset()

    def reset(self) -> None:
        self.region = self._given
        self.fed = False
        """상품 탐지 결과를 한 번이라도 받았나 — 안 받았으면 판정이 기다리지 않는다."""
        self._buffer: list[tuple[float, list, list]] = []
        self._done: list[Seen] = []
        self._run: tuple | None = None
        self._start = self._last = 0.0
        self._used: int | None = None
        """앞 방문이 '뒤'로 쓴 구간 번호 (앞 방문을 못 셌으면 None)."""
        self._prev_end = float("-inf")
        """앞 방문이 끝난 시각 (셌든 못 셌든)."""

    # ------------------------------------------------------------------ 장면 넣기

    def update(self, ms: float, items: list, people: list) -> None:
        """전체 카메라 한 장면. items = 상품 상자 [[종류, x1, y1, x2, y2, 점수], …], people = 사람 상자들."""
        self.fed = True
        if self.region is not None:
            self._step(ms, items, people)
            return
        self._buffer.append((ms, items, people))
        if ms - self._buffer[0][0] < self.start_ms:
            return
        first = self._buffer[0][0]
        boxes = [it for t, its, _ in self._buffer if t - first <= self.start_ms for it in its if it[5] >= SCORE]
        self.region = region_of(boxes)
        if self.region is None:
            self._buffer = self._buffer[-1:]   # 아직 상품을 못 봤다 — 볼 때까지 미룬다
            return
        for t, its, ppl in self._buffer:
            self._step(t, its, ppl)
        self._buffer = []

    def _step(self, ms: float, items: list, people: list) -> None:
        key = None
        if not blocked(people, self.region):
            count = Counter(it[0] for it in inside(items, self.region, self.score, self.merge_iou))
            key = tuple(sorted(count.items()))
        if key != self._run:
            self._close()
            self._run, self._start = key, ms
        self._last = ms

    def _close(self) -> None:
        if self._run is not None and self._last - self._start >= self.stable_ms:
            self._done.append(Seen(self._start, self._last, self._run))

    # ------------------------------------------------------------------ 묻기

    def seen(self) -> list[Seen]:
        """지금까지 본 개수 구간들 (이어지는 중인 구간도 stable_ms 를 넘겼으면 든다)."""
        if self._run is not None and self._last - self._start >= self.stable_ms:
            return [*self._done, Seen(self._start, self._last, self._run)]
        return list(self._done)

    def ready(self, end_ms: float) -> bool:
        """end_ms 에 끝난 방문의 '뒤 개수'가 될 만한 구간을 이미 봤나 (더 기다릴 것이 없나)."""
        seen = self.seen()
        return bool(seen) and seen[-1].end_ms >= end_ms - self.edge_ms

    def _pick(self, start_ms: float, end_ms: float) -> tuple[int, int] | None:
        seen = self.seen()
        edge = self.edge_ms
        before = [i for i in range(len(seen)) if seen[i].start_ms <= start_ms and seen[i].end_ms >= self._prev_end - edge]
        if before:
            i = before[-1]
        elif self._used is not None:
            i = self._used   # 앞 방문 뒤로 가림이 안 풀린 채(또는 풀리자마자) 이 방문이 시작했다 — 앞 방문의 '뒤'가 이 방문의 '전'
        else:
            return None
        after = [j for j in range(i, len(seen)) if seen[j].end_ms >= end_ms - edge and seen[j].start_ms <= end_ms + self.late_ms]
        return (i, after[0]) if after else None

    def change(self, start_ms: float, end_ms: float) -> CountChange | None:
        """[start_ms, end_ms] 방문 사이의 개수 변화. 전이나 뒤 개수를 못 봤으면 None. 방문은 시간 순서로 물어야 한다."""
        picked = self._pick(start_ms, end_ms)
        self._prev_end = end_ms
        self._used = picked[1] if picked else None
        if picked is None:
            return None
        seen = self.seen()
        before, after = seen[picked[0]], seen[picked[1]]
        b, a = Counter(dict(before.counts)), Counter(dict(after.counts))
        return CountChange(sum((b - a).values()), sum((a - b).values()), before, after)
