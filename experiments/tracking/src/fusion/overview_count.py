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

from collections import Counter, deque
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

    @property
    def removed_by(self) -> dict[str, int]:
        """종류별로 줄어든 개수."""
        return dict(Counter(dict(self.before.counts)) - Counter(dict(self.after.counts)))

    @property
    def added_by(self) -> dict[str, int]:
        """종류별로 늘어난 개수."""
        return dict(Counter(dict(self.after.counts)) - Counter(dict(self.before.counts)))


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


# ---------------------------------------------------------------------- 상품을 든 사람

HOLD_FRAMES = 5
"""그 종류의 상품을 든 것으로 보려면 이만큼의 장면에서 그 사람 손에 있어야 한다 (초당 15장이면 0.3초).
3차 정상(N) 영상에서 **그 방문에서 줄지 않은 종류**가 손에 붙은 장면은 방문마다 최대 1장이었다 — 그 위로 잡았다.
복합 · 절도 영상에는 6 ~ 7장 붙은 방문이 셋 있다 (C01 · C19 · T05) — 줄지 않은 종류는 묻지 않으므로 결과에는 안 들어갔다.
3 ~ 20장에서 3차 39편의 판정 점수는 같았다 (docs/shelf-map.md)."""
HOLD_WITHIN_MS = 2000.0
"""바로 세기(fresh): HOLD_FRAMES 장면이 이 시간 안에 모여야 '지금 들었다'로 본다. 맞춰 정한 값이 아니다."""
REACH = 1.5
"""상품 상자 가운데에서 손목까지의 거리가 상품 상자 대각선의 이 배수 안이면 그 사람이 든 것으로 본다.
맞춰 정한 값이 아니다 — 0(손목을 안 쓰고 상자만) · 1 · 1.5 · 2.5 에서 3차 39편의 판정 점수는 같았다."""
WRIST_CONF = 0.3
WRISTS = (9, 10)


def holder_of(item: list, people: list, reach: float = REACH) -> int | None:
    """이 상품 상자를 든 사람. people = [(신원, 사람 상자, 관절 또는 None), …]

    ① 손목이 가장 가까운 사람 (reach 안). 둘째로 가까운 사람도 reach 안이고 첫째의 1.5배보다 가까우면 못 정한다.
    ② 손목이 아무도 안 닿으면, 상품 가운데가 **한 사람의 상자 안에만** 있을 때 그 사람.
    """
    cx, cy = center(item)
    diag = ((item[3] - item[1]) ** 2 + (item[4] - item[2]) ** 2) ** 0.5 or 1.0
    near = []
    for pid, _, kp in people:
        ds = [((kp[k][0] - cx) ** 2 + (kp[k][1] - cy) ** 2) ** 0.5 for k in WRISTS
              if kp is not None and len(kp) > k and kp[k][2] >= WRIST_CONF]
        if ds:
            near.append((min(ds) / diag, pid))
    near.sort()
    if near and near[0][0] <= reach:
        if len(near) > 1 and near[1][0] <= reach and near[1][0] < near[0][0] * 1.5:
            return None
        return near[0][1]
    boxed = [pid for pid, b, _ in people if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]]
    return boxed[0] if len(boxed) == 1 else None


class HolderTracker:
    """전체 카메라에서 선반 자리 **밖**에 보인 상품이 누구 손에 있었는지 적어 둔다 → 선반에서 사라진 상품을 처음 든 사람.

    쓰임 (fusion.overview_owner): 두 사람이 함께 선반을 가려 주인을 못 정한 방문의 주인.
    '들고 있다'는 '이 매장 선반에서 꺼냈다'가 아니다 (들고 들어온 손님, 건네받은 일행) — 그래서
      - 전체 카메라 개수로 **그 종류가 줄어든 방문**에서만 묻고,
      - 방문 전에 이미 그 종류를 들고 있던 사람은 뺀다.
    여러 사람이 들었으면 **먼저 든 사람**이 꺼낸 사람이다 (나중에 든 사람은 건네받았다 — 3차 T09).
    """

    def __init__(self, min_frames: int = HOLD_FRAMES, reach: float = REACH, score: float = SCORE, merge_iou: float = MERGE) -> None:
        self.min_frames, self.reach, self.score, self.merge_iou = min_frames, reach, score, merge_iou
        self.resolve = lambda pid: pid
        """적어 둔 신원 → 지금의 신원 (그 뒤 합쳐진 신원을 한 사람으로 보게). 판정이 손님 장부의 것을 넣는다."""
        self.confused = None
        """(신원, ms) → 그 장면에 전체 카메라가 그 사람을 다른 사람과 헷갈리고 있었나. 판정이 넣는다 (fusion.swap_since_confusion).
        헷갈린 장면의 '든 사람'은 따로 적는다 (unsure) — 두 상자가 겹치면 상품이 누구 손에 있는지도 못 가른다 (4차 C05, 후속 68).
        '방금 들었다'(fresh)에는 안 쓰고, 방문의 주인을 나눌 때 **확실한 장면만으로는 든 사람이 모자란 종류**에만 쓴다 (owners)."""
        self.reset()

    def reset(self) -> None:
        self.samples: list[tuple[float, int, str]] = []
        """(ms, 신원, 종류) — 그 장면에서 그 사람이 그 종류를 들고 있었다."""
        self.unsure: list[tuple[float, int, str]] = []
        """같은 꼴 — 전체 카메라가 그 사람을 다른 사람과 헷갈리던 장면의 것."""
        self._recent: dict[tuple[int, str], deque] = {}
        self._fresh: list[tuple[int, str, float, float]] = []
        self._told: set[tuple[int, str]] = set()

    def update(self, ms: float, items: list, people: list, region) -> None:
        """전체 카메라 한 장면. people = [(신원, 사람 상자, 관절), …]. region = 선반 자리 (아직 없으면 아무것도 안 적는다)."""
        if region is None:
            return
        x1, y1, x2, y2 = region
        for it in merge([i for i in items if i[5] >= self.score], self.merge_iou):
            cx, cy = center(it)
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                continue
            pid = holder_of(it, people, self.reach)
            if pid is not None and self.confused is not None and self.confused(pid, ms):
                self.unsure.append((ms, pid, it[0]))
                continue
            if pid is not None:
                self.samples.append((ms, pid, it[0]))
                recent = self._recent.setdefault((pid, it[0]), deque(maxlen=self.min_frames))
                recent.append(ms)
                if len(recent) == self.min_frames and ms - recent[0] <= HOLD_WITHIN_MS:
                    self._fresh.append((pid, it[0], recent[0], ms))

    def swap(self, a: int, b: int, since_ms: float) -> None:
        """since_ms 부터 적어 둔 '든 사람'에서 신원 a · b 를 서로 바꾼다 — 그때부터 전체 카메라가 두 사람의 번호를 뒤바꿔 붙였다
        (fusion.swap_since_confusion). 아직 안 알린 '방금 든 것'도 바꾼다. 이미 알린 것은 판정이 장부에서 옮긴다."""
        flip = {a: b, b: a}
        self.samples = [(ms, flip.get(pid, pid) if ms >= since_ms else pid, name) for ms, pid, name in self.samples]
        self.unsure = [(ms, flip.get(pid, pid) if ms >= since_ms else pid, name) for ms, pid, name in self.unsure]
        recent: dict[tuple[int, str], deque] = {}
        for ms, pid, name in self.samples:
            recent.setdefault((pid, name), deque(maxlen=self.min_frames)).append(ms)
        self._recent = recent
        self._fresh = [(flip.get(pid, pid) if first >= since_ms else pid, name, first, ms) for pid, name, first, ms in self._fresh]

    def mark_told(self, pid: int, name: str) -> None:
        """그 사람의 그 종류는 이미 '방금 든 것'으로 알렸다 — 다시 알리지 않는다 (번호를 바꿔 읽어 주인이 옮겨진 것)."""
        self._told.add((self.resolve(pid), name))

    def fresh(self) -> list[tuple[int, str, float, float]]:
        """지난번 물은 뒤로 '들었다'고 확정된 것 — (신원, 종류, 손에서 처음 본 ms, 확정한 ms).

        그 사람 손에서 그 종류를 HOLD_WITHIN_MS 안에 min_frames 장면 본 순간 한 번 나온다 ('처음 본 ms'는 그 장면들의 첫 장면 —
        한참 전에 한 장면 스친 것은 안 친다: 3차 N06 은 1.1초에 한 장면, 7.5초부터 이어서 보였다). 같은 사람 · 같은 종류는 다시 안 나온다
        (같은 종류 둘째 것은 여기서 못 센다 — 선반 방문이 끝나면 그 개수로 바뀐다). fusion.overview_instant 가 쓴다."""
        out = []
        for pid, name, first, ms in self._fresh:
            key = (self.resolve(pid), name)
            if key not in self._told:
                self._told.add(key)
                out.append((key[0], name, first, ms))
        self._fresh = []
        return out

    def holds(self, gap_ms: float = HOLD_WITHIN_MS) -> list[tuple[float, float, int, str]]:
        """확정된 '들고 있음' 구간들 — (시작 ms, 끝 ms, 신원, 종류), 시작 순. 확실한 장면만 쓴다 (헷갈린 장면은 뺀다).

        구간 = 그 사람 손에서 그 종류를 본 장면이 gap_ms 넘게 끊기지 않고 이어진 것. 그 안에서 min_frames 장면이 HOLD_WITHIN_MS 안에
        모인 적이 있어야 하고, 시작은 처음 그렇게 모인 장면들의 첫 장면이다 (fresh 와 같은 기준). fresh 와 달리 같은 사람 · 같은 종류도
        끊겼다 다시 들면 또 나온다 — 건넸다 돌려받은 것을 보려고 (fusion.handover)."""
        by: dict[tuple[int, str], list[float]] = {}
        for ms, pid, name in self.samples:
            by.setdefault((self.resolve(pid), name), []).append(ms)
        out = []
        for (pid, name), times in by.items():
            times.sort()
            run: list[float] = []
            for t in times + [None]:
                if t is not None and (not run or t - run[-1] <= gap_ms):
                    run.append(t)
                    continue
                first = next((run[i] for i in range(len(run) - self.min_frames + 1)
                              if run[i + self.min_frames - 1] - run[i] <= HOLD_WITHIN_MS), None)
                if first is not None:
                    out.append((first, run[-1], pid, name))
                run = [t] if t is not None else []
        return sorted(out)

    def count(self, pid: int, name: str, start_ms: float, end_ms: float) -> int:
        """[start_ms, end_ms] 에 그 사람 손에서 그 종류를 본 (확실한) 장면 수."""
        return sum(1 for ms, p, cls in self.samples if cls == name and start_ms <= ms <= end_ms and self.resolve(p) == pid)

    def last_seen(self, pid: int, name: str, before_ms: float) -> float | None:
        """before_ms 전에 그 사람 손에서 그 종류를 마지막으로 본 (확실한) 장면의 시각. 없으면 None."""
        times = [ms for ms, p, cls in self.samples if cls == name and ms < before_ms and self.resolve(p) == pid]
        return max(times) if times else None

    def first_seen(self, pid: int, name: str, start_ms: float, until_ms: float) -> float | None:
        """[start_ms, until_ms] 에 그 사람 손에서 그 종류를 처음 본 (확실한) 장면의 시각. 없으면 None."""
        times = [ms for ms, p, cls in self.samples if cls == name and start_ms <= ms <= until_ms and self.resolve(p) == pid]
        return min(times) if times else None

    def first_holders(self, name: str, start_ms: float, until_ms: float, unsure: bool = False) -> list[int]:
        """[start_ms, until_ms] 에 그 종류를 든 사람들 — 먼저 든 순서. 방문 전에 이미 들고 있던 사람은 뺀다."""
        when = self.first_held(name, start_ms, until_ms, unsure)
        return sorted(when, key=lambda pid: when[pid])

    def first_held(self, name: str, start_ms: float, until_ms: float, unsure: bool = False) -> dict[int, float]:
        """[start_ms, until_ms] 에 그 종류를 든 사람 → 든 것으로 본 시각 (min_frames 번째 장면). 방문 전에 이미 들고 있던 사람은 뺀다.
        unsure = 전체 카메라가 그 사람을 헷갈리던 장면의 것도 넣는다."""
        seen: dict[int, list[float]] = {}
        had: Counter = Counter()
        for ms, pid, cls in (self.samples + self.unsure if unsure else self.samples):
            if cls != name or ms > until_ms:
                continue
            pid = self.resolve(pid)
            if ms < start_ms:
                had[pid] += 1
            else:
                seen.setdefault(pid, []).append(ms)
        return {pid: sorted(times)[self.min_frames - 1] for pid, times in seen.items()
                if len(times) >= self.min_frames and had[pid] < self.min_frames}

    def owners(self, removed_by: dict[str, int], start_ms: float, until_ms: float) -> list[tuple[str, int | None]]:
        """줄어든 상품 하나마다 (종류, 꺼낸 사람 또는 None). 그 종류를 든 사람 가운데 **아직 하나도 못 받은 사람**에게 먼저 든 순서로
        하나씩, 다 받았으면 첫 사람에게. 든 사람이 적은 종류부터 정한다.

        한 사람이 든 상품 하나가 장면마다 다른 종류로 읽힐 수 있다 (4차 C19: 애플사이다 하나를 든 손님이 칸타타 · 동과루로도 읽혀
        '칸타타를 먼저 든 사람'까지 됐다 — 두 사람이 하나씩 꺼냈는데 2개가 한 사람에게 붙었다). 그래서 종류마다 따로 '먼저 든 사람'을
        고르지 않고, 든 사람들에게 하나씩 돌아가게 나눈다. 한 종류만 줄었을 때는 예전과 같다 (먼저 든 사람부터 — 3차 T09)."""
        holders = {name: self.first_holders(name, start_ms, until_ms) for name in removed_by}
        if self.unsure:
            # 확실한 장면에서 든 사람이 줄어든 개수보다 적은 종류만 — 헷갈리던 장면에서 든 사람을 뒤에 붙인다 (4차 L1: 셋째 손님은
            # 다른 손님과 겹친 채로만 들고 있었다). 모자라지 않으면 안 쓴다 (4차 C05: 앞에 선 손님 손목 곁에 뒷사람의 상품이 보였다)
            for name, n in removed_by.items():
                if n > len(holders[name]):
                    holders[name] += [p for p in self.first_holders(name, start_ms, until_ms, True) if p not in holders[name]]
        given: Counter = Counter()
        picked: dict[str, list[int | None]] = {}
        for name in sorted(removed_by, key=lambda n: (len(holders[n]), n)):
            who = holders[name]
            for _ in range(removed_by[name]):
                if not who:
                    picked.setdefault(name, []).append(None)
                    continue
                pid = next((p for p in who if not given[p]), who[0])
                given[pid] += 1
                picked.setdefault(name, []).append(pid)
        return [(name, pid) for name in sorted(picked) for pid in picked[name]]
