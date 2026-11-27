"""물건 단위 추적 — 전체 카메라의 상품 상자를 장면마다 이어 **물건 하나**로 따라간다 (fusion.item_tracks, 후속 73).

기준은 하나다: **물건의 시작과 끝은 선반이다.**

  - 선반 자리에 머물던 상자가 밖으로 나와 손에 들리면 = 집음 (그 손님의 물건 +1)
  - 든 물건의 상자가 선반 자리로 들어가 멈춰 머물고 손이 떠나면 = 놓음 (−1)
  - 선반 밖에서 처음 나타난 상자는 새 물건이 아니다. 그 손님이 들고 있다가 안 보이던 물건이 다시 보인 것이거나
    (개수 그대로), 상품이 아니다 (유리의 스티커 · 손님 소지품 · 헛것 — 안 센다)

개수와 종류를 가른다 (사용자 제안): 개수는 물건 수이고, 종류는 그 물건이 **선반 밖에서** 가장 많이 읽힌 것이다.
한 물건이 잠깐 다른 종류로 읽혀도 개수는 안 바뀐다. (선반 안에서 읽힌 종류는 안 쓴다 — 물건이 빽빽해 이웃 칸의 상자와 이어지기도 한다.)

HolderTracker('그 손에 그 종류가 보였나', 사람 · 종류 단위)와 다른 점: 자리(카메라 좌표)나 '선반 앞에 있던 사람'으로 거르지 않는다.
상자 자신이 선반에서 나왔는지만 본다 — 4차 문 옆 유리의 스티커(애플사이다로 읽힘)는 선반 앞에 있다 지나가는 손님 손에 붙어 세어졌다.

못 보는 것: 선반에서 나오는 순간을 못 본 물건 (몸에 가림 · 손 안에 쥠). 그것은 여기서 안 세고 선반 방문(센서 + 선반 지도)이 센다.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field

from .overview_count import HOLD_FRAMES, HOLD_WITHIN_MS, REACH, SCORE, WRIST_CONF, WRISTS, center, holder_of, iou

SAME_BOX = 0.2
"""종류와 상관없이 이만큼 겹친 상자는 한 물건이다 (한 물건에 두 종류의 상자가 겹쳐 나온다)."""
LINK_GAP_MS = 600.0
"""이만큼 안 보인 추적은 끊는다 (그 뒤에 보인 상자는 새 추적) — **시작을 아는 추적 둘은 빼고** (후속 81, 사용자 지적:
'추적이 시작을 안 따진다 — 놓는다는 것은 들고 있던 물건을 놓는 것인데, 그 물건을 따라가다 선반 위의 다른 물건을 따라가게 된다').

  - 진열 상자 (진열 자리에 멈춰 있던 추적): 그 자리가 **가려져 있는 동안**은 안 끊는다. 가린 것(사람 · 다른 상자)이 없는데
    이만큼 안 보이면 없는 것이다. 오래 안 보인 뒤에는 **제 자리의 상자**(마지막 상자와 SAME_BOX 이상 겹침)만 잇는다
  - 든 물건 (집어서 아직 안 놓은 물건의 추적): 안 끊는다. 오래 안 보인 뒤에는 **그 손님 손에 든, 선반 밖의 상자**만 잇는다
    (놓으려고 진열 자리에 멈춘 채 가려진 것은 그 자리의 상자도 — 후속 83)

전에는 둘 다 이만큼 지나면 끊었다. 든 물건이 손에 가려 0.6초 넘게 안 보이다 진열 상자 옆에서 다시 보이면, 제 추적은 없고
진열 상자의 추적만 살아 있어 그 추적이 든 물건을 가져갔다 → '선반에서 나온 새 물건' = 틀린 집음 (4차 N11 전체 카메라 11.88초).
둘은 함께 넣어야 한다 — 하나만 넣으면 남은 쪽이 상자를 가져간다 (진열 상자만: 4차 N05 를 하나 더 셈, 든 물건만: 3차 C04 틀린 놓음 +1)."""
LINK_STEP_MS = 150.0
LINK_JUMP = 2.5
"""추적을 상자에 이을 수 있는 한계 = min(1 + 안 보인 시간 / LINK_STEP_MS, LINK_JUMP). 이것과 견주는 값은 '달라진 정도'(changed)다:
가운데 점이 옮겨 간 거리 ÷ 상자 대각선 + 크기가 바뀐 정도(대각선 비의 로그). 크기가 그대로면 전처럼 '상자 대각선의 몇 배까지'다.
한계가 없으면 선반 위 상자가 잠깐 가려진 사이 멀리(계산대)의 상자와 이어져 '선반에서 나온 물건'이 된다 (4차 T06 · C14).

후속 80 — 전에는 거리만 봤다 (가장 가까운 상자에 잇는다). 든 물건이 진열 상자 앞을 지나가면 두 추적이 서로 바뀌었다
(3차 N06 선반 카메라 9.80초: 왼쪽으로 가던 든 물건(대각선 87)의 상자 하나만 보였는데, 9픽셀 떨어진 진열 상자(63)의 추적이
18픽셀 떨어진 든 물건의 추적보다 가까워 그 상자를 가져갔다 → 든 물건의 추적은 진열 상자에 붙어 '놓음', 진열 상자의 추적은
손에 들려 '집음'). 그래서 둘을 더 본다:
  - 크기: 같은 물건은 한 장면 사이에 크기가 갑자기 안 바뀐다. 크기가 달라진 만큼 '달라진 정도'가 커진다
  - 움직임: 움직이던 추적은 그 방향으로 한 걸음 더 간 자리(ahead)에서 잰다. 순서(어느 추적이 그 상자를 갖나)에만 쓰고,
    이을 수 있는지는 마지막으로 본 자리에서 잰다 — 예측으로 닿는 거리를 늘리지 않는다
맞춘 값은 없다 (거리와 크기 변화를 그대로 더한다 — 상자 하나만큼 옮겨 간 것과 크기가 e배 달라진 것을 같게 친다)."""
ON_PLACE = 0.2
"""상자가 선반 위 물건 자리(places)와 이만큼 겹치면 그 자리에 있는 것이다. 0.35 · 0.5 로 올리면 선반 위 상자가 조금 흔들릴 때마다
'선반 밖'이 되어 많이 센다 (4차 손님 한 명 21명: 많이 센 사람 0 → 2 → 4명, 3차 19명: 0 → 2 → 6명)."""
NEAR = 1.5
"""선반 자리에서 상자 대각선의 이 배수 안에서 처음 나타난 상자는 '선반 바로 옆'이다 — 나오는 순간 손에 가려 줄이 끊긴 것으로 본다."""
IN_PLACES = 0.9
"""상자의 넓이 가운데 이만큼이 진열 자리들 안에 있으면 '선반 밖에서 본 장면'으로 안 친다 (ON_PLACE 에 못 미쳐도).
작은 조각은 자리 안에 다 들어 있어도 겹침(IoU)이 작다 — 겹침은 합친 넓이로 나누기 때문이다 (3차 N11: 겹침 0.08 ~ 0.16, 자리 안 0.83 ~ 0.93).
NESTED 와 같은 뜻의 값이다 ('거의 다 안에 있다'). 0.5 로 낮추면 선반 앞에서 꺼내는 중인 진짜 물건의 장면까지 빠져 늦게 센다
(3차 손님 한 명 19편: 정답보다 적게 안 시간 37.0 → 39.5초, 많이 안 시간은 29.9 → 29.7초로 거의 같다)."""
NESTED = 0.9
"""두 상자 가운데 작은 쪽이 큰 쪽 안에 이만큼 들어 있으면 한 물건이다 — 한 물건을 손 · 팔까지 넓게 두른 상자 (4차 N02: 품는 비율 1.0, 넓이 4.5배).
나란히 든 두 물건은 서로 품지 않는다 (4차 손님 한 명 21편의 진짜 둘째 물건 7번: 0 ~ 0.35, 먼 선반 카메라의 한 번만 0.83).
맞춰 고른 값이 아니다 — '거의 다 품는다'의 뜻이고 가짜(1.0)와 진짜의 가장 큰 값(0.83) 사이다."""


def _diag(box: list) -> float:
    return ((box[3] - box[1]) ** 2 + (box[4] - box[2]) ** 2) ** 0.5 or 1.0


def _gap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _minmod(a: float, b: float) -> float:
    """두 값이 같은 쪽이면 작은 쪽, 아니면 0."""
    return 0.0 if a * b <= 0 else (min(a, b) if a > 0 else max(a, b))


def changed(last: list, box: list, at: tuple[float, float]) -> float:
    """추적의 마지막 상자(last)가 이 상자(box)가 되려면 달라져야 하는 정도 — at(추적이 있을 자리)에서 상자 가운데까지의 거리 ÷
    큰 쪽 대각선 + 크기가 바뀐 정도(대각선 비의 로그)."""
    a, b = _diag(last), _diag(box)
    return _gap(center(box), at) / max(a, b) + abs(math.log(b / a))


def nested(a: list, b: list) -> float:
    """두 상자 가운데 작은 쪽이 큰 쪽 안에 들어 있는 비율 (겹친 넓이 / 작은 쪽 넓이)."""
    w = min(a[3], b[3]) - max(a[1], b[1])
    h = min(a[4], b[4]) - max(a[2], b[2])
    if w <= 0 or h <= 0:
        return 0.0
    small = min((a[3] - a[1]) * (a[4] - a[2]), (b[3] - b[1]) * (b[4] - b[2]))
    return w * h / small if small > 0 else 0.0


def covered(box: list, places: list) -> float:
    """상자의 넓이 가운데 진열 자리들(합친 것)이 덮은 비율."""
    cut = [(max(box[1], p[1]), max(box[2], p[2]), min(box[3], p[3]), min(box[4], p[4])) for p in places]
    cut = [c for c in cut if c[2] > c[0] and c[3] > c[1]]
    area = (box[3] - box[1]) * (box[4] - box[2])
    if not cut or area <= 0:
        return 0.0
    xs, ys = sorted({v for c in cut for v in (c[0], c[2])}), sorted({v for c in cut for v in (c[1], c[3])})
    got = 0.0
    for x1, x2 in zip(xs, xs[1:]):
        for y1, y2 in zip(ys, ys[1:]):
            if any(c[0] <= x1 and x2 <= c[2] and c[1] <= y1 and y2 <= c[3] for c in cut):
                got += (x2 - x1) * (y2 - y1)
    return got / area


def shelf_distance(region, point: tuple[float, float]) -> float:
    """선반 자리(네모)까지의 거리. 안이면 0."""
    x1, y1, x2, y2 = region
    dx, dy = max(x1 - point[0], 0.0, point[0] - x2), max(y1 - point[1], 0.0, point[1] - y2)
    return (dx * dx + dy * dy) ** 0.5


@dataclass
class Track:
    """이어 본 상자 한 줄."""

    first_ms: float
    origin: str
    """처음 나타난 자리: shelf(선반 자리에 머묾) · near(선반 바로 옆) · away(선반 밖)."""
    last: list
    last_ms: float
    reads: Counter = field(default_factory=Counter)
    """선반 밖에서 누군가의 손에 있을 때 읽힌 종류."""
    held: dict[int, deque] = field(default_factory=dict)
    """신원 → 그 손에서 본 최근 장면들 (물건으로 정해지기 전까지)."""
    inside: deque = field(default_factory=lambda: deque(maxlen=HOLD_FRAMES))
    """선반 자리 안에서 **잇달아** 본 최근 장면들 (때, 상자). 선반 밖에서 한 번 보이면 비운다."""
    item: "Item | None" = None
    step: tuple = (0.0, 0.0)
    """바로 앞 걸음의 빠르기 (픽셀 / ms). 잇달아 본 장면(LINK_STEP_MS 안) 사이에서만 잰다."""
    vel: tuple = (0.0, 0.0)
    """믿는 빠르기 = 지난 두 걸음이 같은 쪽일 때 그 가운데 작은 쪽 (minmod), 아니면 0. 한 걸음만 튄 것(가만히 있던 진열 상자의
    추적이 옆 상자로 한 번 옮겨 붙음 · 겹친 두 상자를 번갈아 읽음)은 움직임으로 안 친다."""
    dead: bool = False
    """상품이 아닌 것으로 정했다 (선반에서 안 나왔고, 다시 보인 물건도 아니다)."""
    resting: bool = False
    """진열 자리에 멈춰 머문 뒤로 선반 밖에서 본 적이 없다 = 지금 선반 위의 물건이다."""
    kept_ms: float = 0.0
    """안 보이는 동안 그 자리가 가려져 있던 마지막 때."""
    waiting: bool = False
    """든 물건이 진열 자리에 멈췄는데 아직 손이 닿아 있다 — 손이 떠나면 놓음이 된다 (다시 선반 밖에서 보이면 없던 일)."""

    def move(self, ms: float, box: list) -> None:
        """새 장면의 상자로 옮긴다."""
        dt, before = ms - self.last_ms, self.step
        if 0 < dt <= LINK_STEP_MS:
            a, b = center(self.last), center(box)
            self.step = ((b[0] - a[0]) / dt, (b[1] - a[1]) / dt)
        else:
            self.step = (0.0, 0.0)
        self.vel = (_minmod(before[0], self.step[0]), _minmod(before[1], self.step[1]))
        self.last, self.last_ms = box, ms

    def ahead(self, ms: float) -> tuple[float, float]:
        """움직이던 대로 갔다면 ms 에 있을 자리 — 한 걸음(LINK_STEP_MS)까지만 내다본다. 그 뒤는 모른다 (손은 멈추기도 한다)."""
        x, y = center(self.last)
        step = min(ms - self.last_ms, LINK_STEP_MS)
        return x + self.vel[0] * step, y + self.vel[1] * step


@dataclass
class Item:
    """선반에서 나온 물건 하나."""

    id: int
    owner: int
    """처음 든 사람 (그 뒤 합쳐진 신원은 ItemTracker.resolve 로 읽는다)."""
    seen_ms: float
    """그 손에서 처음 본 때."""
    ms: float
    """집은 것으로 정한 때."""
    reads: Counter
    track: Track
    origin: str
    name: str = ""
    """지금까지 가장 많이 읽힌 종류."""
    returned_ms: float | None = None


@dataclass(frozen=True)
class ItemEvent:
    """kind = pick(집음) · back(놓음) · kind(종류가 바뀌어 읽힘) · reject(상품 아님 — 안 셈)."""

    kind: str
    pid: int
    name: str
    seen_ms: float
    ms: float
    item: int | None = None
    why: str = ""


class ItemTracker:
    def __init__(self, min_frames: int = HOLD_FRAMES, reach: float = REACH, score: float = SCORE, near: float = NEAR) -> None:
        self.min_frames, self.reach, self.score, self.near = min_frames, reach, score, near
        self.resolve = lambda pid: pid
        """적어 둔 신원 → 지금의 신원. 판정이 손님 장부의 것을 넣는다."""
        self.confused = None
        """(신원, ms) → 그 장면에 전체 카메라가 그 사람을 다른 사람과 헷갈리고 있었나. 헷갈린 장면은 '든 장면'으로 안 친다."""
        self.reset()

    def reset(self) -> None:
        self.live: list[Track] = []
        self._places: list | None = None
        self.items: list[Item] = []
        self._events: list[ItemEvent] = []

    # ------------------------------------------------------------------ 한 장면

    def update(self, ms: float, items: list, people: list, region, places: list | None = None) -> None:
        """카메라 한 장면. items = [[종류, x1, y1, x2, y2, 점수], …], people = [(신원, 사람 상자, 관절), …], region = 선반 자리,
        places = 선반 위 물건이 놓여 있던 자리들 (overview_count.places_of). 주면 '선반 위에 있다'를 그 자리와 겹치는지로 따진다 —
        선반 자리(네모 + 여유)의 빈 곳에 떠 있는 손 안의 물건을 선반 위의 물건으로 보지 않는다. 안 주면 선반 자리 안인지로 따진다."""
        if region is None:
            return
        self._places = places
        boxes: list = []
        for it in sorted((i for i in items if i[5] >= self.score), key=lambda i: -i[5]):
            if all(iou(k, it) < SAME_BOX for k in boxes):
                boxes.append(it)
        self.live = [t for t in self.live if self._alive(t, ms, boxes, people)]
        pairs = []
        for bi, it in enumerate(boxes):
            for ti, t in enumerate(self.live):
                if self._may_link(t, it, ms, people, region):
                    pairs.append((changed(t.last, it, t.ahead(ms)), bi, ti))
        used_box: set[int] = set()
        used_track: set[int] = set()
        seen: list[tuple[Track, list]] = []
        for _, bi, ti in sorted(pairs):
            if bi in used_box or ti in used_track:
                continue
            used_box.add(bi)
            used_track.add(ti)
            seen.append((self.live[ti], boxes[bi]))
        for bi, it in enumerate(boxes):
            if bi in used_box:
                continue
            away = shelf_distance(region, center(it)) > self.near * _diag(it)
            t = Track(first_ms=ms, origin="away" if away else "near", last=it, last_ms=ms)
            self.live.append(t)
            seen.append((t, it))
        for t, it in seen:
            self._see(t, ms, it, people, region)

    def _alive(self, t: Track, ms: float, boxes: list, people: list) -> bool:
        """이 추적을 더 들고 있나. 시작을 아는 추적(진열 상자 · 든 물건)은 안 보여도 들고 있는다 (LINK_GAP_MS 의 설명)."""
        if ms - t.last_ms <= LINK_GAP_MS:
            return True
        if t.item is not None:
            return t.item.track is t and t.item.returned_ms is None
        if not t.resting:
            return False
        x, y = center(t.last)
        if (any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for _, b, _ in people)
                or any(iou(it, t.last) >= SAME_BOX for it in boxes)):
            t.kept_ms = ms   # 사람이나 다른 상자가 그 자리를 가리고 있다
        return ms - max(t.last_ms, t.kept_ms) <= LINK_GAP_MS

    def _may_link(self, t: Track, it: list, ms: float, people: list, region) -> bool:
        """이 추적을 이 상자에 이을 수 있나. 오래 안 보인 추적은 제 시작에 맞는 상자만 잇는다."""
        if changed(t.last, it, center(t.last)) > min(1.0 + (ms - t.last_ms) / LINK_STEP_MS, LINK_JUMP):
            return False
        if ms - t.last_ms <= LINK_GAP_MS:
            return True
        if t.item is not None:
            if t.waiting and iou(it, t.last) >= SAME_BOX:
                return True   # 놓으려고 멈춘 채 가려졌다 — 그 자리의 상자도 잇는다 (안 이으면 손이 떠난 뒤의 놓음을 못 읽는다)
            return self._spot(it, region) == "off" and self._hand(it, ms, people) == self.resolve(t.item.owner)
        return iou(it, t.last) >= SAME_BOX

    def _touched(self, it: list, people: list) -> bool:
        """이 상자에 누군가의 손이 닿아 있나. '든 사람'(holder_of)과 같은 순서로 본다 — 손목이 보이는 사람은 손목이 닿는 거리
        (reach) 안인지로, 손목이 안 보이는 사람은 상자 가운데가 그 사람 상자 안인지로. 누구 손인지는 안 따진다 (둘이 쥐고 있어도 닿은 것).
        손목이 보이는데 '사람 상자 안'까지 닿은 것으로 치면, 손님이 선반 앞에 서 있는 내내 놓음을 못 읽는다 (2 ~ 4초 늦다 — 3차 N11)."""
        cx, cy = center(it)
        reach = self.reach * _diag(it)
        for _, b, kp in people:
            wrists = [kp[k] for k in WRISTS if kp is not None and len(kp) > k and kp[k][2] >= WRIST_CONF]
            if any(_gap((w[0], w[1]), (cx, cy)) <= reach for w in wrists):
                return True
            if not wrists and b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
                return True
        return False

    def _spot(self, it: list, region) -> str:
        """상자가 있는 곳 — shelf(진열 자리 위) · edge(자리들 안에 든 조각) · off(선반 밖)."""
        if not self._places:
            return "shelf" if shelf_distance(region, center(it)) == 0 else "off"
        if any(iou(it, p) >= ON_PLACE for p in self._places):
            return "shelf"
        return "edge" if covered(it, self._places) >= IN_PLACES else "off"

    def _hand(self, it: list, ms: float, people: list) -> int | None:
        """이 상자를 든 손님 (지금의 신원). 못 정하거나 그 사람을 헷갈리고 있는 장면이면 None."""
        raw = holder_of(it, people, self.reach)
        if raw is None or (self.confused is not None and self.confused(raw, ms)):
            return None
        return self.resolve(raw)

    def _see(self, t: Track, ms: float, it: list, people: list, region) -> None:
        t.move(ms, it)
        spot = self._spot(it, region)
        if spot == "shelf":
            t.inside.append((ms, it))
            if (len(t.inside) == self.min_frames and ms - t.inside[0][0] <= HOLD_WITHIN_MS
                    and all(iou(b, it) >= SAME_BOX for _, b in t.inside)):
                # 선반 자리에 **멈춰** 머문다 = 선반 위의 물건이다. 들려 있던 물건이면 놓은 것이다.
                # 멈춰 있다 = 그 다섯 장면의 상자가 지금 상자와 한 자리다 (SAME_BOX 이상 겹침). 진열 자리와 겹친 채 지나가는 든 물건은
                # 선반 위의 물건이 아니다 — 선반 카메라에서는 든 물건이 늘 진열 상자 앞에 있어 자리와 겹친다 (후속 80)
                if t.item is not None and t.item.returned_ms is None:
                    if self._touched(it, people):
                        # 멈췄지만 아직 손에 쥐고 있다 — 놓음은 **손이 떠난 때**다 (후속 83). 진열 상자 앞에 든 채 멈춘 것 ·
                        # 건네느라 둘이 쥐고 있는 것은 놓은 것이 아니다 (3차 T09 · 4차 N06). 자리에 닿은 때를 놓은 때로 읽으면
                        # 0.4 ~ 1.2초 이르다 (4차 N06: 22.95초 → 손이 떠난 23.77초, 정답 23.76초)
                        t.waiting = True
                        return
                    t.item.returned_ms = ms
                    self._events.append(ItemEvent("back", t.item.owner, t.item.name, t.item.seen_ms, ms, t.item.id))
                t.item, t.dead, t.origin, t.resting, t.waiting = None, False, "shelf", True, False
                t.held.clear()
                t.reads.clear()
            return
        if spot == "edge":
            # 진열 자리와 겹침은 작은데 상자가 자리들 안에 들어 있다 — 가려진 진열 상자의 드러난 일부일 수 있다.
            # 든 장면으로도, 선반에 머문 장면으로도 안 친다 (3차 N11: 몸 뒤에서 드러난 상자 끝을 새 물건으로 셈, 후속 78)
            return
        # 선반 밖에서 봤다 — '선반 자리에 머문 장면'은 여기서부터 다시 모은다. 안 비우면 집기 전에 선반에 있던 장면이 남아,
        # 집은 뒤 한 장면만 진열 자리와 겹쳐도 '놓음'이 된다 (4차 N05: 집음 6.58초 → 놓음 6.64초, 후속 76)
        t.inside.clear()
        t.resting = t.waiting = False
        pid = self._hand(it, ms, people)
        if pid is None:
            return
        if t.item is not None:
            self._read(t.item, it[0], ms)
            return
        t.reads[it[0]] += 1
        if t.dead:
            return
        times = t.held.setdefault(pid, deque(maxlen=self.min_frames))
        times.append(ms)
        if len(times) == self.min_frames and ms - times[0] <= HOLD_WITHIN_MS:
            self._decide(t, pid, times[0], ms)

    def _decide(self, t: Track, pid: int, seen_ms: float, ms: float) -> None:
        """그 손에 든 것으로 본 줄 — 새 물건인가, 들고 있던 물건이 다시 보인 것인가, 상품이 아닌가."""
        mine = [i for i in self.items if self.resolve(i.owner) == pid and i.returned_ms is None]
        if t.origin != "shelf":
            # 선반에 있는 것을 못 본 상자가 그 손님의 든 물건에 하나 더 붙은 상자인가 (4차 N02: 캔 하나에 상자 둘이 6장면) — 그러면 버린다.
            # 종류는 안 본다 — 한 물건에 두 종류의 상자가 나오기도 한다.
            # 후속 77 — 선반 바로 옆에서 나타난 상자(near)는 든 물건의 상자와 서로 품을 때만 한 물건이다. 옆에 나란히 있으면 둘째 물건이다
            #   ('바로 옆이면 한 물건'으로 보던 때는 나란히 든 둘째 물건을 버렸다 — 4차 21편에서 가짜 1번 · 진짜 7번).
            # 선반 밖에서 나타난 상자(away)는 어차피 새 물건이 못 된다 — 든 물건 바로 옆이면 전처럼 조용히 버린다 ('안 셈' 줄을 안 늘린다)
            for i in mine:
                if i.track is t or ms - i.track.last_ms > 100.0:
                    continue
                beside = _gap(center(t.last), center(i.track.last)) <= max(_diag(t.last), _diag(i.track.last))
                if nested(t.last, i.track.last) >= NESTED or (t.origin == "away" and beside):
                    t.dead = True
                    return
        hidden = [i for i in mine if i.track.last_ms < t.first_ms]   # 이 줄이 나타나기 전에 안 보이게 된 물건
        if t.origin == "shelf" or (t.origin == "near" and not hidden):
            item = Item(len(self.items), pid, seen_ms, ms, Counter(t.reads), t, t.origin)
            item.name = item.reads.most_common(1)[0][0]
            self.items.append(item)
            t.item = item
            self._events.append(ItemEvent("pick", pid, item.name, seen_ms, ms, item.id))
        elif hidden:
            name = t.reads.most_common(1)[0][0]
            same = [i for i in hidden if i.name == name]
            if not same:
                t.dead = True   # 들고 있던 물건과 종류가 다르다 — 잇지 않는다 (종류를 안 섞는다). 개수는 그대로다
                return
            item = max(same, key=lambda i: i.track.last_ms)   # 가장 늦게까지 보이던 것이 다시 보였다
            item.track.item, item.track.dead = None, True   # 한 물건을 두 추적이 갖지 않는다 — 옛 추적은 물건을 내려놓는다
            item.track, t.item = t, item
            for name, n in t.reads.items():
                item.reads[name] += n
            self._rename(item, ms)
        else:
            t.dead = True
            self._events.append(ItemEvent("reject", pid, t.reads.most_common(1)[0][0], seen_ms, ms,
                                          why="선반에서 나온 상자가 아님"))

    def _read(self, item: Item, name: str, ms: float) -> None:
        item.reads[name] += 1
        self._rename(item, ms)

    def _rename(self, item: Item, ms: float) -> None:
        best, n = item.reads.most_common(1)[0]
        if best != item.name and n > item.reads[item.name]:
            item.name = best
            if item.returned_ms is None:
                self._events.append(ItemEvent("kind", item.owner, best, item.seen_ms, ms, item.id))

    # ------------------------------------------------------------------ 묻기

    def fresh(self) -> list[ItemEvent]:
        """지난번 물은 뒤로 생긴 일 (시간순)."""
        out, self._events = self._events, []
        return [ItemEvent(e.kind, self.resolve(e.pid), e.name, e.seen_ms, e.ms, e.item, e.why) for e in out]

    def carried(self, pid: int) -> list[Item]:
        """그 손님이 지금 가진 (집고 안 놓은) 물건들."""
        return [i for i in self.items if self.resolve(i.owner) == pid and i.returned_ms is None]

    def swap(self, a: int, b: int, since_ms: float) -> None:
        """since_ms 부터 든 물건의 주인을 신원 a · b 사이에서 서로 바꾼다 — 그때부터 전체 카메라가 두 사람의 번호를 뒤바꿔 붙였다."""
        flip = {a: b, b: a}
        for i in self.items:
            if i.seen_ms >= since_ms and i.owner in flip:
                i.owner = flip[i.owner]
        for t in self.live:
            if t.item is None and t.held:
                t.held = {flip.get(pid, pid) if times and times[0] >= since_ms else pid: times for pid, times in t.held.items()}
        self._events = [ItemEvent(e.kind, flip.get(e.pid, e.pid) if e.seen_ms >= since_ms else e.pid, e.name, e.seen_ms, e.ms, e.item, e.why)
                        for e in self._events]
