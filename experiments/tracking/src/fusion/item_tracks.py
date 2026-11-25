"""물건 단위 추적 — 전체 카메라의 상품 상자를 장면마다 이어 **물건 하나**로 따라간다 (fusion.item_tracks, 후속 73).

기준은 하나다: **물건의 시작과 끝은 선반이다.**

  - 선반 자리에 머물던 상자가 밖으로 나와 손에 들리면 = 집음 (그 손님의 물건 +1)
  - 든 물건의 상자가 선반 자리로 들어가 머물면              = 놓음 (−1)
  - 선반 밖에서 처음 나타난 상자는 새 물건이 아니다. 그 손님이 들고 있다가 안 보이던 물건이 다시 보인 것이거나
    (개수 그대로), 상품이 아니다 (유리의 스티커 · 손님 소지품 · 헛것 — 안 센다)

개수와 종류를 가른다 (사용자 제안): 개수는 물건 수이고, 종류는 그 물건이 **선반 밖에서** 가장 많이 읽힌 것이다.
한 물건이 잠깐 다른 종류로 읽혀도 개수는 안 바뀐다. (선반 안에서 읽힌 종류는 안 쓴다 — 물건이 빽빽해 이웃 칸의 상자와 이어지기도 한다.)

HolderTracker('그 손에 그 종류가 보였나', 사람 · 종류 단위)와 다른 점: 자리(카메라 좌표)나 '선반 앞에 있던 사람'으로 거르지 않는다.
상자 자신이 선반에서 나왔는지만 본다 — 4차 문 옆 유리의 스티커(애플사이다로 읽힘)는 선반 앞에 있다 지나가는 손님 손에 붙어 세어졌다.

못 보는 것: 선반에서 나오는 순간을 못 본 물건 (몸에 가림 · 손 안에 쥠). 그것은 여기서 안 세고 선반 방문(센서 + 선반 지도)이 센다.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

from .overview_count import HOLD_FRAMES, HOLD_WITHIN_MS, REACH, SCORE, center, holder_of, iou

SAME_BOX = 0.2
"""종류와 상관없이 이만큼 겹친 상자는 한 물건이다 (한 물건에 두 종류의 상자가 겹쳐 나온다)."""
LINK_GAP_MS = 600.0
"""이만큼 안 보인 줄은 끊는다 (그 뒤에 보인 상자는 새 줄)."""
LINK_STEP_MS = 150.0
LINK_JUMP = 2.5
"""이을 수 있는 거리 = 상자 대각선 × min(1 + 안 보인 시간 / LINK_STEP_MS, LINK_JUMP).
한계가 없으면 선반 위 상자가 잠깐 가려진 사이 멀리(계산대)의 상자와 이어져 '선반에서 나온 물건'이 된다 (4차 T06 · C14)."""
NEAR = 1.5
"""선반 자리에서 상자 대각선의 이 배수 안에서 처음 나타난 상자는 '선반 바로 옆'이다 — 나오는 순간 손에 가려 줄이 끊긴 것으로 본다."""


def _diag(box: list) -> float:
    return ((box[3] - box[1]) ** 2 + (box[4] - box[2]) ** 2) ** 0.5 or 1.0


def _gap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


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
    """선반 자리 안에서 본 최근 장면들."""
    item: "Item | None" = None
    dead: bool = False
    """상품이 아닌 것으로 정했다 (선반에서 안 나왔고, 다시 보인 물건도 아니다)."""


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
        self.items: list[Item] = []
        self._events: list[ItemEvent] = []

    # ------------------------------------------------------------------ 한 장면

    def update(self, ms: float, items: list, people: list, region) -> None:
        """전체 카메라 한 장면. items = [[종류, x1, y1, x2, y2, 점수], …], people = [(신원, 사람 상자, 관절), …], region = 선반 자리."""
        if region is None:
            return
        boxes: list = []
        for it in sorted((i for i in items if i[5] >= self.score), key=lambda i: -i[5]):
            if all(iou(k, it) < SAME_BOX for k in boxes):
                boxes.append(it)
        self.live = [t for t in self.live if ms - t.last_ms <= LINK_GAP_MS]
        pairs = []
        for bi, it in enumerate(boxes):
            for ti, t in enumerate(self.live):
                allow = max(_diag(it), _diag(t.last)) * min(1.0 + (ms - t.last_ms) / LINK_STEP_MS, LINK_JUMP)
                d = _gap(center(it), center(t.last))
                if d <= allow:
                    pairs.append((d / allow, bi, ti))
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

    def _see(self, t: Track, ms: float, it: list, people: list, region) -> None:
        t.last, t.last_ms = it, ms
        if shelf_distance(region, center(it)) == 0:
            t.inside.append(ms)
            if len(t.inside) == self.min_frames and ms - t.inside[0] <= HOLD_WITHIN_MS:
                # 선반 자리에 머문다 = 선반 위의 물건이다. 들려 있던 물건이면 놓은 것이다
                if t.item is not None and t.item.returned_ms is None:
                    t.item.returned_ms = ms
                    self._events.append(ItemEvent("back", t.item.owner, t.item.name, t.item.seen_ms, ms, t.item.id))
                t.item, t.dead, t.origin = None, False, "shelf"
                t.held.clear()
                t.reads.clear()
            return
        raw = holder_of(it, people, self.reach)
        if raw is None or (self.confused is not None and self.confused(raw, ms)):
            return
        pid = self.resolve(raw)
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
            # 선반에 있는 것을 못 본 상자가 그 손님의 든 물건 바로 옆에 있다 — 그 물건에 상자가 하나 더 붙은 것으로 보고 버린다
            # (4차 N02: 캔 하나에 상자 둘이 6장면). 종류도 안 섞는다 — 진짜 둘째 물건이면 종류가 다를 수 있다
            for i in mine:
                if i.track is not t and ms - i.track.last_ms <= 100.0 and \
                        _gap(center(t.last), center(i.track.last)) <= max(_diag(t.last), _diag(i.track.last)):
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
