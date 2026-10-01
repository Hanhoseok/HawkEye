"""선반 지도 — 선반에 무엇이 어디 있는지 기억했다가, 사람이 다녀간 뒤 어느 자리가 비었는지 찾는다.

    선반 앞에 아무도 없음   -> 약 1초 동안 본 것을 모아 '선반 지도'를 확정한다
                             (예: 1단 1번째 병, 1단 2번째 스프레이, 1단 3번째 펌프병)
    손님 3 이 선반을 가림   -> 지도 갱신을 멈추고, 가린 사람을 기록한다
    손님 3 이 떠남          -> 다시 1초 동안 보고 새 지도를 만든다
    두 지도를 자리로 맞춤   -> "1단 2번째 스프레이가 사라짐 -> 손님 3"

같은 종류가 여러 개여도 '자리(위치)'로 구별한다. 종류 이름은 짝을 맞출 때 보조로만 쓴다
(지금은 일반 YOLO 라 이름이 자주 틀린다 — 큰 통을 '변기'로 보기도 했다).

이 모듈은 탐지기도 추적기도 모른다. 물건 상자 목록과 사람(번호, 상자)만 받는다.
형식(ShelfItem, ShelfChange)은 실험 단계라 core/types.py 가 아니라 여기 둔다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ShelfItem:
    label: str
    bbox: tuple[float, float, float, float]
    score: float = 1.0

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def size(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(1.0, ((x2 - x1) * (y2 - y1)) ** 0.5)


@dataclass(frozen=True)
class Slot:
    """선반 지도 위의 한 자리."""

    item: ShelfItem
    row: int          # 위에서부터 1단, 2단 ...
    index: int        # 그 단에서 왼쪽부터 1번째, 2번째 ...

    @property
    def name(self) -> str:
        return f"{self.row}단 {self.index}번째 {self.item.label}"


@dataclass
class ShelfChange:
    """선반 지도가 바뀌었다. 무엇이 어디서 사라졌고/생겼는지, 그때 누가 선반에 있었는지."""

    shelf: str
    removed: list[Slot]
    added: list[Slot]
    start_ms: float
    """선반이 가려지기 시작한 시각 (가림이 없었으면 이전 지도를 확정한 시각)."""
    end_ms: float
    """새 지도를 확정한 시각."""
    people: list[int] = field(default_factory=list)
    """그 사이 선반을 가린 사람들."""

    @property
    def person_id(self) -> int | None:
        """누구 것인지 정할 수 있으면 그 사람. 아무도 없었거나 여럿이면 None."""
        return self.people[0] if len(self.people) == 1 else None

    def describe(self) -> str:
        who = (f"손님 {self.person_id}" if self.person_id is not None
               else "누군지 모름 (사람이 안 보였음)" if not self.people
               else f"정하지 못함 (손님 {', '.join(map(str, self.people))} 이 함께 있었음)")
        parts = []
        if self.removed:
            parts.append("사라짐: " + ", ".join(s.name for s in self.removed))
        if self.added:
            parts.append("생김: " + ", ".join(s.name for s in self.added))
        return f"[{self.shelf}] {' / '.join(parts)} -> {who}"


def layout(items: list[ShelfItem], row_gap: float = 0.6) -> list[Slot]:
    """물건들을 단(높이)과 순서(왼쪽부터)로 정리한다.

    단 나누기: 물건 중심 높이를 위에서부터 훑어, 앞 물건과 높이 차이가 물건 크기의 row_gap 배를 넘으면 새 단.
    """
    if not items:
        return []
    ordered = sorted(items, key=lambda it: it.center[1])
    rows: list[list[ShelfItem]] = [[ordered[0]]]
    for it in ordered[1:]:
        prev = rows[-1]
        mean_y = np.mean([p.center[1] for p in prev])
        mean_size = np.mean([p.size for p in prev] + [it.size])
        if it.center[1] - mean_y > row_gap * mean_size:
            rows.append([it])
        else:
            prev.append(it)
    slots = []
    for r, row in enumerate(rows, start=1):
        for i, it in enumerate(sorted(row, key=lambda it: it.center[0]), start=1):
            slots.append(Slot(it, r, i))
    return slots


def match(before: list[Slot], after: list[Slot], gate: float = 0.8,
          label_penalty: float = 0.3) -> tuple[list[Slot], list[Slot]]:
    """두 지도를 자리로 맞춘다. (사라진 자리, 새로 생긴 자리)를 돌려준다.

    비용 = 중심 거리 / 물건 크기 (+ 종류 이름이 다르면 label_penalty).
    gate 를 넘는 짝은 맺지 않는다 — 같은 종류라도 멀리 떨어진 자리는 다른 물건이다.
    작은 문제(선반 한 칸에 수십 개)라 탐욕적으로 가장 싼 짝부터 맺는다.
    """
    pairs = []
    for i, b in enumerate(before):
        for j, a in enumerate(after):
            (bx, by), (ax, ay) = b.item.center, a.item.center
            size = (b.item.size + a.item.size) / 2.0
            cost = np.hypot(bx - ax, by - ay) / size
            if b.item.label != a.item.label:
                cost += label_penalty
            if cost <= gate:
                pairs.append((cost, i, j))
    used_b, used_a = set(), set()
    for _, i, j in sorted(pairs):
        if i in used_b or j in used_a:
            continue
        used_b.add(i)
        used_a.add(j)
    removed = [b for i, b in enumerate(before) if i not in used_b]
    added = [a for j, a in enumerate(after) if j not in used_a]
    return removed, added


class _Consensus:
    """최근 몇 장면의 탐지를 모아, 대부분의 장면에서 보인 물건만 남긴다 (한 장면만 튀는 오탐 제거)."""

    def __init__(self, presence: float, gate: float) -> None:
        self.presence = presence
        self.gate = gate
        self.frames: list[list[ShelfItem]] = []

    def clear(self) -> None:
        self.frames = []

    def add(self, items: list[ShelfItem], keep: int) -> None:
        self.frames.append(items)
        self.frames = self.frames[-keep:]

    def result(self) -> list[ShelfItem]:
        clusters: list[list[ShelfItem]] = []
        for items in self.frames:
            for it in items:
                best, best_cost = None, self.gate
                for cl in clusters:
                    ref = cl[-1]
                    cost = np.hypot(ref.center[0] - it.center[0], ref.center[1] - it.center[1]) / ((ref.size + it.size) / 2)
                    if cost < best_cost:
                        best, best_cost = cl, cost
                if best is None:
                    clusters.append([it])
                else:
                    best.append(it)
        need = self.presence * len(self.frames)
        out = []
        for cl in clusters:
            if len(cl) < need:
                continue
            boxes = np.array([c.bbox for c in cl])
            labels = [c.label for c in cl]
            label = max(set(labels), key=labels.count)
            out.append(ShelfItem(label, tuple(float(v) for v in np.median(boxes, axis=0)),
                                 float(np.mean([c.score for c in cl]))))
        return out


class ShelfMonitor:
    """선반 하나를 지켜본다.

    shelf_box: 선반 영역 (x1, y1, x2, y2, 화면 픽셀). 중심이 이 안에 있는 물건만 선반 물건으로 본다.
    사람 상자가 이 영역과 겹치면 '가림'으로 본다.
    """

    def __init__(
        self,
        name: str,
        shelf_box: tuple[float, float, float, float],
        stable_seconds: float = 1.0,
        presence: float = 0.6,
        match_gate: float = 0.8,
    ) -> None:
        self.name = name
        self.box = shelf_box
        self.stable_ms = stable_seconds * 1000.0
        self.match_gate = match_gate
        self._cons = _Consensus(presence, gate=0.5)
        self.reset()

    def reset(self) -> None:
        self.state: list[Slot] | None = None
        """확정된 선반 지도. 처음 확정되기 전에는 None."""
        self.state_ms = 0.0
        self._cons.clear()
        self._clear_since: float | None = None
        self._occluded_since: float | None = None
        self._people: list[int] = []
        self._pending: tuple | None = None
        """바뀐 것 같은 상태와 처음 본 시각. 손이 잠깐 가려 생긴 깜빡임을 거르려고, 바뀐 상태가
        stable_seconds 동안 그대로여야 확정한다."""
        self.changes: list[ShelfChange] = []

    def _inside(self, item: ShelfItem) -> bool:
        x, y = item.center
        x1, y1, x2, y2 = self.box
        return x1 <= x <= x2 and y1 <= y <= y2

    def _overlaps(self, bbox) -> bool:
        x1, y1, x2, y2 = self.box
        a1, b1, a2, b2 = bbox
        return min(x2, a2) > max(x1, a1) and min(y2, b2) > max(y1, b1)

    def update(self, now_ms: float, items: list[ShelfItem],
               people: list[tuple[int, tuple]], frames_per_window: int) -> list[ShelfChange]:
        """한 장면을 넣는다. 확정된 변화가 있으면 돌려준다.

        people: (사람 번호, 상자) 목록. frames_per_window: stable_seconds 동안 들어오는 장면 수.
        """
        blocking = [pid for pid, bbox in people if self._overlaps(bbox)]
        if blocking:
            # 가려졌다 — 지금 장면으로는 셀 수 없다. 누가 가렸는지만 기록한다.
            if self._occluded_since is None:
                self._occluded_since = now_ms
            for pid in blocking:
                if pid not in self._people:
                    self._people.append(pid)
            self._cons.clear()
            self._clear_since = None
            return []

        if self._clear_since is None:
            self._clear_since = now_ms
        self._cons.add([it for it in items if self._inside(it)], keep=frames_per_window)
        if now_ms - self._clear_since < self.stable_ms:
            return []   # 아직 충분히 오래 안 봤다

        current = layout(self._cons.result())
        if self.state is None:
            self.state, self.state_ms = current, now_ms
            return []
        removed, added = match(self.state, current, gate=self.match_gate)
        if not removed and not added:
            if self._occluded_since is not None:
                # 가렸지만 아무것도 안 바뀌었다 (보기만 함). 다음 방문을 위해 기록을 비운다.
                self._occluded_since, self._people = None, []
            self._pending = None
            self.state, self.state_ms = current, now_ms
            return []
        signature = (tuple(sorted(s.name for s in removed)), tuple(sorted(s.name for s in added)))
        if self._pending is None or self._pending[0] != signature:
            self._pending = (signature, now_ms)
            return []
        if now_ms - self._pending[1] < self.stable_ms:
            return []
        self._pending = None
        change = ShelfChange(
            shelf=self.name, removed=removed, added=added,
            start_ms=self._occluded_since if self._occluded_since is not None else self.state_ms,
            end_ms=now_ms, people=list(self._people),
        )
        self.changes.append(change)
        self.state, self.state_ms = current, now_ms
        self._occluded_since, self._people = None, []
        return [change]
