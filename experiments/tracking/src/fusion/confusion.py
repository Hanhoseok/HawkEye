"""한 카메라가 두 사람을 헷갈리는 구간을 장면이 들어오는 대로 잡는다 (후속 68, 한계 C-26 · C-28).

'헷갈린다' = 그 카메라 혼자서는 어느 상자가 누구인지 믿기 어렵다. 사람 상자만 본다:
  겹침    두 사람의 상자가 limit 이상 겹친다 (겹친 넓이 ÷ 작은 상자 넓이 — fusion.swap_overlap 과 같은 식)
  합쳐짐  한 사람의 상자가 사라졌는데, 사라지기 직전 상자의 limit 이상이 남은 다른 사람 상자에 덮여 있다.
          그 사람이 다시 보이거나 merged_max_ms 가 지날 때까지 (4차 C05: 두 사람이 상자 하나가 됐다)

번호가 실제로 틀렸는지는 모른다 — 틀릴 수 있는 구간을 알려 줄 뿐이다. 무엇을 할지는 판정(DualJudge)이 정한다.
세는 스크립트(scripts/confusion_count.py)와 같은 정의다. 스크립트는 끝까지 본 뒤 1초보다 짧게 산 번호를 빼지만 여기는 못 뺀다.
"""

from __future__ import annotations

from dataclasses import dataclass


def overlap(a, b) -> float:
    """두 상자가 겹친 넓이 ÷ 작은 상자 넓이."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return w * h / small if small > 0 else 0.0


def covered(mine, other) -> float:
    """mine 상자 넓이 가운데 other 가 덮은 비율."""
    w = min(mine[2], other[2]) - max(mine[0], other[0])
    h = min(mine[3], other[3]) - max(mine[1], other[1])
    area = (mine[2] - mine[0]) * (mine[3] - mine[1])
    return w * h / area if w > 0 and h > 0 and area > 0 else 0.0


@dataclass
class Episode:
    """두 사람(a, b)을 헷갈린 한 구간."""

    a: int
    b: int
    start_ms: float
    end_ms: float
    """마지막으로 헷갈린 장면의 시각."""
    peak_ms: float
    """두 상자가 가장 많이 겹친 장면의 시각 (겹친 장면이 없으면 시작)."""
    peak: float = 0.0
    merged: bool = False
    """상자 하나로 합쳐진 적이 있나."""
    back_ms: float | None = None
    """합쳐졌던 사람의 상자가 다시 생긴 시각 (여러 번이면 마지막)."""
    gone_ms: float | None = None
    """그 사람의 상자가 사라진 첫 장면의 시각 (back_ms 와 같은 합쳐짐의 것)."""
    hidden: int | None = None
    """그때 상자가 사라졌던 번호 (남은 상자의 번호는 a · b 가운데 다른 쪽)."""

    @property
    def switch_ms(self) -> float | None:
        """번호가 바뀌었다면 이때부터로 본다 — 합쳐졌다 갈라진 구간에서 새 상자가 생긴 때. 그때 새 상자가 번호를 다시 받는다
        (4차 C05 8.87초). 겹치기만 한 구간은 언제 바뀌었는지 알 수 없어 None (본 사례가 없다 — 3차 C20 은 조각 시작 그대로)."""
        return self.back_ms

    def has(self, pid: int) -> bool:
        return pid in (self.a, self.b)

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "start_ms": round(self.start_ms, 1), "end_ms": round(self.end_ms, 1),
                "switch_ms": None if self.switch_ms is None else round(self.switch_ms, 1),
                "peak": round(self.peak, 3), "merged": self.merged,
                "gone_ms": None if self.gone_ms is None else round(self.gone_ms, 1), "hidden": self.hidden}


class ConfusionWatch:
    """한 카메라의 사람 상자를 장면마다 받아 헷갈린 구간(Episode)을 쌓는다."""

    def __init__(self, limit: float = 0.5, merged_max_ms: float = 3000.0, join_ms: float = 500.0,
                 keep_ms: float = 600000.0) -> None:
        self.limit, self.merged_max_ms, self.join_ms, self.keep_ms = limit, merged_max_ms, join_ms, keep_ms
        self.reset()

    def reset(self) -> None:
        self.episodes: list[Episode] = []
        self._last: dict[int, tuple] = {}
        self._merged: dict[int, tuple[int, float]] = {}
        """사라진 번호 → (덮은 번호, 사라진 ms)."""
        self._open: dict[frozenset, Episode] = {}
        self.returned: list[tuple[int, int, Episode]] = []
        """방금 장면에서 상자가 다시 생긴 사람들: (사라졌던 번호, 그동안 남아 있던 번호, 구간)."""

    def update(self, ms: float, boxes: dict[int, tuple]) -> None:
        """boxes = 이 장면의 {번호: 상자}. 사람이 없는 장면은 안 불러도 된다."""
        back = [pid for pid in boxes if pid in self._merged]
        self.returned = []
        for pid in back:
            into, since = self._merged.pop(pid)
            ep = self._open.get(frozenset((pid, into)))
            if ep is not None:
                ep.back_ms, ep.gone_ms, ep.hidden = ms, since, pid
                if into in boxes:
                    self.returned.append((pid, into, ep))
        for pid, box in self._last.items():
            if pid in boxes or pid in self._merged:
                continue
            into = [other for other, b in boxes.items() if covered(box, b) >= self.limit]
            if into:
                self._merged[pid] = (into[0], ms)
        for pid, (into, since) in list(self._merged.items()):
            if into not in boxes or ms - since > self.merged_max_ms:
                self._merged.pop(pid)
        ids = sorted(boxes)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                amount = overlap(boxes[a], boxes[b])
                if amount >= self.limit:
                    self._mark(a, b, ms, amount, False)
        for pid, (into, _) in self._merged.items():
            self._mark(pid, into, ms, 0.0, True)
        self._last = dict(boxes)
        if self.episodes and ms - self.episodes[0].end_ms > self.keep_ms:
            self.episodes = [e for e in self.episodes if ms - e.end_ms <= self.keep_ms]

    def _mark(self, a: int, b: int, ms: float, amount: float, merged: bool) -> None:
        key = frozenset((a, b))
        ep = self._open.get(key)
        if ep is None or ms - ep.end_ms > self.join_ms:
            ep = Episode(min(a, b), max(a, b), ms, ms, ms)
            self._open[key] = ep
            self.episodes.append(ep)
        ep.end_ms = ms
        ep.merged = ep.merged or merged
        if amount > ep.peak:
            ep.peak, ep.peak_ms = amount, ms

    def find(self, a: int, b: int, start_ms: float, end_ms: float, slack_ms: float | None = None) -> Episode | None:
        """그 시간([start_ms, end_ms], 앞뒤 slack_ms 여유 — 안 주면 join_ms)에 닿는 a · b 의 구간 — 여럿이면 가장 늦은 것."""
        slack = self.join_ms if slack_ms is None else slack_ms
        hits = [e for e in self.episodes if e.has(a) and e.has(b)
                and e.start_ms <= end_ms + slack and e.end_ms >= start_ms - slack]
        return hits[-1] if hits else None

    def touching(self, pid: int, start_ms: float, end_ms: float) -> list[Episode]:
        """그 시간에 그 사람이 낀 구간들."""
        return [e for e in self.episodes if e.has(pid) and e.start_ms <= end_ms and e.end_ms >= start_ms]

    def rename(self, a: int, b: int) -> None:
        """두 번호를 서로 바꿔 읽기로 했다 — 이 뒤로 들어오는 상자는 바뀐 번호다. 직전 장면 기억만 맞춘다 (구간의 두 사람은 그대로 한 쌍)."""
        flip = {a: b, b: a}
        self._last = {flip.get(pid, pid): box for pid, box in self._last.items()}
        self._merged = {flip.get(pid, pid): (flip.get(into, into), since) for pid, (into, since) in self._merged.items()}
        self.returned = [(flip.get(pid, pid), flip.get(into, into), ep) for pid, into, ep in self.returned]
