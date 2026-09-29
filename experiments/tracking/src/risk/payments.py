"""결제(POS) 입력.

실제 POS 가 알려주는 것은 "몇 시에, 몇 개를 결제했다"뿐이다. **누가** 결제했는지는 모른다.
그래서 결제 순간 계산대(CHECKOUT) 구역에 서 있던 손님에게 연결한다(customers.py).

지금은 실제 POS 가 없으므로 결제 기록을 CSV 파일로 받는다. 실제 POS 를 붙일 때는
`PaymentFeed` 와 같은 모양(`due(now_ms) -> list[PaymentEvent]`)만 지키면 된다.

CSV 형식 (첫 줄은 머리글):

    time_sec,items,person_id
    31.5,1,
    58.0,2,3

- time_sec  : 영상 재생 시각(초). 영상 속 계산대 장면의 시각을 적는다.
- items     : 결제한 품목 수.
- person_id : 비워 두면 계산대 구역으로 손님을 찾는다(실제 POS 와 같은 조건).
              숫자를 적으면 그 손님에게 바로 연결한다 — 계산대 구역이 없는 영상으로
              로직을 시험할 때만 쓴다. 실제 POS 는 이 정보를 주지 않는다.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PaymentEvent:
    time_sec: float
    items: int
    person_id: int | None = None
    """시험용. 실제 POS 결제에는 없다."""


@dataclass(frozen=True, slots=True)
class PaymentRecord:
    """결제 한 건을 누구에게 연결했는지. 연결 실패도 기록한다(놓친 결제는 오경보의 원인이다)."""

    time_sec: float
    items: int
    person_id: int | None
    """연결된 손님. 계산대에 아무도 없었으면 None."""
    method: str
    """"given"(파일에 적힌 손님) / "checkout"(계산대 구역) / "unmatched"(연결 실패)."""
    candidates: int
    """결제 순간 계산대 근처에 있던 손님 수. 2 이상이면 연결이 틀렸을 수 있다."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PaymentFeed:
    """시각 순으로 결제를 하나씩 내보낸다."""

    def __init__(self, events: list[PaymentEvent]) -> None:
        self._events = sorted(events, key=lambda e: e.time_sec)
        self._cursor = 0

    @classmethod
    def load(cls, path: str | Path) -> "PaymentFeed":
        events = []
        with open(path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if not (row.get("time_sec") or "").strip():
                    continue
                pid = (row.get("person_id") or "").strip()
                events.append(
                    PaymentEvent(
                        time_sec=float(row["time_sec"]),
                        items=int(row.get("items") or 1),
                        person_id=int(pid) if pid else None,
                    )
                )
        return cls(events)

    @classmethod
    def empty(cls) -> "PaymentFeed":
        return cls([])

    def reset(self) -> None:
        self._cursor = 0

    def __len__(self) -> int:
        return len(self._events)

    def due(self, now_ms: float) -> list[PaymentEvent]:
        """지금 시각까지 도착한, 아직 내보내지 않은 결제."""
        out = []
        while self._cursor < len(self._events) and self._events[self._cursor].time_sec * 1000.0 <= now_ms:
            out.append(self._events[self._cursor])
            self._cursor += 1
        return out
