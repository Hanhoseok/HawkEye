"""사건 규칙 — 들어온 경보가 사건 상태를 어떻게 바꾸고, 폰 알림을 띄울지 정한다.

파이프라인은 한 손님에 대해 경보를 여러 번 낸다 (출구 접근 WARNING → 퇴장 HIGH_RISK, 또는 REVIEW/CLEAR).
규칙 표는 docs/design.md §4.
"""

from __future__ import annotations

from dataclasses import dataclass

RANK = {"CLEAR": 0, "WARNING": 1, "REVIEW": 2, "HIGH_RISK": 3}
ALARM_LEVELS = {"WARNING", "REVIEW", "HIGH_RISK"}


@dataclass(frozen=True, slots=True)
class Current:
    level: str
    state: str


@dataclass(frozen=True, slots=True)
class Decision:
    level: str
    state: str
    notify: bool
    reopened: bool = False


def decide(current: Current | None, incoming: str) -> Decision:
    if incoming not in RANK:
        raise ValueError(f"알 수 없는 경보 단계: {incoming!r}")

    if current is None:
        if incoming in ALARM_LEVELS:
            return Decision(incoming, "active", notify=True)
        return Decision(incoming, "pass", notify=False)

    if RANK[incoming] > RANK[current.level]:
        return Decision(incoming, "active", notify=True, reopened=current.state == "resolved")

    if incoming == "CLEAR" and current.level == "WARNING" and current.state != "resolved":
        return Decision("CLEAR", "auto_cleared", notify=False)

    return Decision(current.level, current.state, notify=False)
