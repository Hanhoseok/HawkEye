"""경보 서버로 보내는 경보 하나(JSON)를 만든다. 형식: ../docs/design.md §5.

어떤 탐지 방식이든 이 형식만 맞추면 앱에 경보가 뜬다.
필수는 손님 번호·단계·감지 개수·결제 개수이고, 나머지는 모르면 기본값으로 채운다.
"""

from __future__ import annotations

from typing import Any

LEVELS = ("CLEAR", "WARNING", "REVIEW", "HIGH_RISK")


def build_event(
    person_id: int,
    level: str,
    taken: int,
    paid: int,
    *,
    zone: str = "EXIT",
    frame: int = 0,
    time_sec: float = 0.0,
    bbox: tuple[float, float, float, float] | list[float] = (0.0, 0.0, 0.0, 0.0),
    reason: str = "",
    **extra: Any,
) -> dict[str, Any]:
    """extra 에는 identity_check, take_frames, items(품목별 개수) 등을 넣을 수 있다."""
    if level not in LEVELS:
        raise ValueError(f"level 은 {LEVELS} 중 하나여야 합니다: {level!r}")
    minutes, seconds = divmod(time_sec, 60.0)
    event: dict[str, Any] = {
        "person_id": person_id,
        "level": level,
        "zone": zone,
        "frame": frame,
        "timestamp": f"00:{int(minutes):02d}:{seconds:06.3f}",
        "time_sec": time_sec,
        "bbox": list(bbox),
        "taken": taken,
        "paid": paid,
        "unpaid": max(0, taken - paid),
        "take_frames": [],
        "reason": reason,
    }
    event.update(extra)
    return event
