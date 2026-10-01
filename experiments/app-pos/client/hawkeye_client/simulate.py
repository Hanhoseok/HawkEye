"""시뮬레이터 — 탐지 파이프라인 없이 시나리오 경보를 보내 앱·서버를 시험하고 시연한다.

    python -m hawkeye_client.simulate --server http://127.0.0.1:8000 --scenario all
    python -m hawkeye_client.simulate --scenario theft --image scene.jpg --delay 3

API 키가 필요하면 환경변수 HAWKEYE_API_KEY 로 준다.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .events import build_event
from .publisher import AlertPublisher

ALARMS = {"WARNING", "REVIEW", "HIGH_RISK"}


@dataclass(frozen=True)
class Step:
    level: str
    taken: int
    paid: int
    reason: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Scenario:
    description: str
    steps: tuple[Step, ...]


SNACK_UNPAID = [{"name": "과자", "taken": 1, "paid": 0}, {"name": "음료", "taken": 1, "paid": 0}]

SCENARIOS: dict[str, Scenario] = {
    "theft": Scenario(
        "출구에 다가선 뒤(주의) 결제 없이 나감(위험)",
        (
            Step("WARNING", 2, 0, "미결제로 출구 접근", {"items": SNACK_UNPAID}),
            Step("HIGH_RISK", 2, 0, "결제 없이 퇴장", {"items": SNACK_UNPAID, "identity_check": 0.9}),
        ),
    ),
    "partial": Scenario(
        "3개 가져가 2개만 결제하고 나감(확인 필요)",
        (Step("REVIEW", 3, 2, "결제가 감지보다 적음 — 관리자 확인 필요", {
            "items": [{"name": "과자", "taken": 2, "paid": 1}, {"name": "음료", "taken": 1, "paid": 1}],
        }),),
    ),
    "paid": Scenario(
        "가져간 만큼 결제하고 통과(알림 없음)",
        (Step("CLEAR", 2, 2, "전부 결제"),),
    ),
    "came_back": Scenario(
        "출구에 다가섰다가(주의) 되돌아가 결제하고 나감(자동 해제)",
        (
            Step("WARNING", 1, 0, "미결제로 출구 접근"),
            Step("CLEAR", 1, 1, "되돌아가 결제 후 퇴장"),
        ),
    ),
}


def run_scenario(
    alerts: AlertPublisher,
    name: str,
    person_id: int,
    delay: float = 2.0,
    jpeg: bytes | None = None,
) -> int:
    """시나리오 한 개를 보낸다. 경보 단계(주의·확인 필요·위험)에는 장면 사진을 붙인다."""
    scenario = SCENARIOS[name]
    for i, step in enumerate(scenario.steps):
        if i and delay:
            time.sleep(delay)
        event = build_event(
            person_id=person_id, level=step.level, taken=step.taken, paid=step.paid,
            time_sec=10.0 * (i + 1), reason=step.reason, **step.extra,
        )
        alerts.publish(event, jpeg=jpeg if step.level in ALARMS else None)
    return person_id


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="HawkEye 경보 시뮬레이터")
    p.add_argument("--server", default=os.environ.get("HAWKEYE_SERVER", "http://127.0.0.1:8000"))
    p.add_argument("--camera", default="cam1")
    p.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="all")
    p.add_argument("--person", type=int, default=1, help="첫 손님 번호 (all 이면 1씩 늘린다)")
    p.add_argument("--delay", type=float, default=2.0, help="단계 사이 간격(초)")
    p.add_argument("--image", type=Path, help="경보에 붙일 장면 사진(JPEG)")
    a = p.parse_args(argv)
    # 한국어 Windows 터미널(cp949)이 못 찍는 글자가 있어도 죽지 않게 한다.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    jpeg = a.image.read_bytes() if a.image else None
    alerts = AlertPublisher(a.server, camera_id=a.camera, api_key=os.environ.get("HAWKEYE_API_KEY") or None)
    names = list(SCENARIOS) if a.scenario == "all" else [a.scenario]
    for offset, name in enumerate(names):
        print(f"손님 {a.person + offset}: {name} - {SCENARIOS[name].description}")
        run_scenario(alerts, name, a.person + offset, delay=a.delay, jpeg=jpeg)
        if a.delay and offset < len(names) - 1:
            time.sleep(a.delay)
    alerts.close()
    print(f"전송 성공 {alerts.sent} / 실패 {alerts.failed} (실행 {alerts.run_id})")
    return 0 if alerts.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
