"""손님 상태 · 결제 · 위험 판정 테스트.

화면(400x400)을 이렇게 나눈다.

    +-----------------------+
    |        (통로)         |
    |                       |
    |  [계산대 y 250~320]   |
    |  [출구   y 340~400]   |
    +-----------------------+

손님의 위치는 발(bbox 아래쪽 가운데)로 판단하므로, bbox 아래쪽 y 값만 바꿔 가며 움직인다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import RiskConfig  # noqa: E402
from src.core.types import (  # noqa: E402
    Frame, IdentityObservation, RiskLevel, TakeCandidate, Zone, ZoneType,
)
from src.risk.engine import RiskEngine  # noqa: E402
from src.risk.payments import PaymentEvent, PaymentFeed  # noqa: E402
from src.zones.zone_map import ZoneMap  # noqa: E402

AISLE, CHECKOUT, EXIT = 150.0, 300.0, 380.0   # 발의 y 위치


def zones() -> ZoneMap:
    zm = ZoneMap([
        Zone("counter", ZoneType.CHECKOUT, ((0, 250), (400, 250), (400, 320), (0, 320))),
        Zone("door", ZoneType.EXIT, ((0, 340), (400, 340), (400, 400), (0, 400))),
    ])
    zm.resolve(400, 400)
    return zm


def at(t: float) -> Frame:
    return Frame(index=int(t * 10), timestamp=f"t{t}", pts_ms=t * 1000.0, image=None)


def person(pid: int, foot_y: float, x: float = 100.0) -> IdentityObservation:
    return IdentityObservation(
        person_id=pid, track_id=pid, bbox=(x, foot_y - 120, x + 50, foot_y), frame=0, timestamp="t"
    )


def take(pid: int, start: int = 10) -> TakeCandidate:
    return TakeCandidate(pid, "shelf", start, start + 20, "a", "b", 2.0, 20, 0.3, 0.01)


def engine(payments=(), **cfg) -> RiskEngine:
    return RiskEngine(RiskConfig(**cfg), zones(), PaymentFeed(list(payments)))


def walk(eng: RiskEngine, pid: int, foot_y: float, start: float, seconds: float, **kw):
    """pid 가 foot_y 위치에 seconds 초 동안 서 있다 (0.1초 간격). 나온 판정을 모두 돌려준다."""
    events = []
    for i in range(int(seconds * 10)):
        ev, _ = eng.update(at(start + i * 0.1), [person(pid, foot_y)], **kw)
        events += ev
    return events


# ---------------------------------------------------------------- 기본 시나리오

def test_take_and_leave_without_paying_is_high_risk():
    """핵심 — 집고 계산 없이 출구로 가면 HIGH_RISK."""
    eng = engine()
    eng.update(at(1.0), [person(1, AISLE)], takes=[take(1)])
    events = walk(eng, 1, EXIT, 5.0, 1.0)
    assert len(events) == 1
    assert events[0].level == RiskLevel.HIGH_RISK
    assert events[0].unpaid == 1 and events[0].take_frames == ((10, 30),)


def test_take_pay_at_checkout_then_leave_is_clear():
    """집고 → 계산대에서 결제(POS 는 누가 결제했는지 모른다) → 출구 = CLEAR."""
    eng = engine(payments=[PaymentEvent(time_sec=4.0, items=1)])
    eng.update(at(1.0), [person(1, AISLE)], takes=[take(1)])
    walk(eng, 1, CHECKOUT, 3.0, 2.0)                 # 3~5초 계산대. 4초에 결제
    events = walk(eng, 1, EXIT, 6.0, 1.0)
    assert [e.level for e in events] == [RiskLevel.CLEAR]
    assert eng.book.payments[0].method == "checkout"


def test_nothing_taken_is_clear():
    """아무것도 안 집고 나가면 CLEAR (경보가 울리면 안 된다)."""
    events = walk(engine(), 1, EXIT, 0.0, 1.0)
    assert [e.level for e in events] == [RiskLevel.CLEAR]


def test_paying_for_fewer_items_is_review_not_alarm():
    """두 번 집고 하나만 결제 -> REVIEW. 탐지가 과하게 셌을 수도 있어 경보 대신 확인으로 돌린다."""
    eng = engine(payments=[PaymentEvent(time_sec=4.0, items=1)])
    eng.update(at(1.0), [person(1, AISLE)], takes=[take(1, 10), take(1, 40)])
    walk(eng, 1, CHECKOUT, 3.0, 2.0)
    events = walk(eng, 1, EXIT, 6.0, 1.0)
    assert events[0].level == RiskLevel.REVIEW and events[0].unpaid == 1


def test_overlapping_candidates_count_as_one_take():
    """한 번 멈춰 두 선반에 걸친 후보 두 건은 집기 한 번이다 — 1개 결제하면 CLEAR."""
    eng = engine(payments=[PaymentEvent(time_sec=4.0, items=1)])
    overlapping = [take(1, 10), TakeCandidate(1, "shelf2", 20, 50, "a", "b", 3.0, 30, 0.3, 0.01)]
    eng.update(at(1.0), [person(1, AISLE)], takes=overlapping)
    walk(eng, 1, CHECKOUT, 3.0, 2.0)
    events = walk(eng, 1, EXIT, 6.0, 1.0)
    assert events[0].taken == 1 and events[0].level == RiskLevel.CLEAR


# ---------------------------------------------------------------- 결제 연결

def test_payment_goes_to_the_person_standing_longest_at_checkout():
    """계산대에 두 명이 있으면 오래 서 있던 쪽(계산하는 사람)이 결제자다. 뒤에 선 사람이 아니다."""
    eng = engine(payments=[PaymentEvent(time_sec=5.0, items=1)])
    eng.update(at(0.5), [person(1, AISLE), person(2, AISLE, x=250)], takes=[take(1), take(2)])
    for i in range(32):                              # 2~5.1초: 1 은 계속 계산대, 2 는 4초부터 합류
        t = 2.0 + i * 0.1
        obs = [person(1, CHECKOUT)] + ([person(2, CHECKOUT, x=250)] if t >= 4.0 else [])
        eng.update(at(t), obs)
    record = eng.book.payments[0]
    assert record.person_id == 1 and record.candidates == 2
    assert eng.book.customers[1].unpaid == 0 and eng.book.customers[2].unpaid == 1


def test_payment_with_nobody_at_checkout_is_recorded_as_unmatched():
    """계산대에 아무도 없을 때 들어온 결제는 버리지 않고 '연결 실패'로 남긴다 (오경보 원인 추적용)."""
    eng = engine(payments=[PaymentEvent(time_sec=2.0, items=1)])
    walk(eng, 1, AISLE, 0.0, 3.0)
    assert eng.book.payments[0].method == "unmatched"


def test_recent_checkout_visit_within_grace_counts():
    """결제 기록이 계산대를 막 떠난 뒤(여유 시간 안) 도착해도 그 손님에게 연결한다."""
    eng = engine(payments=[PaymentEvent(time_sec=5.0, items=1)], checkout_grace_seconds=3.0)
    eng.update(at(0.5), [person(1, AISLE)], takes=[take(1)])
    walk(eng, 1, CHECKOUT, 2.0, 1.5)                 # 3.5초에 계산대를 떠남
    walk(eng, 1, AISLE, 3.5, 2.0)                    # 5초에 결제 도착
    assert eng.book.payments[0].person_id == 1


# ---------------------------------------------------------------- 출구 판정의 세부

def test_brushing_past_exit_is_not_judged():
    """출구 구역을 스치기만 하면(0.5초 미만) 판정하지 않는다."""
    eng = engine()
    eng.update(at(1.0), [person(1, AISLE)], takes=[take(1)])
    events = walk(eng, 1, EXIT, 2.0, 0.3) + walk(eng, 1, AISLE, 2.3, 1.0)
    assert events == []


def test_one_judgment_per_exit_visit():
    """출구에 오래 서 있어도 판정은 한 번. 경계에서 잠깐 벗어났다 들어와도 같은 방문이다."""
    eng = engine()
    eng.update(at(1.0), [person(1, AISLE)], takes=[take(1)])
    events = walk(eng, 1, EXIT, 2.0, 3.0)
    events += walk(eng, 1, AISLE, 5.0, 0.3)          # 경계에서 잠깐 이탈 (2초 미만)
    events += walk(eng, 1, EXIT, 5.3, 1.0)
    assert len(events) == 1


def test_returning_to_exit_later_is_judged_again():
    """출구를 확실히 떠났다가 다시 오면 새 방문으로 다시 판정한다 (그 사이 결제했을 수 있다)."""
    eng = engine(payments=[PaymentEvent(time_sec=5.0, items=1)])
    eng.update(at(0.5), [person(1, AISLE)], takes=[take(1)])
    first = walk(eng, 1, EXIT, 1.0, 1.0)
    walk(eng, 1, CHECKOUT, 3.0, 3.0)                 # 결제하러 돌아감
    second = walk(eng, 1, EXIT, 7.0, 1.0)
    assert [e.level for e in first + second] == [RiskLevel.HIGH_RISK, RiskLevel.CLEAR]


# ---------------------------------------------------------------- 신원 합침

def test_merged_identity_carries_its_takes_to_the_exit():
    """A 가 집은 뒤 쪼개져 B 로 등록됐다가 합쳐진다. 출구에 간 B 는 A 가 집은 것을 들고 있어야 한다."""
    eng = engine()
    eng.update(at(1.0), [person(1, AISLE)], takes=[take(1)])
    walk(eng, 2, AISLE, 10.0, 1.0)                    # 쪼개져서 2 로 등장
    merges = [{"from": 2, "into": 1}]
    eng.update(at(11.0), [person(1, AISLE)], merges=merges)   # 계층 2 가 합침
    events = walk(eng, 1, EXIT, 12.0, 1.0, merges=merges)
    assert events[0].person_id == 1 and events[0].level == RiskLevel.HIGH_RISK


def test_late_take_under_old_id_reaches_merged_customer():
    """합쳐지기 전 번호로 늦게 도착한 TAKE 후보도 합쳐진 손님에게 간다."""
    eng = engine()
    walk(eng, 1, AISLE, 0.0, 1.0)
    walk(eng, 2, AISLE, 5.0, 1.0)
    merges = [{"from": 2, "into": 1}]
    eng.update(at(6.0), [person(1, AISLE)], takes=[take(2)], merges=merges)
    events = walk(eng, 1, EXIT, 7.0, 1.0, merges=merges)
    assert events[0].level == RiskLevel.HIGH_RISK


# ---------------------------------------------------------------- 화면 가장자리 출구

def test_vanishing_inside_exit_zone_is_judged():
    """출구가 화면 가장자리면 0.5초를 머물지 않고 곧장 사라진다. 사라진 뒤 1초가 지나면 판정한다."""
    eng = engine()
    eng.update(at(1.0), [person(1, AISLE)], takes=[take(1)])
    events = walk(eng, 1, EXIT, 2.0, 0.3)            # 출구에 0.3초만 보이고
    for i in range(25):                              # 사라진다 (2.5초간 아무도 안 보임)
        ev, _ = eng.update(at(2.3 + i * 0.1), [])
        events += ev
    assert [e.level for e in events] == [RiskLevel.HIGH_RISK]
    assert "사라짐" in events[0].reason


def test_vanishing_outside_exit_zone_is_not_judged():
    """통로에서 사라진 것(가려짐, 탐지 실패)은 나간 것이 아니다."""
    eng = engine()
    eng.update(at(1.0), [person(1, AISLE)], takes=[take(1)])
    events = walk(eng, 1, AISLE, 2.0, 0.3)
    for i in range(30):
        ev, _ = eng.update(at(2.3 + i * 0.1), [])
        events += ev
    assert events == []


def test_take_closed_just_after_vanishing_is_counted():
    """사라지기 직전의 TAKE 후보는 사람이 안 보인 뒤에 끝난다. 판정은 그것까지 반영해야 한다."""
    eng = engine()
    walk(eng, 1, EXIT, 2.0, 0.3)
    eng.update(at(2.8), [], takes=[take(1)])         # 사라지고 0.5초 뒤 후보가 도착
    events = []
    for i in range(20):
        ev, _ = eng.update(at(2.9 + i * 0.1), [])
        events += ev
    assert [e.level for e in events] == [RiskLevel.HIGH_RISK]


def test_reappearing_after_vanish_judgment_is_retracted_and_rejudged():
    """출구 구역에서 탐지가 끊겨 '나갔다'고 판정했는데 다시 보이면, 그 판정을 되돌리고 다음에 다시 판정한다.

    UCF-Crime Shoplifting031 실측: 탐지가 2~4초씩 끊겨 이른 판정(CLEAR)이 나갔고,
    그 뒤 물건을 집고 실제로 나갔을 때는 이미 판정했다는 이유로 판정이 없었다.
    """
    eng = engine()
    first = walk(eng, 1, EXIT, 1.0, 0.3)
    for i in range(25):                               # 탐지 끊김 2.5초 -> 이른 판정 (아직 안 집음)
        ev, _ = eng.update(at(1.3 + i * 0.1), [])
        first += ev
    assert [e.level for e in first] == [RiskLevel.CLEAR]
    eng.update(at(4.0), [person(1, AISLE)], takes=[take(1)])   # 다시 나타나 물건을 집음
    assert eng.retracted and eng.retracted[0][0] == 1
    second = walk(eng, 1, EXIT, 5.0, 1.0)
    assert [e.level for e in second] == [RiskLevel.HIGH_RISK]
