"""손님 상태 · 결제 · 위험 판정 테스트.

화면(400x400)을 이렇게 나눈다.

    +-----------------------+
    |        (통로)         |
    |                       |
    |  [계산대 y 250~320]   |
    |  [출구   y 340~400]   |
    +-----------------------+

손님의 위치는 발(bbox 아래쪽 가운데)로 판단하므로, bbox 아래쪽 y 값만 바꿔 가며 움직인다.

판정은 두 단계다. 출구에 다가서면(0.5초) WARNING, 출구 구역에서 사라져 3초간 다시 나타나지 않으면
최종 판정(HIGH_RISK / REVIEW / CLEAR). 테스트의 leave() 가 '출구로 걸어 나가 사라지는' 것을 흉내낸다.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import RiskConfig  # noqa: E402
from src.core.types import (  # noqa: E402
    Frame, IdentityObservation, RiskLevel, TakeCandidate, Zone, ZoneType,
)
from src.risk.engine import RiskEngine  # noqa: E402
from src.risk.payments import PaymentEvent, PaymentFeed  # noqa: E402
from src.zones.zone_map import ZoneMap  # noqa: E402

AISLE, CHECKOUT, EXIT = 150.0, 300.0, 380.0   # 발의 y 위치
STREET_X, STREET_Y = 320.0, 80.0               # 문 밖 (x 도 함께 옮겨야 한다)
H, W, R, C = RiskLevel.HIGH_RISK, RiskLevel.WARNING, RiskLevel.REVIEW, RiskLevel.CLEAR


def zones() -> ZoneMap:
    zm = ZoneMap([
        Zone("counter", ZoneType.CHECKOUT, ((0, 250), (400, 250), (400, 320), (0, 320))),
        Zone("door", ZoneType.EXIT, ((0, 340), (400, 340), (400, 400), (0, 400))),
        # 유리문 너머로 보이는 바깥 (화면 오른쪽 위)
        Zone("street", ZoneType.OUTSIDE, ((300, 0), (400, 0), (400, 100), (300, 100))),
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


def engine(payments=(), consistency=None, **cfg) -> RiskEngine:
    return RiskEngine(RiskConfig(**cfg), zones(), PaymentFeed(list(payments)), consistency=consistency)


def walk(eng: RiskEngine, pid: int, foot_y: float, start: float, seconds: float, **kw):
    """pid 가 foot_y 위치에 seconds 초 동안 서 있다 (0.1초 간격). 나온 판정을 모두 돌려준다."""
    events = []
    for i in range(int(round(seconds * 10))):
        ev, _ = eng.update(at(start + i * 0.1), [person(pid, foot_y)], **kw)
        events += ev
    return events


def nobody(eng: RiskEngine, start: float, seconds: float, **kw):
    """seconds 초 동안 아무도 안 보인다."""
    events = []
    for i in range(int(round(seconds * 10))):
        ev, _ = eng.update(at(start + i * 0.1), [], **kw)
        events += ev
    return events


def leave(eng: RiskEngine, pid: int, start: float, **kw):
    """출구에 1초 서 있다가 사라지고 4초가 지난다. 다가섬 + 나감 판정을 모두 돌려준다."""
    return walk(eng, pid, EXIT, start, 1.0, **kw) + nobody(eng, start + 1.0, 4.0, **kw)


def levels(events):
    return [e.level for e in events]


# ---------------------------------------------------------------- 기본 시나리오

def test_take_and_leave_without_paying_is_warning_then_high_risk():
    """핵심 — 집고 계산 없이 출구에 다가서면 WARNING, 나가면 HIGH_RISK."""
    eng = engine()
    eng.update(at(1.0), [person(1, AISLE)], takes=[take(1)])
    events = leave(eng, 1, 5.0)
    assert levels(events) == [W, H]
    assert events[1].unpaid == 1 and events[1].take_frames == ((10, 30),)


def test_standing_at_exit_without_leaving_is_only_warning():
    """출구에 서 있기만 하고 나가지 않으면 WARNING 에서 멈춘다. HIGH_RISK 는 나가야 확정된다."""
    eng = engine()
    eng.update(at(1.0), [person(1, AISLE)], takes=[take(1)])
    assert levels(walk(eng, 1, EXIT, 5.0, 10.0)) == [W]


def test_take_pay_at_checkout_then_leave_is_clear():
    """집고 → 계산대에서 결제(POS 는 누가 결제했는지 모른다) → 나가면 CLEAR. WARNING 도 없다."""
    eng = engine(payments=[PaymentEvent(time_sec=4.0, items=1)])
    eng.update(at(1.0), [person(1, AISLE)], takes=[take(1)])
    walk(eng, 1, CHECKOUT, 3.0, 2.0)                 # 3~5초 계산대. 4초에 결제
    assert levels(leave(eng, 1, 6.0)) == [C]
    assert eng.book.payments[0].method == "checkout"


def test_nothing_taken_is_clear():
    """아무것도 안 집고 나가면 CLEAR (경보가 울리면 안 된다)."""
    assert levels(leave(engine(), 1, 0.0)) == [C]


def test_paying_for_fewer_items_is_review_not_alarm():
    """두 번 집고 하나만 결제 -> REVIEW. 탐지가 과하게 셌을 수도 있어 경보 대신 확인으로 돌린다."""
    eng = engine(payments=[PaymentEvent(time_sec=4.0, items=1)])
    eng.update(at(1.0), [person(1, AISLE)], takes=[take(1, 10), take(1, 40)])
    walk(eng, 1, CHECKOUT, 3.0, 2.0)
    events = leave(eng, 1, 6.0)
    assert levels(events) == [R] and events[0].unpaid == 1


def test_overlapping_candidates_count_as_one_take():
    """한 번 멈춰 두 선반에 걸친 후보 두 건은 집기 한 번이다 — 1개 결제하면 CLEAR."""
    eng = engine(payments=[PaymentEvent(time_sec=4.0, items=1)])
    overlapping = [take(1, 10), TakeCandidate(1, "shelf2", 20, 50, "a", "b", 3.0, 30, 0.3, 0.01)]
    eng.update(at(1.0), [person(1, AISLE)], takes=overlapping)
    walk(eng, 1, CHECKOUT, 3.0, 2.0)
    events = leave(eng, 1, 6.0)
    assert events[0].taken == 1 and levels(events) == [C]


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
    """출구 구역을 스치기만 하고 안으로 돌아오면 아무 판정도 없다."""
    eng = engine()
    eng.update(at(1.0), [person(1, AISLE)], takes=[take(1)])
    events = walk(eng, 1, EXIT, 2.0, 0.3) + walk(eng, 1, AISLE, 2.3, 5.0)
    assert events == []


def test_one_warning_per_exit_visit():
    """출구에 오래 서 있어도 WARNING 은 한 번. 경계에서 잠깐 벗어났다 들어와도 같은 방문이다."""
    eng = engine()
    eng.update(at(1.0), [person(1, AISLE)], takes=[take(1)])
    events = walk(eng, 1, EXIT, 2.0, 3.0)
    events += walk(eng, 1, AISLE, 5.0, 0.3)          # 경계에서 잠깐 이탈 (2초 미만)
    events += walk(eng, 1, EXIT, 5.3, 1.0)
    assert levels(events) == [W]


def test_warned_then_went_back_to_pay_is_clear():
    """출구에 다가서 WARNING 을 받았다가 돌아가 계산하고 나가면 최종은 CLEAR."""
    eng = engine(payments=[PaymentEvent(time_sec=5.0, items=1)])
    eng.update(at(0.5), [person(1, AISLE)], takes=[take(1)])
    first = walk(eng, 1, EXIT, 1.0, 1.0)
    walk(eng, 1, CHECKOUT, 3.0, 3.0)                 # 결제하러 돌아감
    second = leave(eng, 1, 7.0)
    assert levels(first) == [W] and levels(second) == [C]


# ---------------------------------------------------------------- 신원 합침

def test_merged_identity_carries_its_takes_to_the_exit():
    """A 가 집은 뒤 쪼개져 B 로 등록됐다가 합쳐진다. 나가는 B 는 A 가 집은 것을 들고 있어야 한다."""
    eng = engine()
    eng.update(at(1.0), [person(1, AISLE)], takes=[take(1)])
    walk(eng, 2, AISLE, 10.0, 1.0)                    # 쪼개져서 2 로 등장
    merges = [{"from": 2, "into": 1}]
    eng.update(at(11.0), [person(1, AISLE)], merges=merges)   # 계층 2 가 합침
    events = leave(eng, 1, 12.0, merges=merges)
    assert levels(events) == [W, H] and events[1].person_id == 1


def test_late_take_under_old_id_reaches_merged_customer():
    """합쳐지기 전 번호로 늦게 도착한 TAKE 후보도 합쳐진 손님에게 간다."""
    eng = engine()
    walk(eng, 1, AISLE, 0.0, 1.0)
    walk(eng, 2, AISLE, 5.0, 1.0)
    merges = [{"from": 2, "into": 1}]
    eng.update(at(6.0), [person(1, AISLE)], takes=[take(2)], merges=merges)
    assert levels(leave(eng, 1, 7.0, merges=merges)) == [W, H]


# ---------------------------------------------------------------- 화면 가장자리 출구 · 끊김

def test_vanishing_inside_exit_zone_without_lingering_is_judged():
    """출구가 화면 가장자리면 0.5초를 머물지 않고 곧장 사라진다. WARNING 없이도 나감은 확정된다."""
    eng = engine()
    eng.update(at(1.0), [person(1, AISLE)], takes=[take(1)])
    events = walk(eng, 1, EXIT, 2.0, 0.3) + nobody(eng, 2.3, 4.0)
    assert levels(events) == [H]
    assert "사라짐" in events[0].reason


def test_vanishing_outside_exit_zone_is_not_judged():
    """통로에서 사라진 것(가려짐, 탐지 실패)은 나간 것이 아니다."""
    eng = engine()
    eng.update(at(1.0), [person(1, AISLE)], takes=[take(1)])
    events = walk(eng, 1, AISLE, 2.0, 0.3) + nobody(eng, 2.3, 5.0)
    assert events == []


def test_take_closed_just_after_vanishing_is_counted():
    """사라지기 직전의 TAKE 후보는 사람이 안 보인 뒤에 끝난다. 판정은 그것까지 반영해야 한다."""
    eng = engine()
    walk(eng, 1, EXIT, 2.0, 0.3)
    eng.update(at(2.8), [], takes=[take(1)])         # 사라지고 0.5초 뒤 후보가 도착
    assert levels(nobody(eng, 2.9, 4.0)) == [H]


def test_short_occlusion_at_exit_does_not_confirm():
    """문 앞 인파에 2초 가려졌다 다시 보이면 '나감'이 아니다 — HIGH_RISK 가 나가면 안 된다.

    UCF-Crime Shoplifting047 실측: 인파에 가려 사라진 것을 나간 것으로 보고 경보가 울렸다.
    """
    eng = engine()
    eng.update(at(1.0), [person(1, AISLE)], takes=[take(1)])
    events = walk(eng, 1, EXIT, 2.0, 1.0)            # 다가섬 -> WARNING
    events += nobody(eng, 3.0, 2.0)                  # 2초 가려짐
    events += walk(eng, 1, EXIT, 5.0, 1.0)           # 다시 보임
    assert levels(events) == [W]


def test_reappearing_after_departure_is_retracted_and_rejudged():
    """확정 대기보다 길게 끊겨 '나갔다'고 판정했는데 다시 보이면, 되돌리고 다음에 다시 판정한다."""
    eng = engine()
    first = walk(eng, 1, EXIT, 1.0, 0.3) + nobody(eng, 1.3, 4.0)
    assert levels(first) == [C]                      # 이른 판정 (아직 안 집음)
    eng.update(at(6.0), [person(1, AISLE)], takes=[take(1)])   # 다시 나타나 물건을 집음
    assert eng.retracted and eng.retracted[0][0] == 1
    assert levels(leave(eng, 1, 7.0)) == [W, H]


# ---------------------------------------------------------------- 판정 직전 신원 확인

def test_swapped_identity_is_downgraded_to_review():
    """나가는 사람의 최근 모습이 그 신원의 이전 모습과 다르면(번호가 옮겨 붙음) HIGH_RISK 대신 REVIEW.

    UCF-Crime Shoplifting047 실측: 집기 기록을 가진 번호가 문가의 다른 사람으로 옮겨 붙어 오경보가 났다.
    """
    eng = engine(consistency=lambda pid: 0.55)
    eng.update(at(1.0), [person(1, AISLE)], takes=[take(1)])
    events = leave(eng, 1, 5.0)
    assert levels(events) == [W, R]
    assert events[1].identity_check == 0.55 and "뒤바뀌었을 수" in events[1].reason


def test_consistent_identity_stays_high_risk():
    """최근 모습이 이전과 같으면 HIGH_RISK 그대로. 확인 값도 기록에 남는다."""
    eng = engine(consistency=lambda pid: 0.93)
    eng.update(at(1.0), [person(1, AISLE)], takes=[take(1)])
    events = leave(eng, 1, 5.0)
    assert levels(events) == [W, H] and events[1].identity_check == 0.93


def test_identity_check_unavailable_keeps_high_risk_but_says_so():
    """기록이 모자라 확인할 수 없으면 HIGH_RISK 를 유지하되, 확인하지 못했다고 남긴다."""
    eng = engine(consistency=lambda pid: None)
    eng.update(at(1.0), [person(1, AISLE)], takes=[take(1)])
    events = leave(eng, 1, 5.0)
    assert levels(events) == [W, H] and "못 함" in events[1].reason


# ---------------------------------------------------------------- 문 밖(OUTSIDE) 구역

def outside(eng: RiskEngine, pid: int, start: float, seconds: float, **kw):
    events = []
    for i in range(int(round(seconds * 10))):
        ev, _ = eng.update(at(start + i * 0.1), [person(pid, STREET_Y, x=STREET_X)], **kw)
        events += ev
    return events


def test_seen_outside_after_being_inside_confirms_departure():
    """유리문 너머 문 밖에서 보이면 사라지길 기다릴 필요 없이 나감이 확정된다 (Shoplifting039).

    문을 나선 뒤에도 유리 너머로 몇 초 더 보여서, '출구 구역에서 사라짐' 규칙만으로는 확정되지 않았다.
    """
    eng = engine()
    eng.update(at(1.0), [person(1, AISLE)], takes=[take(1)])
    events = walk(eng, 1, EXIT, 2.0, 1.0) + outside(eng, 1, 3.0, 1.0)
    assert levels(events) == [W, H] and "문 밖" in events[1].reason


def test_passerby_seen_only_outside_is_never_judged():
    """매장에 들어온 적 없이 문 밖에서만 보인 행인은 판정하지 않는다."""
    eng = engine()
    events = outside(eng, 9, 0.0, 5.0) + nobody(eng, 5.0, 5.0)
    assert events == []


def test_vanishing_just_after_leaving_exit_zone_counts():
    """출구 구역을 막 지나(2초 안) 사라진 것도 나간 것이다. 문턱 너머가 구역 밖이어도 놓치지 않는다."""
    eng = engine()
    eng.update(at(1.0), [person(1, AISLE)], takes=[take(1)])
    events = walk(eng, 1, EXIT, 2.0, 1.0)
    events += walk(eng, 1, CHECKOUT, 3.0, 0.5)      # 구역 경계를 살짝 벗어난 곳에서 마지막으로 보이고
    events += nobody(eng, 3.5, 4.0)                  # 사라짐
    assert levels(events) == [W, H]


def test_going_back_into_store_then_vanishing_is_not_departure():
    """출구에 다가섰다가 매장 안으로 한참 돌아간 뒤 가려진 것은 나간 것이 아니다."""
    eng = engine()
    eng.update(at(1.0), [person(1, AISLE)], takes=[take(1)])
    events = walk(eng, 1, EXIT, 2.0, 1.0) + walk(eng, 1, AISLE, 3.0, 3.0) + nobody(eng, 6.0, 5.0)
    assert levels(events) == [W]


def test_identity_check_uses_value_from_inside_not_at_the_bright_door():
    """출구 쪽에서만 값이 떨어지면(역광) 뒤바뀜으로 보지 않는다 — 매장 안에서 잰 값을 쓴다 (Shoplifting039).

    실측: 진열대에서 0.82~0.90 이던 값이 밝은 유리문 쪽으로 걸어가며 0.68 까지 떨어졌다.
    나가는 순간의 값을 썼더니 실제 절도 인물이 HIGH_RISK 가 아니라 REVIEW 로 내려갔다.
    """
    value = {"v": 0.88}
    eng = engine(consistency=lambda pid: value["v"])
    eng.update(at(1.0), [person(1, AISLE)], takes=[take(1)])
    walk(eng, 1, AISLE, 1.1, 1.0)                    # 매장 안: 0.88
    value["v"] = 0.65                                # 문 쪽: 역광으로 떨어짐
    events = leave(eng, 1, 5.0)
    assert levels(events) == [W, H] and events[1].identity_check == 0.88


def test_persistent_drop_inside_is_still_caught():
    """매장 안에서 5초 넘게 계속 낮으면(번호가 다른 사람을 따라가는 중) 뒤바뀜으로 본다."""
    value = {"v": 0.9}
    eng = engine(consistency=lambda pid: value["v"])
    eng.update(at(1.0), [person(1, AISLE)], takes=[take(1)])
    walk(eng, 1, AISLE, 1.1, 2.0)                    # 원래 사람: 0.9
    value["v"] = 0.6                                 # 뒤바뀜: 이후 계속 낮음
    walk(eng, 1, AISLE, 3.1, 6.0)
    assert levels(leave(eng, 1, 10.0)) == [W, R]
