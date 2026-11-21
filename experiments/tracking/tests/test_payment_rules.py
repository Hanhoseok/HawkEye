"""결제 기록으로 판정할 때의 규칙 (후속 61): 남는 결제로 메우기 (risk.share_surplus_payments) · 일부만 결제 = 경보 (risk.partial_payment_alert) · 겹친 상자에 붙은 번호 버리기 (risk.return_cover_ignore)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.risk.customers import ShelfVisit  # noqa: E402
from src.risk.payments import PaymentEvent  # noqa: E402
from test_risk import AISLE, CHECKOUT, EXIT, C, H, R, IdentityObservation, at, engine, leave, levels, person, take, walk  # noqa: E402


def visit(pid, removed=0, added=0, start=10, end=30):
    return ShelfVisit(pid, "shelf1", start, end, removed, added, zone="shelf", detail="test")


def pays_for_the_other(**cfg):
    """손님 2 가 집고, 손님 1 이 (아무것도 안 집고) 계산대에서 1개를 결제한다 — 일행이 대신 결제 (N9)."""
    eng = engine(payments=[PaymentEvent(time_sec=4.0, items=1)], **cfg)
    eng.update(at(1.0), [person(1, AISLE), person(2, AISLE, x=250)], takes=[take(2)])
    walk(eng, 1, CHECKOUT, 3.0, 2.0)
    return eng


def test_surplus_payment_of_a_companion_covers_the_taker():
    eng = pays_for_the_other(share_surplus_payments=True)
    events = leave(eng, 2, 6.0)
    assert levels(events) == [C] and "다른 손님의 남는 결제 1" in events[0].reason
    assert eng.book.customers[2].covered_items == 1


def test_without_the_rule_the_taker_is_still_an_alarm():
    assert levels(leave(pays_for_the_other(), 2, 6.0))[-1] == H


def test_unmatched_payment_covers_too():
    eng = engine(payments=[PaymentEvent(time_sec=2.0, items=1)], share_surplus_payments=True)
    eng.update(at(1.0), [person(1, AISLE)], takes=[take(1)])
    walk(eng, 1, AISLE, 1.1, 2.0)                       # 결제 때 계산대에 아무도 없었다 — 주인 못 찾은 결제
    assert eng.book.payments[0].method == "unmatched"
    assert levels(leave(eng, 1, 6.0)) == [C]


def test_one_surplus_covers_only_one_item():
    eng = engine(payments=[PaymentEvent(time_sec=4.0, items=1)], share_surplus_payments=True)
    eng.update(at(1.0), [person(1, AISLE), person(2, AISLE, x=250), person(3, AISLE, x=320)], takes=[take(2), take(3)])
    walk(eng, 1, CHECKOUT, 3.0, 2.0)
    assert levels(leave(eng, 2, 6.0)) == [C]
    assert levels(leave(eng, 3, 12.0))[-1] == H             # 남는 결제 1개는 손님 2 가 이미 썼다


def test_surplus_is_not_shared_when_payment_is_guessed_from_standing_at_the_counter():
    eng = engine(share_surplus_payments=True, payment_mode="dwell")
    eng.update(at(1.0), [person(2, AISLE, x=250)], takes=[take(2)])
    eng.book.customers[2].covered_items = 0
    eng.book._get(1, 1000.0).paid_items = 1                 # 남는 결제가 있어 보여도
    assert levels(leave(eng, 2, 6.0))[-1] == H


def paid_for_one_of_two(**cfg):
    eng = engine(payments=[PaymentEvent(time_sec=4.0, items=1)], **cfg)
    eng.update(at(1.0), [person(1, AISLE)], shelf_visits=[visit(1, removed=2)])
    walk(eng, 1, CHECKOUT, 3.0, 2.0)
    return leave(eng, 1, 6.0)


def test_partial_payment_is_review_by_default_and_alarm_with_the_rule():
    assert levels(paid_for_one_of_two())[-1] == R
    events = paid_for_one_of_two(partial_payment_alert=True)
    assert events[-1].level == H and "물건 2 / 결제 1" in events[-1].reason and "확인 필요" not in events[-1].reason


def test_partial_payment_counted_by_reaches_only_stays_review():
    # 집기 동작 수는 물건 수가 아니다 — 선반 기록이 없으면 규칙을 켜도 확인 필요
    eng = engine(payments=[PaymentEvent(time_sec=4.0, items=1)], partial_payment_alert=True)
    eng.update(at(1.0), [person(1, AISLE)], takes=[take(1, 10), take(1, 40)])
    walk(eng, 1, CHECKOUT, 3.0, 2.0)
    assert levels(leave(eng, 1, 6.0)) == [R]


# ---------------------------------------------------------------- 나간 손님이 다시 보일 때 (risk.return_cover_ignore, 4차 L1)

def cover(pid, like):
    """다른 손님(like)의 상자를 다 덮는 큰 상자에 pid 번호가 붙은 관측 — 한 사람에게 상자가 둘 생겼다."""
    x0, y0, x1, y1 = like.bbox
    return IdentityObservation(person_id=pid, track_id=90 + pid, bbox=(x0 - 5, y0 - 5, x1 + 60, y1 + 120), frame=0, timestamp="t")


def left_then_ghost(**cfg):
    """손님 1 이 집고 결제 없이 나간다. 판정 뒤, 매장 안 손님 2 에게 겹친 상자에 손님 1 의 번호가 붙는다."""
    eng = engine(**cfg)
    eng.update(at(1.0), [person(1, AISLE), person(2, AISLE, x=400)], takes=[take(1)])
    events = []
    for k in range(10):                                   # 손님 1 이 출구에 1초 서 있다
        t = 2.0 + k * 0.1
        events += eng.update(at(t), [person(1, EXIT), person(2, AISLE, x=400)])[0]
    for k in range(40):                                   # 사라지고 4초 — 손님 2 는 그대로
        t = 3.0 + k * 0.1
        events += eng.update(at(t), [person(2, AISLE, x=400)])[0]
    assert levels(events)[-1] == H
    other = person(2, AISLE, x=400)
    eng.update(at(7.0), [other, cover(1, other)])
    eng.update(at(7.1), [cover(1, other)])                # 두 번째 장에는 겹친 상자만 있다 — 방금 본 손님 2 의 상자로 견준다
    return eng


def test_ghost_box_over_another_customer_does_not_bring_back_someone_who_left():
    eng = left_then_ghost(return_cover_ignore=0.8)
    assert eng.book.customers[1].departed and eng.book.reappeared == []


def test_without_the_rule_the_ghost_box_undoes_the_exit():
    assert not left_then_ghost().book.customers[1].departed


def test_a_real_return_apart_from_others_still_undoes_the_exit():
    eng = left_then_ghost(return_cover_ignore=0.8)
    eng.update(at(8.0), [person(2, AISLE, x=400), person(1, AISLE, x=100)])
    assert not eng.book.customers[1].departed
