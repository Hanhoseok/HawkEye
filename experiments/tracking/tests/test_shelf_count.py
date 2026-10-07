"""선반 확인을 손님 기록에 연결하는 테스트 — '몇 번 집었나'가 아니라 '몇 개 가져갔나'.

선반 지도(src/shelf)가 만든 방문 결과(ShelfVisit)를 손님 장부에 붙이고, 같은 시간의 집기 동작은
선반 기록으로 대신한다. 화면·구역·걷기 도우미는 test_risk.py 와 같다.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import ShelfConfig  # noqa: E402
from src.core.types import Detection, Frame, IdentityObservation, TakeCandidate  # noqa: E402
from src.risk.customers import ShelfVisit  # noqa: E402
from src.risk.payments import PaymentEvent  # noqa: E402
from src.shelf.watcher import ShelfWatcher, parse_area  # noqa: E402
from test_risk import AISLE, CHECKOUT, C, H, R, W, at, engine, leave, levels, person, take, walk  # noqa: E402


def visit(pid: int, removed: int = 0, added: int = 0, start: int = 5, end: int = 40, zone=None) -> ShelfVisit:
    return ShelfVisit(pid, "shelf1", start, end, removed, added, zone=zone, detail="test")


def shop(eng, pid: int, takes=(), visits=(), t: float = 1.0):
    """손님이 통로에 서 있고, 그 순간 집기 동작과 선반 방문 결과가 들어온다."""
    eng.update(at(t), [person(pid, AISLE)], takes=list(takes), shelf_visits=list(visits))


# ---------------------------------------------------------------- 개수 세기

def test_one_grab_two_items_counts_two():
    """한 번 손을 뻗어 두 개를 집었다 — 동작은 1번이지만 선반에서 자리 2개가 비었다 -> 2개."""
    eng = engine()
    shop(eng, 1, takes=[take(1, start=10)], visits=[visit(1, removed=2)])
    assert eng.book.customers[1].taken == 2


def test_browsing_cancels_false_take():
    """집기 동작이 잡혔지만 선반은 그대로다(구경만 함) -> 0개. 출구로 나가도 경보 없음."""
    eng = engine()
    shop(eng, 1, takes=[take(1, start=10)], visits=[visit(1)])
    assert eng.book.customers[1].taken == 0
    events = leave(eng, 1, 5.0)
    assert levels(events) == [C] and "만지기만 1번" in events[0].reason


def test_put_back_is_subtracted():
    """두 개 가져갔다가 하나를 다시 놓았다 -> 1개. 1개 결제하고 나가면 통과."""
    eng = engine(payments=[PaymentEvent(time_sec=4.0, items=1)])
    shop(eng, 1, takes=[take(1, start=10)], visits=[visit(1, removed=2, start=5, end=40)])
    shop(eng, 1, visits=[visit(1, added=1, start=60, end=80)], t=2.0)
    assert eng.book.customers[1].taken == 1
    walk(eng, 1, CHECKOUT, 3.0, 2.0)
    events = leave(eng, 1, 6.0)
    assert levels(events) == [C] and "놓음 1개" in events[0].reason


def test_take_elsewhere_still_counts():
    """지켜보지 않는 시간·선반의 집기 동작은 그대로 센다 -> 선반 1개 + 동작 1번 = 2개."""
    eng = engine()
    shop(eng, 1, takes=[take(1, start=100)], visits=[visit(1, removed=1, start=5, end=40)])
    customer = eng.book.customers[1]
    assert customer.taken == 2 and "선반으로 확인 못 한 집기 동작 1번" in customer.count_detail()


def test_zone_limits_which_takes_are_replaced():
    """선반에 구역 이름을 주면, 같은 시간이라도 다른 구역의 집기 동작은 대신하지 않는다."""
    eng = engine()
    other = TakeCandidate(1, "shelf_far", 10, 30, "a", "b", 2.0, 20, 0.3, 0.01)
    shop(eng, 1, takes=[other], visits=[visit(1, removed=1, zone="shelf_near")])
    assert eng.book.customers[1].taken == 2


def test_unattributed_change_falls_back_to_takes():
    """두 사람이 함께 있어 누구 것인지 못 정한 변화는 붙이지 않는다 — 집기 동작 수 그대로."""
    eng = engine()
    shop(eng, 1, takes=[take(1)], visits=[visit(-1, removed=3)])
    assert eng.book.customers[1].taken == 1
    assert eng.book.shelf_unassigned == 1


def test_without_shelf_nothing_changes():
    """선반 지도를 안 쓰면 지금까지와 똑같다 (문구도 '집기 행동')."""
    eng = engine()
    shop(eng, 1, takes=[take(1)])
    events = leave(eng, 1, 5.0)
    assert levels(events) == [W, H] and events[1].reason.startswith("집기 행동 1번")


def test_shelf_count_drives_verdict():
    """선반에서 2개 확인, 1개만 결제 -> REVIEW (동작만 셌다면 1번 = 1개라 통과였을 것)."""
    eng = engine(payments=[PaymentEvent(time_sec=4.0, items=1)])
    shop(eng, 1, takes=[take(1)], visits=[visit(1, removed=2)])
    walk(eng, 1, CHECKOUT, 3.0, 2.0)
    events = leave(eng, 1, 6.0)
    assert levels(events) == [R]
    assert events[0].taken == 2 and events[0].unpaid == 1 and "집음 2개" in events[0].reason


def test_no_payment_with_shelf_count_is_high_risk():
    eng = engine()
    shop(eng, 1, visits=[visit(1, removed=1)])
    events = leave(eng, 1, 5.0)
    assert levels(events) == [W, H] and events[1].reason.startswith("물건 1개, 결제 없음")


def test_merge_keeps_shelf_visits():
    """신원이 나중에 합쳐지면 선반 기록도 합친다."""
    eng = engine()
    shop(eng, 1, visits=[visit(1, removed=1)])
    shop(eng, 2, visits=[visit(2, removed=1, start=50, end=70)], t=2.0)
    eng.update(at(3.0), [person(1, AISLE)], merges=[{"from": 2, "into": 1}])
    assert eng.book.customers[1].taken == 2 and 2 not in eng.book.customers


# ---------------------------------------------------------------- 행동 판정 (집음 / 놓음 / 만짐)

def kinds(eng, pid):
    return [(a.kind, a.count, a.evidence) for a in eng.book.customers[pid].actions]


def test_take_with_reach_is_take_backed_by_both():
    """손 뻗기 + 자리 2개 비움 -> 집음 2개, 근거 '선반+동작'."""
    eng = engine()
    shop(eng, 1, takes=[take(1, start=10)], visits=[visit(1, removed=2)])
    assert kinds(eng, 1) == [("TAKE", 2, "선반+동작")]
    assert eng.book.customers[1].actions[0].reach_frames == ((10, 30),)


def test_reach_without_change_is_touch():
    """손을 뻗었지만 선반이 그대로 -> 만지기만."""
    eng = engine()
    shop(eng, 1, takes=[take(1, start=10)], visits=[visit(1)])
    assert kinds(eng, 1) == [("TOUCH", 0, "선반+동작")]


def test_no_reach_no_change_is_look():
    """선반을 가렸지만 손 뻗기도 변화도 없음 -> 구경 (개수 변화 없음)."""
    eng = engine()
    shop(eng, 1, visits=[visit(1)])
    assert kinds(eng, 1) == [("LOOK", 0, "선반만")]


def test_refilled_slot_is_return():
    """자리가 다시 참 -> 놓음, 개수 -1."""
    eng = engine()
    shop(eng, 1, visits=[visit(1, added=1)])
    assert kinds(eng, 1) == [("RETURN", -1, "선반만")]


def test_removed_and_added_is_move():
    """한 방문에 한 자리가 비고 다른 자리가 참 -> 자리 바뀜, 순변화 0."""
    eng = engine()
    shop(eng, 1, takes=[take(1, start=10)], visits=[visit(1, removed=1, added=1)])
    assert kinds(eng, 1) == [("MOVE", 0, "선반+동작")] and eng.book.customers[1].taken == 0


def test_reach_without_shelf_record_is_unverified():
    """선반 기록이 없는 손 뻗기 -> 확인 못 함, 지금처럼 1개로 센다."""
    eng = engine()
    shop(eng, 1, takes=[take(1, start=100)])
    assert kinds(eng, 1) == [("UNVERIFIED", 1, "동작만")] and eng.book.customers[1].taken == 1


def test_actions_are_in_time_order_and_sum_to_count():
    """집음 2 -> 놓음 1 -> 확인 못 한 손 뻗기 1: 시간순, 개수 합 = 2."""
    eng = engine()
    shop(eng, 1, takes=[take(1, start=10)], visits=[visit(1, removed=2, start=5, end=40)])
    shop(eng, 1, visits=[visit(1, added=1, start=60, end=80)], t=2.0)
    shop(eng, 1, takes=[take(1, start=120)], t=3.0)
    c = eng.book.customers[1]
    assert [a.kind for a in c.actions] == ["TAKE", "RETURN", "UNVERIFIED"]
    assert c.taken == 2


def test_engine_reports_new_action_when_visit_arrives():
    """방문이 끝나는 프레임에 행동 판정이 나온다 (화면 표시용)."""
    eng = engine()
    eng.update(at(1.0), [person(1, AISLE)], takes=[take(1, start=10)])
    eng.update(at(2.0), [person(1, AISLE)], shelf_visits=[visit(1, removed=1)])
    assert [(a.kind, a.person_id) for a in eng.new_actions] == [("TAKE", 1)]
    assert "집음 1개" in eng.new_actions[0].describe()
    eng.update(at(2.1), [person(1, AISLE)])
    assert eng.new_actions == []


def test_merged_identity_action_uses_final_person():
    """합쳐진 신원의 옛 번호로 온 방문도 최종 번호로 판정한다."""
    eng = engine()
    shop(eng, 1, visits=[visit(1)])
    eng.update(at(2.0), [person(1, AISLE)], merges=[{"from": 2, "into": 1}])
    eng.update(at(3.0), [person(1, AISLE)], shelf_visits=[visit(2, removed=1, start=50, end=70)])
    assert eng.new_actions[0].person_id == 1 and eng.book.customers[1].taken == 1


# ---------------------------------------------------------------- 파이프라인 연결

class FakeItems:
    """정해진 시각부터 2번째 물건이 사라지는 탐지기."""

    def __init__(self, gone_from_ms: float):
        self.gone_from_ms = gone_from_ms

    def detect(self, frame):
        boxes = [(20, 50, 60, 110), (80, 50, 120, 110), (140, 50, 180, 110)]
        if frame.pts_ms >= self.gone_from_ms:
            boxes = [boxes[0], boxes[2]]
        return [Detection(bbox=b, score=0.9, class_id=39, class_name="bottle") for b in boxes]


def test_watcher_turns_shelf_change_into_customer_visit():
    """선반(화면 위쪽)에 손님 7 이 다녀간 뒤 2번째 병이 사라졌다 -> 손님 7 의 방문 결과 '1개 가져감'."""
    watcher = ShelfWatcher(ShelfConfig(areas=[parse_area("shelf1=0,0,0.5,0.5@shelf_a", 0)]), FakeItems(3000))
    watcher.resolve(400, 400, fps=10)
    blocker = IdentityObservation(person_id=9, track_id=9, bbox=(90, 0, 200, 300), frame=0, timestamp="t")
    visits = []
    for i in range(80):
        t = i / 10
        people = [blocker] if 2.0 <= t < 4.0 else []
        v, _ = watcher.update(Frame(i, "", t * 1000.0, None), people, resolve=lambda pid: 7 if pid == 9 else pid)
        visits += v
    assert len(visits) == 1
    v = visits[0]
    assert (v.person_id, v.removed, v.added, v.zone) == (7, 1, 0, "shelf_a")
    assert v.start_frame == 20 and v.end_frame > 40 and "1단 2번째 bottle" in v.detail


def test_watcher_reports_browse_visit():
    """다녀갔지만 아무것도 안 바뀌면 '구경' 방문으로 나온다 (집기 동작을 지우는 근거)."""
    watcher = ShelfWatcher(ShelfConfig(areas=[parse_area("0,0,0.5,0.5", 0)]), FakeItems(10_000))
    watcher.resolve(400, 400, fps=10)
    blocker = IdentityObservation(person_id=3, track_id=3, bbox=(90, 0, 200, 300), frame=0, timestamp="t")
    visits = []
    for i in range(60):
        t = i / 10
        v, _ = watcher.update(Frame(i, "", t * 1000.0, None), [blocker] if 2.0 <= t < 3.0 else [])
        visits += v
    assert [(v.person_id, v.removed, v.added) for v in visits] == [(3, 0, 0)]


def test_parse_area_rejects_pixels():
    import pytest
    with pytest.raises(ValueError):
        parse_area("120,40,300,200", 0)
