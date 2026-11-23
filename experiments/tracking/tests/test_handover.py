"""손님끼리 건넨 물건 — 물건 단위 장부 (fusion.handover, 후속 70, 한계 D-13).

물건마다 '지금 든 사람'과 '결제됨'이 따라다닌다. 건네면 든 사람만 바뀌고 결제됨은 물건과 함께 간다.
나갈 때 결제 안 된 물건을 **든** 손님이 경보, 그것을 건넨 손님은 확인 필요 (사용자 결정 2026-11-23).
본 사례: 4차 T09 (건넨 뒤 둘 다 결제 없이 나감) · 4차 N09 (골라서 일행에게 들려 주고 자기가 결제)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.core.config import FusionConfig, RiskConfig  # noqa: E402
from src.fusion.overview_count import HolderTracker  # noqa: E402
from src.risk.customers import UNKNOWN_KIND, CustomerBook, HeldItem, ShelfVisit  # noqa: E402
from src.risk.engine import RiskEngine  # noqa: E402
from src.risk.payments import PaymentEvent, PaymentFeed  # noqa: E402
from test_overview_count import STEP  # noqa: E402
from test_overview_instant import frame_at, make_judge  # noqa: E402
from test_overview_owner import A_BOX, B_BOX, REGION, held, hold, someone  # noqa: E402
from test_risk import AISLE, CHECKOUT, EXIT, C, H, R, W, at, engine, levels, person, zones  # noqa: E402


def visit(pid: int, start: int = 10, kinds=("cantata",)) -> ShelfVisit:
    return ShelfVisit(pid, "shelf", start, start + 20, len(kinds), 0, "shelf", "[shelf] 사라짐: " + ", ".join(kinds),
                      removed_kinds=tuple(kinds))


def book(**cfg) -> CustomerBook:
    return CustomerBook(RiskConfig(**cfg), zones())


def picked(b: CustomerBook, giver: int = 1, receiver: int = 2, kinds=("cantata",)) -> None:
    """giver 가 선반에서 집었고 receiver 는 매장에 있다."""
    b.add_shelf_visit(visit(giver, kinds=kinds), 2000.0)
    b._get(receiver, 2000.0)


# ---------------------------------------------------------------------- 손님 장부

def test_handing_over_moves_the_item_to_the_receiver():
    b = book()
    picked(b)
    h = b.hand_over(1, 2, "cantata", 5000.0, 50, 5500.0)
    a, c = b.customers[1], b.customers[2]
    assert h is not None and not h.paid and not h.void
    assert (a.taken, a.unpaid) == (0, 0) and (c.taken, c.unpaid) == (1, 1)
    assert [x.kind for x in a.actions] == ["TAKE", "GIVE"] and [x.kind for x in c.actions] == ["GET"]
    assert a.count_detail() == "집음 1개, 건넴 1개 (손님 2 에게)"
    assert c.count_detail() == "건네받음 1개 (손님 1 에게서)"
    assert a.kinds == {} and c.kinds == {"cantata": 1}
    assert c.actions[0].describe() == "손님 2: 건네받음 — cantata ← 손님 1 (전체 카메라)"


def test_nothing_is_handed_over_when_the_giver_does_not_have_it():
    b = book()
    b._get(1, 1000.0), b._get(2, 1000.0)
    assert b.hand_over(1, 2, "cantata", 5000.0, 50, 5500.0) is None            # 가진 것이 없다
    b.add_shelf_visit(visit(1, kinds=("donggwaru",)), 2000.0)
    assert b.hand_over(1, 2, "cantata", 5000.0, 50, 5500.0) is None            # 다른 종류만 가졌다
    assert b.handovers == [] and b.customers[1].taken == 1 and b.customers[2].taken == 0
    assert b.hand_over(1, 1, "donggwaru", 5000.0, 50, 5500.0) is None          # 자기에게 건넬 수는 없다


def test_an_item_of_unknown_kind_can_be_the_one_handed_over():
    b = book()
    b.add_shelf_visit(ShelfVisit(1, "shelf", 10, 30, 1, 0, "shelf", "센서 L1 -1개", source="sensor"), 2000.0)   # 종류 목록이 없는 방문
    b._get(2, 2000.0)
    assert b.customers[1].picked_kinds() == {UNKNOWN_KIND: 1}
    assert b.hand_over(1, 2, "cantata", 5000.0, 50, 5500.0) is not None
    assert b.customers[1].taken == 0 and b.customers[2].kinds == {"cantata": 1}


def test_a_paid_item_takes_its_payment_along():
    b = book()
    picked(b)
    b.pay(PaymentEvent(3.0, 1, person_id=1), 3000.0)
    h = b.hand_over(1, 2, "cantata", 5000.0, 50, 5500.0)
    a, c = b.customers[1], b.customers[2]
    assert h.paid
    assert (a.taken, a.paid, a.unpaid) == (0, 0, 0) and (c.taken, c.paid, c.unpaid) == (1, 1, 0)
    assert a.paid_items == 1 and c.paid_items == 0                             # 결제 기록은 그대로 — 옮긴 것은 따로 적는다


def test_payment_made_after_the_hand_over_moment_does_not_make_the_item_paid():
    # 건넴은 몇 초 늦게 확인된다 — 그 사이의 결제를 '건넬 때 이미 결제됨'으로 치면 안 된다
    b = book()
    picked(b)
    b.pay(PaymentEvent(6.0, 1, person_id=1), 6000.0)
    h = b.hand_over(1, 2, "cantata", 5000.0, 50, 7000.0)
    a, c = b.customers[1], b.customers[2]
    assert not h.paid and (a.taken, a.paid) == (0, 1) and c.unpaid == 1
    assert b.given_unpaid(a) == 1
    assert b.cover_from_surplus(c, givers_only=True) == 1 and c.unpaid == 0    # 건넨 손님이 낸 결제가 건넨 물건을 메운다


def test_with_an_unpaid_item_left_the_handed_item_counts_as_unpaid():
    # 2개를 집어 1개만 결제하고 1개를 건넸다 — 어느 것을 건넸는지 모른다. 결제 안 된 것으로 본다
    b = book()
    picked(b, kinds=("cantata", "cantata"))
    b.pay(PaymentEvent(3.0, 1, person_id=1), 3000.0)
    h = b.hand_over(1, 2, "cantata", 5000.0, 50, 5500.0)
    assert not h.paid and b.customers[1].unpaid == 0 and b.customers[2].unpaid == 1


def test_handing_it_back_returns_the_item():
    b = book()
    picked(b)
    b.hand_over(1, 2, "cantata", 5000.0, 50, 5500.0)
    b.hand_over(2, 1, "cantata", 8000.0, 80, 8500.0)
    a, c = b.customers[1], b.customers[2]
    assert a.taken == 1 and c.taken == 0 and b.given_unpaid(a) == 0
    assert a.count_detail() == "집음 1개, 건네받음 1개 (손님 2 에게서), 건넴 1개 (손님 2 에게)"


def test_hand_over_is_void_while_the_giver_has_no_such_item():
    # 건넨 손님의 물건이 '든 것을 봄'이었는데 선반 방문이 대신하며 지워졌다 — 그 방문이 장부에 들어올 때까지 건넴은 무효다
    b = book()
    b.add_held(HeldItem(1, "cantata", 20, 40, 2000.0), 2500.0)
    b._get(2, 2500.0)
    h = b.hand_over(1, 2, "cantata", 5000.0, 50, 5500.0)
    assert b.customers[2].taken == 1
    b.drop_held(1000.0, 4000.0)
    assert h.void and b.customers[1].taken == 0 and b.customers[2].taken == 0 and b.customers[2].got == []
    b.add_shelf_visit(visit(1), 6000.0)
    assert not h.void and b.customers[1].taken == 0 and b.customers[2].taken == 1


def test_dropping_a_hand_over_restores_the_counts():
    b = book()
    picked(b)
    h = b.hand_over(1, 2, "cantata", 5000.0, 50, 5500.0)
    b.drop_handover(h)
    a, c = b.customers[1], b.customers[2]
    assert b.handovers == [] and a.taken == 1 and c.taken == 0 and a.gave == [] and c.got == [] and c.kinds == {}


def test_hand_overs_follow_a_merged_identity():
    b = book()
    picked(b)
    b._get(7, 2000.0)
    b.hand_over(1, 7, "cantata", 5000.0, 50, 5500.0)
    b.merge(7, 2)
    assert b.customers[2].taken == 1 and b.customers[2].got[0].receiver == 2
    b.merge(2, 1)                                                              # 둘이 한 사람이었다 — 건넴이 아니다
    assert b.customers[1].taken == 1 and b.handovers[0].void


def test_items_picked_until_a_frame():
    b = book()
    b.add_shelf_visit(visit(1, start=10), 2000.0)
    b.add_shelf_visit(visit(1, start=100, kinds=("donggwaru",)), 12000.0)
    c = b.customers[1]
    assert c.picked_kinds() == {"cantata": 1, "donggwaru": 1}
    assert c.picked_kinds(50) == {"cantata": 1} and c.picked_kinds(5) == {}


def test_book_without_hand_overs_counts_as_before():
    b = book()
    picked(b)
    a = b.customers[1]
    assert (a.taken, a.paid, a.unpaid, a.gave, a.got, a.kinds) == (1, 0, 1, [], [], {})
    assert [x.kind for x in a.actions] == ["TAKE"] and a.count_detail() == "집음 1개"


# ---------------------------------------------------------------------- 위험 판정

def step(eng: RiskEngine, start: float, seconds: float, people: dict[int, float], **kw):
    """people = {손님: 발의 y} 가 seconds 초 동안 그 자리에 있다. 나온 판정을 모두 돌려준다."""
    events = []
    for i in range(int(round(seconds * 10))):
        ev, _ = eng.update(at(start + i * 0.1), [person(pid, y, x=100.0 + 60.0 * pid) for pid, y in people.items()], **kw)
        events += ev
    return events


def handed(eng: RiskEngine) -> None:
    """1번이 집어 2번에게 건넸다 (0 ~ 2초)."""
    step(eng, 0.0, 1.0, {1: AISLE, 2: AISLE})
    eng.book.add_shelf_visit(visit(1, start=2), 1000.0)
    eng.book.hand_over(1, 2, "cantata", 1500.0, 15, 1800.0)
    step(eng, 1.0, 1.0, {1: AISLE, 2: AISLE})


def last_by_person(events) -> dict:
    return {e.person_id: e.level for e in events if e.level != W}


def test_receiver_leaving_unpaid_is_the_alarm_and_the_giver_is_review():
    eng = engine()
    handed(eng)
    events = step(eng, 2.0, 1.0, {1: EXIT, 2: EXIT}) + step(eng, 3.0, 4.0, {})
    assert last_by_person(events) == {1: R, 2: H}
    assert len([e for e in events if e.level != W]) == 2                       # 한 손님에게 판정 하나
    giver = next(e for e in events if e.person_id == 1 and e.level == R)
    taker = next(e for e in events if e.person_id == 2 and e.level == H)
    assert "건넨 물건이 결제 없이 나감 (받은 사람: 손님 2) — 확인 필요" in giver.reason and giver.taken == 0
    assert "물건 1개, 결제 없음 [건네받음 1개 (손님 1 에게서)]" in taker.reason and taker.unpaid == 1


def test_giver_who_left_first_gets_review_when_the_receiver_leaves_unpaid():
    eng = engine()
    handed(eng)
    first = step(eng, 2.0, 1.0, {1: EXIT, 2: AISLE}) + step(eng, 3.0, 4.0, {2: AISLE})
    assert levels(first) == [C] and "가져간 물건 없음 [집음 1개, 건넴 1개 (손님 2 에게)]" in first[0].reason
    later = step(eng, 7.0, 1.0, {2: EXIT}) + step(eng, 8.0, 4.0, {})
    assert [(e.person_id, e.level) for e in later] == [(2, W), (2, H), (1, R)]
    assert "건넴을 반영해 다시 판정" in later[-1].reason
    assert levels(step(eng, 12.0, 1.0, {})) == []                              # 같은 판정을 되풀이하지 않는다


def test_receiver_who_pays_clears_both():
    eng = engine(payments=[PaymentEvent(3.0, 1)])
    handed(eng)
    step(eng, 2.0, 2.0, {1: AISLE, 2: CHECKOUT})
    events = step(eng, 4.0, 1.0, {1: EXIT, 2: EXIT}) + step(eng, 5.0, 4.0, {})
    assert last_by_person(events) == {1: C, 2: C}
    assert "건넨 물건이 결제 없이" not in "".join(e.reason for e in events)


def test_giver_who_pays_covers_the_item_they_handed_over():
    # 4차 N09: 골라서 일행에게 들려 주고 자기가 결제했다. 남는 결제를 매장 전체로 나누지 않아도(share_surplus_payments 꺼짐) 메운다
    eng = engine(payments=[PaymentEvent(3.0, 1)])
    handed(eng)
    step(eng, 2.0, 2.0, {1: CHECKOUT, 2: AISLE})
    events = step(eng, 4.0, 1.0, {1: EXIT, 2: EXIT}) + step(eng, 5.0, 4.0, {})
    assert last_by_person(events) == {1: C, 2: C}
    taker = next(e for e in events if e.person_id == 2 and e.level == C)
    assert "모두 결제함 (물건 1 / 결제 0 + 다른 손님의 남는 결제 1)" in taker.reason


def test_a_strangers_spare_payment_does_not_cover_without_sharing():
    eng = engine(payments=[PaymentEvent(3.0, 1)])
    handed(eng)
    step(eng, 2.0, 2.0, {1: AISLE, 2: AISLE, 3: CHECKOUT})                     # 결제한 것은 상관없는 3번
    events = step(eng, 4.0, 1.0, {1: EXIT, 2: EXIT, 3: AISLE}) + step(eng, 5.0, 4.0, {3: AISLE})
    assert last_by_person(events) == {1: R, 2: H}


def test_giver_standing_at_the_counter_pays_for_the_handed_item_in_dwell_mode():
    eng = RiskEngine(RiskConfig(payment_mode="dwell"), zones(), PaymentFeed.empty())
    handed(eng)
    step(eng, 2.0, 2.5, {1: CHECKOUT, 2: AISLE})                               # 건넨 손님이 계산대에 섰다 — 가진 것은 0개다
    assert eng.book.customers[1].paid == 1
    events = step(eng, 4.5, 1.0, {1: EXIT, 2: EXIT}) + step(eng, 5.5, 4.0, {})
    assert last_by_person(events) == {1: C, 2: C}


def test_nobody_at_the_counter_in_dwell_mode_is_still_the_alarm():
    eng = RiskEngine(RiskConfig(payment_mode="dwell"), zones(), PaymentFeed.empty())
    handed(eng)
    events = step(eng, 2.0, 1.0, {1: EXIT, 2: EXIT}) + step(eng, 3.0, 4.0, {})
    assert last_by_person(events) == {1: R, 2: H}


def test_hand_over_found_after_the_receiver_left_rejudges_them():
    eng = engine()
    step(eng, 0.0, 1.0, {1: AISLE, 2: AISLE})
    eng.book.add_shelf_visit(visit(1, start=2), 1000.0)
    first = step(eng, 1.0, 1.0, {1: AISLE, 2: EXIT}) + step(eng, 2.0, 4.0, {1: AISLE})
    assert [(e.person_id, e.level) for e in first] == [(2, C)]                 # 받은 손님이 '집은 물건 없음'으로 나갔다
    eng.book.hand_over(1, 2, "cantata", 900.0, 9, 6000.0)                      # 건넴이 뒤늦게 확인됐다
    later = step(eng, 6.0, 0.2, {1: AISLE})
    assert [(e.person_id, e.level) for e in later] == [(2, H)] and "건넴을 반영해 다시 판정" in later[0].reason
    assert levels(step(eng, 6.2, 1.0, {1: EXIT}) + step(eng, 7.2, 4.0, {})) == [R]


def test_engine_without_hand_overs_is_untouched():
    eng = engine()
    step(eng, 0.0, 1.0, {1: AISLE})
    eng.book.add_shelf_visit(visit(1, start=2), 1000.0)
    events = step(eng, 1.0, 1.0, {1: EXIT}) + step(eng, 2.0, 4.0, {})
    assert levels(events) == [W, H] and eng.before_judge is None and eng._handover_people == set()


# ---------------------------------------------------------------------- 든 구간 (HolderTracker.holds)

A = someone(1, A_BOX, (690, 540))
B = someone(2, B_BOX, (910, 540))
A_ITEM = [held("cantata", 680)]
B_ITEM = [held("cantata", 900)]


def test_holds_are_runs_of_frames_with_the_item_in_hand():
    tr = HolderTracker(min_frames=5)
    hold(tr, 1.0, 2.0, A_ITEM, [A])
    hold(tr, 6.0, 7.0, A_ITEM, [A])                                            # 4초 끊겼다 다시 들었다 — 구간이 둘이다
    runs = tr.holds()
    assert [(p, n) for _, _, p, n in runs] == [(1, "cantata"), (1, "cantata")]
    assert runs[0][0] == 1000.0 and 1900.0 < runs[0][1] < 2000.0 and runs[1][0] == 6000.0
    assert tr.count(1, "cantata", 1000.0, 1000.0 + 4 * STEP + 1.0) == 5 and tr.count(2, "cantata", 0.0, 9000.0) == 0
    assert tr.last_seen(1, "cantata", 5000.0) == runs[0][1] and tr.last_seen(1, "cantata", 500.0) is None


def test_a_few_frames_are_not_a_hold():
    tr = HolderTracker(min_frames=5)
    for i in range(4):
        tr.update(1000 + i * STEP, A_ITEM, [A], REGION)
    assert tr.holds() == []
    tr.update(1100.0 + 4 * STEP, A_ITEM, [A], REGION)
    assert len(tr.holds()) == 1


def test_hold_starts_at_the_first_dense_frames_not_at_a_stray_one():
    tr = HolderTracker(min_frames=5)
    tr.update(1000.0, A_ITEM, [A], REGION)                                     # 한 장면 스침 (1.9초 앞)
    hold(tr, 2.9, 3.5, A_ITEM, [A])
    assert [round(s) for s, _, _, _ in tr.holds()] == [2900]


# ---------------------------------------------------------------------- 건넴 찾기 (DualJudge._hand_overs)

def judge_with_pick(giver_kinds=("cantata",), **cfg):
    """1번이 선반에서 집었다 (0.3초에 시작한 방문). 2번은 매장에 있다. 두 사람의 상자는 4 ~ 6초에 닿아 있었다."""
    judge, risk = make_judge(handover=True, **cfg)
    b = risk.book
    b.add_shelf_visit(visit(1, start=10, kinds=giver_kinds), 3000.0)
    b._get(2, 1000.0)
    for c in b.customers.values():
        c.first_seen_ms, c.last_seen_ms = 1000.0, 9000.0
    for t in range(4000, 6000, 66):
        judge._note_touch(float(t), {1: A_BOX, 2: B_BOX})
    return judge, b


def found(judge, b) -> list:
    judge._hand_overs(frame_at(9000.0))
    return [(h.giver, h.receiver, h.name, h.ms) for h in b.handovers]


def test_item_seen_in_the_hand_of_someone_who_picked_nothing_was_handed_over():
    judge, b = judge_with_pick()
    hold(judge.holders, 5.0, 6.0, B_ITEM, [A, B])
    assert found(judge, b) == [(1, 2, "cantata", 5000.0)]
    assert b.customers[1].taken == 0 and b.customers[2].taken == 1
    assert [(c["say"], c["giver"]) for c in judge.hand_checks] == [("건넴", 1)]
    assert found(judge, b) == [(1, 2, "cantata", 5000.0)]                      # 다시 따져도 같은 건넴 하나


def test_holding_your_own_item_is_not_a_hand_over():
    judge, b = judge_with_pick()
    hold(judge.holders, 5.0, 6.0, A_ITEM, [A, B])                              # 집은 사람이 자기 것을 들고 있다
    assert found(judge, b) == [] and judge.hand_checks == []


def test_no_hand_over_between_people_who_never_touched():
    judge, b = judge_with_pick()
    judge._touch.clear()
    hold(judge.holders, 5.0, 6.0, B_ITEM, [A, B])
    assert found(judge, b) == []
    assert judge.hand_checks[0]["say"] == "줄 사람 없음" and "닿지 않았음" in judge.hand_checks[0]["why"]


def test_no_hand_over_when_both_hold_that_kind_at_once():
    # 둘이 함께 들고 있으면 물건이 둘이다 — 받은 쪽의 방문을 놓친 것이지 건넨 것이 아니다
    judge, b = judge_with_pick()
    hold(judge.holders, 5.0, 6.0, A_ITEM + B_ITEM, [A, B])
    assert found(judge, b) == []
    assert "그때 들고 있었음" in next(c for c in judge.hand_checks if c["pid"] == 2)["why"]


def test_a_kind_nobody_picked_is_ignored():
    judge, b = judge_with_pick(giver_kinds=("donggwaru",))
    hold(judge.holders, 5.0, 6.0, B_ITEM, [A, B])                              # 칸타타로 읽혔지만 칸타타를 집은 사람이 없다
    assert found(judge, b) == [] and judge.hand_checks[0]["say"] == "줄 사람 없음"


def test_someone_with_their_own_item_is_not_a_receiver_unless_kinds_must_match():
    judge, b = judge_with_pick()
    b.add_shelf_visit(visit(2, start=10, kinds=("donggwaru",)), 3000.0)        # 2번도 다른 종류를 집었다
    hold(judge.holders, 5.0, 6.0, B_ITEM, [A, B])
    assert found(judge, b) == []                                               # 든 상품이 다른 종류로 읽힌 것일 수 있다 (4차 C19)
    judge, b = judge_with_pick(handover_own_any=False)
    b.add_shelf_visit(visit(2, start=10, kinds=("donggwaru",)), 3000.0)
    hold(judge.holders, 5.0, 6.0, B_ITEM, [A, B])
    assert found(judge, b) == [(1, 2, "cantata", 5000.0)]


def test_waits_while_a_shelf_visit_that_could_explain_it_is_open():
    judge, b = judge_with_pick()
    hold(judge.holders, 5.0, 6.0, B_ITEM, [A, B])
    judge._open_visits = [(4000.0, "shelf")]                                   # 그 순간을 덮는 방문이 아직 안 끝났다
    assert found(judge, b) == [] and judge.hand_checks[0]["say"] == "기다림"
    judge._open_visits = []
    assert found(judge, b) == [(1, 2, "cantata", 5000.0)]


def test_hand_over_is_taken_back_when_the_receiver_turns_out_to_have_picked_it():
    judge, b = judge_with_pick()
    hold(judge.holders, 5.0, 6.0, B_ITEM, [A, B])
    assert len(found(judge, b)) == 1
    b.add_shelf_visit(visit(2, start=20), 9500.0)                              # 늦게 들어온 방문 — 2번이 직접 집었다
    assert found(judge, b) == []
    assert b.customers[1].taken == 1 and b.customers[2].taken == 1 and b.customers[2].got == []


def test_item_picked_after_the_hold_cannot_have_been_handed_over():
    judge, risk = make_judge(handover=True)
    b = risk.book
    b.add_shelf_visit(visit(1, start=700), 30000.0)                            # 1번은 한참 뒤에 집었다
    b._get(2, 1000.0)
    for c in b.customers.values():
        c.first_seen_ms, c.last_seen_ms = 1000.0, 40000.0
    judge._note_touch(5000.0, {1: A_BOX, 2: B_BOX})
    hold(judge.holders, 5.0, 6.0, B_ITEM, [A, B])
    assert found(judge, b) == []


def test_handing_back_is_a_second_hand_over():
    judge, b = judge_with_pick()
    hold(judge.holders, 5.0, 6.0, B_ITEM, [A, B])
    for t in range(9000, 11000, 66):
        judge._note_touch(float(t), {1: A_BOX, 2: B_BOX})
    hold(judge.holders, 10.0, 11.0, A_ITEM, [A, B])                            # 4초 뒤 1번이 다시 들고 있다
    for c in b.customers.values():
        c.last_seen_ms = 12000.0
    judge._hand_overs(frame_at(12000.0))
    assert [(h.giver, h.receiver) for h in b.handovers] == [(1, 2), (2, 1)]
    assert b.customers[1].taken == 1 and b.customers[2].taken == 0


def test_hand_over_search_is_off_by_default_and_needs_the_holder_record():
    assert FusionConfig().handover is False and FusionConfig().handover_own_any is True
    judge, risk = make_judge()
    assert judge.handover_on is False and risk.before_judge is None
    from src.fusion.judge import DualJudge
    from src.fusion.shelf_events import TimeAxis
    from src.zones.floor import FloorMap

    floor = FloorMap.load(ROOT / "zones" / "store01_dual" / "floor.yaml")
    risk = RiskEngine(RiskConfig(), zones(), PaymentFeed.empty())
    judge = DualJudge(FusionConfig(handover=True), risk, floor, (1.0, 0.0), TimeAxis(30.0), TimeAxis(60.0), 1920, (1920, 1080),
                      {}, None)
    assert judge.handover_on is False                                          # 전체 카메라가 든 사람을 안 보면 찾을 수 없다
    judge, risk = make_judge(handover=True)
    assert judge.handover_on is True and risk.before_judge == judge._hand_overs
