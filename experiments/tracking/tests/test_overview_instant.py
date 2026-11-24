"""전체 카메라가 손에 든 상품을 보면 바로 센다 (fusion.overview_instant, 후속 59).

선반 방문은 선반을 가린 사람이 비켜야 끝나 개수가 늦게 온다 — 그 전에 '든 것을 봄'으로 먼저 세고,
그 시간을 덮는 방문이 끝나면 방문의 개수 · 주인으로 바꾼다."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.core.config import FusionConfig, RiskConfig  # noqa: E402
from src.core.types import Frame  # noqa: E402
from src.fusion.overview_count import HOLD_WITHIN_MS, HolderTracker  # noqa: E402
from src.fusion.shelf_events import TimeAxis  # noqa: E402
from src.risk.customers import Customer, CustomerBook, HeldItem, ShelfVisit  # noqa: E402
from src.risk.engine import RiskEngine  # noqa: E402
from src.risk.payments import PaymentFeed  # noqa: E402
from test_dwell_payment import SHELF_X, zones  # noqa: E402
from test_overview_count import STEP  # noqa: E402
from test_overview_owner import A_BOX, B_BOX, REGION, dwell_engine, held, hold, someone, stand  # noqa: E402
from test_risk import AISLE, CHECKOUT, EXIT, C, H, levels, nobody, take  # noqa: E402

A = someone(1, A_BOX, (690, 540))           # 1번 손목이 상품(가운데 700, 530) 옆
ITEM = [held("cantata", 680)]


# ---------------------------------------------------------------------- 방금 든 것 (HolderTracker.fresh)

def test_fresh_reports_a_hold_once_when_enough_frames_are_seen():
    tr = HolderTracker(min_frames=5)
    for i in range(4):
        tr.update(1000 + i * STEP, ITEM, [A], REGION)
    assert tr.fresh() == []                                                 # 4장면 — 아직
    tr.update(1000 + 4 * STEP, ITEM, [A], REGION)
    assert tr.fresh() == [(1, "cantata", 1000.0, 1000 + 4 * STEP)]          # (신원, 종류, 처음 본 때, 확정한 때)
    hold(tr, 2.0, 3.0, ITEM, [A])
    assert tr.fresh() == []                                                 # 같은 사람 · 같은 종류는 다시 안 나온다
    hold(tr, 3.0, 3.5, [held("donggwaru", 680)], [A])
    assert [(p, n) for p, n, _, _ in tr.fresh()] == [(1, "donggwaru")]      # 다른 종류는 나온다


def test_fresh_ignores_a_stray_frame_long_before():
    # 3차 N06: 1.1초에 한 장면 스치고 7.5초부터 이어서 보였다 — '처음 본 때'가 1.1초면 그 시간을 덮는 방문이 없어 두 번 세어진다
    tr = HolderTracker(min_frames=5)
    tr.update(1100.0, ITEM, [A], REGION)
    for i in range(4):
        tr.update(7500 + i * STEP, ITEM, [A], REGION)
    assert tr.fresh() == []                                                 # 다섯 장면이 HOLD_WITHIN_MS 안에 안 모였다
    tr.update(7500 + 4 * STEP, ITEM, [A], REGION)
    assert tr.fresh() == [(1, "cantata", 7500.0, 7500 + 4 * STEP)]
    assert 7500 + 4 * STEP - 7500 <= HOLD_WITHIN_MS


def test_dropped_hold_is_reported_again_only_when_it_is_picked_up_anew():
    # 4차 C13: 들어오는 길의 잘못 읽힌 든 것을 버린 뒤, 선반에서 진짜 든 것은 새 든 구간으로 다시 알린다 (후속 72)
    tr = HolderTracker(min_frames=5)
    hold(tr, 1.0, 1.5, ITEM, [A])
    assert len(tr.fresh()) == 1
    tr.drop(1, "cantata")                                                   # 판정이 안 셌다 — 선반에서 시작한 든 구간이 아니다
    hold(tr, 1.5, 2.5, ITEM, [A])
    assert tr.fresh() == []                                                 # 같은 든 구간이 이어지는 동안은 다시 안 알린다
    start = 2500 + HOLD_WITHIN_MS + 500
    hold(tr, start / 1000, start / 1000 + 0.5, ITEM, [A])                   # HOLD_WITHIN_MS 넘게 끊겼다 다시 들었다
    again = tr.fresh()
    assert [(p, n) for p, n, _, _ in again] == [(1, "cantata")] and again[0][2] >= start
    hold(tr, start / 1000 + 0.5, start / 1000 + 1.5, ITEM, [A])
    assert tr.fresh() == []                                                 # 이번에는 안 버렸다 — 전처럼 한 번만


def test_hold_that_was_not_dropped_is_never_reported_twice():
    tr = HolderTracker(min_frames=5)
    hold(tr, 1.0, 1.5, ITEM, [A])
    assert len(tr.fresh()) == 1
    hold(tr, 6.0, 6.5, ITEM, [A])                                           # 끊겼다 다시 들어도 — 이미 센 사람 · 종류
    assert tr.fresh() == []


def test_fresh_treats_merged_identities_as_one_person():
    tr = HolderTracker(min_frames=5)
    tr.resolve = lambda pid: 1 if pid == 7 else pid                        # 7번은 나중에 1번으로 합쳐졌다
    hold(tr, 1.0, 1.5, ITEM, [A])
    assert len(tr.fresh()) == 1
    hold(tr, 2.0, 2.5, ITEM, [someone(7, A_BOX, (690, 540))])
    assert tr.fresh() == []


# ---------------------------------------------------------------------- 손님 장부

def book() -> CustomerBook:
    return CustomerBook(RiskConfig(), zones())


def item_held(pid=1, name="cantata", start=20, end=40, seen_ms=2000.0) -> HeldItem:
    return HeldItem(pid, name, start, end, seen_ms)


def test_held_item_counts_as_one_item_at_once():
    b = book()
    c = b.add_held(item_held(), 2500.0)
    assert c.taken == 1 and c.unpaid == 1
    assert [(a.kind, a.name, a.count, a.evidence, a.detail) for a in c.actions] == [("HELD", "든 것을 봄", 1, "전체 카메라", "cantata")]
    assert c.count_detail() == "든 것을 봄 1개"
    assert c.actions[0].describe() == "손님 1: 든 것을 봄 — cantata (전체 카메라)"


def test_held_item_stands_in_for_the_hand_touch_just_before_it():
    c = Customer(1, 0.0, 0.0)
    c.takes.append(take(1, 15))                                             # 선반 카메라의 손-칸 접촉 — 선반 기록이 아직 없다
    assert c.taken == 1 and c.actions[0].kind == "UNVERIFIED"
    c.held.append(item_held(start=20, end=40))
    assert c.taken == 1 and [a.kind for a in c.actions] == ["HELD"]         # 같은 집기를 두 번 세지 않는다
    c.takes.append(take(1, 200))                                            # 한참 뒤의 손 접촉은 따로다
    assert c.taken == 2


def test_visit_replaces_the_held_items_seen_during_it():
    b = book()
    b.add_held(item_held(1, "cantata", seen_ms=2000.0), 2500.0)
    b.add_held(item_held(2, "donggwaru", seen_ms=3000.0), 3500.0)           # 건네받은 일행도 먼저 세어져 있었다
    b.add_held(item_held(1, "applecider", seen_ms=9000.0), 9500.0)          # 방문 밖 — 남는다
    gone = b.drop_held(1000.0, 5000.0)
    assert sorted(h.name for h in gone) == ["cantata", "donggwaru"]
    assert [h.name for h in b.customers[1].held] == ["applecider"] and b.customers[2].held == []
    b.add_shelf_visit(ShelfVisit(1, "shelf", 10, 50, 1, 0, "shelf", "[shelf] 사라짐: cantata"), 6500.0)
    assert b.customers[1].taken == 2 and b.customers[2].taken == 0          # 방문의 개수 · 주인으로 바뀌었다


def test_held_items_follow_a_merged_identity():
    b = book()
    b.add_held(item_held(7, seen_ms=2000.0), 2500.0)
    b._get(1, 2500.0)
    b.merge(7, 1)
    assert [h.name for h in b.customers[1].held] == ["cantata"] and b.customers[1].taken == 1


def test_item_paid_while_held_is_not_paid_again_when_its_visit_arrives():
    # 손에 든 것을 먼저 세어 둔 물건을 계산대에서 결제했다 — 그 뒤 방문이 들어와도 한 번만 결제된 것으로 남는다
    eng = dwell_engine()
    stand(eng, AISLE, SHELF_X, 1.0, 1.0)
    eng.book.add_held(item_held(1, seen_ms=1500.0), 1800.0)
    stand(eng, CHECKOUT, 250.0, 2.0, 2.0)                                   # 2 ~ 4초 계산대: 든 것 1개 결제
    assert eng.book.customers[1].paid_items == 1
    eng.book.drop_held(1000.0, 4000.0)
    stand(eng, AISLE, 250.0, 4.5, 1.0, visits=[ShelfVisit(1, "shelf", 10, 60, 1, 0, "shelf", "[shelf] 꺼냄: cantata", held_ms=1500.0)])
    events = stand(eng, EXIT, 250.0, 6.0, 1.0) + nobody(eng, 7.0, 4.0)
    assert levels(events) == [C]
    assert eng.book.customers[1].paid_items == 1 and [(p.method, p.items) for p in eng.book.payments] == [("dwell", 1)]


def test_customer_who_leaves_with_a_held_item_and_no_payment_is_flagged():
    eng = RiskEngine(RiskConfig(payment_mode="dwell"), zones(), PaymentFeed.empty())
    stand(eng, AISLE, SHELF_X, 1.0, 1.0)
    eng.book.add_held(item_held(1, seen_ms=1500.0), 1800.0)                 # 방문은 아직 안 끝났다 (일행이 선반을 가리고 있다)
    events = stand(eng, EXIT, 250.0, 2.5, 1.0) + nobody(eng, 3.5, 4.0)
    assert levels(events)[-1] == H and "든 것을 봄 1개" in events[-1].reason


# ---------------------------------------------------------------------- 판정 (DualJudge)

def make_judge(**cfg):
    from src.fusion.judge import DualJudge
    from src.zones.floor import FloorMap

    floor = FloorMap.load(ROOT / "zones" / "store01_dual" / "floor.yaml")
    risk = RiskEngine(RiskConfig(), zones(), PaymentFeed.empty())
    judge = DualJudge(FusionConfig(overview_count=True, overview_owner=True, **cfg), risk, floor, (1.0, 0.0),
                      TimeAxis(30.0), TimeAxis(60.0), 1920, (1920, 1080), {}, None)
    return judge, risk


def frame_at(ms: float) -> Frame:
    return Frame(index=int(ms * 30 / 1000), timestamp="", pts_ms=ms, image=None, camera_id="angle1")


def test_judge_counts_a_held_item_only_for_someone_who_was_at_the_shelf():
    judge, risk = make_judge(overview_instant=True)
    judge.fuser.note_band(4200.0, 1)                                        # 1번은 선반 앞에 있었다, 2번은 아니다
    B = someone(2, B_BOX, (910, 540))
    for i in range(5):
        ms = 5000 + i * STEP
        judge.holders.update(ms, ITEM + [held("donggwaru", 900)], [A, B], REGION)
    judge._count_held(frame_at(5000 + 4 * STEP))
    assert risk.book.customers[1].taken == 1 and 2 not in risk.book.customers
    rows = {h["pid"]: h for h in judge.held}
    assert rows[1]["counted"] and rows[1]["seen_ms"] == 5000.0 and rows[1]["until_ms"] is None
    assert not rows[2]["counted"] and rows[2]["why"] == "선반 앞에 있지 않았음"


def test_judge_counts_a_hold_that_starts_at_the_shelf_after_one_that_did_not():
    # 4차 C13: 1.8초(들어오는 길)에 잘못 읽힌 칸타타는 안 세고, 5.5초에 선반 앞에서 든 칸타타는 센다
    for per_run, taken in ((True, 1), (False, 0)):
        judge, risk = make_judge(overview_instant=True, hold_per_run=per_run)
        for i in range(5):
            judge.holders.update(1800 + i * STEP, ITEM, [A], REGION)
        judge._count_held(frame_at(1800 + 4 * STEP))                        # 선반 앞에 있은 적이 없다
        judge.fuser.note_band(5200.0, 1)
        for i in range(5):
            judge.holders.update(5500 + i * STEP, ITEM, [A], REGION)
        judge._count_held(frame_at(5500 + 4 * STEP))
        got = risk.book.customers[1].taken if 1 in risk.book.customers else 0
        assert got == taken, (per_run, got)
        assert [h["counted"] for h in judge.held] == ([False, True] if per_run else [False])


def test_judge_swaps_the_held_item_for_the_visit_that_covers_it():
    judge, risk = make_judge(overview_instant=True)
    judge.fuser.note_band(4200.0, 1)
    for i in range(5):
        judge.holders.update(5000 + i * STEP, ITEM, [A], REGION)
    judge._count_held(frame_at(5000 + 4 * STEP))
    judge._settled.append((9000.0, 12000.0))                                # 다른 시간의 방문 — 안 건드린다
    judge._settle_held()
    assert risk.book.customers[1].taken == 1 and judge.held[0]["until_ms"] is None
    judge._settled.append((3000.0, 8000.0))                                 # 그 시간을 덮는 방문이 끝났다
    judge._settle_held()
    assert risk.book.customers[1].taken == 0 and judge.held[0]["until_ms"] == 8000.0


def test_instant_count_is_off_by_default():
    assert FusionConfig().overview_instant is False and FusionConfig().overview_instant_near_seconds == 2.0
    judge, risk = make_judge()
    judge.fuser.note_band(4200.0, 1)
    frame = frame_at(5000.0)
    for i in range(6):
        judge.overview_step(frame_at(5000 + i * STEP), [], [], products=ITEM)
    assert judge.held == [] and risk.book.customers == {}
    assert frame.pts_ms == 5000.0
