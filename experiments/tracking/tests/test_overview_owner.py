"""주인을 못 정한 방문의 주인 = 전체 카메라에서 그 상품을 먼저 든 사람 (fusion.overview_owner),
전체 카메라의 '선반 앞 머묾'을 판정에서 빼는 설정 (fusion.overview_dwell), 늦게 들어온 방문의 계산대 결제 (후속 55)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.core.config import FusionConfig, RiskConfig  # noqa: E402
from src.core.types import Frame, IdentityObservation  # noqa: E402
from src.fusion.cross_view import CrossViewLinker  # noqa: E402
from src.fusion.overview_count import HolderTracker, OverviewCounter, holder_of  # noqa: E402
from src.fusion.shelf_events import SensorEvent, ShelfEventFuser, TimeAxis  # noqa: E402
from src.risk.customers import ShelfVisit  # noqa: E402
from src.risk.engine import RiskEngine  # noqa: E402
from src.risk.payments import PaymentFeed  # noqa: E402
from test_dwell_payment import SHELF_X, zones  # noqa: E402
from test_fusion import RecordingRisk, kp_with_wrist  # noqa: E402
from test_overview_count import PERSON, STEP, feed, item, shelf  # noqa: E402
from test_risk import AISLE, CHECKOUT, EXIT, C, H, at, levels, nobody, person, take  # noqa: E402

REGION = (1400.0, 560.0, 1640.0, 800.0)     # 선반 자리
A_BOX = [600, 200, 800, 900]
B_BOX = [820, 200, 1020, 900]


def held(name: str, x: float, y: float = 500) -> list:
    """선반 밖에 보인 상품 상자 (40 × 60, 대각선 72)."""
    return item(name, x, y)


def someone(pid: int, box: list, wrist: tuple[float, float] | None = None):
    return (pid, box, kp_with_wrist(*wrist) if wrist else None)


def hold(tracker: HolderTracker, start: float, end: float, items: list, people: list) -> None:
    t = start * 1000.0
    while t < end * 1000.0:
        tracker.update(t, items, people, REGION)
        t += STEP


# ---------------------------------------------------------------------- 누가 들었나

def test_holder_is_the_person_whose_wrist_is_nearest():
    it = held("cantata", 780)                               # 가운데 (800, 530) — 두 사람 상자의 경계
    a, b = someone(1, A_BOX, (790, 540)), someone(2, B_BOX, (1000, 540))
    assert holder_of(it, [a, b]) == 1
    assert holder_of(it, [someone(1, A_BOX, (790, 540)), someone(2, B_BOX, (812, 540))]) is None    # 두 손목이 비슷하게 가깝다


def test_without_a_wrist_the_holder_is_the_only_box_around_the_item():
    it = held("cantata", 680)                               # 가운데 (700, 530) — 1번 상자 안에만
    assert holder_of(it, [someone(1, A_BOX), someone(2, B_BOX)]) == 1
    assert holder_of(it, [someone(1, A_BOX), someone(2, [650, 200, 1020, 900])]) is None            # 두 상자에 걸침
    assert holder_of(it, [someone(1, A_BOX, (300, 300)), someone(2, B_BOX, (1000, 300))]) == 1      # 손목이 다 멀면 상자로
    assert holder_of(held("cantata", 100), [someone(1, A_BOX)]) is None                             # 아무도 없는 곳


def test_products_on_the_shelf_are_not_held():
    tracker = HolderTracker()
    hold(tracker, 0, 2, shelf(), [someone(1, PERSON)])       # 선반 자리 안의 상품 — 선반을 가린 사람 상자 안이어도 든 게 아니다
    assert tracker.samples == []
    tracker.update(0.0, [held("cantata", 680)], [someone(1, A_BOX)], None)      # 선반 자리를 아직 모르면 안 적는다
    assert tracker.samples == []


def test_first_holder_took_it_and_a_later_holder_was_handed_it():
    # 3차 T09: 한 사람이 꺼내 1초쯤 들고 있다 일행에게 건넸다 — 일행이 더 오래 들었지만 꺼낸 사람은 먼저 든 사람
    tracker = HolderTracker(min_frames=5)
    people = [someone(1, A_BOX), someone(2, B_BOX)]
    hold(tracker, 4.0, 5.0, [held("donggwaru", 680)], people)
    hold(tracker, 5.0, 8.0, [held("donggwaru", 900)], people)
    assert tracker.first_holders("donggwaru", 3000.0, 9000.0) == [1, 2]
    assert tracker.owners({"donggwaru": 1}, 3000.0, 9000.0) == [("donggwaru", 1)]
    assert tracker.first_holders("cantata", 3000.0, 9000.0) == []


def test_a_few_frames_are_not_holding():
    tracker = HolderTracker(min_frames=5)
    hold(tracker, 4.0, 4.0 + 4 * STEP / 1000 - 0.001, [held("donggwaru", 680)], [someone(1, A_BOX)])
    assert len(tracker.samples) == 4 and tracker.first_holders("donggwaru", 3000.0, 9000.0) == []
    assert tracker.owners({"donggwaru": 1}, 3000.0, 9000.0) == [("donggwaru", None)]


def test_someone_who_already_held_that_kind_before_the_visit_did_not_take_it_now():
    # 3차 C04: 앞 방문에서 꺼낸 손님이 같은 종류를 든 채 옆에 서 있다 — 이번 방문에서 꺼낸 사람은 새로 든 사람
    tracker = HolderTracker(min_frames=5)
    people = [someone(1, A_BOX), someone(2, B_BOX)]
    hold(tracker, 1.0, 9.0, [held("donggwaru", 680)], people)
    hold(tracker, 6.0, 9.0, [held("donggwaru", 900)], people)
    assert tracker.first_holders("donggwaru", 5000.0, 9000.0) == [2]


def test_only_holding_seen_until_the_given_time_counts():
    tracker = HolderTracker(min_frames=5)
    hold(tracker, 7.0, 9.0, [held("donggwaru", 680)], [someone(1, A_BOX)])
    assert tracker.first_holders("donggwaru", 3000.0, 6500.0) == []
    assert tracker.first_holders("donggwaru", 3000.0, 9000.0) == [1]


def test_two_of_a_kind_go_to_the_first_two_holders_and_merged_ids_are_one_person():
    tracker = HolderTracker(min_frames=5)
    people = [someone(1, A_BOX), someone(2, B_BOX)]
    hold(tracker, 4.0, 6.0, [held("donggwaru", 680)], people)
    hold(tracker, 5.0, 7.0, [held("donggwaru", 900)], people)
    assert tracker.owners({"donggwaru": 2}, 3000.0, 9000.0) == [("donggwaru", 1), ("donggwaru", 2)]
    assert tracker.owners({"donggwaru": 3}, 3000.0, 9000.0)[2] == ("donggwaru", 1)     # 든 사람보다 많이 줄었으면 첫 사람에게
    tracker.resolve = lambda pid: 1                                                    # 2번이 1번으로 합쳐졌다
    assert tracker.owners({"donggwaru": 2}, 3000.0, 9000.0) == [("donggwaru", 1), ("donggwaru", 1)]


# ---------------------------------------------------------------------- 방문의 주인

def fuser_with(plan, tracker, sensors=None, band=(7, 8)) -> ShelfEventFuser:
    cfg = FusionConfig(overview_count=True, overview_owner=True)
    counter = OverviewCounter()
    feed(counter, plan)
    fuser = ShelfEventFuser(cfg, CrossViewLinker(cfg), TimeAxis(30.0), TimeAxis(60.0), {"L2": "shelf"}, sensors, counter, tracker)
    fuser.hold_wait_ms = 1500.0
    for pid in band:                                  # 선반 앞에 두 사람 — 다른 근거로는 주인을 못 정한다
        fuser.note_band(4000.0, pid)
    return fuser


TOOK_ONE = [(0, 3, shelf(), []), (3, 6, [], [PERSON]), (6, 9, shelf(2, 2, 1), [])]
TOOK_TWO = [(0, 3, shelf(), []), (3, 6, [], [PERSON]), (6, 9, shelf(1, 2, 1), [])]       # cantata 1 + donggwaru 1
VISIT = ShelfVisit(-1, "shelf", 90, 180, 1, 0, "shelf", "[shelf] 사라짐: 1단 1번째 donggwaru")
PAIR = [someone(7, A_BOX), someone(8, B_BOX)]


def test_unknown_owner_becomes_the_first_holder():
    tracker = HolderTracker()
    hold(tracker, 4.0, 5.0, [held("donggwaru", 680)], PAIR)
    fuser = fuser_with(TOOK_ONE, tracker)
    fused = fuser.detail_visit(VISIT, 3000.0, 6000.0)
    assert fused.person_id == 7 and (fused.removed, fused.added) == (1, 0)
    assert fuser.records[-1].how == "전체 카메라에서 먼저 든 사람 (선반 앞 여러 명 또는 없음 — 모름)"
    assert fused.held_ms is not None and 4000.0 < fused.held_ms < 4500.0
    assert fuser.take_shares() == []


def test_owner_stays_unknown_when_nobody_was_seen_holding_it():
    tracker = HolderTracker()
    hold(tracker, 4.0, 5.0, [held("cantata", 680)], PAIR)            # 줄어든 종류(donggwaru)가 아니다
    fuser = fuser_with(TOOK_ONE, tracker)
    fused = fuser.detail_visit(VISIT, 3000.0, 6000.0)
    assert fused.person_id == -1 and fuser.records[-1].how == "선반 앞 여러 명 또는 없음 — 모름" and fused.held_ms is None


def test_visit_is_split_when_each_kind_was_held_by_a_different_person():
    # 3차 C12: 두 사람이 함께 선반을 가리고 하나씩 꺼냈다
    tracker = HolderTracker()
    hold(tracker, 4.0, 5.0, [held("donggwaru", 680)], PAIR)
    hold(tracker, 4.5, 5.5, [held("cantata", 900)], PAIR)
    fuser = fuser_with(TOOK_TWO, tracker)
    fused = fuser.detail_visit(VISIT, 3000.0, 6000.0)
    rest = fuser.take_shares()
    assert [(v.person_id, v.removed, v.added) for v in [fused, *rest]] == [(8, 1, 0), (7, 1, 0)]     # 종류 이름순 — cantata 먼저
    assert "꺼냄: cantata" in fused.detail and "-> 손님 8" in fused.detail and "꺼냄: donggwaru" in rest[0].detail
    assert fused.held_ms > rest[0].held_ms                                # 7번이 먼저 들었다
    assert fuser.take_shares() == []                                      # 한 번만 꺼내 간다
    record = fuser.records[-1]
    assert record.how.startswith("전체 카메라에서 먼저 든 사람 — 사람별로 나눔") and "든 사람: cantata → 손님 8, donggwaru → 손님 7" in record.detail


def test_item_nobody_held_stays_unassigned_in_a_split_visit():
    tracker = HolderTracker()
    hold(tracker, 4.0, 5.0, [held("donggwaru", 680)], PAIR)
    fuser = fuser_with(TOOK_TWO, tracker)
    fused = fuser.detail_visit(VISIT, 3000.0, 6000.0)
    rest = fuser.take_shares()
    assert [(v.person_id, v.removed) for v in [fused, *rest]] == [(7, 1), (-1, 1)]


def test_owner_found_by_other_evidence_is_kept():
    tracker = HolderTracker()
    hold(tracker, 4.0, 5.0, [held("donggwaru", 900)], PAIR)           # 든 사람은 8번
    fuser = fuser_with(TOOK_ONE, tracker, band=(7,))                  # 선반 앞에는 7번 한 명
    fused = fuser.detail_visit(VISIT, 3000.0, 6000.0)
    assert fused.person_id == 7 and fuser.records[-1].how == "선반 앞 한 명" and fused.held_ms is None
    hold(tracker, 5.0, 6.0, [held("donggwaru", 680)], PAIR)           # 7번도 들었으면 그 시각을 적어 둔다
    assert fuser_with(TOOK_ONE, tracker, band=(7,)).detail_visit(VISIT, 3000.0, 6000.0).held_ms > 5000.0


def test_visit_is_not_split_when_the_sensor_count_differs():
    tracker = HolderTracker()
    hold(tracker, 4.0, 5.0, [held("donggwaru", 680)], PAIR)
    hold(tracker, 4.5, 5.5, [held("cantata", 900)], PAIR)
    fuser = fuser_with(TOOK_TWO, tracker, [SensorEvent(4000.0, "L2", -1.0)])     # 센서는 1개, 전체 카메라는 2개
    fused = fuser.detail_visit(VISIT, 3000.0, 6000.0)
    assert (fused.person_id, fused.removed, fused.source) == (-1, 1, "sensor+vision") and fuser.take_shares() == []


def test_owner_setting_is_off_by_default_and_needs_the_count():
    from src.fusion.judge import DualJudge
    from src.zones.floor import FloorMap

    floor = FloorMap.load(ROOT / "zones" / "store01_dual" / "floor.yaml")

    def judge(**cfg):
        return DualJudge(FusionConfig(**cfg), RecordingRisk(), floor, (1.0, 0.0), TimeAxis(30.0), TimeAxis(60.0),
                         1920, (1920, 1080), {}, None)

    assert FusionConfig().overview_owner is False and FusionConfig().overview_dwell is True
    assert judge().holders is None and judge(overview_owner=True).holders is None
    both = judge(overview_count=True, overview_owner=True, overview_owner_frames=7)
    assert both.holders.min_frames == 7 and both.fuser.holders is both.holders and both.fuser.hold_wait_ms == DualJudge.VISIT_DELAY_MS


# ---------------------------------------------------------------------- 선반 앞 머묾을 판정에서 빼기

def test_overview_dwell_off_keeps_the_overview_takes_out_of_the_judge():
    from src.fusion.judge import DualJudge
    from src.zones.floor import FloorMap

    floor = FloorMap.load(ROOT / "zones" / "store01_dual" / "floor.yaml")

    def run(**cfg):
        risk = RecordingRisk()
        judge = DualJudge(FusionConfig(**cfg), risk, floor, (1.0, 0.0), TimeAxis(30.0), TimeAxis(60.0),
                          1920, (1920, 1080), {}, None)
        judge.reset()
        frame = Frame(index=0, timestamp="", pts_ms=0.0, image=None, camera_id="angle1")
        obs = IdentityObservation(person_id=3, track_id=3, bbox=(600, 200, 800, 900), frame=0, timestamp="")
        judge.overview_step(frame, [obs], [take(3)], [])
        return risk.takes

    assert len(run()) == 1                           # 기본: 선반 앞 머묾이 집기 동작으로 들어간다
    assert run(overview_dwell=False) == []


# ---------------------------------------------------------------------- 늦게 들어온 방문의 계산대 결제

def dwell_engine() -> RiskEngine:
    return RiskEngine(RiskConfig(payment_mode="dwell"), zones(), PaymentFeed.empty())


def stand(eng, foot_y, x, start, seconds, visits=()):
    out = []
    for i in range(int(round(seconds * 10))):
        ev, _ = eng.update(at(start + i * 0.1), [person(1, foot_y, x=x)], shelf_visits=visits if i == 0 else ())
        out += ev
    return out


def late_visit(held_ms):
    return ShelfVisit(1, "shelf", 10, 60, 1, 0, "shelf", "[shelf] 꺼냄: applecider", held_ms=held_ms)


def test_visit_that_arrives_after_the_checkout_stay_is_paid_if_the_item_was_in_hand_then():
    # 3차 C19: 두 사람이 함께 선반을 가려 방문이 늦게 끝났다 — 먼저 계산하고 떠난 손님의 물건이 계산 뒤에 장부에 들어온다
    eng = dwell_engine()
    stand(eng, AISLE, SHELF_X, 1.0, 1.0)
    stand(eng, CHECKOUT, 250.0, 2.0, 2.0)                                   # 2 ~ 4초 계산대 (장부에는 아직 물건이 없다)
    stand(eng, AISLE, 250.0, 4.5, 1.0, visits=[late_visit(1500.0)])         # 방문이 4.5초에 들어옴 — 물건은 1.5초에 손에 있었다
    events = stand(eng, EXIT, 250.0, 6.0, 1.0) + nobody(eng, 7.0, 4.0)
    assert levels(events) == [C]
    assert [(p.method, p.items) for p in eng.book.payments] == [("dwell", 1)]


def test_late_visit_is_not_paid_when_the_item_came_into_hand_after_the_stay():
    eng = dwell_engine()
    stand(eng, AISLE, SHELF_X, 1.0, 1.0)
    stand(eng, CHECKOUT, 250.0, 2.0, 2.0)
    stand(eng, AISLE, 250.0, 4.5, 1.0, visits=[late_visit(4200.0)])         # 계산대를 떠난 뒤에 들었다
    events = stand(eng, EXIT, 250.0, 6.0, 1.0) + nobody(eng, 7.0, 4.0)
    assert levels(events)[-1] == H and eng.book.payments == []


def test_late_visit_without_a_holding_time_is_judged_as_before():
    eng = dwell_engine()
    stand(eng, AISLE, SHELF_X, 1.0, 1.0)
    stand(eng, CHECKOUT, 250.0, 2.0, 2.0)
    stand(eng, AISLE, 250.0, 4.5, 1.0, visits=[late_visit(None)])
    events = stand(eng, EXIT, 250.0, 6.0, 1.0) + nobody(eng, 7.0, 4.0)
    assert levels(events)[-1] == H and eng.book.payments == []


def test_brief_pass_by_the_counter_does_not_pay_a_late_visit():
    eng = dwell_engine()
    stand(eng, AISLE, SHELF_X, 1.0, 1.0)
    stand(eng, CHECKOUT, 250.0, 2.0, 1.0)                                   # 1초만 스쳐 감
    stand(eng, AISLE, 250.0, 3.5, 1.0, visits=[late_visit(1500.0)])
    events = stand(eng, EXIT, 250.0, 5.0, 1.0) + nobody(eng, 6.0, 4.0)
    assert levels(events)[-1] == H


def test_visit_that_arrives_while_standing_at_the_counter_is_paid_once():
    eng = dwell_engine()
    stand(eng, AISLE, SHELF_X, 1.0, 1.0)
    stand(eng, CHECKOUT, 250.0, 2.0, 2.0)
    stand(eng, CHECKOUT, 250.0, 4.0, 1.0, visits=[late_visit(1500.0)])      # 아직 서 있다 — 평소 방식으로 결제
    events = stand(eng, EXIT, 250.0, 5.5, 1.0) + nobody(eng, 6.5, 4.0)
    assert levels(events) == [C]
    assert [(p.method, p.items) for p in eng.book.payments] == [("dwell", 1)] and eng.book.customers[1].paid_items == 1
