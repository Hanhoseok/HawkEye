"""전체 카메라 상품 개수로 선반 방문의 꺼냄 · 놓음을 센다 (src/fusion/overview_count.py, fusion.overview_count) — 영상 · 모델 없이."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import product_overview_items as po  # noqa: E402
from src.core.config import FusionConfig  # noqa: E402
from src.fusion.cross_view import CrossViewLinker  # noqa: E402
from src.fusion.overview_count import OverviewCounter, blocked, region_of  # noqa: E402
from src.fusion.shelf_events import SensorEvent, ShelfEventFuser, TimeAxis  # noqa: E402
from src.risk.customers import ShelfVisit, classify_visit  # noqa: E402

STEP = 1000 / 15           # 전체 카메라는 초당 15장을 본다 (3차)
PERSON = [1300, 300, 1520, 1000]        # 선반 자리와 겹친 사람 상자
FAR = [200, 300, 400, 900]              # 선반에서 먼 사람


def item(name: str, x: float, y: float = 650, score: float = 0.9) -> list:
    return [name, x, y, x + 40, y + 60, score]


def shelf(cantata: int = 2, donggwaru: int = 3, applecider: int = 1) -> list:
    return ([item("cantata", 1500 + 45 * i, 700) for i in range(cantata)]
            + [item("donggwaru", 1480 + 45 * i, 620) for i in range(donggwaru)]
            + [item("applecider", 1440 - 45 * i, 660) for i in range(applecider)])


def feed(counter: OverviewCounter, plan: list[tuple[float, float, list, list]]) -> None:
    """plan = [(시작 초, 끝 초, 상품 상자들, 사람 상자들), …] — 이어 붙여 장면으로 넣는다."""
    for start, end, items, people in plan:
        t = start * 1000.0
        while t < end * 1000.0:
            counter.update(t, items, people)
            t += STEP


def test_region_ignores_a_stray_box_and_blocked_means_a_person_box_overlaps_it():
    region = region_of([*shelf(), item("cantata", 300, 200)])
    assert region[0] > 1300 and region[2] < 1700                # 멀리 떨어진 상자 하나가 선반 자리를 키우지 않는다
    assert blocked([PERSON], region) and not blocked([FAR], region) and not blocked([], region)
    assert region_of([]) is None


def test_visit_change_is_count_before_minus_count_after():
    c = OverviewCounter()
    feed(c, [(0, 3, shelf(), []), (3, 6, shelf(2, 1, 1), [PERSON]), (6, 9, shelf(2, 1, 1), [])])
    seen = c.seen()
    assert [s.text() for s in seen] == ["applecider 1 cantata 2 donggwaru 3", "applecider 1 cantata 2 donggwaru 1"]
    change = c.change(2900.0, 6100.0)
    assert (change.removed, change.added) == (2, 0)             # 푸셔 줄에서 둘을 꺼냄 — 자리가 아니라 개수로 센다
    assert "donggwaru 3" in change.text() and "donggwaru 1" in change.text()


def test_counts_seen_while_a_person_blocks_the_shelf_are_not_trusted():
    c = OverviewCounter()
    # 사람이 가린 동안에만 상품이 덜 보였다 — 꺼낸 것이 아니다
    feed(c, [(0, 3, shelf(), []), (3, 6, shelf(1, 1, 0), [PERSON]), (6, 9, shelf(), [])])
    change = c.change(2900.0, 6100.0)
    assert (change.removed, change.added) == (0, 0)


def test_short_flicker_is_not_a_count():
    c = OverviewCounter()
    feed(c, [(0, 3, shelf(), []), (3, 3.3, shelf(2, 2, 1), []), (3.3, 6, shelf(), [])])
    assert {s.text() for s in c.seen()} == {"applecider 1 cantata 2 donggwaru 3"}      # 0.3초짜리 개수는 구간이 못 된다


def test_take_and_return_by_class_in_one_visit():
    c = OverviewCounter()
    feed(c, [(0, 3, shelf(2, 3, 1), []), (3, 6, [], [PERSON]), (6, 9, shelf(1, 3, 2), [])])
    change = c.change(3000.0, 6000.0)
    assert (change.removed, change.added) == (1, 1)             # 칸타타 하나 꺼내고 애플사이다 하나 놓음 — 종류별로 따로 센다


def test_one_change_is_not_counted_for_two_visits():
    c = OverviewCounter()
    feed(c, [(0, 3, shelf(), []), (3, 6, [], [PERSON]), (6, 8, shelf(2, 2, 1), []),
             (8, 10, [], [PERSON]), (10, 12, shelf(2, 2, 1), [])])
    first, second = c.change(3000.0, 6000.0), c.change(8000.0, 10000.0)
    assert (first.removed, second.removed) == (1, 0)


def test_before_count_must_have_been_seen_after_the_previous_visit():
    # 3차 N11: 손님이 선반 앞에 서서 세 방문 내내 가렸다. 마지막 방문 뒤에야 개수가 보였다 —
    # 그 '전'은 첫 방문 전의 개수뿐이라 앞 두 방문의 꺼냄을 모른다. 세 방문 모두 못 센다 (선반 지도 개수 그대로).
    c = OverviewCounter()
    feed(c, [(0, 3, shelf(), []), (3, 20, [], [PERSON]), (20, 24, shelf(), [])])
    assert c.change(3000.0, 7000.0) is None
    assert c.change(8000.0, 12000.0) is None
    assert c.change(13000.0, 20000.0) is None


def test_previous_after_count_is_the_before_count_when_the_shelf_stayed_blocked():
    c = OverviewCounter()
    # 앞 방문 뒤 개수가 6.0초에야 보이기 시작했는데 다음 방문은 5.8초에 시작했다 (두 카메라의 가림이 다르다)
    feed(c, [(0, 3, shelf(), []), (3, 6, [], [PERSON]), (6, 8, shelf(2, 2, 1), []),
             (8, 10, [], [PERSON]), (10, 12, shelf(1, 2, 1), [])])
    first, second = c.change(3000.0, 5700.0), c.change(5800.0, 10000.0)
    assert (first.removed, second.removed) == (1, 1)


def test_visit_end_may_lag_the_overview_block_by_the_edge():
    # 3차 N08-1: 선반 카메라는 방문이 18.8초에 끝났다고 봤는데 전체 카메라는 18.7초부터 가렸다 (다음 방문의 손님이 다가옴).
    # 어긋남을 안 봐주면 다음 방문(19.0 ~ 23.4초)의 꺼냄이 이 방문에 붙는다.
    plan = [(0, 16.7, shelf(), []), (16.7, 18.7, shelf(), []), (18.7, 20.8, [], [PERSON]), (20.8, 24, shelf(1, 3, 1), [])]
    c = OverviewCounter(edge_ms=1000.0)
    feed(c, plan)
    a, b = c.change(16700.0, 18800.0), c.change(19000.0, 23400.0)
    assert (a.removed, b.removed) == (0, 1)
    strict = OverviewCounter(edge_ms=0.0)
    feed(strict, plan)
    assert strict.change(16700.0, 18800.0).removed == 1         # 안 봐주면 틀린다


def test_only_frames_seen_so_far_are_used():
    c = OverviewCounter()
    feed(c, [(0, 3, shelf(), []), (3, 6, [], [PERSON])])
    assert not c.ready(6000.0) and c.change(3000.0, 6000.0) is None     # 뒤 개수를 아직 못 봤다
    c = OverviewCounter()
    feed(c, [(0, 3, shelf(), []), (3, 6, [], [PERSON]), (6, 6.4, shelf(2, 2, 1), [])])
    assert not c.ready(6000.0)                                           # 0.4초 본 것으로는 아직
    feed(c, [(6.4, 7, shelf(2, 2, 1), [])])
    assert c.ready(6000.0) and c.change(3000.0, 6000.0).removed == 1


def test_counter_without_products_at_the_start_waits_for_them():
    c = OverviewCounter()
    feed(c, [(0, 2, [], [])])
    assert c.fed and c.region is None and c.seen() == []
    feed(c, [(2, 5, shelf(), [])])
    assert c.region is not None and c.seen()[-1].text() == "applecider 1 cantata 2 donggwaru 3"


# ---------------------------------------------------------------------- 선반 사건 묶기

def fuser_with(counter, sensors=None, **config) -> ShelfEventFuser:
    cfg = FusionConfig(**config)
    return ShelfEventFuser(cfg, CrossViewLinker(cfg), TimeAxis(30.0), TimeAxis(60.0), {"L2": "shelf"}, sensors, counter)


def counted(plan) -> OverviewCounter:
    c = OverviewCounter()
    feed(c, plan)
    return c


TOOK_TWO = [(0, 3, shelf(), []), (3, 6, [], [PERSON]), (6, 9, shelf(2, 1, 1), [])]


def test_fuser_replaces_the_visit_count_with_the_overview_count():
    fuser = fuser_with(counted(TOOK_TWO))
    visit = ShelfVisit(1, "shelf", 90, 180, 1, 0, "shelf", "[shelf] 사라짐: 1단 1번째 donggwaru -> 손님 1")
    fused = fuser.detail_visit(visit, 3000.0, 6000.0)
    assert (fused.removed, fused.added, fused.source) == (2, 0, "overview+vision")
    assert fused.detail.startswith("[shelf] 전체 카메라 개수 ") and "선반 지도는 사라짐: 1단 1번째 donggwaru -> 손님 1" in fused.detail
    action = classify_visit(fused, [])
    assert action.kind == "TAKE" and action.count == 2 and action.evidence.startswith("전체+선반")
    assert action.detail.startswith("전체 카메라 개수") and "손님 1" not in action.detail


def test_fuser_turns_a_no_change_visit_into_a_take():
    # 3차 C18: 푸셔 줄에서 하나를 꺼냈는데 선반 지도는 '변화 없음' — 뒤 상품이 앞으로 와서 자리가 그대로였다
    fuser = fuser_with(counted([(0, 3, shelf(), []), (3, 6, [], [PERSON]), (6, 9, shelf(2, 2, 1), [])]))
    fused = fuser.detail_visit(ShelfVisit(1, "shelf", 90, 180, 0, 0, "shelf", "[shelf] 변화 없음 -> 손님 1"), 3000.0, 6000.0)
    assert (fused.removed, fused.added) == (1, 0) and "선반 지도는 변화 없음" in fused.detail


def test_fuser_keeps_the_detail_when_both_agree():
    fuser = fuser_with(counted(TOOK_TWO))
    visit = ShelfVisit(1, "shelf", 90, 180, 2, 0, "shelf", "[shelf] 사라짐: 1단 1번째 donggwaru, 1단 2번째 donggwaru -> 손님 1")
    fused = fuser.detail_visit(visit, 3000.0, 6000.0)
    assert fused.source == "overview+vision" and fused.detail.startswith(visit.detail) and "선반 지도와 다름" not in fused.detail


def test_fuser_keeps_the_shelf_map_count_when_the_overview_could_not_count():
    fuser = fuser_with(counted([(0, 3, shelf(), []), (3, 9, [], [PERSON])]))
    visit = ShelfVisit(1, "shelf", 90, 180, 1, 0, "shelf", "[shelf] 사라짐: 1단 1번째 donggwaru -> 손님 1")
    fused = fuser.detail_visit(visit, 3000.0, 6000.0)
    assert (fused.removed, fused.added, fused.source) == (1, 0, "vision") and "전체 카메라 개수 못 봄" in fused.detail


def test_sensor_count_comes_before_the_overview_count():
    fuser = fuser_with(counted(TOOK_TWO), [SensorEvent(4000.0, "L2", -1.0)])
    visit = ShelfVisit(1, "shelf", 90, 180, 0, 0, "shelf", "[shelf] 변화 없음 -> 손님 1")
    fused = fuser.detail_visit(visit, 3000.0, 6000.0)
    assert (fused.removed, fused.source) == (1, "sensor+vision") and "전체 카메라 개수" in fused.detail     # 적어만 둔다


def test_without_a_counter_nothing_changes():
    fuser = fuser_with(None)
    visit = ShelfVisit(1, "shelf", 90, 180, 1, 0, "shelf", "[shelf] 사라짐: 1단 1번째 donggwaru -> 손님 1")
    fused = fuser.detail_visit(visit, 3000.0, 6000.0)
    assert (fused.removed, fused.source) == (1, "vision") and "전체 카메라" not in fused.detail
    assert FusionConfig().overview_count is False


def test_sensor_record_of_a_waiting_visit_is_not_sent_out_alone():
    fuser = fuser_with(None, [SensorEvent(4000.0, "L2", -1.0)])
    assert fuser.sensor_only(8000.0, hold=[(3000.0, 6000.0, "shelf")]) == []      # 그 방문이 묶어 갈 기록
    assert len(fuser.sensor_only(8000.0, hold=[(3000.0, 6000.0, "다른 선반")])) == 1


# ---------------------------------------------------------------------- 판정이 기다리는 시간

def test_judge_waits_for_the_after_count_but_not_forever():
    from src.fusion.judge import DualJudge

    judge = DualJudge.__new__(DualJudge)
    judge.config = FusionConfig(overview_count=True, overview_count_wait_seconds=3.0)
    judge.counter = None
    assert judge._count_ready(6000.0, 6500.0)                   # 기능을 안 쓰면 안 기다린다
    judge.counter = OverviewCounter()
    assert judge._count_ready(6000.0, 6500.0)                   # 상품 탐지 결과를 한 번도 못 받았으면 안 기다린다
    feed(judge.counter, [(0, 3, shelf(), []), (3, 8, [], [PERSON])])
    assert not judge._count_ready(6000.0, 6500.0)               # 뒤 개수가 아직 없다
    assert judge._count_ready(6000.0, 9000.0)                   # 3초를 넘기면 선반 지도 개수로 넘긴다
    assert judge._count_ready(6000.0, float("inf"))
    feed(judge.counter, [(8, 9, shelf(2, 2, 1), [])])
    assert judge._count_ready(8000.0, 8100.0)


# ---------------------------------------------------------------------- 방문별로 재 보기 (scripts/product_overview_items.py --visits)

def frames_of(plan) -> list:
    out = []
    for start, end, items, people in plan:
        t = start * 1000.0
        while t < end * 1000.0:
            out.append((t, items, people))
            t += STEP
    return out


def test_visit_changes_asks_in_the_judge_order():
    plan = [(0, 3, shelf(), []), (3, 6, [], [PERSON]), (6, 8, shelf(2, 2, 1), []), (8, 20, [], [PERSON]), (20, 22, shelf(2, 1, 1), [])]
    got = po.visit_changes(frames_of(plan), [(3000.0, 6000.0), (8000.0, 12000.0)], wait_ms=3000.0)
    assert got[0].removed == 1 and got[1] is None               # 둘째 방문은 3초 안에 뒤 개수가 안 나와 못 센다
    late = po.visit_changes(frames_of(plan), [(3000.0, 6000.0), (8000.0, 12000.0)], wait_ms=10000.0)
    assert late[1].removed == 1                                 # 더 기다리면 센다


def test_after_count_seen_too_late_is_not_used_whenever_it_is_asked():
    # 방문이 6초에 끝났는데 개수는 12초에야 보였다 — 한참 뒤에 물어도(기록을 다 넣은 뒤) 그 개수는 안 쓴다
    c = OverviewCounter(late_ms=3000.0)
    feed(c, [(0, 3, shelf(), []), (3, 12, [], [PERSON]), (12, 15, shelf(2, 2, 1), [])])
    assert c.ready(6000.0) and c.change(3000.0, 6000.0) is None


def test_splits_reads_pair_to_split(tmp_path):
    path = tmp_path / "frames.csv"
    lines = ["image,pair,split", "a.jpg,C01,train", "b.jpg,C01,train", "c.jpg,C03,test"]
    path.write_text("".join(line + chr(10) for line in lines), encoding="utf-8")
    assert po.splits(path) == {"C01": "train", "C03": "test"} and po.splits(tmp_path / "없음.csv") == {}


def test_assign_puts_each_action_into_one_visit():
    visits = [(3000.0, 6000.0), (8000.0, 12000.0)]
    per, lost = po.assign([(4000.0, "take"), (6500.0, "take"), (9000.0, "return"), (20000.0, "take")], visits)
    assert per == [[2, 0], [0, 1]] and lost == 1                # 6.5초는 1초 안의 가까운 방문에, 20초는 어디에도 못 넣는다
