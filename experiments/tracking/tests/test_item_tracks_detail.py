"""선반 카메라의 상품 상자도 물건 단위로 따라가 같은 장부에 넣는다 (fusion.item_tracks_detail, 후속 74).

기준은 전체 카메라와 같다 (물건의 시작과 끝은 선반). 두 카메라가 본 것은 한 장부로 묶는다:
같은 손님의 물건은 본 순서대로 같은 물건이고, 개수는 더 많이 본 카메라의 개수다.
'선반 위에 있다'는 선반 자리(네모 + 여유)가 아니라 물건이 놓여 있던 자리와 겹치는지로 따진다.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.core.config import FusionConfig  # noqa: E402
from src.fusion.item_tracks import ON_PLACE, ItemTracker, covered  # noqa: E402
from src.fusion.overview_count import OverviewCounter, iou, places_of  # noqa: E402
from test_item_tracks import OUT, P, REGION, SHELF, Run, box, feed  # noqa: E402
from test_overview_count import STEP  # noqa: E402
from test_overview_instant import frame_at, make_judge  # noqa: E402
from test_swap_repair import person  # noqa: E402

D = (11, P[1], None)                # 선반 카메라 화면의 사람 11 — 두 화면 짝으로 손님 1
LINK = {11: 1, 12: 2}


def two_cameras(link: dict | None = None, **cfg):
    judge, risk = make_judge(overview_instant=True, item_tracks=True, item_tracks_detail=True, **cfg)
    table = LINK if link is None else link
    judge.linker.owner = lambda d, start, end=None: table.get(d)
    return judge, risk


def feed_shelf(judge, run: Run, items: list, people: list | None = None, times: int = 1) -> None:
    """선반 카메라 한 장면 + 그 뒤의 전체 카메라 장면 (세는 것은 전체 카메라 장면에서 한다)."""
    for _ in range(times):
        judge.items_detail.update(run.ms, items, [D] if people is None else people, REGION)
        judge._item_wait += judge.items_detail.fresh()
        judge._count_held(frame_at(run.ms))
        run.ms += STEP


def pick_on_shelf_camera(judge, run: Run, name: str = "cantata", y: float = 650) -> None:
    feed_shelf(judge, run, [box(name, (SHELF[0], y))], times=6)
    for x, _ in OUT:
        feed_shelf(judge, run, [box(name, (x, y))])


def pick_on_overview(judge, run: Run, name: str = "cantata", y: float = 650) -> None:
    feed(judge, run, [box(name, (SHELF[0], y))], times=6)
    for x, _ in OUT:
        feed(judge, run, [box(name, (x, y))])


# ---------------------------------------------------------------------- 선반 위 = 물건이 놓여 있던 자리

def test_places_are_the_boxes_the_shelf_region_was_made_from():
    boxes = [box("cantata", (1450, 650)), box("cantata", (1452, 651)), box("donggwaru", (1500, 650)), box("applecider", (300, 300))]
    places = places_of(boxes)
    assert len(places) == 2                                   # 같은 자리 둘은 하나로, 멀리 떨어진 상자(스티커)는 뺀다
    counter = OverviewCounter()
    for i in range(20):
        counter.update(1000.0 + i * STEP, boxes[:3], [])
    assert counter.region is not None and len(counter.places) == 2


def test_item_hovering_in_the_empty_part_of_the_shelf_region_is_not_a_shelf_item():
    # 4차 N10-2 선반 카메라: 손에 든 캔이 선반 자리의 여유 띠에 머물다 나와 '선반에서 나온 새 물건'으로 한 번 더 세어졌다
    places = [box("cantata", SHELF)]
    hover = (1420, 760)                                       # 선반 자리(네모) 안이지만 물건 자리와는 안 겹친다
    def run(use_places: bool) -> list[str]:
        r = Run().pick()                                      # 물건 하나를 집어 들고 있다
        for at in [hover] * 6 + [(1350, 760), (1300, 760), (1250, 760), (1200, 760), (1150, 760), (1150, 760)]:
            r.tracker.update(r.ms, [box("cantata", at)], [P], REGION, places if use_places else None)
            r.events += r.tracker.fresh()
            r.ms += STEP
        return r.kinds()
    assert run(False) == ["pick", "pick"]                     # 네모로 따지면: 선반에서 나온 새 물건이 하나 더 생긴다
    assert run(True) == ["pick"]                              # 물건 자리로 따지면: 들고 있던 그 물건 그대로


def test_item_put_back_on_a_place_is_a_return():
    places = [box("cantata", SHELF)]
    r = Run()
    for at in [SHELF] * 6 + OUT + [(1250, 650), (1350, 650)] + [SHELF] * 6:
        r.tracker.update(r.ms, [box("cantata", at)], [P], REGION, places)
        r.events += r.tracker.fresh()
        r.ms += STEP
    assert r.kinds() == ["pick", "back"]


WIDE = (1, [1000, 200, 1700, 900], None)                      # 선반 앞에 서서 선반을 가린 손님 1
TIP = ["donggwaru", 1440, 620, 1460, 640, 0.8]                # 진열 자리(1430 ~ 1470, 620 ~ 680) 안의 작은 조각


def test_covered_is_the_share_of_the_box_inside_the_places():
    places = [box("applecider", SHELF), box("donggwaru", (SHELF[0] + 40, SHELF[1]))]      # 나란한 두 자리
    assert covered(TIP, places) == 1.0 and iou(TIP, places[0]) < ON_PLACE     # 다 들어 있어도 겹침은 작다
    assert covered(["donggwaru", 1460, 620, 1480, 640, 0.8], places) == 1.0  # 두 자리에 걸쳐 있다 — 합친 자리로 따진다
    assert covered(["donggwaru", 1420, 620, 1440, 640, 0.8], places) == 0.5  # 반은 자리 밖
    assert covered(box("cantata", (1350, 650)), places) == 0.0 and covered(TIP, []) == 0.0


def test_tip_of_a_shelf_item_showing_inside_its_place_is_not_a_pick():
    # 3차 N11: 손님 몸에 가린 진열 상자의 위쪽 끝만 드러났다 — 자리와의 겹침이 작아 '선반 밖'이 되고, 선반 옆에서 나타난 새 물건으로 세어졌다
    places = [box("applecider", SHELF)]
    def run(item: list) -> list[str]:
        r = Run()
        for _ in range(8):
            r.tracker.update(r.ms, [item], [WIDE], REGION, places)
            r.events += r.tracker.fresh()
            r.ms += STEP
        return r.kinds()
    assert run(TIP) == []                                                   # 자리 안에 든 조각: 든 것으로 안 친다
    assert run(["donggwaru", 1380, 620, 1400, 640, 0.8]) == ["pick"]        # 같은 크기라도 자리 밖(선반 바로 옆)이면 집은 물건이다


# ---------------------------------------------------------------------- 두 카메라를 한 장부로

def test_off_by_default_and_needs_item_tracks():
    assert FusionConfig().item_tracks_detail is False
    judge, _ = make_judge(overview_instant=True, item_tracks=True)
    assert judge.items_detail is None
    judge, _ = make_judge(overview_instant=True, item_tracks_detail=True)
    assert judge.items_detail is None


def test_item_only_the_shelf_camera_saw_is_counted():
    judge, risk = two_cameras()
    pick_on_shelf_camera(judge, Run())
    assert risk.book.customers[1].taken == 1
    assert [(h["pid"], h["name"], h["counted"], h["cam"]) for h in judge.held] == [(1, "cantata", True, "detail")]


def test_same_item_seen_by_both_cameras_is_one_item():
    judge, risk = two_cameras()
    run = Run()
    pick_on_overview(judge, run)
    pick_on_shelf_camera(judge, run)                          # 같은 손님의 첫 물건을 선반 카메라도 봤다
    assert risk.book.customers[1].taken == 1
    assert [(h["counted"], h["cam"]) for h in judge.held] == [(True, "both")]


def test_count_is_the_larger_of_the_two_cameras():
    judge, risk = two_cameras()
    run = Run()
    pick_on_overview(judge, run)
    pick_on_shelf_camera(judge, run)
    pick_on_shelf_camera(judge, run, y=720)                   # 선반 카메라만 본 둘째 물건
    assert risk.book.customers[1].taken == 2
    assert [h["cam"] for h in judge.held] == ["both", "detail"]
    pick_on_overview(judge, run, y=720)                       # 전체 카메라가 뒤늦게 둘째를 봐도 그 물건이다
    assert risk.book.customers[1].taken == 2 and [h["cam"] for h in judge.held] == ["both", "both"]


def test_items_of_different_customers_are_not_bound():
    judge, risk = two_cameras()
    run = Run()
    pick_on_overview(judge, run)                              # 손님 1
    other = (12, P[1], None)                                  # 선반 카메라 화면의 12 = 손님 2
    feed_shelf(judge, run, [box("cantata", SHELF)], [other], times=6)
    for x, _ in OUT:
        feed_shelf(judge, run, [box("cantata", (x, 650))], [other])
    assert risk.book.customers[1].taken == 1 and risk.book.customers[2].taken == 1


def test_kind_is_what_both_cameras_read_most():
    judge, risk = two_cameras()
    run = Run()
    pick_on_overview(judge, run, "cantata")                   # 전체 카메라: 칸타타로 6장면
    pick_on_shelf_camera(judge, run, "donggwaru")
    feed_shelf(judge, run, [box("donggwaru", (1150, 650))], times=10)   # 선반 카메라: 동과루로 훨씬 많이
    assert risk.book.customers[1].taken == 1
    assert judge.held[0]["name"] == "donggwaru" and [h.name for h in risk.book.customers[1].held] == ["donggwaru"]


BACK = [(1150, 650), (1200, 650), (1250, 650), (1300, 650), (1350, 650)] + [SHELF] * 5   # 든 물건을 선반에 도로 놓는 길


def test_return_takes_the_item_back_only_when_no_camera_still_holds_it():
    # 4차 N08-1 선반 카메라: 막 꺼낸 캔이 선반 앞에 머문 것을 '놓음'으로 봤다 — 전체 카메라는 계속 들고 있는 것으로 본다
    judge, risk = two_cameras()
    run = Run()
    pick_on_overview(judge, run)
    pick_on_shelf_camera(judge, run)
    for at in BACK:
        feed_shelf(judge, run, [box("cantata", at)])          # 선반 카메라만 도로 놓는 것을 봤다
    assert risk.book.customers[1].taken == 1 and judge.held[0]["until_ms"] is None
    assert judge.held[0]["holding"] == ["overview"]
    for at in BACK:
        feed(judge, run, [box("cantata", at)])                # 전체 카메라도 봤다 — 이제 놓은 것이다
    assert risk.book.customers[1].taken == 0
    assert judge.held[0]["until_ms"] is not None and judge.held[0]["end"] == "놓음"


def test_return_of_an_item_only_one_camera_saw_takes_it_back():
    judge, risk = two_cameras()
    run = Run()
    pick_on_shelf_camera(judge, run)
    for at in BACK:
        feed_shelf(judge, run, [box("cantata", at)])
    assert risk.book.customers[1].taken == 0 and judge.held[0]["end"] == "놓음"


def test_camera_that_saw_a_return_binds_its_next_item_to_the_same_row():
    judge, risk = two_cameras()
    run = Run()
    pick_on_overview(judge, run)
    pick_on_shelf_camera(judge, run)
    for at in BACK:
        feed_shelf(judge, run, [box("cantata", at)])
    pick_on_shelf_camera(judge, run)                          # 선반 카메라가 다시 든 것을 봤다 — 전체 카메라가 들고 있다고 보는 그 물건
    assert risk.book.customers[1].taken == 1 and len([h for h in judge.held if h["counted"]]) == 1


def test_box_not_from_the_shelf_is_not_logged_when_the_other_camera_counted_that_customers_item():
    judge, risk = two_cameras()
    run = Run()
    pick_on_overview(judge, run)
    feed_shelf(judge, run, [box("cantata", (1150, 650))], times=8)   # 선반 카메라는 나오는 순간을 못 보고 손에 든 것만 봤다
    assert risk.book.customers[1].taken == 1 and len(judge.held) == 1


def test_shelf_camera_person_without_a_link_waits_then_is_not_counted():
    link: dict = {}
    judge, risk = two_cameras(link)
    run = Run()
    pick_on_shelf_camera(judge, run)
    assert judge.held == [] and len(judge._item_wait) == 1    # 아직 손님에게 못 이었다 — 기다린다
    link[11] = 1
    feed_shelf(judge, run, [box("cantata", (1150, 650))])
    assert risk.book.customers[1].taken == 1 and judge._item_wait == []

    link2: dict = {}
    judge, risk = two_cameras(link2)
    run = Run()
    pick_on_shelf_camera(judge, run)
    run.ms += judge.visit_delay_ms + 1000.0
    feed_shelf(judge, run, [])
    assert [(h["counted"], h["why"]) for h in judge.held] == [(False, "선반 카메라의 사람을 손님에게 못 이음")]
    assert not risk.book.customers or all(c.taken == 0 for c in risk.book.customers.values())


def test_item_that_arrives_after_its_visit_was_handed_over_is_not_counted_again():
    judge, risk = two_cameras()
    run = Run(start_ms=5000.0)
    judge._settled.append((3000.0, 9000.0))                   # 그 시간을 덮는 방문을 먼저 넘겼다 (방문의 개수가 이미 세었다)
    judge._settle_held()
    pick_on_shelf_camera(judge, run)
    assert 1 not in risk.book.customers or risk.book.customers[1].taken == 0
    assert [(h["counted"], h["until_ms"]) for h in judge.held] == [(True, 9000.0)]


def settle(judge, start_ms: float, end_ms: float, put_back: bool = False) -> None:
    """그 시간의 선반 방문을 넘긴다 — 방문의 개수가 그 사이에 본 든 것을 대신한다. put_back = 방문이 돌려놓음을 확인했다."""
    judge._settled.append((start_ms, end_ms))
    if put_back:
        judge._put_back.append((start_ms, end_ms))
    judge._settle_held()


def test_return_seen_after_the_visit_took_the_item_over_is_not_a_new_item_when_seen_again():
    # 3차 T08 선반 카메라: 방문이 확정한 캔을 든 채 선반 앞에 다시 서자 '놓음 → 다시 집음'으로 봤다.
    # 놓음은 방문 개수에서 안 빼는데 다시 집음만 더하면 1개가 늘어난다
    judge, risk = two_cameras()
    run = Run(start_ms=5000.0)
    pick_on_shelf_camera(judge, run)
    settle(judge, 3000.0, run.ms)                             # 방문을 넘겼다 — 든 것 줄은 방문의 개수로 바뀐다
    assert risk.book.customers[1].taken == 0 and judge.held[0]["until_ms"] is not None
    run.ms += judge.visit_delay_ms + 2000.0                   # 그 방문이 덮는 시간이 지난 뒤
    for at in BACK:
        feed_shelf(judge, run, [box("cantata", at)])          # 선반 카메라가 '놓음'으로 봤다
    assert judge.held[0]["left"].keys() == {"detail"} and judge.held[0].get("end") is None
    pick_on_shelf_camera(judge, run)                          # 다시 든 것을 봤다 — 방문이 이미 센 그 물건이다
    assert risk.book.customers[1].taken == 0 and len(judge.held) == 1
    assert judge.held[0]["left"] == {} and judge.held[0]["holding"] == ["detail"]
    pick_on_shelf_camera(judge, run, y=720)                   # 그 뒤의 다른 물건은 새 물건이다
    assert risk.book.customers[1].taken == 1 and len(judge.held) == 2


def test_return_one_camera_saw_before_the_visit_was_handed_over_still_binds_after_it():
    # 4차 N08-2 선반 카메라: 막 꺼낸 캔을 '놓음'으로 봤고 (전체 카메라는 들고 있다고 봄), 방문을 넘긴 뒤 그 캔을 새로 집은 것으로 봤다
    judge, risk = two_cameras()
    run = Run(start_ms=5000.0)
    pick_on_overview(judge, run)
    pick_on_shelf_camera(judge, run)
    for at in BACK:
        feed_shelf(judge, run, [box("cantata", at)])
    assert risk.book.customers[1].taken == 1 and judge.held[0]["left"].keys() == {"detail"}
    settle(judge, 3000.0, run.ms)
    run.ms += judge.visit_delay_ms + 2000.0
    pick_on_shelf_camera(judge, run)
    assert risk.book.customers[1].taken == 0 and len(judge.held) == 1   # 든 것은 그대로 0 — 방문의 개수가 그 물건이다


def test_return_the_visit_confirmed_is_real_so_the_next_item_is_new():
    judge, risk = two_cameras()
    run = Run(start_ms=5000.0)
    pick_on_shelf_camera(judge, run)
    settle(judge, 3000.0, run.ms)
    run.ms += judge.visit_delay_ms + 2000.0
    start = run.ms
    for at in BACK:
        feed_shelf(judge, run, [box("cantata", at)])
    settle(judge, start - 1000.0, run.ms, put_back=True)      # 방문 (센서 · 선반 지도) 이 돌려놓음을 확인했다 — 그 놓음은 진짜다
    assert judge.held[0]["left"] == {}
    run.ms += judge.visit_delay_ms + 2000.0
    pick_on_shelf_camera(judge, run)
    assert risk.book.customers[1].taken == 1 and len(judge.held) == 2


def test_single_camera_return_after_the_visit_then_seen_again_is_the_same_item():
    judge, risk = make_judge(overview_instant=True, item_tracks=True)   # 선반 카메라 없이도 같은 규칙이다
    run = Run(start_ms=5000.0)
    pick_on_overview(judge, run)
    settle(judge, 3000.0, run.ms)
    run.ms += judge.visit_delay_ms + 2000.0
    for at in BACK:
        feed(judge, run, [box("cantata", at)])
    pick_on_overview(judge, run)
    assert risk.book.customers[1].taken == 0 and len(judge.held) == 1


def test_detail_step_feeds_the_shelf_camera_tracker():
    judge, risk = two_cameras()
    judge.linker.owner = lambda d, start, end=None: 1
    stock = [box("cantata", SHELF), box("donggwaru", (1500, 650))]
    ms = 1000.0
    def step(items: list) -> None:
        nonlocal ms
        judge.detail_step(ms, [person(1, P[1])], [], [], products=items)
        judge._count_held(frame_at(ms))
        ms += STEP
    for _ in range(20):                                       # 처음 1초: 선반 자리 · 물건 자리를 잡는다
        step(stock)
    assert judge.detail_region.region is not None and len(judge.detail_region.places) == 2
    for x, _ in OUT:
        step([box("cantata", (x, 650)), stock[1]])
    assert risk.book.customers[1].taken == 1 and judge.held[0]["cam"] == "detail"


def test_tracker_without_places_keeps_the_region_rule():
    r = Run(ItemTracker()).pick()
    assert r.kinds() == ["pick"]
