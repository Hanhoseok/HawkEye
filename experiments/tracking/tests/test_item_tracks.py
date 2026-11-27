"""물건 단위 추적 (src/fusion/item_tracks.py, fusion.item_tracks, 후속 73).

기준: 물건의 시작과 끝은 선반이다. 선반 자리에 머물던 상자가 나와 손에 들리면 집음, 들어가 멈춰 머물면 놓음.
선반 밖에서 처음 나타난 상자는 새 물건이 아니다. 종류는 개수에 영향이 없다.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.core.config import FusionConfig  # noqa: E402
import math  # noqa: E402

from src.fusion.item_tracks import LINK_GAP_MS, LINK_STEP_MS, ItemTracker, Track, _minmod, changed, nested, shelf_distance  # noqa: E402
from src.risk.customers import HeldItem  # noqa: E402
from test_overview_count import STEP  # noqa: E402
from test_overview_instant import book, frame_at, make_judge  # noqa: E402

REGION = (1400.0, 560.0, 1640.0, 800.0)     # 선반 자리
P = (1, [1000, 200, 1380, 900], None)       # 선반 왼쪽에 선 손님 1 (관절 없음 — 상품 가운데가 이 상자 안이면 이 손님이 든 것)
Q = (2, [200, 200, 420, 900], None)         # 멀리 선 손님 2
SHELF = (1450, 650)                         # 선반 위의 자리
OUT = [(1350, 650), (1300, 650), (1250, 650), (1200, 650), (1150, 650), (1150, 650)]   # 선반에서 손으로 나오는 길 (6장면)


def box(name: str, at: tuple[float, float], score: float = 0.9) -> list:
    """상품 상자 40 × 60 (대각선 72)."""
    return [name, at[0] - 20, at[1] - 30, at[0] + 20, at[1] + 30, score]


class Run:
    """장면을 차례로 넣고 생긴 일을 모은다."""

    def __init__(self, tracker: ItemTracker | None = None, start_ms: float = 1000.0) -> None:
        self.tracker, self.ms, self.events = tracker or ItemTracker(), start_ms, []

    def see(self, items: list, people: list | None = None, times: int = 1) -> "Run":
        for _ in range(times):
            self.tracker.update(self.ms, items, [P] if people is None else people, REGION)
            self.events += self.tracker.fresh()
            self.ms += STEP
        return self

    def wait(self, ms: float) -> "Run":
        self.ms += ms
        return self

    def kinds(self) -> list[str]:
        return [e.kind for e in self.events]

    def pick(self, name: str = "cantata", shelf: tuple[float, float] = SHELF, dy: float = 0.0) -> "Run":
        """선반 위에 있던 상품을 손님 1 이 집어 든다."""
        self.see([box(name, shelf)], times=6)
        for x, y in OUT:
            self.see([box(name, (x, y + dy))])
        return self


def test_shelf_distance_is_zero_inside():
    assert shelf_distance(REGION, (1500, 600)) == 0 and shelf_distance(REGION, (1400, 600)) == 0
    assert shelf_distance(REGION, (1300, 600)) == 100 and shelf_distance(REGION, (1300, 500)) > 100


# ---------------------------------------------------------------------- 집음 · 놓음

def test_item_that_leaves_the_shelf_into_a_hand_is_one_pick():
    run = Run().pick()
    assert run.kinds() == ["pick"]
    e = run.events[0]
    assert (e.pid, e.name, e.item) == (1, "cantata", 0)
    assert [i.name for i in run.tracker.carried(1)] == ["cantata"] and run.tracker.carried(2) == []


def test_item_still_on_the_shelf_is_not_a_pick():
    # 손님 상자가 선반을 덮어도 선반 자리 안의 상자는 든 것이 아니다
    run = Run().see([box("cantata", SHELF)], [(1, [1000, 200, 1700, 900], None)], times=20)
    assert run.events == []


def test_item_put_back_on_the_shelf_is_returned():
    run = Run().pick()
    for x in (1200, 1250, 1300, 1350, 1400):
        run.see([box("cantata", (x, 650))])
    run.see([box("cantata", SHELF)], times=5)
    assert run.kinds() == ["pick", "back"] and run.tracker.carried(1) == []
    assert run.events[1].item == 0
    for at in OUT:                                   # 다시 집으면 새 물건이다
        run.see([box("cantata", at)])
    assert run.kinds() == ["pick", "back", "pick"] and run.events[2].item == 1


def test_one_frame_over_the_shelf_after_a_pick_is_not_a_return():
    # 집기 전에 선반에 있던 장면은 놓음에 안 친다 — 든 물건이 한 장면 선반 자리와 겹친 것은 놓음이 아니다 (4차 N05, 후속 76)
    run = Run().pick()
    for x in (1200, 1250, 1300, 1350, 1400, 1350, 1300):     # 1400 = 선반 자리 가장자리 (한 장면)
        run.see([box("cantata", (x, 650))])
    assert run.kinds() == ["pick"] and [i.name for i in run.tracker.carried(1)] == ["cantata"]


def test_return_needs_the_frames_after_the_item_left_the_shelf():
    run = Run().pick()
    for x in (1200, 1250, 1300, 1350):
        run.see([box("cantata", (x, 650))])
    run.see([box("cantata", (1400, 650))], times=4)           # 선반 자리에 4장면 — 아직 놓음이 아니다
    assert run.kinds() == ["pick"]
    run.see([box("cantata", (1400, 650))])                    # 5장면째
    assert run.kinds() == ["pick", "back"]


def test_two_items_taken_together_are_two_picks():
    run = Run().see([box("donggwaru", (1450, 620)), box("donggwaru", (1450, 720))], times=6)
    for x, _ in OUT:
        run.see([box("donggwaru", (x, 620)), box("donggwaru", (x, 720))])
    assert run.kinds() == ["pick", "pick"]            # 같은 종류 둘도 둘로 센다 (사람 · 종류마다 1개가 아니다)
    assert len(run.tracker.carried(1)) == 2


# ---------------------------------------------------------------------- 선반 밖에서 처음 나타난 상자

def test_box_first_seen_away_from_the_shelf_is_not_counted():
    # 4차: 문 옆 유리의 스티커가 애플사이다로 읽힌다 — 그 앞을 지나는 손님 손에 붙지만 선반에서 나온 상자가 아니다
    sticker = box("applecider", (300, 300))
    run = Run().see([sticker], [], times=10).see([sticker], [Q], times=8)
    assert run.kinds() == ["reject"]
    e = run.events[0]
    assert (e.pid, e.name, e.item) == (2, "applecider", None) and "선반" in e.why
    assert run.tracker.carried(2) == []
    run.see([sticker], [], times=5).see([sticker], [Q], times=8)      # 다시 지나가도 또 적지 않는다
    assert run.kinds() == ["reject"]


def test_sticker_is_not_counted_even_right_after_a_pick():
    # 방금 선반에서 집은 손님이 스티커 앞을 지난다 — '선반 앞에 있던 사람'으로 거르면 세어졌다 (4차 T02 · S1 · T10)
    run = Run().pick()
    sticker = box("applecider", (1050, 300))
    run.see([box("cantata", (1150, 650)), sticker], times=8)
    assert run.kinds() == ["pick", "reject"] and len(run.tracker.carried(1)) == 1   # '안 셈'으로만 적는다


def test_far_box_is_not_linked_to_a_shelf_item():
    # 선반 위 상자가 가려진 사이 멀리 나타난 상자와 이으면 '선반에서 나온 물건'이 된다 (4차 T06 · C14)
    run = Run().see([box("donggwaru", SHELF)], times=6).see([box("donggwaru", (1100, 500))], times=6)
    assert run.kinds() == ["reject"]


def test_box_first_seen_right_beside_the_shelf_is_a_pick():
    # 나오는 순간 손에 가려 줄이 끊겼다 — 선반 바로 옆에서 나타났으면 집은 것으로 본다
    run = Run().see([box("cantata", (1340, 650))], times=6)
    assert run.kinds() == ["pick"] and run.tracker.items[0].origin == "near"


# ---------------------------------------------------------------------- 다시 보인 물건

def test_item_seen_again_after_a_break_is_not_counted_twice():
    run = Run().pick().wait(LINK_GAP_MS + 500)
    run.see([box("cantata", (1100, 400))], times=8)   # 주머니에서 꺼내 계산대에 — 선반 밖에서 다시 보였다
    assert run.kinds() == ["pick"] and len(run.tracker.carried(1)) == 1


def test_box_beside_the_shelf_is_the_hidden_item_when_there_is_one():
    # 든 물건이 안 보이는 동안 선반 바로 옆에 나타난 상자는 그 물건이다 (4차 N10-2: 돌려놓으러 가는 길)
    run = Run().pick().wait(LINK_GAP_MS + 500).see([box("cantata", (1340, 650))], times=6)
    assert run.kinds() == ["pick"]


def test_item_of_another_kind_is_not_merged_into_the_hidden_one():
    run = Run().pick().wait(LINK_GAP_MS + 500)
    run.see([box("donggwaru", (1100, 400))], times=8)
    assert run.kinds() == ["pick"]                    # 개수 그대로, 종류도 그대로
    assert [i.name for i in run.tracker.carried(1)] == ["cantata"]


def test_second_box_next_to_the_held_item_is_the_same_item():
    # 선반 밖(away)에서 든 물건 옆에 상자가 하나 더 보였다 — 선반에서 나온 것이 아니니 새 물건이 아니고, '안 셈' 줄도 안 늘린다
    run = Run().pick()
    run.see([box("cantata", (1150, 650)), box("cantata", (1150, 590))], times=8)
    assert run.kinds() == ["pick"] and len(run.tracker.carried(1)) == 1


def held_beside_the_shelf() -> Run:
    """칸타타를 집어 선반 바로 옆(1350, 650)에 들고 있다."""
    run = Run().see([box("cantata", SHELF)], times=6)
    return run.see([box("cantata", (1350, 650))], times=6)


def test_nested_is_the_share_of_the_smaller_box_inside_the_other():
    a = box("cantata", (1350, 650))
    assert nested(a, ["cantata", 1240, 600, 1380, 700, 0.6]) == 1.0      # 큰 상자가 다 품는다
    assert nested(a, box("cantata", (1350, 720))) == 0.0                  # 나란히 있다
    assert nested(a, box("cantata", (1370, 650))) == 0.5                  # 반이 겹친다


def test_wide_box_around_the_held_item_is_the_same_item():
    # 4차 N02: 캔 하나에 캔만 두른 상자와 손 · 팔까지 두른 큰 상자가 나왔다 — 큰 상자가 든 물건의 상자를 다 품는다
    run = held_beside_the_shelf()
    assert run.kinds() == ["pick"]
    run.see([box("cantata", (1350, 650)), ["cantata", 1240, 600, 1380, 700, 0.6]], times=8)
    assert run.kinds() == ["pick"] and len(run.tracker.carried(1)) == 1


def test_second_item_held_beside_the_first_is_a_second_pick():
    # 4차 N05 · T04 · T08 · N08-2: 든 물건 바로 옆에 선반 옆에서 나타난 상자 — 서로 안 품으면 둘째 물건이다 (후속 77)
    run = held_beside_the_shelf()
    run.see([box("cantata", (1350, 650)), box("donggwaru", (1350, 720))], times=8)
    assert run.kinds() == ["pick", "pick"]
    assert sorted(i.name for i in run.tracker.carried(1)) == ["cantata", "donggwaru"]


# ---------------------------------------------------------------------- 추적 잇기 · 자리에 멈춤 (후속 80)

WIDE = (1, [1000, 200, 1700, 900], None)    # 선반 앞에 서서 선반까지 덮은 손님 1 (선반 위로 뻗은 손)
A, B = (1620, 650), (1450, 650)             # 진열 자리 둘


def big(name: str, at: tuple[float, float], score: float = 0.95) -> list:
    """카메라 가까이 든 물건의 상자 60 × 90 (대각선 108)."""
    return [name, at[0] - 30, at[1] - 45, at[0] + 30, at[1] + 45, score]


def play(frames: list, places: list) -> list[str]:
    """frames = 장면마다의 상자들. 손님 1 이 선반 앞에 서 있다."""
    r = Run()
    for items in frames:
        r.tracker.update(r.ms, items, [WIDE], REGION, places)
        r.events += r.tracker.fresh()
        r.ms += STEP
    return r.kinds()


def test_changed_adds_the_distance_and_the_size_change():
    a = box("cantata", (100, 100))
    assert changed(a, a, (100, 100)) == 0.0
    assert math.isclose(changed(a, box("cantata", (160, 180)), (100, 100)), 100 / math.hypot(40, 60))   # 옮겨 간 거리 ÷ 대각선
    assert math.isclose(changed(a, big("cantata", (100, 100)), (100, 100)), math.log(1.5))             # 그 자리에서 크기만 1.5배
    assert changed(a, a, (140, 100)) > 0                                                               # 있을 자리(예측)에서 잰다


def test_only_two_steps_the_same_way_are_a_movement():
    assert (_minmod(2, 3), _minmod(-2, -3), _minmod(2, -3), _minmod(0, 3)) == (2, -2, 0, 0)
    t = Track(first_ms=0.0, origin="near", last=box("cantata", (100, 100)), last_ms=0.0)
    t.move(STEP, box("cantata", (120, 100)))
    assert t.ahead(2 * STEP) == (120, 100)                    # 한 걸음만 옮겨 갔다 — 움직임으로 안 친다 (옆 상자로 한 번 옮겨 붙은 추적)
    t.move(2 * STEP, box("cantata", (140, 100)))
    assert math.isclose(t.ahead(3 * STEP)[0], 160)            # 두 걸음을 같은 쪽으로 갔다 — 한 걸음 더 간 자리
    assert math.isclose(t.ahead(2 * STEP + 5000)[0], 140 + 20 / STEP * LINK_STEP_MS)   # 오래 안 보여도 한 걸음까지만 내다본다
    t.move(3 * STEP, box("cantata", (130, 100)))
    assert t.ahead(4 * STEP) == (130, 100)                    # 되돌아왔다 — 방향이 엇갈리면 0


def test_held_item_passing_in_front_of_a_shelf_item_keeps_its_track():
    # 3차 N06 선반 카메라 9.80초: 든 물건이 진열 상자 앞을 지나가며 그 상자를 가렸다. 가까운 것만 보고 이으면 진열 상자의 추적이
    # 든 물건을 가져가고 (→ 선반에서 나온 새 물건, 집음), 든 물건의 추적은 진열 상자에 붙는다 (→ 놓음)
    places = [box("cantata", A), box("donggwaru", B)]
    stock = box("donggwaru", B)
    frames = [[big("cantata", A), stock]] * 6
    frames += [[big("cantata", (x, 650)), stock] for x in (1600, 1580, 1560, 1540, 1520, 1500, 1480)]    # 왼쪽으로 꺼내 든다
    frames += [[big("cantata", (x, 650))] for x in (1460, 1440, 1420, 1400)]                             # 진열 상자를 가리며 지나간다
    frames += [[big("cantata", (x, 650)), stock] for x in (1380, 1360, 1340, 1320, 1300, 1280, 1260)]    # 진열 상자가 다시 보인다
    assert play(frames, places) == ["pick"]


def test_item_moving_across_the_places_is_not_put_back_until_it_rests():
    # 선반 카메라에서는 든 물건이 진열 상자 앞에 있어 자리와 겹친다 — 움직이며 지나가는 다섯 장면은 '자리에 머묾'이 아니다
    places = [box("cantata", (x, 650)) for x in (1450, 1490, 1530, 1570)]
    start = [[box("cantata", SHELF)]] * 6 + [[box("cantata", at)] for at in OUT + [(1200, 650), (1250, 650), (1300, 650), (1350, 650)]]
    across = [[box("cantata", (x, 650))] for x in (1430, 1450, 1470, 1490, 1510, 1530, 1550)]            # 자리들 위를 지나간다
    away = [[box("cantata", at)] for at in [(1550, 720), (1500, 760), (1450, 760), (1400, 760), (1350, 760), (1300, 760)]]
    assert play(start + across + away, places) == ["pick"]                                                # 놓지 않았다 (다시 집음도 없다)
    rest = [[box("cantata", (x, 650))] for x in (1430, 1450, 1470, 1490, 1510)] + [[box("cantata", (1530, 650))]] * 5
    assert play(start + rest, places) == ["pick", "back"]                                                 # 한 자리에 멈추면 놓은 것이다


def test_track_does_not_hop_to_a_box_of_another_size():
    # 4차 N04 선반 카메라 6.59초: 든 물건(대각선 76)이 한 장면 안 보인 사이 그 추적이 1.2배 떨어진 작은 진열 상자(34)에 붙어 '놓음'이 됐다
    places = [box("cantata", A), box("donggwaru", B)]
    frames = [[big("cantata", A)]] * 6 + [[big("cantata", (x, 650))] for x in (1600, 1580, 1560, 1540, 1520, 1500)]
    frames += [[["donggwaru", 1430, 635, 1450, 665, 0.8]]] * 6      # 든 물건은 손에 가렸고, 가려져 있던 진열 상자의 한쪽만 드러났다
    assert play(frames, places) == ["pick"]


# ---------------------------------------------------------------------- 종류와 개수

def test_kind_read_differently_for_a_moment_changes_nothing():
    # 4차 C12 손님 2: 칸타타가 6장면 동과루로 읽혀 2개가 됐다 — 물건 단위로는 1개다
    run = Run().pick().see([box("donggwaru", (1150, 650))], times=3).see([box("cantata", (1150, 650))], times=5)
    assert run.kinds() == ["pick"]
    assert [i.name for i in run.tracker.carried(1)] == ["cantata"]


def test_kind_follows_the_majority_without_changing_the_count():
    run = Run().pick("donggwaru").see([box("cantata", (1150, 650))], times=12)
    assert run.kinds() == ["pick", "kind"]
    assert (run.events[1].name, run.events[1].item) == ("cantata", 0)
    assert [i.name for i in run.tracker.carried(1)] == ["cantata"]


def test_kind_read_on_the_shelf_is_not_used():
    # 선반 안에서는 이웃 칸의 상자와 이어지기도 한다 — 손에 들린 뒤 읽힌 종류만 쓴다
    run = Run().see([box("applecider", SHELF)], times=30)
    for at in OUT:
        run.see([box("cantata", at)])
    assert [e.name for e in run.events] == ["cantata"]


# ---------------------------------------------------------------------- 신원

def test_merged_identities_are_one_customer():
    tracker = ItemTracker()
    tracker.resolve = lambda pid: 1 if pid == 7 else pid
    run = Run(tracker).see([box("cantata", SHELF)], times=6)
    for at in OUT:
        run.see([box("cantata", at)], [(7, P[1], None)])
    assert run.events[0].pid == 1 and len(tracker.carried(1)) == 1


def test_confused_frames_are_not_held_frames():
    tracker = ItemTracker()
    tracker.confused = lambda pid, ms: True
    assert Run(tracker).pick().events == []


def test_swap_moves_items_picked_since_then():
    run = Run().pick()
    run.tracker.swap(1, 2, 0.0)
    assert run.tracker.carried(1) == [] and len(run.tracker.carried(2)) == 1


# ---------------------------------------------------------------------- 손님 장부

def test_book_renames_and_removes_an_item():
    b = book()
    b.add_held(HeldItem(1, "donggwaru", 20, 40, 2000.0, item=0), 3000.0)
    b.add_held(HeldItem(1, "donggwaru", 30, 50, 2500.0, item=1), 3000.0)
    assert b.customers[1].taken == 2
    assert b.rename_held(1, "cantata") and sorted(h.name for h in b.customers[1].held) == ["cantata", "donggwaru"]
    assert b.remove_held(0) and b.customers[1].taken == 1
    assert not b.remove_held(0) and not b.rename_held(9, "cantata")


def test_swap_keeps_two_items_of_one_kind():
    b = book()
    b.add_held(HeldItem(1, "donggwaru", 20, 40, 2000.0, item=0), 3000.0)
    b.add_held(HeldItem(1, "donggwaru", 30, 50, 2500.0, item=1), 3000.0)
    assert b.swap_held(1, 2, 0.0, 3000.0) == 2 and b.customers[2].taken == 2


# ---------------------------------------------------------------------- 판정

def feed(judge, run: Run, items: list, people: list | None = None, times: int = 1) -> None:
    for _ in range(times):
        judge.items.update(run.ms, items, [P] if people is None else people, REGION)
        judge._count_held(frame_at(run.ms))
        run.ms += STEP


def test_judge_counts_items_not_kinds():
    judge, risk = make_judge(overview_instant=True, item_tracks=True)
    run = Run()
    feed(judge, run, [box("donggwaru", (1450, 620)), box("donggwaru", (1450, 720))], times=6)
    for x, _ in OUT:
        feed(judge, run, [box("donggwaru", (x, 620)), box("donggwaru", (x, 720))])
    assert risk.book.customers[1].taken == 2
    assert [(h["counted"], h["name"]) for h in judge.held] == [(True, "donggwaru")] * 2


def test_judge_does_not_count_a_box_that_never_was_on_the_shelf():
    judge, risk = make_judge(overview_instant=True, item_tracks=True)
    judge.fuser.note_band(900.0, 2)                   # 선반 앞에 있었어도 — 상자가 선반에서 안 나왔다
    feed(judge, Run(), [box("applecider", (300, 300))], [Q], times=8)
    assert 2 not in risk.book.customers or risk.book.customers[2].taken == 0
    assert [(h["counted"], h["why"]) for h in judge.held] == [(False, "선반에서 나온 상자가 아님")]


def test_judge_fixes_the_kind_and_takes_back_a_returned_item():
    judge, risk = make_judge(overview_instant=True, item_tracks=True)
    run = Run()
    feed(judge, run, [box("donggwaru", SHELF)], times=6)
    for at in OUT:
        feed(judge, run, [box("donggwaru", at)])
    feed(judge, run, [box("cantata", (1150, 650))], times=12)
    assert risk.book.customers[1].taken == 1 and judge.held[0]["name"] == "cantata"
    assert [h.name for h in risk.book.customers[1].held] == ["cantata"]
    for x in (1200, 1250, 1300, 1350, 1400):
        feed(judge, run, [box("cantata", (x, 650))])
    feed(judge, run, [box("cantata", SHELF)], times=5)
    assert risk.book.customers[1].taken == 0
    assert judge.held[0]["until_ms"] is not None and judge.held[0]["end"] == "놓음"


def test_visit_replaces_items_like_before():
    judge, risk = make_judge(overview_instant=True, item_tracks=True)
    run = Run(start_ms=5000.0)
    feed(judge, run, [box("cantata", SHELF)], times=6)
    for at in OUT:
        feed(judge, run, [box("cantata", at)])
    assert risk.book.customers[1].taken == 1
    judge._settled.append((3000.0, 8000.0))           # 그 시간을 덮는 방문이 끝났다 — 방문의 개수 · 주인이 대신한다
    judge._settle_held()
    assert risk.book.customers[1].taken == 0 and judge.held[0]["until_ms"] == 8000.0


def test_item_tracks_is_off_by_default():
    assert FusionConfig().item_tracks is False
    judge, _ = make_judge(overview_instant=True)
    assert judge.items is None
