"""선반 지도 테스트 — 어느 자리가 비었는지, 누구에게 붙이는지.

선반(화면 x 0~400, y 0~200)에 물건을 한 줄 또는 두 줄로 놓는다. 물건 하나는 40x60 상자.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.shelf.monitor import ShelfItem, ShelfMonitor, layout, match  # noqa: E402

SHELF = (0.0, 0.0, 400.0, 200.0)
FPS = 10


def item(label: str, x: float, y: float = 50.0) -> ShelfItem:
    return ShelfItem(label, (x, y, x + 40, y + 60))


ROW = [item("bottle", 20), item("spray", 80), item("bottle", 140), item("jug", 200)]
PERSON = (90.0, 0.0, 200.0, 400.0)       # 선반을 가리는 사람
AWAY = (500.0, 0.0, 600.0, 400.0)        # 선반과 떨어진 사람


def run(mon: ShelfMonitor, start: float, seconds: float, items, people=()):
    changes = []
    for i in range(int(round(seconds * FPS))):
        changes += mon.update((start + i / FPS) * 1000.0, list(items), list(people), FPS)
    return changes


def monitor(**kw) -> ShelfMonitor:
    return ShelfMonitor("shelf", SHELF, **kw)


# ---------------------------------------------------------------- 지도 정리

def test_layout_orders_rows_top_down_and_items_left_to_right():
    top = [item("a", 200, 20), item("b", 20, 22)]
    bottom = [item("c", 100, 130), item("d", 20, 128)]
    names = [s.name for s in layout(top + bottom)]
    assert names == ["1단 1번째 b", "1단 2번째 a", "2단 1번째 d", "2단 2번째 c"]


def test_match_uses_position_so_same_kind_items_are_told_apart():
    """같은 '병'이 두 개여도 자리로 구별한다 — 1번째 병이 사라졌는지 3번째 병이 사라졌는지."""
    before = layout(ROW)
    after = layout([ROW[1], ROW[2], ROW[3]])            # 1번째 병만 사라짐
    removed, added = match(before, after)
    assert [s.name for s in removed] == ["1단 1번째 bottle"] and added == []


def test_match_tolerates_detector_label_flicker():
    """같은 자리에서 종류 이름만 바뀐 것(탐지 흔들림)은 사라짐/생김이 아니다."""
    before = layout(ROW)
    after = layout([ROW[0], ROW[1], ROW[2], item("toilet", 202)])   # 'jug' 를 'toilet' 으로 봄
    assert match(before, after) == ([], [])


# ---------------------------------------------------------------- 지켜보기

def test_person_takes_item_and_it_is_attributed():
    """핵심 — 사람이 가린 동안 2번째 스프레이가 사라졌다 -> 그 사람에게."""
    mon = monitor()
    run(mon, 0.0, 2.0, ROW)                                          # 지도 확정
    run(mon, 2.0, 3.0, ROW, people=[(7, PERSON)])                    # 손님 7 이 가림
    changes = run(mon, 5.0, 3.0, [ROW[0], ROW[2], ROW[3]])           # 떠난 뒤
    assert len(changes) == 1
    c = changes[0]
    assert [s.name for s in c.removed] == ["1단 2번째 spray"] and c.person_id == 7
    assert c.start_ms == 2000.0


def test_browsing_without_taking_is_not_a_change():
    """가렸지만 아무것도 안 가져가면 변화가 없다."""
    mon = monitor()
    run(mon, 0.0, 2.0, ROW)
    run(mon, 2.0, 3.0, ROW, people=[(7, PERSON)])
    assert run(mon, 5.0, 3.0, ROW) == []


def test_returning_item_is_added():
    """가져갔다가 다시 놓으면 '생김'으로 나온다 (되돌려놓기)."""
    mon = monitor()
    run(mon, 0.0, 2.0, [ROW[0], ROW[2], ROW[3]])
    run(mon, 2.0, 2.0, [ROW[0], ROW[2], ROW[3]], people=[(7, PERSON)])
    changes = run(mon, 4.0, 3.0, ROW)
    assert [s.name for s in changes[0].added] == ["1단 2번째 spray"] and changes[0].person_id == 7


def test_two_people_at_shelf_is_not_attributed():
    """두 명이 함께 가렸으면 누구 것인지 정하지 않는다."""
    mon = monitor()
    run(mon, 0.0, 2.0, ROW)
    run(mon, 2.0, 2.0, ROW, people=[(7, PERSON), (8, PERSON)])
    changes = run(mon, 4.0, 3.0, ROW[1:])
    assert changes[0].person_id is None and sorted(changes[0].people) == [7, 8]


def test_person_away_from_shelf_does_not_block_or_get_credit():
    """선반과 떨어진 사람은 가림도 아니고, 변화의 주인도 아니다."""
    mon = monitor()
    run(mon, 0.0, 2.0, ROW)
    changes = run(mon, 2.0, 4.0, ROW[1:], people=[(9, AWAY)])
    assert len(changes) == 1 and changes[0].people == []


def test_single_frame_misses_are_ignored():
    """탐지가 한두 장면 물건을 놓쳐도 변화로 보지 않는다."""
    mon = monitor()
    run(mon, 0.0, 2.0, ROW)
    changes = []
    for i in range(40):
        items = ROW if i % 5 else ROW[1:]                            # 다섯 장면에 한 번 1번째 병을 놓침
        changes += mon.update((2.0 + i / FPS) * 1000.0, items, [], FPS)
    assert changes == []


def test_brief_hand_flicker_without_person_is_ignored():
    """사람으로 안 잡히는 손이 잠깐(1초 미만) 물건을 가렸다가 빠지면 변화가 아니다."""
    mon = monitor()
    run(mon, 0.0, 2.0, ROW)
    changes = run(mon, 2.0, 0.8, ROW[1:]) + run(mon, 2.8, 3.0, ROW)
    assert changes == []


def test_change_without_visible_person_is_reported_unattributed():
    """손만 들어와 사람이 안 잡혀도 물건이 사라진 건 알린다 — 주인은 '모름'."""
    mon = monitor()
    run(mon, 0.0, 2.0, ROW)
    changes = run(mon, 2.0, 4.0, ROW[1:])
    assert len(changes) == 1 and changes[0].person_id is None and changes[0].people == []
