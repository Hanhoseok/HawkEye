"""상자가 다시 생긴 순간 선반 카메라의 앞뒤 순서로 번호를 정한다 (fusion.swap_ask_detail, 후속 69, 한계 C-28).

배경 (4차 C05 8.87초 · C06 7.20초): 전체 카메라에서 두 손님이 상자 하나로 합쳐졌다. 남은 상자는 앞사람을 따라갔는데 번호는
뒷사람의 것이었다. 뒷사람의 상자가 새로 생기며 남은 번호(앞사람의 것)를 받았다 — 두 번호가 서로 바뀌었다.
선반 카메라는 그동안 두 사람을 따로 보고 있었고, 누가 전체 카메라 쪽에 가까운지 바닥 자리로 안다.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.core.config import FusionConfig  # noqa: E402
from src.core.types import Frame, IdentityObservation  # noqa: E402
from src.fusion.confusion import ConfusionWatch  # noqa: E402
from src.fusion.cross_view import CrossViewLinker  # noqa: E402
from src.fusion.judge import DualJudge  # noqa: E402
from src.fusion.shelf_events import TimeAxis  # noqa: E402
from src.zones.floor import FloorMap  # noqa: E402
from test_swap_repair import Risk  # noqa: E402

W = H = 1000
CORNERS = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
OVERVIEW = [(0.0, 0.0), (400.0, 0.0), (400.0, 400.0), (0.0, 400.0)]
"""전체 카메라: 화면 (x, y) → 바닥 (400x, 400y) cm. 화면 아래쪽(바닥 y 가 큰 쪽)이 카메라에 가깝다."""
DETAIL = [(0.0, 0.0), (0.0, 400.0), (400.0, 400.0), (400.0, 0.0)]
"""선반 카메라는 옆에서 본다: 화면 (x, y) → 바닥 (400y, 400x) cm. 전체 카메라에서 앞뒤로 선 두 사람이 여기서는 좌우로 갈린다."""

FRONT, BACK, AWAY = (200.0, 300.0), (200.0, 220.0), (60.0, 220.0)
"""바닥 자리 (cm): 앞사람 · 그 뒤에 선 자리 (전체 카메라에서 0.8 겹침) · 뒷사람이 처음 서 있던 떨어진 자리."""
NEAR_MS, ONE_MS, BACK_MS, AGAIN_MS, BACK2_MS = 1254, 1584, 1848, 2772, 3036
"""뒷사람이 앞사람 뒤로 옴 → 상자 하나 → 뒷사람 상자가 다시 생김 → (두 번째) 상자 하나 → 다시 생김."""


def overview_box(spot):
    x, y = spot[0] / 400.0 * W, spot[1] / 400.0 * H
    return (x - 100.0, y - 500.0, x + 100.0, y)


def detail_box(spot):
    x, y = spot[1] / 400.0 * W, spot[0] / 400.0 * H
    return (x - 100.0, max(0.0, y - 400.0), x + 100.0, y)


def person(pid, bbox, track=None, ankles=True):
    foot = [(bbox[0] + bbox[2]) / 2.0, bbox[3], 0.9]
    keypoints = [[0.0, 0.0, 0.0]] * 15 + [foot, foot] if ankles else None
    return IdentityObservation(person_id=pid, track_id=track or pid, bbox=bbox, frame=0, timestamp="", keypoints=keypoints)


def play(ask=True, kept=12, shelf_floor=True, shelf_ankles=True, shelf_touch=False, twice=False, until=4200, **cfg):
    """선반 카메라 1번 = 앞사람, 2번 = 뒷사람. 전체 카메라 11번(추적 101) = 앞사람, 12번(102) = 뒷사람.
    ONE_MS 에 상자 하나가 된다 — 상자는 앞사람 자리에 있고 번호는 kept. BACK_MS 에 뒷사람의 상자가 나머지 번호로 다시 생긴다.
    kept=12: 번호가 서로 바뀐다 (앞사람이 12, 뒷사람이 11). kept=11: 그대로.
    twice: AGAIN_MS 에 한 번 더 합쳐지며 번호가 되돌아온다 (4차 C06)."""
    risk = Risk()
    judge = DualJudge(FusionConfig(swap_ask_detail=ask, **{"position_mode": "floor", **cfg}), risk, FloorMap("overview", CORNERS, OVERVIEW),
                      (1.0, 0.0), TimeAxis(15.0), TimeAxis(15.0), W, (W, H), {}, None, None,
                      detail_floor=FloorMap("detail", CORNERS, DETAIL) if shelf_floor else None, detail_height=H)
    judge.reset()
    other = {11: 12, 12: 11}[kept]
    for t in range(0, until, 66):
        behind = BACK if t >= NEAR_MS else AWAY
        shelf_back = detail_box(behind)
        if shelf_touch and t >= NEAR_MS:
            x1, y1, x2, y2 = detail_box(FRONT)
            shelf_back = (x1 - 60.0, y1, x2 - 60.0, y2)            # 선반 카메라에서도 두 상자가 0.7 겹친다
        judge.detail_step(float(t), [person(1, detail_box(FRONT), ankles=shelf_ankles),
                                     person(2, shelf_back, ankles=shelf_ankles)], [], [])
        if t < ONE_MS:
            people = [person(11, overview_box(FRONT), 101), person(12, overview_box(behind), 102)]
        elif t < BACK_MS:
            people = [person(kept, overview_box(FRONT), 100 + kept % 10)]
        elif not twice or t < AGAIN_MS:
            people = [person(kept, overview_box(FRONT), 100 + kept % 10), person(other, overview_box(BACK), 103)]
        elif t < BACK2_MS:
            people = [person(other, overview_box(FRONT), 103)]            # 이번에는 뒷사람에게 붙어 있던 번호가 앞사람 상자에 남는다
        else:
            people = [person(other, overview_box(FRONT), 103), person(kept, overview_box(BACK), 104)]
        frame = Frame(index=t, timestamp="", pts_ms=float(t), image=None, camera_id="angle1")
        judge.overview_step(frame, people, [])
    return judge, risk


def first(ms: int) -> float:
    """66ms 간격 장면 가운데 ms 뒤 첫 장면."""
    return float(-(-ms // 66) * 66)


def test_numbers_are_read_swapped_the_moment_the_hidden_box_comes_back():
    judge, risk = play()
    assert [a["say"] for a in judge.asks] == ["바뀜"]
    assert judge.asks[0]["store"] == [11, 12] and judge.asks[0]["detail"] == [1, 2]
    assert judge.asks[0]["gap_cm"] == 80.0                       # 사라졌던 번호(11)의 사람이 80cm 앞
    assert len(judge.swaps) == 1
    swap = judge.swaps[0]
    assert swap["how"] == "ask" and swap["ms"] == swap["found_ms"] == first(BACK_MS)   # 짝이 뒤집히기를 안 기다린다
    assert swap["seen_ms"] == first(ONE_MS)                      # 본 것은 상자가 사라진 때부터 — 그동안 남은 상자는 앞사람이었다
    assert judge.store_id(12) == 11 and judge.store_id(11) == 12
    # 손님 장부에는 처음부터 끝까지 앞사람 = 11, 뒷사람 = 12
    after = [ids for ms, ids in risk.seen if ms >= first(BACK_MS)]
    assert after and all(ids == {102: 11, 103: 12} for ids in after)
    # 자리 표본: 매장 신원 11 은 내내 앞사람 자리, 12 는 합쳐진 동안 없다가 뒷사람 자리
    spots = {pid: {tuple(round(v, 6) for v in s.floor) for s in judge.linker._overview[pid].samples if s.pts_ms >= first(ONE_MS)}
             for pid in (11, 12)}
    assert spots == {11: {FRONT}, 12: {BACK}}
    assert not [s for s in judge.linker._overview[12].samples if first(ONE_MS) <= s.pts_ms < first(BACK_MS)]
    # 두 화면 짝은 한 번도 안 바뀐다
    assert {o for _, _, o in judge.linker._history[1]} == {11} and {o for _, _, o in judge.linker._history[2]} == {12}


def test_numbers_stay_when_the_hidden_number_belongs_to_the_person_behind():
    judge, risk = play(kept=11)
    assert [(a["say"], a["store"], a["gap_cm"]) for a in judge.asks] == [("그대로", [12, 11], -80.0)]
    assert judge.swaps == [] and judge._store == {}
    assert all(ids == {101: 11, 103: 12} for ms, ids in risk.seen if ms >= first(BACK_MS))


def test_a_second_merge_that_puts_the_numbers_back_is_caught_too():
    """4차 C06: 7.20초에 바뀌고 7.93초에 되돌아왔다 — 짝이 뒤집힐 틈(세 조각)이 없어 예전 방식으로는 못 본다."""
    judge, risk = play(twice=True)
    assert [a["say"] for a in judge.asks] == ["바뀜", "바뀜"]
    assert [s["ms"] for s in judge.swaps] == [first(BACK_MS), first(BACK2_MS)]
    assert judge._store == {}                                    # 화면 번호 = 매장 신원으로 돌아왔다
    for ms, ids in risk.seen:                                    # 앞사람 = 11, 뒷사람 = 12 가 내내 이어진다
        if first(BACK_MS) <= ms < first(AGAIN_MS):
            assert ids == {102: 11, 103: 12}
        elif first(AGAIN_MS) <= ms < first(BACK2_MS):
            # 합쳐진 동안에는 아직 모른다 — 손님 장부는 남은 번호(뒷사람 것)로 본다. 상자가 다시 생긴 순간에 고치고,
            # 자리 표본 · 선반 앞 기록 · 든 기록만 거슬러 고친다 (장부의 그 사이 구역 상태는 안 되돌린다 — 한계)
            assert ids == {103: 12}
        elif ms >= first(BACK2_MS):
            assert ids == {103: 11, 104: 12}


def test_not_asked_without_a_shelf_camera_floor_map():
    judge, _ = play(shelf_floor=False, position_mode="band")
    assert [a["say"] for a in judge.asks] == ["못 물음"] and "바닥 지도" in judge.asks[0]["why"]
    assert judge.swaps == []


def test_not_asked_when_the_shelf_camera_does_not_see_the_ankles():
    judge, _ = play(shelf_ankles=False)
    assert [a["say"] for a in judge.asks] == ["못 물음"] and "발목" in judge.asks[0]["why"]
    assert judge.swaps == []


def test_not_asked_when_the_shelf_camera_confuses_the_two_as_well():
    judge, _ = play(shelf_touch=True)
    assert judge.watch["detail"].episodes
    assert [a["say"] for a in judge.asks] == ["못 물음"] and "헷갈림" in judge.asks[0]["why"]
    assert judge.swaps == []


def test_a_small_gap_is_not_trusted():
    judge, _ = play(swap_ask_min_gap_cm=100.0)
    assert [a["say"] for a in judge.asks] == ["못 물음"] and "작음" in judge.asks[0]["why"]
    assert judge.swaps == []


def test_off_by_default():
    judge, risk = play(ask=False)
    assert judge.asks == [] and judge.swaps == []
    assert all(ids == {102: 12, 103: 11} for ms, ids in risk.seen if ms >= first(BACK_MS))   # 바뀐 번호 그대로 읽는다


# ---------------------------------------------------------------------- 부품

def test_watch_reports_whose_box_came_back():
    watch = ConfusionWatch()
    one, two = (0, 0, 100, 300), (20, 0, 120, 300)
    for t in range(5):
        watch.update(t * 100.0, {1: one, 2: two})
    for t in range(5, 8):
        watch.update(t * 100.0, {2: two})
        assert watch.returned == []
    watch.update(800.0, {1: one, 2: two})
    assert [(hidden, kept) for hidden, kept, _ in watch.returned] == [(1, 2)]
    ep = watch.returned[0][2]
    assert (ep.gone_ms, ep.back_ms, ep.hidden) == (500.0, 800.0, 1)
    watch.update(900.0, {1: one, 2: two})
    assert watch.returned == []                                   # 그 장면에만


def test_detail_feet_leave_out_guessed_spots():
    linker = CrossViewLinker(FusionConfig())
    linker.observe("detail", 1, 100.0, None, None, (10.0, 20.0), None)
    linker.observe("detail", 1, 200.0, None, None, (11.0, 21.0), (0.0, 1.0))    # 발목이 가려 방향만 아는 표본
    linker.observe("detail", 1, 300.0, None, None, None, None)
    assert linker.detail_feet(1, 0.0, 1000.0) == {100.0: (10.0, 20.0)}
    assert linker.detail_feet(1, 150.0, 1000.0) == {} and linker.detail_feet(9, 0.0, 1000.0) == {}
    assert linker.detail_ids(250.0) == [1] and linker.detail_ids(350.0) == []
