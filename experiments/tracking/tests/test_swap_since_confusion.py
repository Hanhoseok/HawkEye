"""헷갈리는 순간은 덜 헷갈리는 카메라가 정한다 — 번호 맞바뀜을 되돌릴 때 '언제부터'와 그 사이의 든 기록
(fusion.swap_since_confusion, 후속 68, 한계 C-28).

배경 (4차 C05): 전체 카메라에서 두 손님이 상자 하나로 합쳐졌다가, 가려졌던 손님의 상자가 새로 생기며 남은 번호를 받았다 (8.87초).
두 화면 짝은 두 사람의 자리가 갈린 조각에서야 뒤집혔다. 그 사이 손님 1 손의 상품이 손님 2 가 든 것으로 적혀 남았다.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.core.config import FusionConfig  # noqa: E402
from src.core.types import Frame  # noqa: E402
from src.fusion.judge import DualJudge  # noqa: E402
from src.fusion.overview_count import HolderTracker  # noqa: E402
from src.fusion.shelf_events import TimeAxis  # noqa: E402
from test_overview_instant import book, item_held  # noqa: E402
from test_overview_owner import A_BOX, B_BOX, held, hold, someone  # noqa: E402
from test_swap_repair import BAND, H, LEFT, RIGHT, W, Risk, person  # noqa: E402

NEAR_L, NEAR_R = (401.0, 100.0, 601.0, 900.0), (451.0, 100.0, 651.0, 900.0)   # 가까이 선 두 사람 (0.75 겹침)
ONE = (400.0, 100.0, 652.0, 900.0)   # 두 사람을 함께 덮은 상자
BAND.update({NEAR_L: 0.49, NEAR_R: 0.51, ONE: 0.5})
"""가까이 선 동안은 띠 위치가 거의 같다 — 자리로는 누가 누구인지 못 가른다 (4차 C05: 위치 점수가 네 짝 모두 1.00)."""
NEAR_MS, ONE_MS, BACK_MS, APART_MS = 1254, 1584, 1848, 2772
"""가까이 섬 → 상자 하나 → 가려졌던 사람의 상자가 새로 생김 (번호가 뒤바뀌어 붙는다) → 떨어짐."""


def play(since_confusion: bool, until: int = 4500):
    """선반 카메라 1 · 2 번은 왼쪽 · 오른쪽에 그대로. 전체 카메라: 왼쪽 사람(추적 101)이 11, 오른쪽 사람(102)이 12 였다가
    상자 하나(11)로 합쳐지고, 다시 갈릴 때 11 이 오른쪽 사람에게 · 12 가 왼쪽 사람에게 붙는다."""
    floor = SimpleNamespace(foot_mode="bbox", foot_on_floor=lambda bbox, w, h, kp=None: tuple(bbox),
                            band_position=lambda point: BAND[point], zones_at=lambda point: [])
    risk = Risk()
    # max_band_gap 1.0: 가까이 선 동안에도 짝이 끊기지 않게 (끊기면 '맞바뀜'을 볼 조각이 없다)
    judge = DualJudge(FusionConfig(swap_repair=True, swap_since_confusion=since_confusion, max_band_gap=1.0), risk, floor, (1.0, 0.0),
                      TimeAxis(15.0), TimeAxis(15.0), W, (W, H), {}, None, None,
                      detail_point=lambda bbox, width: BAND[tuple(bbox)])
    judge.reset()
    for t in range(0, until, 66):
        judge.detail_step(float(t), [person(1, LEFT), person(2, RIGHT)], [], [])
        if t < NEAR_MS:
            people = [person(11, LEFT, track=101), person(12, RIGHT, track=102)]
        elif t < ONE_MS:
            people = [person(11, NEAR_L, track=101), person(12, NEAR_R, track=102)]
        elif t < BACK_MS:
            people = [person(11, ONE, track=101)]
        elif t < APART_MS:
            people = [person(12, NEAR_L, track=103), person(11, NEAR_R, track=101)]
        else:
            people = [person(12, LEFT, track=103), person(11, RIGHT, track=101)]
        frame = Frame(index=t, timestamp="", pts_ms=float(t), image=None, camera_id="angle1")
        judge.overview_step(frame, people, [])
    return judge


def test_overview_records_are_reread_from_when_the_hidden_persons_box_came_back():
    judge = play(True)
    assert len(judge.swaps) == 1
    swap = judge.swaps[0]
    assert swap["seen_ms"] == BACK_MS and swap["ms"] > APART_MS - 600      # 짝은 두 사람이 떨어진 뒤에야 뒤집혔다
    assert swap["episode"]["merged"] and swap["episode"]["switch_ms"] == BACK_MS
    # 다시 갈린 뒤의 자리 표본: 매장 신원 11 = 왼쪽 사람 (띠 0.2), 12 = 오른쪽 사람 (띠 0.8) — 조각 시작 전 것도
    for pid, band in ((11, 0.49), (12, 0.51)):
        after = [s.band for s in judge.linker._overview[pid].samples if BACK_MS <= s.pts_ms < APART_MS]
        assert after and set(after) == {band}
    # 짝 기록은 선반 카메라 번호를 따라 그대로 이어진다
    assert [o for _, _, o in judge.linker._history[1]] == [11] and [o for _, _, o in judge.linker._history[2]] == [12]


def test_without_the_setting_only_the_flipped_window_is_reread():
    judge = play(False)
    swap = judge.swaps[0]
    assert swap["seen_ms"] == swap["ms"]
    before = [s.band for s in judge.linker._overview[11].samples if BACK_MS <= s.pts_ms < swap["ms"]]
    assert set(before) == {0.51}     # 그 사이는 오른쪽 사람의 자리가 11 로 남는다


def test_both_cameras_watch_for_confusion():
    judge = play(True)
    assert [(e.a, e.b) for e in judge.watch["overview"].episodes] == [(11, 12)]
    assert judge.watch["detail"].episodes == []       # 선반 카메라는 두 사람을 내내 따로 봤다


# ---------------------------------------------------------------------- 든 기록

def tracker(confused=None) -> HolderTracker:
    t = HolderTracker()
    t.confused = confused
    return t


def test_holding_seen_in_a_confused_frame_is_kept_apart_and_not_announced():
    t = tracker(lambda pid, ms: ms >= 1000.0)
    hold(t, 0.0, 0.2, [held("cantata", 780)], [someone(1, A_BOX, (790, 540))])
    hold(t, 1.0, 2.0, [held("cantata", 780)], [someone(1, A_BOX, (790, 540))])
    assert {ms < 1000.0 for ms, _, _ in t.samples} == {True} and len(t.unsure) >= 5
    assert t.fresh() == []                                   # 확실한 장면은 5장에 못 미친다
    assert t.first_holders("cantata", 0.0, 3000.0) == [] and t.first_holders("cantata", 0.0, 3000.0, True) == [1]


def test_unsure_holding_is_used_only_when_sure_holders_are_fewer_than_the_items():
    """4차 L1: 셋째 손님은 다른 손님과 겹친 채로만 들고 있었다 → 모자란 하나는 그 손님. 4차 C05: 모자라지 않으면 안 쓴다."""
    t = tracker(lambda pid, ms: pid == 2)
    hold(t, 0.0, 1.0, [held("donggwaru", 780)], [someone(1, A_BOX, (790, 540))])
    hold(t, 1.0, 2.0, [held("donggwaru", 1000)], [someone(2, B_BOX, (1010, 540))])
    assert t.owners({"donggwaru": 1}, 0.0, 3000.0) == [("donggwaru", 1)]
    assert t.owners({"donggwaru": 2}, 0.0, 3000.0) == [("donggwaru", 1), ("donggwaru", 2)]


def test_swap_rereads_holding_from_the_given_time():
    t = tracker(lambda pid, ms: 1000.0 <= ms < 1500.0)
    hold(t, 0.0, 0.5, [held("cantata", 780)], [someone(1, A_BOX, (790, 540))])       # 바뀌기 전 — 그대로 1번
    hold(t, 1.0, 2.0, [held("donggwaru", 780)], [someone(2, A_BOX, (790, 540))])     # 번호가 바뀐 뒤 — 2번으로 적혔다
    t.swap(1, 2, 900.0)
    assert {pid for _, pid, _ in t.samples} == {1} and {pid for _, pid, _ in t.unsure} == {1}
    assert t.first_holders("donggwaru", 0.0, 3000.0) == [1]
    assert [(pid, name) for pid, name, _, _ in t.fresh()] == [(1, "cantata"), (1, "donggwaru")]


def test_book_moves_counted_holding_between_the_two_customers():
    b = book()
    b.add_held(item_held(1, "cantata", seen_ms=500.0), 800.0)          # 바뀌기 전
    b.add_held(item_held(2, "cantata", seen_ms=2000.0), 2300.0)        # 바뀐 뒤 — 1번의 것 (같은 종류라 하나만 남는다)
    b.add_held(item_held(2, "donggwaru", seen_ms=2100.0), 2400.0)      # 바뀐 뒤 — 1번의 것
    assert b.swap_held(1, 2, 1000.0, 3000.0) == 2
    one, two = b.customers[1], b.customers[2]
    assert sorted((h.name, h.seen_ms, h.person_id) for h in one.held) == [("cantata", 500.0, 1), ("donggwaru", 2100.0, 1)]
    assert two.held == []
