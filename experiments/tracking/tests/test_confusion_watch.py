"""한 카메라가 두 사람을 헷갈리는 구간 (src/fusion/confusion.py, 후속 68)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fusion.confusion import ConfusionWatch, covered, overlap  # noqa: E402

LEFT, RIGHT = (0, 0, 100, 300), (200, 0, 300, 300)
CLOSE = (30, 0, 130, 300)    # LEFT 와 70% 겹침
BOTH = (40, 0, 300, 300)     # 두 사람을 함께 덮은 상자
INNER = (60, 0, 160, 300)    # BOTH 가 덮는 자리


def feed(watch, spec):
    for seconds, boxes in spec:
        watch.update(seconds * 1000.0, boxes)
    return watch


def test_overlap_is_shared_area_over_the_smaller_box():
    assert overlap(LEFT, CLOSE) == 0.7 and overlap(LEFT, RIGHT) == 0.0
    assert covered(INNER, BOTH) == 1.0 and covered(BOTH, INNER) < 0.5


def test_separate_people_make_no_episode():
    watch = feed(ConfusionWatch(), [(t / 10, {1: LEFT, 2: RIGHT}) for t in range(20)])
    assert watch.episodes == []


def test_overlapping_boxes_make_one_episode_without_a_switch_time():
    spec = ([(t / 10, {1: LEFT, 2: RIGHT}) for t in range(10)] + [(1.0 + t / 10, {1: LEFT, 2: CLOSE}) for t in range(5)]
            + [(1.5 + t / 10, {1: LEFT, 2: RIGHT}) for t in range(10)])
    watch = feed(ConfusionWatch(), spec)
    assert len(watch.episodes) == 1
    ep = watch.episodes[0]
    assert (ep.a, ep.b, ep.start_ms, ep.end_ms) == (1, 2, 1000.0, 1400.0)
    assert not ep.merged and ep.switch_ms is None and ep.peak == 0.7


def test_person_swallowed_by_another_box_switches_when_their_box_comes_back():
    """4차 C05: 두 사람이 상자 하나가 됐다가, 가려졌던 사람의 상자가 새로 생긴다 — 그때 번호를 다시 받는다."""
    spec = ([(t / 10, {1: RIGHT, 2: INNER}) for t in range(12)] + [(1.2 + t / 10, {1: BOTH}) for t in range(6)]
            + [(1.8 + t / 10, {1: RIGHT, 2: LEFT}) for t in range(12)])
    watch = feed(ConfusionWatch(), spec)
    assert len(watch.episodes) == 1
    ep = watch.episodes[0]
    assert ep.merged and ep.start_ms == 1200.0 and ep.end_ms == 1700.0 and ep.switch_ms == 1800.0


def test_swallowed_state_stops_after_the_limit():
    spec = [(t / 10, {1: RIGHT, 2: INNER}) for t in range(12)] + [(1.2 + t / 10, {1: BOTH}) for t in range(50)]
    watch = feed(ConfusionWatch(merged_max_ms=3000.0), spec)
    assert watch.episodes[0].end_ms == 4200.0 and watch.episodes[0].switch_ms is None


def test_frames_close_in_time_join_into_one_episode():
    spec = ([(1.0, {1: LEFT, 2: CLOSE}), (1.1, {1: LEFT, 2: RIGHT}), (1.4, {1: LEFT, 2: CLOSE}),
             (3.0, {1: LEFT, 2: CLOSE})])
    watch = feed(ConfusionWatch(join_ms=500.0), spec)
    assert [(e.start_ms, e.end_ms) for e in watch.episodes] == [(1000.0, 1400.0), (3000.0, 3000.0)]


def test_find_and_touching_look_up_by_time():
    watch = feed(ConfusionWatch(), [(1.0 + t / 10, {1: LEFT, 2: CLOSE, 3: RIGHT}) for t in range(5)])
    assert watch.find(2, 1, 1600.0, 2000.0) is watch.episodes[0]   # 0.5초 여유 안
    assert watch.find(1, 2, 2000.0, 2500.0) is None and watch.find(1, 3, 1000.0, 1400.0) is None
    assert watch.touching(1, 1200.0, 1200.0) and not watch.touching(3, 1200.0, 1200.0)
    assert not watch.touching(1, 1500.0, 1600.0)
