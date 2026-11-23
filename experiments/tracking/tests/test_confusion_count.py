"""scripts/confusion_count.py — 한 카메라가 사람을 헷갈린 구간을 세는 규칙."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import confusion_count as cc  # noqa: E402

LEFT, RIGHT, BOTH = (0, 0, 100, 300), (200, 0, 300, 300), (40, 0, 300, 300)


def frames(spec):
    """spec = [(초, {번호: 상자}), …] → camera_states 입력."""
    return [(s * 1000.0, list(boxes.items())) for s, boxes in spec]


def states(spec):
    return [state for _, state in cc.camera_states(frames(spec))]


def test_two_separate_people_are_clear():
    spec = [(t / 10, {1: LEFT, 2: RIGHT}) for t in range(20)]
    assert set(states(spec)) == {"갈라 봄"}
    assert cc.episodes(cc.camera_states(frames(spec))) == []


def test_overlapping_boxes_are_confused():
    near = (60, 0, 160, 300)   # LEFT 와 40% 겹침 → 기준 아래
    close = (30, 0, 130, 300)  # LEFT 와 70% 겹침
    spec = ([(t / 10, {1: LEFT, 2: RIGHT}) for t in range(10)] + [(1.0, {1: LEFT, 2: near})]
            + [(1.1 + t / 10, {1: LEFT, 2: close}) for t in range(5)] + [(1.6 + t / 10, {1: LEFT, 2: RIGHT}) for t in range(10)])
    got = states(spec)
    assert got[10] == "갈라 봄" and got[11:16] == ["헷갈림"] * 5 and got[16] == "갈라 봄"
    assert cc.episodes(cc.camera_states(frames(spec))) == [(1100.0, 1500.0)]


def test_person_swallowed_by_another_box_is_confused_until_they_return():
    """두 사람이 상자 하나가 된다 (4차 C05) — 번호 2 가 사라지고 그 자리를 번호 1 의 상자가 덮는다."""
    inner = (60, 0, 160, 300)
    spec = ([(t / 10, {1: RIGHT, 2: inner}) for t in range(12)]
            + [(1.2 + t / 10, {1: BOTH}) for t in range(6)]
            + [(1.8 + t / 10, {1: RIGHT, 2: LEFT}) for t in range(12)])
    got = states(spec)
    assert got[12:18] == ["헷갈림"] * 6 and got[18] == "갈라 봄"


def test_swallowed_state_ends_after_the_limit():
    inner = (60, 0, 160, 300)
    spec = [(t / 10, {1: RIGHT, 2: inner}) for t in range(12)] + [(1.2 + t / 10, {1: BOTH}) for t in range(50)]
    got = states(spec)
    assert got[12] == "헷갈림" and got[-1] == "한 명 이하"


def test_short_lived_box_is_not_a_person():
    ghost = (10, 0, 90, 300)   # LEFT 안에 0.3초만 생긴 상자
    spec = [(t / 10, {1: LEFT, **({9: ghost} if 5 <= t < 8 else {})}) for t in range(20)]
    assert set(states(spec)) == {"한 명 이하"}


def test_other_view_takes_the_longest_state():
    view = [(0.0, "갈라 봄"), (100.0, "갈라 봄"), (200.0, "헷갈림"), (900.0, "헷갈림")]
    assert cc.other_view(view, 0.0, 250.0) == "갈라 봄"
    assert cc.other_view(view, 150.0, 1000.0) == "함께 헷갈림"
    assert cc.other_view(view, 2000.0, 3000.0) == "두 사람이 안 보임"


def test_pair_table_has_one_row_per_video():
    rows = [{"pair": "C05", "cam": "angle1", "start": 8600.0, "end": 9600.0, "other": "갈라 봄"},
            {"pair": "C05", "cam": "angle0", "start": 7000.0, "end": 7500.0, "other": "함께 헷갈림"}]
    table = cc.pair_table(rows, ["C05", "C06"], {"C05": {"people": 2}, "C06": {"people": 2}})
    assert table[2] == "| C05 | 2 | 1번 1.0초 | 1 / 0 / 0 | 1번 0.5초 | 0 / 1 / 0 |"
    assert table[3] == "| C06 | 2 | 0 | - | 0 | - |"
