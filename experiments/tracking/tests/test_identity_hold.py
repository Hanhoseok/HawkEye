"""신원 보류 두 가지 (후속 43, docs/multiview-progress.md).

- identity.nested_hold_overlap: 새 신원을 만들려는 상자가 이미 신원이 있는 상자와 포개져 있으면 만들지 않고 보류.
  3차 C03: 한 사람 위에 상자가 둘(머리 상자 + 큰 상자, 윗몸 상자 + 온몸 상자) 나와 두 번째 상자가 새 신원이 됐다.
- identity.min_identity_score: 탐지 점수가 낮은 상자로는 신원을 만들지도, 되살아난 번호를 다시 판정하지도 않고 보류.
  3차 C03: 놓친 손님 번호가 사람 없는 창가 상자(0.26)에 되살아나 그 손님 신원이 됐다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import IdentityConfig  # noqa: E402
from src.core.types import Frame, TrackObservation  # noqa: E402
from src.identity.registry import IdentityRegistry  # noqa: E402

DIM = 16
IMAGE = np.zeros((680, 920, 3), dtype=np.uint8)
A_BODY = (400.0, 200.0, 500.0, 600.0)     # 손님 A 의 온몸 상자
A_HEAD = (420.0, 200.0, 480.0, 280.0)     # A 의 머리만 덮은 상자 (온몸 상자 안에 통째로)
A_UPPER = (400.0, 200.0, 500.0, 400.0)    # A 의 윗몸 상자
AWAY = (700.0, 200.0, 800.0, 600.0)       # 떨어진 자리
WINDOW = (100.0, 50.0, 200.0, 350.0)      # 사람 없는 창가 상자


def axis(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


class SpotEmbedder:
    """상자마다 정해진 외형. A 의 상자들은 A(0), 떨어진 자리는 B(1), 머리·창가 상자는 누구와도 안 닮음."""

    looks = {A_BODY: axis(0), A_UPPER: axis(0), A_HEAD: axis(2), AWAY: axis(1), WINDOW: axis(3)}

    def embed(self, image, boxes):
        return np.array([self.looks[tuple(float(v) for v in b)] for b in boxes])


def registry(**cfg) -> IdentityRegistry:
    base = dict(min_tensor_height=0, stitch_enabled=False, late_merge_enabled=False, edge_margin=0.0)
    return IdentityRegistry(IdentityConfig(**{**base, **cfg}), embedder=SpotEmbedder())


def show(reg: IdentityRegistry, t: float, tracks: list[tuple]):
    """tracks: (추적 번호, 상자) 또는 (추적 번호, 상자, 탐지 점수)."""
    frame = Frame(index=int(t * 10), timestamp="t", pts_ms=t * 1000.0, image=IMAGE)
    obs = [TrackObservation(track_id=t_[0], bbox=t_[1], frame=frame.index, timestamp="t",
                            score=t_[2] if len(t_) > 2 else 0.9) for t_ in tracks]
    return {o.track_id: o.person_id for o in reg.assign(frame, obs)}


# ---------------------------------------------------------------- 포개진 상자

def test_nested_measure_is_overlap_over_the_smaller_box():
    n = IdentityRegistry._nested
    assert n(A_HEAD, A_BODY) == 1.0                       # 통째로 안에 듦
    assert n(A_BODY, A_HEAD) == 1.0                       # 순서와 무관
    assert n(A_BODY, AWAY) == 0.0
    assert abs(n(A_UPPER, A_BODY) - 1.0) < 1e-9
    half = (450.0, 200.0, 550.0, 600.0)                    # 옆으로 반만 걸침
    assert abs(n(A_BODY, half) - 0.5) < 1e-9


def test_a_box_nested_in_a_visible_identity_gets_no_new_identity():
    reg = registry(nested_hold_overlap=0.7)
    a = show(reg, 0.0, [(1, A_BODY)])[1]
    out = show(reg, 0.1, [(1, A_BODY), (2, A_HEAD)])
    assert out == {1: a, 2: -1}, "머리 상자가 새 신원이 됐다"
    assert reg.holds[-1]["why"] == "nested" and reg.holds[-1]["track"] == 2
    # 꺼 두면 지금까지처럼 새 신원
    off = registry()
    a = show(off, 0.0, [(1, A_BODY)])[1]
    assert show(off, 0.1, [(1, A_BODY), (2, A_HEAD)])[2] not in (a, -1)


def test_held_box_takes_the_identity_once_the_first_track_is_gone():
    """C03: 윗몸 상자(기존 번호)와 온몸 상자(새 번호)가 함께 나오다가 추적기가 온몸 상자만 이어 간다 → 같은 신원이어야 한다."""
    reg = registry(nested_hold_overlap=0.7)
    a = show(reg, 0.0, [(1, A_UPPER)])[1]
    assert show(reg, 0.1, [(1, A_UPPER), (2, A_BODY)])[2] == -1
    assert show(reg, 0.2, [(2, A_BODY)])[2] == a


def test_nested_box_may_still_join_an_existing_identity_that_is_not_visible():
    """포개져 있어도 기존 신원과 외형이 맞으면 잇는다 — 미루는 것은 '새로 만드는 것'뿐."""
    near = (410.0, 200.0, 510.0, 600.0)
    emb = SpotEmbedder()
    emb.looks = {**SpotEmbedder.looks, near: axis(1)}       # A 바로 옆에 포개 선 B
    reg = IdentityRegistry(IdentityConfig(min_tensor_height=0, stitch_enabled=False, late_merge_enabled=False,
                                          edge_margin=0.0, nested_hold_overlap=0.7), embedder=emb)
    b = show(reg, 0.0, [(1, AWAY)])[1]
    a = show(reg, 1.0, [(2, A_BODY)])[2]
    assert show(reg, 2.0, [(2, A_BODY), (3, near)]) == {2: a, 3: b}


def test_a_box_apart_from_everyone_still_gets_a_new_identity():
    reg = registry(nested_hold_overlap=0.7)
    a = show(reg, 0.0, [(1, A_BODY)])[1]
    b = show(reg, 0.1, [(1, A_BODY), (2, AWAY)])[2]
    assert b not in (a, -1)
    assert not reg.holds


def test_two_new_nested_boxes_in_one_frame_make_one_identity():
    reg = registry(nested_hold_overlap=0.7)
    out = show(reg, 0.0, [(1, A_BODY), (2, A_HEAD)])
    assert out[1] > 0 and out[2] == -1


def test_revived_track_held_as_nested_keeps_its_owner_for_the_next_look():
    """되살아난 번호가 포개져 보류되면, 다음 장면에도 '되살아난 번호'로 다시 본다 (원래 주인 우대가 남는다)."""
    reg = registry(nested_hold_overlap=0.7, track_gap_recheck_seconds=1.0)
    b = show(reg, 0.0, [(1, AWAY)])[1]
    a = show(reg, 0.1, [(1, AWAY), (2, A_BODY)])[2]
    show(reg, 0.2, [(2, A_BODY)])
    # 2초 뒤 번호 1 이 A 의 머리 상자에 되살아남: B 와도 안 닮았고 A 상자 안이다 → 보류
    assert show(reg, 2.2, [(2, A_BODY), (1, A_HEAD)]) == {2: a, 1: -1}
    assert reg._recheck.get(1) == b
    # 번호 1 이 제자리(B)로 돌아오면 B 로 이어진다
    assert show(reg, 2.3, [(2, A_BODY), (1, AWAY)])[1] == b


# ---------------------------------------------------------------- 낮은 탐지 점수

def test_low_score_box_gets_no_identity_until_the_score_rises():
    reg = registry(min_identity_score=0.4)
    assert show(reg, 0.0, [(1, WINDOW, 0.26)])[1] == -1
    assert reg.holds[-1] == {"track": 1, "ms": 0.0, "why": "score"}
    assert show(reg, 0.1, [(1, WINDOW, 0.39)])[1] == -1
    assert show(reg, 0.2, [(1, WINDOW, 0.40)])[1] > 0
    off = registry()
    assert show(off, 0.0, [(1, WINDOW, 0.26)])[1] > 0


def test_a_track_that_already_has_an_identity_keeps_it_when_the_score_dips():
    reg = registry(min_identity_score=0.4)
    a = show(reg, 0.0, [(1, A_BODY, 0.9)])[1]
    assert show(reg, 0.1, [(1, A_BODY, 0.26)])[1] == a
    assert not reg.holds


def test_revived_track_on_a_low_score_box_does_not_bring_the_identity_back():
    """C03 13.33초: 놓친 손님 번호가 창가의 낮은 점수 상자에 되살아남 → 손님 신원을 주지 않고, 그 신원은 비어 있어야 한다."""
    reg = registry(min_identity_score=0.4, track_gap_recheck_seconds=1.0)
    a = show(reg, 0.0, [(1, A_BODY)])[1]
    out = show(reg, 2.0, [(1, WINDOW, 0.26), (2, A_BODY)])   # 진짜 A 는 새 번호 2 로 같은 장면에
    assert out == {1: -1, 2: a}
    assert reg.holds[-1]["why"] == "score"
    # 점수가 오르면 그때 다시 판정한다 — A 는 이미 보이는 중이라 창가 상자는 새 신원
    later = show(reg, 2.1, [(1, WINDOW, 0.6), (2, A_BODY)])
    assert later[2] == a and later[1] not in (a, -1)


def test_a_short_gap_is_not_a_revival():
    """1초 안에 이어진 번호는 다시 판정하지 않으므로 점수가 낮아도 신원이 그대로다."""
    reg = registry(min_identity_score=0.4, track_gap_recheck_seconds=1.0)
    a = show(reg, 0.0, [(1, A_BODY)])[1]
    assert show(reg, 0.5, [(1, A_BODY, 0.3)])[1] == a
