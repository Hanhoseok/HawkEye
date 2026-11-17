"""자리를 건너뛴 추적 번호 다시 판정(identity.track_jump_recheck)과 한 장면 한 신원 한 상자(identity.one_box_per_identity).

배경 (3차 T09, 후속 47·48): 창가 그림에 붙어 있던 번호가 0.6초 끊겼다가 문 앞의 진짜 손님에게 붙었다.
1초가 안 돼 다시 판정하지 않았고, 손님이 '창가 상자의 신원'을 물려받아 그 뒤 창가 상자까지 손님으로 그려졌다.
그 사이 진짜 손님과 창가 상자가 같은 신원을 달고 함께 나왔다.
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
WINDOW = (100.0, 100.0, 180.0, 300.0)     # 창가 그림 자리
DOOR = (500.0, 100.0, 580.0, 300.0)       # 문 앞 — 창가 상자와 전혀 겹치지 않는다
DOOR_NEXT = (510.0, 100.0, 590.0, 300.0)
SHIFTED = (150.0, 100.0, 230.0, 300.0)    # 창가 상자와 겹치는 자리 (조금 움직임)


def axis(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def weak(look: np.ndarray, other: int, cos: float) -> np.ndarray:
    return (cos * look + (1 - cos**2) ** 0.5 * axis(other)).astype(np.float32)


class SpotEmbedder:
    """자리마다 정해진 외형."""

    def __init__(self, looks: dict):
        self.looks = looks

    def embed(self, image, boxes):
        return np.array([self.looks[tuple(float(v) for v in b)] for b in boxes])


def registry(looks: dict, **cfg) -> IdentityRegistry:
    base = dict(min_tensor_height=0, late_merge_enabled=False, edge_margin=0.0, track_gap_recheck_seconds=1.0)
    return IdentityRegistry(IdentityConfig(**{**base, **cfg}), embedder=SpotEmbedder(looks))


def show(reg: IdentityRegistry, t: float, tracks: list[tuple]):
    frame = Frame(index=round(t * 10), timestamp="t", pts_ms=t * 1000.0, image=IMAGE)
    obs = [TrackObservation(track_id=tid, bbox=b, frame=frame.index, timestamp="t", score=s[0] if s else 0.9)
           for tid, b, *s in tracks]
    return {o.track_id: o for o in reg.assign(frame, obs)}


# ---------------------------------------------------------------------- 자리를 건너뛴 번호

def jump(reg: IdentityRegistry, landed=DOOR):
    """창가 상자(track 1) → 0.6초 끊김 → 같은 번호가 landed 자리에 나타남."""
    ghost = show(reg, 0.0, [(1, WINDOW)])[1].person_id
    show(reg, 0.1, [(1, WINDOW)])
    for t in (0.2, 0.3, 0.4, 0.5, 0.6):
        show(reg, t, [])
    return ghost, show(reg, 0.7, [(1, landed)])[1]


def test_off_by_default_the_jumped_track_inherits_the_identity():
    ghost, landed = jump(registry({WINDOW: axis(0), DOOR: axis(1)}))
    assert landed.person_id == ghost


def test_jumped_track_with_a_different_look_becomes_a_new_person():
    reg = registry({WINDOW: axis(0), DOOR: axis(1)}, track_jump_recheck=True)
    ghost, landed = jump(reg)
    assert landed.person_id != ghost and landed.is_new
    assert reg.jumps and reg.jumps[0]["person"] == ghost


def test_jumped_track_gets_no_bonus_from_its_old_identity():
    # 외형 0.5: 번호·위치 가산점이 있으면 넘는 값(0.45 이상)이지만 생김새만으로는 기준(0.65) 미달 → 새 신원
    reg = registry({WINDOW: axis(0), DOOR: weak(axis(0), 5, 0.5)}, track_jump_recheck=True, stitch_max_distance=10.0)
    ghost, landed = jump(reg)
    assert landed.person_id != ghost


def test_jumped_track_that_still_looks_the_same_keeps_the_identity():
    reg = registry({WINDOW: axis(0), DOOR: weak(axis(0), 5, 0.8)}, track_jump_recheck=True)
    ghost, landed = jump(reg)
    assert landed.person_id == ghost and not landed.stitched


def test_track_that_returns_overlapping_its_old_box_is_not_rechecked():
    reg = registry({WINDOW: axis(0), SHIFTED: axis(1)}, track_jump_recheck=True)
    ghost, landed = jump(reg, landed=SHIFTED)
    assert landed.person_id == ghost and not reg.jumps


def test_continuous_track_is_not_rechecked_even_if_the_box_moves_away():
    # 끊기지 않고 이어진 번호는 이 규칙의 대상이 아니다 (끊김이 있어야 한다)
    reg = registry({WINDOW: axis(0), DOOR: axis(1)}, track_jump_recheck=True)
    ghost = show(reg, 0.0, [(1, WINDOW)])[1].person_id
    assert show(reg, 0.1, [(1, DOOR)])[1].person_id == ghost
    assert not reg.jumps


def test_long_gap_still_favors_the_old_identity():
    # 1초 넘게 끊긴 번호는 예전 규칙 그대로 — 원래 신원을 우대해 다시 판정 (C21)
    reg = registry({WINDOW: axis(0), DOOR: weak(axis(0), 5, 0.5)}, track_jump_recheck=True)
    ghost = show(reg, 0.0, [(1, WINDOW)])[1].person_id
    show(reg, 0.1, [(1, WINDOW)])
    back = show(reg, 3.0, [(1, DOOR)])[1]
    assert back.person_id == ghost and back.stitched


def test_denied_identity_stays_denied_until_the_track_is_named():
    # 건너뛴 직후 상자가 너무 작아 보류됐다가 커진 뒤에도, 옛 신원의 가산점은 없다
    tiny = (500.0, 100.0, 520.0, 130.0)
    looks = {WINDOW: axis(0), tiny: axis(0), DOOR: weak(axis(0), 5, 0.5)}
    reg = registry(looks, track_jump_recheck=True, stitch_max_distance=10.0)
    ghost = show(reg, 0.0, [(1, WINDOW)])[1].person_id
    reg._min_bbox_px = 100.0
    show(reg, 0.1, [(1, WINDOW)])
    show(reg, 0.2, [])
    assert show(reg, 0.3, [(1, tiny)])[1].person_id == -1
    grown = show(reg, 0.4, [(1, DOOR)])[1]
    assert grown.person_id not in (-1, ghost)


# ---------------------------------------------------------------------- 한 장면 한 신원 한 상자

def shared(reg: IdentityRegistry):
    """track 1(A) 이 0.3초 끊긴 사이 track 2 가 A 의 신원을 받고, track 1 이 돌아와 둘이 함께 나온다."""
    a = show(reg, 0.0, [(1, WINDOW)])[1].person_id
    show(reg, 0.1, [(1, WINDOW)])
    assert show(reg, 0.2, [(2, DOOR)])[2].person_id == a      # 외형이 같아 A 로 이어짐
    show(reg, 0.3, [(2, DOOR)])
    return a, show(reg, 0.4, [(1, SHIFTED), (2, DOOR_NEXT)])


LOOKS = {WINDOW: axis(0), DOOR: axis(0), DOOR_NEXT: axis(0), SHIFTED: axis(0)}


def test_off_by_default_two_boxes_share_one_identity():
    a, both = shared(registry(dict(LOOKS)))
    assert both[1].person_id == a and both[2].person_id == a


def test_only_the_most_recent_holder_keeps_the_identity():
    reg = registry(dict(LOOKS), one_box_per_identity=True)
    a, both = shared(reg)
    assert both[2].person_id == a                 # 직전 장면까지 그 신원을 이어 온 쪽
    assert both[1].person_id != a and both[1].is_new
    assert [(d["track"], d["ms"], d["person"], d["kept"]) for d in reg.duplicates] == [(1, 400.0, a, 2)]


def test_the_box_that_looks_like_the_identity_keeps_it_even_if_it_returned_later():
    # T09: 창가 상자의 신원을 문 앞 손님이 잠깐 받아 갔다(덜 닮음 0.7). 창가 상자가 돌아오면 더 닮은 창가 상자가 그 신원을 지킨다
    looks = {**LOOKS, DOOR: weak(axis(0), 5, 0.7), DOOR_NEXT: weak(axis(0), 5, 0.7)}
    reg = registry(looks, one_box_per_identity=True, template_refresh_seconds=0.0)
    ghost = show(reg, 0.0, [(1, WINDOW)])[1].person_id
    for t in (0.1, 0.2, 0.3):
        show(reg, t, [(1, WINDOW)])                               # 그 신원의 사진은 대부분 창가 상자
    assert show(reg, 0.4, [(2, DOOR)])[2].person_id == ghost      # 창가 상자가 안 잡힌 사이 손님이 그 신원을 받음
    both = show(reg, 0.5, [(1, SHIFTED), (2, DOOR_NEXT)])
    assert both[1].person_id == ghost             # 늦게 돌아왔지만 그 신원의 평균 외형과 더 닮음
    assert both[2].person_id != ghost and both[2].is_new
    d = reg.duplicates[0]
    assert d["track"] == 2 and d["kept"] == 1 and d["likeness"] < d["kept_likeness"]


def test_the_dropped_track_stays_dropped_afterwards():
    reg = registry(dict(LOOKS), one_box_per_identity=True)
    a, both = shared(reg)
    other = both[1].person_id
    later = show(reg, 0.5, [(1, SHIFTED), (2, DOOR_NEXT)])
    assert later[2].person_id == a and later[1].person_id == other
    assert len(reg.duplicates) == 1


def test_tie_goes_to_the_higher_detection_score():
    # 둘 다 직전 장면에 보였다면(규칙을 켜기 전부터 겹쳐 있던 경우) 탐지 점수가 높은 쪽이 지킨다
    reg = registry(dict(LOOKS))
    a, _ = shared(reg)
    reg.config.one_box_per_identity = True
    both = show(reg, 0.5, [(1, SHIFTED, 0.3), (2, DOOR_NEXT, 0.9)])
    assert both[2].person_id == a and both[1].person_id != a


def test_dropped_box_nested_in_the_keeper_is_held_not_renamed():
    # 지킨 상자와 포개진 상자(한 사람 위의 두 상자)는 새 신원을 만들지 않고 보류한다 (포개짐 보류와 함께 쓸 때)
    inner = (510.0, 100.0, 570.0, 200.0)
    looks = {**LOOKS, inner: axis(3)}
    reg = registry(looks, one_box_per_identity=True, nested_hold_overlap=0.7)
    a = show(reg, 0.0, [(1, WINDOW)])[1].person_id
    show(reg, 0.1, [(1, WINDOW)])
    show(reg, 0.2, [(2, DOOR)])
    show(reg, 0.3, [(2, DOOR)])
    both = show(reg, 0.4, [(1, inner), (2, DOOR_NEXT)])
    assert both[2].person_id == a and both[1].person_id == -1
