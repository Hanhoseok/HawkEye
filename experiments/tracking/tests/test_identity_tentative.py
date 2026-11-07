"""임시 신원 (identity.tentative_seconds, C-11).

배경: 신원 등록에 유지 시간 조건이 없어, 외투·창밖 행인·문 앞 조각 같은 가짜 사람도 외형 사진과 함께
보관함에 들어가 이후 새 track 의 대조 상대·나중에 합치기 상대가 됐다 (2차 38쌍: 신원 105개 중 손님 53).
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
GHOST = (800.0, 300.0, 900.0, 600.0)      # 오른쪽 끝 외투 (매장 안 바닥 아님)
INSIDE = (400.0, 300.0, 480.0, 500.0)     # 매장 안 바닥에 선 손님
INSIDE2 = (300.0, 300.0, 380.0, 500.0)


def axis(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


class SpotEmbedder:
    looks = {GHOST: axis(0), INSIDE: axis(0), INSIDE2: axis(1)}   # 검은 외투와 검은 상의 손님이 닮음

    def embed(self, image, boxes):
        return np.array([self.looks[tuple(float(v) for v in b)] for b in boxes])


def registry(**cfg) -> IdentityRegistry:
    base = dict(min_tensor_height=0, stitch_enabled=False, late_merge_enabled=False, edge_margin=0.0)
    reg = IdentityRegistry(IdentityConfig(**{**base, **cfg}), embedder=SpotEmbedder())
    reg.set_promote_check(lambda bbox: bbox[0] < 700)   # x < 700 = 매장 안 바닥
    return reg


def show(reg: IdentityRegistry, t: float, tracks: list[tuple[int, tuple]]):
    frame = Frame(index=int(t * 10), timestamp="t", pts_ms=t * 1000.0, image=IMAGE)
    obs = [TrackObservation(track_id=tid, bbox=b, frame=frame.index, timestamp="t", score=0.9) for tid, b in tracks]
    return {o.track_id: o for o in reg.assign(frame, obs)}


def ghost_then_customer(reg: IdentityRegistry):
    """외투가 2초 잡혔다 사라지고, 닮은 손님이 새 track 으로 들어옴."""
    for i in range(21):
        g = show(reg, i * 0.1, [(1, GHOST)])[1].person_id
    customer = show(reg, 3.0, [(2, INSIDE)])[2]
    return g, customer


def test_off_by_default_customer_inherits_the_ghost():
    g, customer = ghost_then_customer(registry())
    assert customer.person_id == g          # 예전: 외투 신원과 맺어짐


def test_ghost_stays_tentative_and_is_not_a_match_target():
    reg = registry(tentative_seconds=1.0)
    g, customer = ghost_then_customer(reg)
    assert reg.is_tentative(g)
    assert customer.person_id != g and customer.is_new
    assert customer.person_id > 0             # 임시여도 번호는 그대로 내보냄


def test_customer_is_promoted_after_staying_inside():
    reg = registry(tentative_seconds=1.0)
    a = show(reg, 0.0, [(1, INSIDE)])[1].person_id
    show(reg, 0.5, [(1, INSIDE)])
    assert reg.is_tentative(a)
    show(reg, 1.0, [(1, INSIDE)])
    assert not reg.is_tentative(a) and reg.promotions[0]["person"] == a
    back = show(reg, 5.0, [(2, INSIDE)])[2]   # 정식이 된 뒤 다시 들어오면 이어짐
    assert back.person_id == a


def test_condition_must_hold_continuously():
    reg = registry(tentative_seconds=1.0)
    a = show(reg, 0.0, [(1, INSIDE)])[1].person_id
    show(reg, 0.6, [(1, GHOST)])              # 바닥 밖으로 나감 → 시계 초기화
    show(reg, 0.8, [(1, INSIDE)])
    show(reg, 1.5, [(1, INSIDE)])
    assert reg.is_tentative(a)
    show(reg, 1.8, [(1, INSIDE)])
    assert not reg.is_tentative(a)


def test_revived_track_may_still_return_to_its_tentative_owner():
    # 되살아난 번호의 원래 신원은 임시여도 대조 상대 (C21 처럼 맞게 이어진 번호를 쪼개지 않게)
    reg = registry(tentative_seconds=1.0, track_gap_recheck_seconds=1.0)
    a = show(reg, 0.0, [(1, INSIDE)])[1].person_id
    assert show(reg, 3.0, [(1, INSIDE)])[1].person_id == a


def test_tentative_people_are_left_out_of_late_merge():
    # 외투(임시, 사라짐) 뒤에 닮은 손님이 들어와 정식이 된 뒤 나중에 합치기 확인 — 외투와 합치지 않는다
    reg = registry(tentative_seconds=1.0, late_merge_enabled=True, late_merge_samples=(15,),
                   template_refresh_seconds=0.1)
    for i in range(10):
        g = show(reg, i * 0.1, [(1, GHOST)])[1].person_id
    for i in range(20):
        c = show(reg, 2.0 + i * 0.1, [(2, INSIDE)])[2].person_id
    assert not reg.is_tentative(c)
    assert not reg.merges and g != c
