"""오래 끊겼다 되살아난 track 의 신원 다시 판정 (identity.track_gap_recheck_seconds, C-18).

배경: BoT-SORT 는 사라진 track 을 몇 초 동안 기억했다가 외형이 느슨하게 맞으면 같은 번호로 되살린다.
되살린 상대가 다른 사람이어도 신원 계층은 그 번호의 신원을 그대로 물려줬다 (2차 촬영 C12:
입장 순간 잠깐 잡힌 조각 번호가 3초 뒤 다른 손님에게 되살아나 한 사람이 두 손님이 됨).
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
A_SPOT = (400.0, 300.0, 480.0, 500.0)    # 손님 A 가 처음 서 있던 자리
B_SPOT = (100.0, 300.0, 180.0, 500.0)    # 조각 B 가 잠깐 잡힌 자리
A_LATER = (600.0, 300.0, 680.0, 500.0)   # 몇 초 뒤 A 가 나타난 다른 자리


def axis(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


class SpotEmbedder:
    """자리마다 정해진 외형: A 의 두 자리는 A, B 자리는 B."""

    looks = {A_SPOT: axis(0), A_LATER: axis(0), B_SPOT: axis(1)}

    def embed(self, image, boxes):
        return np.array([self.looks[tuple(float(v) for v in b)] for b in boxes])


def registry(**cfg) -> IdentityRegistry:
    base = dict(min_tensor_height=0, stitch_enabled=False, late_merge_enabled=False, edge_margin=0.0)
    return IdentityRegistry(IdentityConfig(**{**base, **cfg}), embedder=SpotEmbedder())


def show(reg: IdentityRegistry, t: float, tracks: list[tuple[int, tuple]]):
    frame = Frame(index=int(t * 10), timestamp="t", pts_ms=t * 1000.0, image=IMAGE)
    obs = [TrackObservation(track_id=tid, bbox=b, frame=frame.index, timestamp="t", score=0.9) for tid, b in tracks]
    return {o.track_id: o for o in reg.assign(frame, obs)}


def scenario(reg: IdentityRegistry, revive_after: float):
    """A(track 1) 1초, 조각 B(track 2) 0.2초 → 둘 다 사라짐 → track 2 번호가 A 에게 되살아남."""
    a = show(reg, 0.0, [(1, A_SPOT)])[1].person_id
    b = show(reg, 0.1, [(1, A_SPOT), (2, B_SPOT)])[2].person_id
    show(reg, 0.2, [(1, A_SPOT), (2, B_SPOT)])
    show(reg, 1.0, [(1, A_SPOT)])
    revived = show(reg, 0.2 + revive_after, [(2, A_LATER)])[2]
    return a, b, revived


def test_long_gap_revival_is_judged_again():
    a, b, revived = scenario(registry(track_gap_recheck_seconds=1.0), revive_after=3.0)
    assert a != b
    assert revived.person_id == a          # 번호(조각 B)가 아니라 외형(A)을 따름
    assert revived.is_new is False and revived.matched_similarity == 1.0


def test_off_by_default_keeps_old_behaviour():
    a, b, revived = scenario(registry(), revive_after=3.0)
    assert revived.person_id == b          # 예전처럼 번호의 신원을 물려줌


def test_short_gap_keeps_the_identity():
    # 가림 몇 장(1초 이하)은 tracker 가 메우는 정상 공백 — 다시 판정하지 않는다
    reg = registry(track_gap_recheck_seconds=1.0)
    a, b, revived = scenario(reg, revive_after=0.8)
    assert revived.person_id == b


def test_revival_of_a_stranger_becomes_a_new_person():
    # 되살아난 번호에 아무와도 안 닮은 사람이 붙으면 새 신원
    stranger = (700.0, 300.0, 780.0, 500.0)
    SpotEmbedder.looks[stranger] = axis(5)
    reg = registry(track_gap_recheck_seconds=1.0)
    a = show(reg, 0.0, [(1, A_SPOT)])[1].person_id
    assert show(reg, 0.5, [(1, A_SPOT)])[1].person_id == a
    revived = show(reg, 3.0, [(1, stranger)])[1]
    assert revived.person_id != a and revived.is_new


def weak(look: np.ndarray, other: int, cos: float) -> np.ndarray:
    """look 과 코사인 cos 인 사진 (같은 사람인데 한 장 비교로는 기준 0.65 에 못 미침)."""
    v = cos * look + (1 - cos**2) ** 0.5 * axis(other)
    return v.astype(np.float32)


def test_revival_of_the_same_person_keeps_identity_with_weak_photo():
    # C21: tracker 가 맞게 이어 붙였는데 되살아난 순간 사진이 덜 닮음(0.5) — 원래 신원 우대로 유지
    spot = (300.0, 300.0, 380.0, 500.0)
    SpotEmbedder.looks[spot] = weak(axis(0), 7, 0.5)
    reg = registry(track_gap_recheck_seconds=1.0)
    a = show(reg, 0.0, [(1, A_SPOT)])[1].person_id
    revived = show(reg, 3.0, [(1, spot)])[1]
    assert revived.person_id == a and revived.stitched


def test_recheck_waits_until_body_is_off_the_edge():
    edge = (0.0, 300.0, 80.0, 500.0)          # 왼쪽 가장자리에 걸림
    SpotEmbedder.looks[edge] = axis(9)
    reg = registry(track_gap_recheck_seconds=1.0, edge_margin=0.02)
    a = show(reg, 0.0, [(2, B_SPOT)])[2].person_id
    assert show(reg, 3.0, [(2, edge)])[2].person_id == a      # 잘린 동안은 원래 신원
    show(reg, 2.0 + 1.5, [(1, A_SPOT)])                         # (다른 손님 A 등장)
    moved = show(reg, 3.6, [(2, A_LATER)])[2]                   # 몸이 다 보이자 다시 판정 → 외형 A
    assert moved.person_id != a


def test_too_small_to_recheck_is_held_not_inherited():
    # C12: 되살아난 번호가 너무 작은 사람(창밖 행인)에게 붙음 → 신원 보류(-1), 원래 신원은 비어 있게 됨
    tiny = (600.0, 300.0, 620.0, 330.0)
    SpotEmbedder.looks[tiny] = axis(0)
    reg = registry(track_gap_recheck_seconds=1.0, min_tensor_height=100)
    a = show(reg, 0.0, [(1, A_SPOT)])[1].person_id
    assert show(reg, 3.0, [(1, tiny)])[1].person_id == -1
    back = show(reg, 3.1, [(2, A_LATER)])[2]          # 진짜 주인이 다른 번호로 돌아옴
    assert back.person_id == a
    assert show(reg, 3.2, [(1, A_SPOT), (2, A_LATER)])[1].person_id != a   # 커지면 다시 판정 (a 는 점유 중)


def test_edge_sides_skip_a_side_the_camera_always_cuts():
    # 전체 카메라는 머리가 위쪽 끝에 늘 잘린다 — 위쪽에 닿은 새 track 도 즉시 대조해야 재입장을 잇는다 (C-18)
    top_cut = (400.0, 0.0, 480.0, 200.0)
    SpotEmbedder.looks[top_cut] = axis(0)
    for sides, linked in [(("left", "top", "right", "bottom"), False), (("left", "right", "bottom"), True)]:
        reg = registry(edge_margin=0.02, edge_sides=sides)
        a = show(reg, 0.0, [(1, A_SPOT)])[1].person_id
        again = show(reg, 5.0, [(2, top_cut)])[2]
        assert (again.person_id == a) is linked


def test_camera_identity_overrides_reach_the_registry(tmp_path):
    from src.core.config import AppConfig, CameraConfig
    from src.multicam import build_multicam

    made = []

    class Reg:
        def __init__(self, cfg, *a, **k):
            made.append(cfg)
            self.recent_consistency = None

    class Dummy:
        def __init__(self, *a, **k):
            pass

    floor = Path(__file__).resolve().parents[1] / "zones" / "store01_dual" / "floor.yaml"
    cfg = AppConfig()
    cfg.cameras = [CameraConfig("angle1", "없음.mp4", "overview", str(floor), identity={"edge_sides": ["left", "right"]})]
    try:
        build_multicam(cfg, tmp_path, build_detector=lambda c: None, build_tracker=lambda c: None,
                       build_embedder=lambda c: None, build_item_detector=None, IdentityRegistry=Reg,
                       InteractionDetector=Dummy, ShelfWatcher=None, RiskEngine=Dummy, PaymentFeed=None)
    except Exception:
        pass  # 영상이 없어 뒤 단계에서 멈춰도 된다 — 신원 계층이 만들어졌는지만 본다
    assert made and made[0].edge_sides == ("left", "right")
    assert cfg.identity.edge_sides == ("left", "top", "right", "bottom")   # 기본 설정은 그대로
