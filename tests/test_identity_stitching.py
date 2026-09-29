"""위치·시간 근거(track stitching) 테스트.

배경: 천장 시점에서는 같은 사람끼리의 생김새 유사도가 0.58~0.72 로 떨어져
기준(0.65)을 걸친다. 그래서 끊겼다 다시 나타날 때 새 신원으로 등록되어
한 사람이 둘로 쪼개졌다 (docs/merl-evaluation.md §4-E).

확인하는 것 — **합쳐야 할 때는 합치고, 합치면 안 될 때는 안 합치는가.**
뒤쪽이 더 중요하다. 다른 사람을 합치면 A 가 집은 물건이 B 에게 붙어
아무것도 안 집은 손님에게 경보가 울린다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import IdentityConfig  # noqa: E402
from src.core.types import Frame, TrackObservation  # noqa: E402
from src.identity.registry import IdentityRegistry  # noqa: E402

REFERENCE = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def vec(similarity: float) -> np.ndarray:
    """REFERENCE 와의 코사인 유사도가 정확히 similarity 인 단위 벡터."""
    return np.array([similarity, (1 - similarity**2) ** 0.5, 0.0, 0.0], dtype=np.float32)


class ScriptedEmbedder:
    """bbox 마다 돌려줄 임베딩을 테스트가 직접 정한다.

    위치와 생김새를 독립적으로 조절해야 하므로, 좌표로 임베딩을 만들지 않고 표를 쓴다.
    """

    def __init__(self, table: dict[tuple, np.ndarray]):
        self.table = {tuple(round(v, 1) for v in k): v for k, v in table.items()}

    def embed(self, image, boxes):
        return np.array([self.table[tuple(round(float(v), 1) for v in b)] for b in boxes])


def frame(t_sec: float) -> Frame:
    return Frame(
        index=int(t_sec * 30),
        timestamp="t",
        pts_ms=t_sec * 1000.0,
        image=np.zeros((680, 920, 3), dtype=np.uint8),
    )


def obs(track_id: int, bbox: tuple, t_sec: float) -> TrackObservation:
    return TrackObservation(
        track_id=track_id, bbox=bbox, frame=int(t_sec * 30), timestamp="t", score=0.9
    )


# 사람 키 200px 기준 좌표들
HERE = (400.0, 300.0, 480.0, 500.0)          # 사라진 자리
NEARBY = (420.0, 310.0, 500.0, 510.0)        # 발 위치 약 22px 차이 -> 체구의 0.11배
FAR = (50.0, 300.0, 130.0, 500.0)            # 발 위치 350px 차이 -> 체구의 1.75배
OTHER_SPOT = (700.0, 300.0, 780.0, 500.0)


def registry(embedder, **cfg) -> IdentityRegistry:
    base = dict(min_tensor_height=0, match_threshold=0.65, stitch_threshold=0.45,
                stitch_max_seconds=3.0, stitch_max_distance=1.0)
    base.update(cfg)
    return IdentityRegistry(IdentityConfig(**base), embedder, imgsz=640)


def test_reappears_nearby_soon_with_weak_appearance_is_stitched():
    """핵심 시나리오. 생김새 0.55 는 기준(0.65) 미만이지만, 같은 자리 1초 뒤라 잇는다."""
    reg = registry(ScriptedEmbedder({HERE: REFERENCE, NEARBY: vec(0.55)}))
    first = reg.assign(frame(10.0), [obs(1, HERE, 10.0)])[0]
    again = reg.assign(frame(11.0), [obs(2, NEARBY, 11.0)])[0]
    assert again.person_id == first.person_id, "같은 자리에서 1초 뒤 나타났는데 새 손님이 됐다"
    assert again.stitched, "위치·시간 덕분에 이어붙였다는 표시가 없다"


def test_same_case_without_stitching_splits():
    """비교용. stitching 을 끄면 기존처럼 쪼개진다 — 이 기능이 실제로 차이를 만든다는 증거."""
    reg = registry(ScriptedEmbedder({HERE: REFERENCE, NEARBY: vec(0.55)}), stitch_enabled=False)
    first = reg.assign(frame(10.0), [obs(1, HERE, 10.0)])[0]
    again = reg.assign(frame(11.0), [obs(2, NEARBY, 11.0)])[0]
    assert again.person_id != first.person_id


def test_far_away_is_not_stitched():
    """생김새가 같은 수준(0.55)이어도, 멀리서 나타나면 이어붙이지 않는다."""
    reg = registry(ScriptedEmbedder({HERE: REFERENCE, FAR: vec(0.55)}))
    first = reg.assign(frame(10.0), [obs(1, HERE, 10.0)])[0]
    other = reg.assign(frame(11.0), [obs(2, FAR, 11.0)])[0]
    assert other.person_id != first.person_id, "멀리서 나타난 사람을 위치 근거로 합쳤다"


def test_long_gap_is_not_stitched():
    """같은 자리라도 한참 뒤(3초 초과)면 위치 근거를 쓰지 않는다."""
    reg = registry(ScriptedEmbedder({HERE: REFERENCE, NEARBY: vec(0.55)}))
    first = reg.assign(frame(10.0), [obs(1, HERE, 10.0)])[0]
    later = reg.assign(frame(20.0), [obs(2, NEARBY, 20.0)])[0]
    assert later.person_id != first.person_id, "10초 뒤인데 위치 근거로 합쳤다"


def test_different_person_stepping_into_the_spot_is_not_merged():
    """가장 중요한 안전장치.

    A 가 사라진 자리에 B 가 곧바로 들어서는 일은 붐비는 매장에서 실제로 일어난다.
    생김새가 전혀 다르면(0.30 < 완화 기준 0.45) 위치·시간이 맞아도 합치지 않는다.
    """
    reg = registry(ScriptedEmbedder({HERE: REFERENCE, NEARBY: vec(0.30)}))
    a = reg.assign(frame(10.0), [obs(1, HERE, 10.0)])[0]
    b = reg.assign(frame(10.5), [obs(2, NEARBY, 10.5)])[0]
    assert b.person_id != a.person_id, "생김새가 전혀 다른 사람을 자리만 보고 합쳤다"


def test_nearby_recent_person_preferred_over_better_looking_far_one():
    """후보가 둘일 때: 방금 같은 자리에서 사라진 사람(0.55)이
    20초 전 멀리서 사라진 사람(0.66)보다 그럴듯하다."""
    # X 와 Y 는 서로 전혀 안 닮은 사람이어야 하므로 직교하는 벡터로 둔다.
    x_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)   # 사람 X: 20초 전, 먼 곳
    y_vec = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)   # 사람 Y: 방금, 여기
    # 새로 나타난 사람: X 와 0.66 (생김새만으로 기준 통과), Y 와 0.55 (기준 미달)
    rest = (1 - 0.66**2 - 0.55**2) ** 0.5
    new = np.array([0.66, 0.55, rest, 0.0], dtype=np.float32)
    table = {OTHER_SPOT: x_vec, HERE: y_vec, NEARBY: new}
    x_sim = float(x_vec @ new)

    reg = registry(ScriptedEmbedder(table), retire_after_seconds=60.0)
    x = reg.assign(frame(0.0), [obs(1, OTHER_SPOT, 0.0)])[0]
    y_obs = reg.assign(frame(19.0), [obs(2, HERE, 19.0)])[0]
    assert y_obs.person_id != x.person_id
    got = reg.assign(frame(20.0), [obs(3, NEARBY, 20.0)])[0]
    assert got.person_id == y_obs.person_id, (
        f"방금 같은 자리에서 사라진 사람(0.55) 대신 멀리 있던 사람({x_sim:.2f})에게 붙었다"
    )


def test_strong_appearance_match_is_not_marked_as_stitched():
    """생김새만으로 기준을 넘었다면 위치 근거 덕분이 아니므로 stitched=False."""
    reg = registry(ScriptedEmbedder({HERE: REFERENCE, NEARBY: vec(0.90)}))
    first = reg.assign(frame(10.0), [obs(1, HERE, 10.0)])[0]
    again = reg.assign(frame(11.0), [obs(2, NEARBY, 11.0)])[0]
    assert again.person_id == first.person_id
    assert not again.stitched


def test_visible_person_is_never_stitched():
    """지금 보이고 있는 사람에게는 절대 붙이지 않는다 (기존 안전장치 유지)."""
    reg = registry(ScriptedEmbedder({HERE: REFERENCE, NEARBY: vec(0.99)}))
    out = reg.assign(frame(10.0), [obs(1, HERE, 10.0), obs(2, NEARBY, 10.0)])
    assert len({o.person_id for o in out}) == 2, "같은 순간에 보이는 두 사람을 합쳤다"
