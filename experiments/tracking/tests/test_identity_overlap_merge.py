"""나중에 합치기의 짧은 겹침 허용 (identity.late_merge_overlap_seconds, 후속 31).

3차 C14: 손님이 문으로 나가는 0.66초 동안 유리 밖 모습에 새 신원이 먼저 생겨, 원래 신원과 '함께 보인 적이 있다'는
이유로 돌아온 뒤에도 합쳐지지 않았다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import IdentityConfig  # noqa: E402
from src.core.types import Frame, TrackObservation  # noqa: E402
from src.identity.registry import IdentityRegistry  # noqa: E402

IMAGE = np.zeros((680, 920, 3), dtype=np.uint8)
DOOR = (100.0, 300.0, 180.0, 500.0)       # 나가는 손님
GLASS = (500.0, 100.0, 560.0, 250.0)      # 같은 손님이 유리 밖에 보임 (같은 생김새)


class SameLook:
    def embed(self, image, boxes):
        v = np.zeros(16, dtype=np.float32)
        v[0] = 1.0
        return np.array([v for _ in boxes])


def run(overlap: float) -> IdentityRegistry:
    reg = IdentityRegistry(IdentityConfig(min_tensor_height=0, stitch_enabled=False, edge_margin=0.0,
                                          late_merge_enabled=True, late_merge_samples=(20,),
                                          template_refresh_seconds=0.1, late_merge_overlap_seconds=overlap),
                           embedder=SameLook())

    def show(t, tracks):
        frame = Frame(index=int(round(t * 10)), timestamp="t", pts_ms=t * 1000.0, image=IMAGE)
        obs = [TrackObservation(track_id=tid, bbox=b, frame=frame.index, timestamp="t", score=0.9) for tid, b in tracks]
        return {o.track_id: o.person_id for o in reg.assign(frame, obs)}

    for i in range(10):                                   # 0~0.9초: 문 쪽 손님만
        a = show(i * 0.1, [(1, DOOR)])[1]
    for i in range(5):                                    # 1.0~1.4초: 손님 + 유리 밖 모습 함께 (0.5초 겹침)
        b = show(1.0 + i * 0.1, [(1, DOOR), (2, GLASS)])[2]
    for i in range(20):                                   # 1.5초~: 유리 밖 모습만 (손님은 나감)
        show(1.5 + i * 0.1, [(2, GLASS)])
    assert a != b
    return reg


def test_brief_overlap_blocks_merge_by_default():
    assert run(0.0).merges == []


def test_brief_overlap_within_allowance_is_merged():
    merges = run(1.0).merges
    assert len(merges) == 1 and merges[0]["from"] != merges[0]["into"]


def test_longer_overlap_than_allowance_still_blocks():
    assert run(0.3).merges == []


def test_people_still_seen_together_are_never_merged():
    """겹침 허용 1초라도 두 사람이 계속 함께 보이면 합치지 않는다 (사진 수가 1초 전에 차도)."""
    reg = IdentityRegistry(IdentityConfig(min_tensor_height=0, stitch_enabled=False, edge_margin=0.0,
                                          late_merge_enabled=True, late_merge_samples=(5, 10, 20),
                                          template_refresh_seconds=0.1, late_merge_overlap_seconds=1.0),
                           embedder=SameLook())
    ids = set()
    for i in range(40):
        frame = Frame(index=i, timestamp="t", pts_ms=i * 100.0, image=IMAGE)
        obs = [TrackObservation(track_id=tid, bbox=b, frame=i, timestamp="t", score=0.9) for tid, b in ((1, DOOR), (2, GLASS))]
        out = reg.assign(frame, obs)
        ids.add(tuple(sorted(o.person_id for o in out)))
    assert reg.merges == []
    assert all(len(set(t)) == 2 for t in ids)
