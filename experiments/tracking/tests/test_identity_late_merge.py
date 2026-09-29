"""나중에 합치기(late merge) 테스트.

배경: 천장 시점에서는 사진 한 장씩 비교하면 같은 사람끼리도 유사도가 크게 흔들려,
다시 나타난 손님이 새 신원으로 등록되곤 했다(한 사람이 둘로 쪼개짐).
판정은 즉시 하되, 새 신원이 몇 초간 사진을 모으면 직전에 사라진 사람과
평균끼리 다시 비교해 같으면 합친다.

테스트에서는 이를 그대로 흉내낸다 — 사람마다 고유 방향(특징)이 있고,
사진 한 장은 그 방향에 무작위 잡음이 크게 섞인 것이다.
한 장끼리는 안 닮아 보이지만(≈0.36), 여러 장 평균끼리는 닮아 보인다(≈0.85).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import IdentityConfig  # noqa: E402
from src.core.types import Frame, TrackObservation  # noqa: E402
from src.identity.registry import IdentityRegistry  # noqa: E402

DIM = 128
SIGNAL = 0.6  # 사진 한 장이 그 사람 고유 특징과 닮은 정도


def noisy_photos(person_axis: int, count: int, seed: int) -> list[np.ndarray]:
    """고유 특징(person_axis 방향)에 무작위 잡음이 섞인 사진 count 장."""
    rng = np.random.default_rng(seed)
    base = np.zeros(DIM, dtype=np.float32)
    base[person_axis] = 1.0
    out = []
    for _ in range(count):
        noise = rng.standard_normal(DIM).astype(np.float32)
        noise -= (noise @ base) * base
        noise /= np.linalg.norm(noise)
        out.append((SIGNAL * base + (1 - SIGNAL**2) ** 0.5 * noise).astype(np.float32))
    return out


class StreamEmbedder:
    """bbox 마다 정해진 사진 목록을 순서대로 돌려준다 (같은 자리 = 같은 track 으로 본다)."""

    def __init__(self, streams: dict[tuple, list[np.ndarray]]):
        self.streams = {tuple(round(v, 1) for v in k): list(vs) for k, vs in streams.items()}

    def embed(self, image, boxes):
        return np.array([self.streams[tuple(round(float(v), 1) for v in b)].pop(0) for b in boxes])


HERE = (400.0, 300.0, 480.0, 500.0)
FAR = (50.0, 300.0, 130.0, 500.0)
IMAGE = np.zeros((680, 920, 3), dtype=np.uint8)
SAMPLES = 10


def frame_at(t: float) -> Frame:
    return Frame(index=int(t * 10), timestamp="t", pts_ms=t * 1000.0, image=IMAGE)


def feed(reg: IdentityRegistry, tracks: list[tuple[int, tuple]], start: float, frames: int):
    """tracks 를 frames 프레임 동안 함께 보여준다 (0.1초 간격). 프레임별 결과 목록."""
    history = []
    for i in range(frames):
        t = start + i * 0.1
        obs = [TrackObservation(track_id=tid, bbox=b, frame=int(t * 10), timestamp="t", score=0.9)
               for tid, b in tracks]
        history.append(reg.assign(frame_at(t), obs))
    return history


def registry(embedder, **cfg) -> IdentityRegistry:
    base = dict(
        min_tensor_height=0,
        stitch_enabled=False,        # 위치 근거 없이, 나중에 합치기만 본다
        retire_after_seconds=60.0,
        template_refresh_seconds=0.1,  # 프레임마다 사진 한 장 (테스트를 짧게)
        late_merge_samples=(SAMPLES,),
        late_merge_threshold=0.75,   # 이 합성 데이터의 척도에 맞춘 값
    )
    base.update(cfg)
    return IdentityRegistry(IdentityConfig(**base), embedder, imgsz=640)


def test_same_person_is_split_at_first_then_merged():
    """핵심 — 다시 나타난 순간에는 한 장이라 새 신원이 되지만, 사진이 쌓이면 원래 신원으로 합친다."""
    reg = registry(StreamEmbedder({HERE: noisy_photos(0, SAMPLES, seed=1),
                                   FAR: noisy_photos(0, SAMPLES, seed=2)}))
    first = feed(reg, [(1, HERE)], 0.0, SAMPLES)[-1][0]
    again = feed(reg, [(2, FAR)], 20.0, SAMPLES)          # 20초 뒤, 먼 곳

    assert again[0][0].person_id != first.person_id, "한 장 판정에서 쪼개져야 이 테스트가 의미가 있다"
    assert again[-1][0].person_id == first.person_id, "사진이 쌓였는데도 원래 신원으로 합치지 않았다"
    assert reg.merges and reg.merges[0]["into"] == first.person_id
    assert reg.resolve(again[0][0].person_id) == first.person_id, "먼저 내보낸 신원을 최종 신원으로 풀지 못한다"


def test_merged_track_keeps_the_merged_identity():
    """합친 뒤에도 그 track 은 계속 원래 신원으로 나간다."""
    reg = registry(StreamEmbedder({HERE: noisy_photos(0, SAMPLES, seed=1),
                                   FAR: noisy_photos(0, SAMPLES + 5, seed=2)}))
    first = feed(reg, [(1, HERE)], 0.0, SAMPLES)[-1][0]
    later = feed(reg, [(2, FAR)], 20.0, SAMPLES + 5)
    assert all(r[0].person_id == first.person_id for r in later[SAMPLES - 1:])


def test_different_people_are_not_merged():
    """가장 중요한 안전장치 — 다른 사람은 사진이 쌓여도 합치지 않는다."""
    reg = registry(StreamEmbedder({HERE: noisy_photos(0, SAMPLES, seed=1),
                                   FAR: noisy_photos(1, SAMPLES, seed=2)}))
    a = feed(reg, [(1, HERE)], 0.0, SAMPLES)[-1][0]
    b = feed(reg, [(2, FAR)], 20.0, SAMPLES)[-1][0]
    assert b.person_id != a.person_id, "다른 사람을 나중에 합쳐 버렸다"
    assert not reg.merges


def test_people_seen_together_are_never_merged():
    """같은 시각에 함께 보인 적이 있으면, 생김새가 같아도 한 사람일 수 없다."""
    twins = noisy_photos(0, 2 * SAMPLES, seed=1)
    reg = registry(StreamEmbedder({HERE: twins[:SAMPLES] + noisy_photos(0, SAMPLES, seed=9),
                                   FAR: twins[SAMPLES:]}))
    feed(reg, [(1, HERE), (2, FAR)], 0.0, SAMPLES)          # 둘이 함께 보인다
    out = feed(reg, [(1, HERE)], 1.0, SAMPLES)[-1]           # 2 가 사라진 뒤에도 1 은 계속 보인다
    assert not reg.merges, "함께 보였던 두 사람을 합쳤다"
    assert len(reg._people) == 2


def test_gap_longer_than_retire_is_not_merged():
    """매장을 나간 것으로 본 사람(retire 초과)과는 합치지 않는다."""
    reg = registry(StreamEmbedder({HERE: noisy_photos(0, SAMPLES, seed=1),
                                   FAR: noisy_photos(0, SAMPLES, seed=2)}),
                   retire_after_seconds=10.0)
    a = feed(reg, [(1, HERE)], 0.0, SAMPLES)[-1][0]
    b = feed(reg, [(2, FAR)], 30.0, SAMPLES)[-1][0]
    assert b.person_id != a.person_id


def test_disabled_keeps_the_split():
    """비교용 — 끄면 쪼개진 채로 남는다 (이 기능이 실제로 차이를 만든다는 증거)."""
    reg = registry(StreamEmbedder({HERE: noisy_photos(0, SAMPLES, seed=1),
                                   FAR: noisy_photos(0, SAMPLES, seed=2)}),
                   late_merge_enabled=False)
    a = feed(reg, [(1, HERE)], 0.0, SAMPLES)[-1][0]
    b = feed(reg, [(2, FAR)], 20.0, SAMPLES)[-1][0]
    assert b.person_id != a.person_id


def test_cut_off_at_frame_edge_is_not_matched_immediately():
    """화면 가장자리에 걸려 몸이 잘린 첫 사진으로는 기존 신원에 합치지 않는다.

    막 들어오는 순간엔 머리카락만 보여 누구든 닮아 보인다(people-detection 실측).
    생김새가 완전히 같아도 일단 새 신원으로 두고, 판단은 나중에 합치기에 맡긴다.
    """
    at_edge = (400.0, 0.0, 480.0, 150.0)          # 위쪽 가장자리에 닿음
    same = noisy_photos(0, 1, seed=1)[0]
    reg = registry(StreamEmbedder({HERE: [same], at_edge: [same]}))
    first = feed(reg, [(1, HERE)], 0.0, 1)[-1][0]
    entering = feed(reg, [(2, at_edge)], 20.0, 1)[-1][0]
    assert entering.person_id != first.person_id, "잘린 사진만 보고 기존 신원에 합쳤다"

    reg = registry(StreamEmbedder({HERE: [same], at_edge: [same]}), edge_margin=0.0)
    first = feed(reg, [(1, HERE)], 0.0, 1)[-1][0]
    entering = feed(reg, [(2, at_edge)], 20.0, 1)[-1][0]
    assert entering.person_id == first.person_id, "비교용 — 끄면 즉시 합쳐져야 한다"


def test_recent_consistency_drops_when_track_switches_person():
    """판정 직전 신원 확인 — 같은 번호가 도중에 다른 사람을 따라가기 시작하면 '최근 vs 이전' 유사도가 떨어진다."""
    same = noisy_photos(0, 12, seed=1)
    swapped = noisy_photos(0, 9, seed=1) + noisy_photos(1, 3, seed=2)   # 마지막 3장은 다른 사람
    reg_same = registry(StreamEmbedder({HERE: same}), late_merge_enabled=False)
    reg_swap = registry(StreamEmbedder({HERE: swapped}), late_merge_enabled=False)
    pid_same = feed(reg_same, [(1, HERE)], 0.0, 12)[-1][0].person_id
    pid_swap = feed(reg_swap, [(1, HERE)], 0.0, 12)[-1][0].person_id
    ok, bad = reg_same.recent_consistency(pid_same), reg_swap.recent_consistency(pid_swap)
    assert ok is not None and bad is not None
    # 이 합성 사진은 실제 ReID 특징보다 잡음이 커서 절대값(실측 기준 0.78)은 맞지 않는다.
    # 확인하는 것은 '뒤바뀌면 같은 사람일 때보다 뚜렷이 낮아진다'는 성질이다.
    assert ok - bad > 0.1, f"같은 사람 {ok:.2f} / 뒤바뀜 {bad:.2f}"


def test_recent_consistency_needs_enough_history():
    """이전 기록이 모자라면 확인하지 않는다(None) — '괜찮다'가 아니라 '모른다'."""
    reg = registry(StreamEmbedder({HERE: noisy_photos(0, 5, seed=1)}), late_merge_enabled=False)
    pid = feed(reg, [(1, HERE)], 0.0, 5)[-1][0].person_id
    assert reg.recent_consistency(pid) is None
