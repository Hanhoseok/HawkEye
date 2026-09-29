"""신원 레지스트리(계층 2) 로직 테스트.

가짜 임베더를 써서 모델 없이 검증한다. 확인하는 것:
  1. tracker 가 쪼갠 track 을 같은 신원으로 이어붙이는가
  2. 다른 사람을 합치지 않는가 (오병합)
  3. 같은 프레임에 있는 두 track 을 같은 신원으로 보지 않는가
  4. 너무 작은 관측은 신원 판단에 쓰지 않는가
  5. 오래 사라진 사람은 퇴장 처리되는가
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import IdentityConfig  # noqa: E402
from src.core.types import Frame, TrackObservation  # noqa: E402
from src.identity.registry import IdentityRegistry  # noqa: E402


class FakeEmbedder:
    """bbox 의 x 위치로 '사람'을 흉내내는 임베더.

    같은 사람은 같은 방향 벡터를, 다른 사람은 직교 벡터를 돌려준다.
    """

    def __init__(self, person_of):
        self.person_of = person_of  # (x1) -> 사람 번호
        self.calls = 0

    def embed(self, image, boxes):
        self.calls += 1
        out = []
        for b in boxes:
            idx = self.person_of(float(b[0]))
            v = np.zeros(4, dtype=np.float32)
            v[idx] = 1.0
            out.append(v)
        return np.array(out, dtype=np.float32)


def frame(i: int) -> Frame:
    return Frame(index=i, timestamp="2026-01-01T00:00:00", pts_ms=i * 33.3, image=np.zeros((400, 400, 3), dtype=np.uint8))


def obs(track_id: int, x: float, frame_index: int, height: float = 200.0) -> TrackObservation:
    return TrackObservation(
        track_id=track_id,
        bbox=(x, 50.0, x + 50.0, 50.0 + height),  # 화면 가장자리에 닿지 않게 (edge_margin)
        frame=frame_index,
        timestamp="2026-01-01T00:00:00",
        score=0.9,
        class_name="person",
    )


def make_registry(**overrides):
    cfg = IdentityConfig(min_tensor_height=65, **overrides)
    # 사람 번호 = x 좌표를 100 으로 나눈 몫
    return IdentityRegistry(cfg, FakeEmbedder(lambda x: int(x // 100)), imgsz=640)


def test_reconnects_fragmented_track():
    """track 이 끊겨 새 track_id 를 받아도 같은 신원으로 이어야 한다."""
    reg = make_registry()
    out = reg.assign(frame(0), [obs(1, 10, 0)])
    first_pid = out[0].person_id
    assert out[0].is_new

    # track 1 이 사라지고, 한참 뒤 같은 사람이 track 9 로 다시 등장
    out = reg.assign(frame(200), [obs(9, 15, 200)])
    assert out[0].person_id == first_pid, "같은 사람인데 새 신원이 생겼다"
    assert not out[0].is_new
    assert out[0].matched_similarity is not None


def test_does_not_merge_different_people():
    reg = make_registry()
    a = reg.assign(frame(0), [obs(1, 10, 0)])[0]
    b = reg.assign(frame(50), [obs(2, 250, 50)])[0]
    assert a.person_id != b.person_id, "서로 다른 사람이 합쳐졌다"


def test_two_live_tracks_never_share_identity():
    """같은 프레임에 보이는 두 track 은 절대 같은 신원일 수 없다."""
    reg = make_registry()
    # 외형이 완전히 같아도(같은 사람 번호) 동시에 존재하면 나눠야 한다
    reg.embedder = FakeEmbedder(lambda x: 0)
    out = reg.assign(frame(0), [obs(1, 10, 0), obs(2, 300, 0)])
    pids = {o.person_id for o in out}
    assert len(pids) == 2, f"동시에 보이는 두 사람이 같은 신원이 됐다: {pids}"


def test_small_observation_is_not_registered():
    """텐서 65px 미만 관측은 신원 판단에 쓰지 않는다 (docs/scale-limits.md)."""
    reg = make_registry()
    # 프레임 긴 변 400, imgsz 640 -> 임계값 = 65 * 400 / 640 = 40.6px
    tiny = reg.assign(frame(0), [obs(1, 10, 0, height=20.0)])
    assert tiny[0].person_id == -1, "너무 작은 관측에 신원이 배정됐다"
    assert reg.embedder.calls == 0, "쓸 수 없는 관측에 임베딩을 계산했다"

    # 가까워져 커지면 그때 배정된다
    big = reg.assign(frame(1), [obs(1, 10, 1, height=200.0)])
    assert big[0].person_id > 0


def test_retired_person_is_not_matched():
    """오래 사라진 사람은 매장을 나간 것으로 보고 대조하지 않는다."""
    # 테스트 프레임은 33.3ms 간격(30fps)이므로 frame 500 = 약 16.6초
    reg = make_registry(retire_after_seconds=3.0)
    first = reg.assign(frame(0), [obs(1, 10, 0)])[0]
    late = reg.assign(frame(500), [obs(7, 10, 500)])[0]
    assert late.person_id != first.person_id, "이미 나간 사람과 이어붙였다"


def test_templates_keep_scale_diversity():
    """템플릿이 가득 차도 크기 다양성이 유지돼야 한다.

    같은 카메라에서 사람 크기가 4~10배 변하므로(docs/scale-limits.md),
    최근 크기로만 쏠리면 멀리 있을 때를 못 알아본다.
    """
    reg = make_registry(max_templates=3, template_refresh_seconds=0.0)
    heights = [60.0, 120.0, 200.0, 210.0, 215.0, 220.0]
    for i, h in enumerate(heights):
        reg.assign(frame(i * 2), [obs(1, 10, i * 2, height=h)])
    person = reg._people[1]
    assert len(person.templates) == 3
    spread = max(person.template_heights) - min(person.template_heights)
    assert spread > 100, f"템플릿이 한 크기로 쏠렸다: {person.template_heights}"


def test_retirement_counts_from_last_sighting_not_entry():
    """퇴장 판정은 '입장 이후'가 아니라 '마지막으로 보인 이후'를 센다.

    계속 보이는 고객은 몇 분이 지나도 만료되면 안 된다.
    """
    reg = make_registry(retire_after_seconds=3.0)
    pid = reg.assign(frame(0), [obs(1, 10, 0)])[0].person_id

    # 입장 후 오래 지났지만 계속 보이는 중 (0.33초 간격으로 계속 관측)
    for i in range(10, 600, 10):
        reg.assign(frame(i), [obs(1, 10, i)])

    # track 이 끊겨 새 track_id 로 재등장해도 같은 신원이어야 한다
    again = reg.assign(frame(610), [obs(99, 12, 610)])[0]
    assert again.person_id == pid, "계속 보이던 고객이 체류 시간 때문에 만료됐다"


def test_retirement_uses_time_not_frame_count():
    """프레임 번호가 아니라 재생 시각(ms)으로 판단해야 한다.

    같은 프레임 수라도 원본 fps 와 stride 에 따라 실제 시간이 다르기 때문이다.
    """
    from src.core.types import Frame as F
    import numpy as np

    reg = make_registry(retire_after_seconds=3.0)
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    first = reg.assign(F(index=0, timestamp="t", pts_ms=0.0, image=img), [obs(1, 10, 0)])[0]

    # 프레임 번호는 20 밖에 안 늘었지만 재생 시각으로는 10초가 지났다
    late = F(index=20, timestamp="t", pts_ms=10_000.0, image=img)
    out = reg.assign(late, [obs(50, 10, 20)])[0]
    assert out.person_id != first.person_id, "프레임 번호만 보고 아직 안 지났다고 판단했다"
