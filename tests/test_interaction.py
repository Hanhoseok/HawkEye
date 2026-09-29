"""계층 3 · TAKE 후보 판정 테스트.

핵심 질문: **지나가는 사람과 멈춰 선 사람을 구분하는가.**
구분하지 못하면 이 계층은 존재 이유가 없다 — 근접만으로는 모두가 선반에 닿기 때문이다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import InteractionConfig  # noqa: E402
from src.core.types import Frame, IdentityObservation, Zone, ZoneType  # noqa: E402
from src.interaction.detector import InteractionDetector  # noqa: E402
from src.zones.zone_map import ZoneMap  # noqa: E402

FPS = 30.0
SHELF = Zone("shelf", ZoneType.SHELF, ((0.0, 0.0), (400.0, 0.0), (400.0, 400.0), (0.0, 400.0)))


def make_detector(**overrides) -> InteractionDetector:
    zm = ZoneMap([SHELF])
    zm.resolve(800, 400)
    return InteractionDetector(InteractionConfig(**overrides), zm)


def frame(i: int) -> Frame:
    return Frame(
        index=i,
        timestamp=f"2026-01-01T00:00:{i/FPS:06.3f}",
        pts_ms=i * 1000.0 / FPS,
        image=np.zeros((400, 800, 3), dtype=np.uint8),
    )


def person(x: float, i: int, height: float = 200.0, pid: int = 1) -> IdentityObservation:
    return IdentityObservation(
        person_id=pid,
        track_id=pid,
        bbox=(x, 400.0 - height, x + 80.0, 400.0),
        frame=i,
        timestamp=f"2026-01-01T00:00:{i/FPS:06.3f}",
        score=0.9,
        class_name="person",
    )


def run(detector: InteractionDetector, positions: list[float], **kw):
    for i, x in enumerate(positions):
        detector.update(frame(i), [person(x, i, **kw)])
    detector.flush()
    return detector.completed


def test_standing_still_at_shelf_is_a_candidate():
    det = make_detector(dwell_seconds=1.0)
    # 선반 앞에 3초간 정지 (90 프레임)
    got = run(det, [100.0] * 90)
    assert len(got) == 1, f"멈춰 있었는데 후보가 안 나왔다: {got}"
    assert got[0].zone == "shelf"
    assert got[0].duration_sec >= 1.0
    assert got[0].min_speed < 0.05


def test_walking_past_shelf_is_not_a_candidate():
    """가장 중요한 테스트 — 그냥 지나가는 사람을 걸러내지 못하면 이 계층은 의미가 없다."""
    det = make_detector(dwell_seconds=1.0)
    # 체구 높이 200px, 프레임당 20px 이동 = 초당 600px = 3.0 체구/초 (빠른 보행)
    got = run(det, [i * 20.0 for i in range(40)])
    assert got == [], f"지나가는 사람이 후보로 잡혔다: {got}"


def test_brief_pause_is_too_short():
    det = make_detector(dwell_seconds=2.0)
    positions = [i * 20.0 for i in range(10)] + [200.0] * 20 + [200.0 + i * 20.0 for i in range(10)]
    got = run(det, positions)
    assert got == [], "2초 기준인데 0.67초 멈춤이 후보가 됐다"


def test_speed_is_normalized_by_body_height():
    """같은 픽셀 속도라도 멀리 있는(작은) 사람은 '빠르게 걷는 중'이어야 한다.

    정규화하지 않으면 통로 안쪽을 지나가는 사람이 전부 '멈춤'으로 오인된다.
    """
    pixels_per_frame = 8.0
    positions = [i * pixels_per_frame for i in range(60)]

    # 가까운 사람(키 300px): 8*30/300 = 0.8 체구/초 -> 걷는 중
    near = run(make_detector(dwell_seconds=1.0), positions, height=300.0)
    # 먼 사람(키 60px): 8*30/60 = 4.0 체구/초 -> 훨씬 빠름
    far = run(make_detector(dwell_seconds=1.0), positions, height=60.0)

    assert near == [], "가까운 사람이 걷는데 멈춤으로 잡혔다"
    assert far == [], "먼 사람이 걷는데 멈춤으로 잡혔다"


def test_unnormalized_speed_would_have_failed():
    """정규화의 필요성을 수치로 못박는다.

    프레임당 4px 이동은 키 60px 인 사람에게는 초당 2.0 체구(빠른 걸음)지만,
    픽셀 기준으로는 초당 120px 에 불과해 '느리다'고 오판하기 쉽다.
    """
    det = make_detector(dwell_seconds=1.0, max_speed=0.35)
    got = run(det, [i * 4.0 for i in range(60)], height=60.0)
    assert got == [], "작은 사람의 정상 보행이 멈춤으로 잡혔다"

    # 같은 픽셀 속도라도 키가 크면 실제로 느린 것이므로 후보가 되어야 한다
    det2 = make_detector(dwell_seconds=1.0, max_speed=0.35)
    got2 = run(det2, [i * 4.0 for i in range(60)], height=400.0)
    assert got2, "큰 사람의 느린 움직임이 후보가 되지 않았다"


def test_short_interruption_does_not_split_episode():
    """한 프레임 튄 값 때문에 구간이 쪼개지면 체류 시간이 과소평가된다."""
    det = make_detector(dwell_seconds=2.0, gap_tolerance_seconds=0.5)
    positions = [100.0] * 40 + [900.0] + [100.0] * 40  # 중간에 한 프레임만 구역 밖
    got = run(det, positions)
    assert len(got) == 1, f"짧은 끊김으로 구간이 쪼개졌다: {got}"
    assert got[0].duration_sec > 2.0


def test_two_people_tracked_separately():
    det = make_detector(dwell_seconds=1.0)
    for i in range(90):
        det.update(frame(i), [person(100.0, i, pid=1), person(200.0, i, pid=2)])
    det.flush()
    assert {c.person_id for c in det.completed} == {1, 2}


def test_unassigned_identity_is_ignored():
    """신원이 보류(-1)된 관측은 후보 판정에 쓰지 않는다."""
    det = make_detector(dwell_seconds=1.0)
    for i in range(90):
        obs = person(100.0, i)
        det.update(frame(i), [IdentityObservation(**{**obs.to_dict(), "person_id": -1})])
    det.flush()
    assert det.completed == []
