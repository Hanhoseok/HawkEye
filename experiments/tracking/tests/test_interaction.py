"""계층 3 · 손 뻗기 후보 판정 테스트.

핵심 질문: **손이 선반 구역에 닿은 사람만 후보가 되는가.**
몸이 선반 앞에 서 있기만 한 사람(지운 '선반 앞 머묾')은 후보가 되면 안 된다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import InteractionConfig  # noqa: E402
from src.core.types import (  # noqa: E402
    KP_LEFT_WRIST,
    KP_RIGHT_WRIST,
    Frame,
    IdentityObservation,
    TakeCandidate,
    Zone,
    ZoneType,
)
from src.interaction.detector import InteractionDetector  # noqa: E402
from src.zones.zone_map import ZoneMap  # noqa: E402

FPS = 30.0
SHELF = Zone("shelf", ZoneType.SHELF, ((0.0, 0.0), (400.0, 0.0), (400.0, 400.0), (0.0, 400.0)))
IN_SHELF = (200.0, 200.0)
OUT_OF_SHELF = (600.0, 200.0)


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


def wrists(point: tuple[float, float] | None, conf: float = 0.9):
    """왼 손목만 그 점에 둔 관절. None 이면 관절 없음 (일반 탐지기)."""
    if point is None:
        return None
    kps = [(0.0, 0.0, 0.0)] * 17
    kps[KP_LEFT_WRIST] = (point[0], point[1], conf)
    kps[KP_RIGHT_WRIST] = (0.0, 0.0, 0.0)
    return tuple(kps)


def person(i: int, hand: tuple[float, float] | None, x: float = 100.0, pid: int = 1, conf: float = 0.9):
    return IdentityObservation(
        person_id=pid,
        track_id=pid,
        bbox=(x, 200.0, x + 80.0, 400.0),
        frame=i,
        timestamp=f"2026-01-01T00:00:{i/FPS:06.3f}",
        score=0.9,
        class_name="person",
        keypoints=wrists(hand, conf),
    )


def run(detector: InteractionDetector, hands: list, **kw):
    for i, hand in enumerate(hands):
        detector.update(frame(i), [person(i, hand, **kw)])
    detector.flush()
    return detector.completed


def test_hand_in_shelf_is_a_candidate():
    got = run(make_detector(min_seconds=0.3), [OUT_OF_SHELF] * 5 + [IN_SHELF] * 15 + [OUT_OF_SHELF] * 30)
    assert len(got) == 1, f"손이 0.5초 선반 안에 있었는데 후보가 안 나왔다: {got}"
    assert got[0].zone == "shelf"
    assert got[0].duration_sec >= 0.3
    assert (got[0].start_frame, got[0].end_frame) == (5, 19)


def test_standing_at_shelf_without_hand_is_not_a_candidate():
    """지운 근거 — 선반 앞에 3초 서 있기만 한 사람은 후보가 아니다."""
    assert run(make_detector(), [OUT_OF_SHELF] * 90) == []


def test_no_keypoints_gives_no_candidate():
    """관절이 없는 탐지기(yolo)로는 후보가 나오지 않는다."""
    assert run(make_detector(), [None] * 90) == []


def test_brief_touch_is_too_short():
    got = run(make_detector(min_seconds=0.3), [OUT_OF_SHELF] * 5 + [IN_SHELF] * 4 + [OUT_OF_SHELF] * 30)
    assert got == [], "0.3초 기준인데 0.1초 닿은 것이 후보가 됐다"


def test_low_confidence_wrist_is_ignored():
    assert run(make_detector(hand_conf=0.5), [IN_SHELF] * 60, conf=0.2) == []


def test_short_interruption_does_not_split_episode():
    """한 프레임 튄 값 때문에 구간이 쪼개지면 길이가 과소평가된다."""
    det = make_detector(min_seconds=1.0, gap_tolerance_seconds=0.5)
    got = run(det, [IN_SHELF] * 25 + [OUT_OF_SHELF] + [IN_SHELF] * 25)
    assert len(got) == 1, f"짧은 끊김으로 구간이 쪼개졌다: {got}"
    assert got[0].duration_sec > 1.0


def test_long_interruption_splits_episode():
    det = make_detector(min_seconds=0.3, gap_tolerance_seconds=0.5)
    got = run(det, [IN_SHELF] * 20 + [OUT_OF_SHELF] * 30 + [IN_SHELF] * 20)
    assert len(got) == 2


def test_two_people_tracked_separately():
    det = make_detector()
    for i in range(60):
        det.update(frame(i), [person(i, IN_SHELF, pid=1), person(i, IN_SHELF, x=200.0, pid=2)])
    det.flush()
    assert {c.person_id for c in det.completed} == {1, 2}


def test_unassigned_identity_is_ignored():
    """신원이 보류(-1)된 관측은 후보 판정에 쓰지 않는다."""
    det = make_detector()
    for i in range(60):
        det.update(frame(i), [person(i, IN_SHELF, pid=-1)])
    det.flush()
    assert det.completed == []


def test_reach_uses_wrist_distance_to_shelf():
    """reach: 손목이 선반 밖이어도 체구 대비 거리가 wrist_zone_max 안이면 닿은 것으로 본다."""
    near, far = (420.0, 200.0), (600.0, 200.0)      # 선반 끝(x=400)에서 20px · 200px, 체구 200px → 0.1 · 1.0
    kw = dict(signal="reach", wrist_zone_max=0.25, min_seconds=0.3)
    assert len(run(make_detector(**kw), [near] * 30, x=380.0)) == 1
    assert run(make_detector(**kw), [far] * 30, x=380.0) == []


@pytest.mark.parametrize("signal", ["dwell", "both", "raise", "stage"])
def test_dwell_signals_are_gone(signal):
    """머묾 계열 근거는 지웠다 — 예전 값을 주면 조용히 다른 근거로 돌지 않고 멈춘다."""
    with pytest.raises(ValueError):
        make_detector(signal=signal)


def test_dwell_settings_are_gone():
    for key in ("min_overlap", "max_speed", "speed_window_seconds", "dwell_seconds",
                "raise_min_seconds", "best_raise_only", "hand_height_min"):
        with pytest.raises(TypeError):
            InteractionConfig(**{key: 1})
    assert InteractionConfig().signal == "hand"


def test_old_take_records_still_load():
    """예전 기록의 min_speed 칸은 버리고 읽는다."""
    old = {"person_id": 1, "zone": "s", "start_frame": 1, "end_frame": 9, "start_timestamp": "a",
           "end_timestamp": "b", "duration_sec": 0.5, "observations": 8, "max_overlap": 0.2, "min_speed": 0.01}
    take = TakeCandidate.from_dict(old)
    assert take.end_frame == 9 and "min_speed" not in take.to_dict()
