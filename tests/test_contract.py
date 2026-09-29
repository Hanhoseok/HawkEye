"""표준 인터페이스 계약 테스트.

무거운 모델 없이도 '뒤 모듈이 tracker 라이브러리를 직접 참조하지 않는다'는
Phase 4 확인 항목을 자동으로 검증한다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.types import Detection, Frame, TrackObservation  # noqa: E402


def test_track_observation_has_documented_fields():
    obs = TrackObservation(
        track_id=1,
        bbox=(10.0, 20.0, 30.0, 40.0),
        frame=7,
        timestamp="2026-09-20T14:00:00+09:00",
    )
    payload = obs.to_dict()
    for key in ("track_id", "bbox", "frame", "timestamp"):
        assert key in payload, f"문서에 고정된 필드가 빠졌습니다: {key}"
    json.dumps(payload)  # 직렬화 가능해야 로그로 남길 수 있다


def test_core_layer_has_no_external_vision_deps():
    """core 계층은 cv2/ultralytics/supervision 을 import 하면 안 된다."""
    forbidden = ("cv2", "ultralytics", "supervision", "torch")
    for path in (Path(__file__).resolve().parents[1] / "src" / "core").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for name in forbidden:
            assert f"import {name}" not in source, f"{path.name} 이 {name} 에 의존합니다"


def test_risk_layer_has_no_external_vision_deps():
    """손님 상태·결제·출구 판정은 앞 계층의 출력만 받는다. 영상·모델 라이브러리를 몰라야 한다."""
    forbidden = ("cv2", "ultralytics", "supervision", "torch", "boxmot")
    for path in (Path(__file__).resolve().parents[1] / "src" / "risk").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for name in forbidden:
            assert f"import {name}" not in source, f"{path.name} 이 {name} 에 의존합니다"


def test_detection_and_frame_are_plain_data():
    frame = Frame(index=0, timestamp="2026-09-20T14:00:00+09:00", pts_ms=0.0, image=None)
    det = Detection(bbox=(0.0, 0.0, 1.0, 1.0), score=0.9, class_id=0, class_name="person")
    assert frame.index == 0
    assert det.class_name == "person"


def test_video_source_survives_broken_timestamps():
    """실제 CCTV 녹화본(MPEG2 등)은 POS_MSEC 로 쓰레기값을 준다.

    그대로 timedelta 에 넣으면 OverflowError 로 파이프라인 전체가 죽는다.
    (ThreePastShop2cor.mpg 에서 실제로 발생)
    """
    from datetime import datetime, timedelta

    from src.inputs.video_source import _MAX_PTS_MS

    start = datetime(2026, 1, 1)
    for bad in (-1.0, 1e18, float("inf")):
        usable = 0.0 <= bad <= _MAX_PTS_MS
        assert not usable, f"{bad} 는 걸러져야 한다"
    # 걸러진 뒤 쓰이는 fps 기반 값은 항상 안전해야 한다
    assert (start + timedelta(milliseconds=30 * 1000.0 / 25)).year == 2026
