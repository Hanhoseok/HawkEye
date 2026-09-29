"""detector 를 가짜로 바꿔 끼워 파이프라인 전체를 검증한다.

두 가지를 동시에 확인한다.
1. ByteTrack 어댑터가 움직이는 객체에 동일한 track_id 를 유지하는가 (Phase 2)
2. YOLO 가 아닌 detector 를 끼워도 pipeline 이 그대로 동작하는가 (계층 분리 §4)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import AppConfig  # noqa: E402
from src.core.types import Detection, Frame  # noqa: E402
from src.pipeline import TrackingPipeline  # noqa: E402
from src.sinks.observation_log import read_observations  # noqa: E402
from src.trackers.bytetrack_tracker import ByteTrackTracker  # noqa: E402

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "videos" / "synthetic.mp4"


class FakeDetector:
    """오른쪽으로 일정하게 움직이는 두 개의 bbox 를 만들어 내는 detector."""

    def detect(self, frame: Frame) -> list[Detection]:
        offset = frame.index * 4
        return [
            Detection(
                bbox=(10.0 + offset, 100.0, 70.0 + offset, 260.0),
                score=0.9,
                class_id=0,
                class_name="person",
            ),
            Detection(
                bbox=(300.0 + offset, 60.0, 360.0 + offset, 220.0),
                score=0.85,
                class_id=0,
                class_name="person",
            ),
        ]


@pytest.mark.skipif(not SAMPLE.exists(), reason="합성 영상이 없습니다. scripts/make_sample_video.py 먼저 실행")
def test_pipeline_keeps_track_ids(tmp_path):
    config = AppConfig()
    config.video.source = str(SAMPLE)
    config.video.max_frames = 40
    config.output.dir = str(tmp_path)
    config.output.write_video = False  # 테스트에서는 영상 저장 생략

    stats = TrackingPipeline(config, FakeDetector(), ByteTrackTracker(config.tracker)).run()

    assert stats.frames == 40
    assert stats.detections == 80
    assert stats.observations > 0
    # 등장하는 객체가 두 개뿐이므로 track_id 도 두 개여야 한다. 더 많으면 ID switch 다.
    assert len(stats.unique_track_ids) == 2, f"ID switch 발생: {stats.unique_track_ids}"

    observations = list(read_observations(Path(config.output.dir) / config.output.observations_name))
    assert len(observations) == stats.observations
    first = observations[0]
    assert first.frame == 0 or first.frame > 0
    assert len(first.bbox) == 4
    assert first.timestamp.count(":") >= 2  # ISO8601 형태


def test_tracker_returns_standard_type_only():
    """tracker 가 supervision 자료형을 그대로 흘려보내지 않는지 확인한다."""
    tracker = ByteTrackTracker(AppConfig().tracker)
    frame = Frame(index=0, timestamp="2026-09-20T14:00:00+09:00", pts_ms=0.0, image=None)
    detections = [
        Detection(bbox=(0.0, 0.0, 50.0, 100.0), score=0.9, class_id=0, class_name="person")
    ]
    observations = tracker.update(frame, detections)
    for obs in observations:
        assert type(obs).__name__ == "TrackObservation"
        assert isinstance(obs.track_id, int)
        assert not isinstance(obs.bbox, np.ndarray)
