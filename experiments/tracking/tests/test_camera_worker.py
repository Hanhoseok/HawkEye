"""카메라 작업자와 카메라 이름·시간축 — 1단계 1-1 구조 쪼개기 (docs/multiview-design.md M-2·M-9).

카메라가 한 대일 때는 예전과 똑같이 동작해야 하고(기본 이름 cam0, 시차 0),
카메라가 둘이면 각 장면·관측에 카메라 이름이 붙고 시간이 기준 카메라 축으로 밀려야 한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.camera_worker import CameraWorker  # noqa: E402
from src.core.config import AppConfig  # noqa: E402
from src.core.types import Detection, Frame, TrackObservation  # noqa: E402
from src.inputs.video_source import VideoSource, _epoch_ms  # noqa: E402
from src.trackers.bytetrack_tracker import ByteTrackTracker  # noqa: E402


class OneBox:
    def detect(self, frame: Frame) -> list[Detection]:
        x = 10.0 + frame.index * 3
        return [Detection(bbox=(x, 50.0, x + 40.0, 170.0), score=0.9, class_id=0, class_name="person")]


class FakeRegistry:
    """신원 계층 대역: track_id 를 그대로 person_id 로."""

    merges: list = []

    def reset(self):
        self.calls = 0

    def assign(self, frame, observations):
        from src.core.types import IdentityObservation
        self.calls += 1
        return [IdentityObservation(person_id=o.track_id, track_id=o.track_id, bbox=o.bbox, frame=o.frame,
                                    timestamp=o.timestamp, camera_id=o.camera_id) for o in observations]

    def resolve(self, pid):
        return pid


class FakeInteractions:
    def reset(self):
        self.seen = 0

    def update(self, frame, identities):
        self.seen += len(identities)
        return []

    def flush(self):
        return ["열린 구간"]


def frames(camera="cam0"):
    return [Frame(index=i, timestamp="t", pts_ms=i * 100.0, image=None, camera_id=camera) for i in range(6)]


def test_worker_without_registry_only_tracks():
    worker = CameraWorker("cam0", OneBox(), ByteTrackTracker(AppConfig().tracker))
    worker.reset()
    steps = [worker.step(f) for f in frames()]
    assert all(s.identities is None for s in steps)
    assert steps[-1].drawable is steps[-1].observations
    assert len({o.track_id for s in steps for o in s.observations}) == 1
    assert worker.flush() == []


def test_worker_runs_identity_and_interactions_and_keeps_camera_name():
    registry, inter = FakeRegistry(), FakeInteractions()
    worker = CameraWorker("angle1", OneBox(), ByteTrackTracker(AppConfig().tracker), registry, interactions=inter)
    worker.reset()
    steps = [worker.step(f) for f in frames("angle1")]
    assert registry.calls == 6
    assert inter.seen == sum(len(s.identities) for s in steps) > 0
    obs = [o for s in steps for o in s.observations]
    ids = [o for s in steps for o in s.identities]
    assert {o.camera_id for o in obs} == {"angle1"}       # 추적기가 장면의 카메라 이름을 옮김
    assert {o.camera_id for o in ids} == {"angle1"}
    assert worker.flush() == ["열린 구간"]


def test_contract_defaults_keep_single_camera_logs_compatible():
    # 예전 기록(camera_id 없음)을 그대로 읽을 수 있어야 한다
    old = {"track_id": 1, "bbox": (0, 0, 1, 1), "frame": 0, "timestamp": "t"}
    assert TrackObservation(**old).camera_id == "cam0"
    assert Frame(0, "t", 0.0, None).camera_id == "cam0"


def write_video(path: Path, n: int, fps: float) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(n):
        writer.write(np.full((48, 64, 3), i * 10 % 255, np.uint8))
    writer.release()


def test_video_source_names_camera_and_shifts_onto_reference_timeline(tmp_path):
    path = tmp_path / "a.mp4"
    write_video(path, 10, 10.0)
    start = "2026-10-28T14:00:00+09:00"
    with VideoSource(str(path), start_time=start) as src:
        plain = list(src)
    with VideoSource(str(path), start_time=start, camera_id="angle0", offset_ms=-300.0) as src:
        shifted = list(src)
    assert {f.camera_id for f in plain} == {"cam0"}
    assert {f.camera_id for f in shifted} == {"angle0"}
    for a, b in zip(plain, shifted):
        assert b.pts_ms == a.pts_ms - 300.0
        assert b.wall_ms == a.wall_ms - 300.0
    # 실제 시각 = 영상 시작 시각 + 재생 위치 (14:00 KST = 05:00 UTC)
    assert plain[0].wall_ms == _epoch_ms(__import__("datetime").datetime.fromisoformat(start)) + plain[0].pts_ms


def test_stream_address_is_not_treated_as_missing_file(monkeypatch):
    # 실시간 주소는 '파일 없음'으로 멈추지 않아야 한다 (F-5). 여기서는 열기 실패(RuntimeError)까지만 확인.
    # 실제 접속은 시간 제한이 30초라 열기만 가짜로 바꾼다.
    class Closed:
        def __init__(self, *_):
            pass

        def isOpened(self):
            return False

    monkeypatch.setattr("src.inputs.video_source.cv2.VideoCapture", Closed)
    try:
        VideoSource("rtsp://127.0.0.1:1/없는카메라")
    except FileNotFoundError:
        raise AssertionError("rtsp 주소를 파일로 취급했습니다")
    except RuntimeError:
        pass
