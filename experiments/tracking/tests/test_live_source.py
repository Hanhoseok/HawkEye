"""실시간 입력 — 최신 장면만 처리하고, 끊기면 다시 연결한다 (설계 M-9, limitations F-6). 실제 카메라 없이 가짜로."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inputs.live_source import LiveSource  # noqa: E402


class FakeCamera:
    """초당 fps 장 나오는 카메라. fail_after 장 뒤 한 번 끊긴다."""

    opened = 0

    def __init__(self, fps=100.0, fail_after=None, total=60):
        self.fps, self.fail_after, self.total = fps, fail_after, total
        self.count = 0

    def __call__(self, source):
        FakeCamera.opened += 1
        self.count_at_open = self.count
        return self

    def isOpened(self):
        return True

    def get(self, prop):
        return {5: self.fps, 3: 64, 4: 48}.get(prop, 0)

    def read(self):
        time.sleep(1.0 / self.fps)
        if self.count >= self.total:
            return False, None
        self.count += 1
        if self.fail_after is not None and self.count == self.fail_after:
            self.fail_after = None
            return False, None
        return True, np.full((48, 64, 3), self.count % 255, np.uint8)

    def release(self):
        pass


def test_slow_processing_gets_latest_frames_and_drops_the_rest():
    cam = FakeCamera(fps=200.0, total=100)
    src = LiveSource("rtsp://fake", "angle1", opener=cam, max_reconnects=0, idle_timeout_seconds=1.0)
    seen, cams = [], set()
    for frame in src:
        seen.append(int(frame.image[0, 0, 0]))
        cams.add(frame.camera_id)
        time.sleep(0.03)          # 처리가 카메라보다 6배 느림
    src.release()
    assert len(seen) < 60                      # 밀린 장면은 버렸다
    assert src.dropped > 0
    assert seen == sorted(seen)                # 늘 더 새로운 장면
    assert cams == {"angle1"}


def test_reconnects_after_a_dropout_and_keeps_going():
    FakeCamera.opened = 0
    cam = FakeCamera(fps=200.0, fail_after=10, total=40)
    src = LiveSource(0, "angle0", opener=cam, reconnect_seconds=0.05, max_reconnects=3, idle_timeout_seconds=1.0)
    frames = list(src)
    src.release()
    assert src.reconnects >= 1 and FakeCamera.opened >= 2
    assert frames and frames[-1].pts_ms > frames[0].pts_ms
    assert all(f.camera_id == "angle0" and f.wall_ms is not None for f in frames)


# ---- 장면 번호·공통 시계·크기 조회 (실시간 시연 대비, docs/shoot4-demo-plan.md §4-2) ----

def test_index_is_milliseconds_so_time_axis_matches_pts():
    """처리가 느려 장면을 건너뛰어도 index = 0점 이후 밀리초 — 판정의 TimeAxis(1000, 0).to_ms(index) 가 pts 와 맞는다."""
    from src.fusion.shelf_events import TimeAxis
    from src.inputs.live_source import INDEX_PER_SECOND

    cam = FakeCamera(fps=100.0, total=80)
    src = LiveSource("rtsp://fake", "angle1", opener=cam, max_reconnects=0, idle_timeout_seconds=1.0)
    frames = []
    for frame in src:
        frames.append(frame)
        time.sleep(0.03)
    src.release()
    idx = [f.index for f in frames]
    assert all(b > a for a, b in zip(idx, idx[1:]))            # 늘 증가
    axis = TimeAxis(INDEX_PER_SECOND, 0.0)
    for f in frames:
        assert abs(axis.to_ms(f.index) - f.pts_ms) <= 1.0


def test_unknown_fps_camera_consumed_fast_keeps_time():
    """fps 를 안 알려 주는(0) 30fps 카메라를 빠르게 처리해도 번호→시각이 누적해서 어긋나지 않는다 (코드 리뷰 R2)."""
    from src.fusion.shelf_events import TimeAxis
    from src.inputs.live_source import INDEX_PER_SECOND

    cam = FakeCamera(fps=30.0, total=30)
    cam.get = lambda prop: {5: 0.0, 3: 64, 4: 48}.get(prop, 0)
    src = LiveSource("rtsp://fake", "angle1", opener=cam, max_reconnects=0, idle_timeout_seconds=1.0)
    frames = list(src)
    src.release()
    assert src.fps == 15.0 and len(frames) > 20
    axis = TimeAxis(INDEX_PER_SECOND, 0.0)
    assert abs(axis.to_ms(frames[-1].index) - frames[-1].pts_ms) <= 1.0


def test_max_frames_counts_sent_frames_not_index():
    cam = FakeCamera(fps=100.0, total=200)
    src = LiveSource("rtsp://fake", "angle1", opener=cam, max_reconnects=0, max_frames=5, idle_timeout_seconds=1.0)
    n = 0
    for _ in src:
        n += 1
        time.sleep(0.03)                                        # index 는 3씩 건너뜀
    src.release()
    assert n == 5


def test_shared_start_gives_same_pts_for_same_moment():
    """두 카메라를 다른 시각에 열어도 같은 start_s 를 주면 같은 순간이 같은 pts."""
    now = {"t": 100.0}
    clock = lambda: now["t"]
    a = LiveSource("rtsp://a", "angle1", opener=FakeCamera(), max_reconnects=0, clock=clock, start_s=100.0)
    now["t"] = 102.5                                            # 두 번째 카메라는 2.5초 늦게 연결
    b = LiveSource("rtsp://b", "angle0", opener=FakeCamera(), max_reconnects=0, clock=clock, start_s=100.0)
    a.release(); b.release()
    assert a.start_s == b.start_s == 100.0
    c = LiveSource("rtsp://c", "x", opener=FakeCamera(), max_reconnects=0, clock=clock)
    c.release()
    assert c.start_s == 102.5                                   # 안 주면 연 직후 (예전 그대로)


def test_source_helpers():
    from src.inputs.live_source import is_live_source, normalize_source

    assert normalize_source("0") == 0 and normalize_source(" 1 ") == 1 and normalize_source("a.mp4") == "a.mp4"
    assert is_live_source("0") and is_live_source("rtsp://x") and not is_live_source("C:/v/a.mp4")


class _Cap:
    def __init__(self, opened=True, size=(64, 48), fps=0.0):
        self.opened, self.size, self.fps, self.source = opened, size, fps, None

    def __call__(self, source):
        self.source = source
        return self

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return {5: self.fps, 3: self.size[0], 4: self.size[1]}.get(prop, 0)

    def release(self):
        pass


def test_probe_source_normalizes_and_refuses_unknown_size():
    import pytest
    from src.multicam import probe_source

    cap = _Cap(fps=0.0)
    fps, size = probe_source("0", opener=cap)
    assert cap.source == 0 and size == (64, 48) and fps == 30.0
    with pytest.raises(RuntimeError):
        probe_source("0", opener=_Cap(opened=False))
    with pytest.raises(RuntimeError):
        probe_source("0", opener=_Cap(size=(0, 0)))


def test_live_axes_use_opened_fps_and_zero_offset():
    """크기 조회는 fps 0 이면 30 을, LiveSource 는 15 를 쓴다 — 실시간 카메라 시간축은 연 입력 값으로 다시 세운다 (녹화본은 그대로)."""
    from types import SimpleNamespace

    from src.core.config import CameraConfig
    from src.fusion.shelf_events import TimeAxis
    from src.multicam import MultiCameraPipeline

    ov, de = CameraConfig("angle1", "rtsp://a", "overview"), CameraConfig("angle0", "C:/v/b.mp4", "detail", offset_ms=500.0)
    fuser = SimpleNamespace(overview_axis=TimeAxis(30.0, 900.0), detail_axis=TimeAxis(30.0, 500.0))
    pipe = MultiCameraPipeline.__new__(MultiCameraPipeline)
    pipe.detail_cam, pipe.judge = de, SimpleNamespace(fuser=fuser)
    pipe._live_axes([ov, de], {"angle1": SimpleNamespace(fps=15.0), "angle0": SimpleNamespace(fps=30.0)})
    assert fuser.overview_axis == TimeAxis(1000.0, 0.0)
    assert fuser.detail_axis == TimeAxis(30.0, 500.0)
