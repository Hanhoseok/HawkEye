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
