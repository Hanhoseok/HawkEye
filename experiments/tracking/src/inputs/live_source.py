"""실시간 카메라 입력 — 최신 장면만, 끊기면 다시 연결 (설계 M-9, limitations F-6).

녹화본(VideoSource)은 장면을 하나도 빼지 않고 차례로 읽는다. 실시간 카메라에서 그렇게 하면 처리(초당 5~10장)가
카메라(초당 15~30장)보다 느려 지연이 계속 쌓인다. 그래서 읽기 스레드가 카메라를 계속 비우며 마지막 장면만 들고 있고,
처리 쪽은 새 장면이 생길 때마다 그 마지막 장면을 가져간다 — 밀린 장면은 버린다.
읽기가 실패하면(망 끊김, 카메라 재부팅) 잠깐 쉬었다가 다시 연다. 다시 연결된 동안의 시간은 장면이 없을 뿐 시계는 흐른다.

장면 시각(pts_ms)은 '받은 순간의 실제 시계'에서 시작 시각을 뺀 값이다 — 카메라가 여러 대여도 같은 시계를 쓰므로
녹화본처럼 시차를 맞출 필요가 없다 (PC 시계 하나 기준).
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Callable, Iterator

from ..core.types import Frame


class LiveSource:
    def __init__(
        self,
        source: str | int,
        camera_id: str = "cam0",
        reconnect_seconds: float = 2.0,
        max_reconnects: int | None = None,
        max_frames: int | None = None,
        idle_timeout_seconds: float = 30.0,
        opener: Callable | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """opener: source → cv2.VideoCapture 같은 것 (테스트에서 가짜를 넣는다)."""
        if opener is None:
            import cv2
            opener = cv2.VideoCapture
        self.source = int(source) if isinstance(source, str) and source.isdigit() else source
        self.camera_id = camera_id
        self.reconnect_seconds = reconnect_seconds
        self.max_reconnects = max_reconnects
        self.max_frames = max_frames
        self.idle_timeout_seconds = idle_timeout_seconds
        self._opener = opener
        self._clock = clock
        self._cap = self._open()
        self.fps = float(self._cap.get(5) or 15.0)          # cv2.CAP_PROP_FPS
        self.width = int(self._cap.get(3) or 0)             # cv2.CAP_PROP_FRAME_WIDTH
        self.height = int(self._cap.get(4) or 0)            # cv2.CAP_PROP_FRAME_HEIGHT
        self.total_frames = 0
        self.start_s = clock()
        self._latest = None          # (받은 시각 s, 이미지)
        self._seq = 0
        self._taken = 0
        self._lock = threading.Condition()
        self._stop = False
        self.reconnects = 0
        self.dropped = 0
        """처리가 못 따라가 버린 장면 수."""
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _open(self):
        cap = self._opener(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"카메라를 열 수 없습니다: {self.source}")
        return cap

    def _reader(self) -> None:
        while not self._stop:
            ok, image = self._cap.read()
            if ok:
                with self._lock:
                    if self._latest is not None and self._seq > self._taken:
                        self.dropped += 1
                    self._latest = (self._clock(), image)
                    self._seq += 1
                    self._lock.notify_all()
                continue
            # 끊김: 다시 연결
            if self.max_reconnects is not None and self.reconnects >= self.max_reconnects:
                with self._lock:
                    self._stop = True
                    self._lock.notify_all()
                return
            self.reconnects += 1
            try:
                self._cap.release()
            except Exception:
                pass
            time.sleep(self.reconnect_seconds)
            try:
                self._cap = self._open()
            except RuntimeError:
                continue

    def __enter__(self) -> "LiveSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def release(self) -> None:
        with self._lock:
            self._stop = True
            self._lock.notify_all()
        self._thread.join(timeout=2.0)
        try:
            self._cap.release()
        except Exception:
            pass

    def __iter__(self) -> Iterator[Frame]:
        index = 0
        while True:
            with self._lock:
                waited = self._lock.wait_for(lambda: self._stop or self._seq > self._taken, self.idle_timeout_seconds)
                if self._seq <= self._taken:
                    return          # 멈춤 또는 오래 장면이 없음
                self._taken = self._seq
                got_s, image = self._latest
            pts_ms = (got_s - self.start_s) * 1000.0
            wall = got_s * 1000.0
            timestamp = datetime.fromtimestamp(got_s, timezone.utc).astimezone().isoformat()
            yield Frame(index=index, timestamp=timestamp, pts_ms=pts_ms, image=image,
                        camera_id=self.camera_id, wall_ms=wall)
            index += 1
            if self.max_frames is not None and index >= self.max_frames:
                return
            if not waited and self._stop:
                return
