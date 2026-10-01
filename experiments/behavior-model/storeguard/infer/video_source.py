"""영상 수신 스레드 (RTSP / MP4 파일).

설계 원칙
  - 수신과 추론을 분리한다. 수신 스레드는 절대 추론을 기다리지 않는다.
  - 버퍼는 크기가 제한되고, 가득 차면 **오래된 프레임부터 버린다**(실시간성 우선).
  - 연결이 끊기면 지수 백오프로 재접속하고, 재접속 시 버퍼를 비운다(오래된 프레임 재사용 금지).
  - 마지막 프레임 수신 시각을 기록한다. stale_after_sec 을 넘으면 LIVE 가 아니다.
  - URL 의 계정/비밀번호는 상태·로그 어디에도 원문으로 나가지 않는다.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from storeguard.utils import mask_url


@dataclass
class Frame:
    seq: int
    recv_ts: float          # 수신 시각(epoch)
    media_ts: float         # 소스 기준 시각(초). 파일이면 영상 내 시각, RTSP 면 수신 경과 시간.
    image: np.ndarray       # BGR


@dataclass
class SourceStatus:
    url_masked: str
    kind: str
    connected: bool = False
    opening: bool = False
    last_frame_ts: float | None = None
    frames_received: int = 0
    frames_dropped: int = 0
    reconnects: int = 0
    last_error: str | None = None
    measured_fps: float = 0.0
    source_fps: float = 0.0
    eof: bool = False
    _fps_win: deque = field(default_factory=lambda: deque(maxlen=30), repr=False)

    def public(self, stale_after: float) -> dict:
        now = time.time()
        age = None if self.last_frame_ts is None else round(now - self.last_frame_ts, 2)
        live = bool(self.connected and age is not None and age <= stale_after)
        return {
            "url": self.url_masked, "kind": self.kind,
            "connected": self.connected, "opening": self.opening, "live": live,
            "last_frame_age_sec": age,
            "last_frame_at": self.last_frame_ts,
            "frames_received": self.frames_received, "frames_dropped": self.frames_dropped,
            "reconnects": self.reconnects, "last_error": self.last_error,
            "measured_fps": round(self.measured_fps, 2), "source_fps": round(self.source_fps, 2),
            "eof": self.eof,
        }


class VideoSource:
    """백그라운드 스레드로 프레임을 읽어 제한된 버퍼에 넣는다."""

    def __init__(self, url: str, *, kind: str = "auto", buffer_size: int = 64,
                 drop_oldest: bool = True, reconnect_delay: float = 3.0,
                 reconnect_max_delay: float = 30.0, loop_file: bool = False,
                 realtime_file: bool = True, name: str = "cam",
                 rtsp_transport: str = "tcp", rtsp_timeout_sec: float = 8.0):
        self.url = url
        self.name = name
        if kind == "auto":
            kind = "rtsp" if "://" in url and not url.startswith("file:") else "file"
        self.kind = kind
        self.buffer_size = buffer_size
        self.drop_oldest = drop_oldest
        self.reconnect_delay = reconnect_delay
        self.reconnect_max_delay = reconnect_max_delay
        self.loop_file = loop_file
        self.realtime_file = realtime_file
        self.rtsp_transport = rtsp_transport
        self.rtsp_timeout_sec = rtsp_timeout_sec

        self.status = SourceStatus(url_masked=mask_url(url), kind=kind)
        self._buf: deque[Frame] = deque()
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._seq = 0
        self._epoch = 0                     # 재접속마다 증가. 오래된 프레임 구분용
        self.latest: Frame | None = None    # 미리보기용(버퍼와 별개)

    # --- 수명주기 ---
    def start(self) -> "VideoSource":
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"src-{self.name}", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        if self._thread:
            self._thread.join(timeout=5.0)

    @property
    def epoch(self) -> int:
        return self._epoch

    # --- 소비자 API ---
    def read(self, timeout: float = 1.0) -> Frame | None:
        with self._cv:
            if not self._buf:
                self._cv.wait(timeout)
            if not self._buf:
                return None
            return self._buf.popleft()

    def clear(self) -> int:
        with self._cv:
            n = len(self._buf)
            self._buf.clear()
            return n

    def buffered(self) -> int:
        with self._lock:
            return len(self._buf)

    # --- 내부 ---
    def _open(self) -> cv2.VideoCapture | None:
        self.status.opening = True
        if self.kind == "rtsp":
            # OpenCV 기본 RTSP 열기 타임아웃은 30초라 끊김을 늦게 알아챈다.
            # FFmpeg 옵션으로 줄인다. 이 환경변수는 VideoCapture 생성 전에 설정되어야 한다.
            # UDP 는 방화벽/NAT 에서 프레임이 유실되기 쉬워 tcp 를 기본으로 한다.
            us = int(self.rtsp_timeout_sec * 1_000_000)
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"rtsp_transport;{self.rtsp_transport}|timeout;{us}|stimeout;{us}")
        if self.kind == "rtsp":
            # 환경변수만으로는 OpenCV 내부 30초 인터럽트 타임아웃이 우선한다.
            # 열기/읽기 타임아웃은 이 파라미터로만 실제로 줄어든다.
            ms = int(self.rtsp_timeout_sec * 1000)
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, ms,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, ms,
            ])
        else:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if self.kind == "rtsp":
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        self.status.opening = False
        if not cap.isOpened():
            self.status.last_error = "연결 실패"
            self.status.connected = False
            try:
                cap.release()
            except Exception:
                pass
            return None
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.status.source_fps = fps if fps and fps == fps else 0.0
        self.status.connected = True
        self.status.last_error = None
        self._epoch += 1
        self.clear()
        return cap

    def _push(self, frame: Frame) -> None:
        with self._cv:
            if len(self._buf) >= self.buffer_size:
                if self.drop_oldest:
                    self._buf.popleft()
                    self.status.frames_dropped += 1
                else:
                    self.status.frames_dropped += 1
                    return
            self._buf.append(frame)
            self.latest = frame
            self._cv.notify()

    def _run(self) -> None:
        delay = self.reconnect_delay
        while not self._stop.is_set():
            cap = self._open()
            if cap is None:
                if self._stop.wait(delay):
                    break
                self.status.reconnects += 1
                delay = min(delay * 2, self.reconnect_max_delay)
                continue
            delay = self.reconnect_delay
            t_start = time.time()
            first_media = None
            last_emit = time.perf_counter()
            src_fps = self.status.source_fps or 0.0

            while not self._stop.is_set():
                ok, img = cap.read()
                now = time.time()
                if not ok:
                    if self.kind == "file":
                        self.status.eof = True
                        if self.loop_file:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            self.status.eof = False
                            t_start = now
                            first_media = None
                            # 반복 재생은 media_ts 가 0 으로 되돌아가는 불연속이다.
                            # epoch 을 올려 소비자가 클립 버퍼와 사건 상태를 버리게 한다.
                            # (안 그러면 샘플링 기준 시각이 과거로 돌아가 프레임이 전부 버려진다)
                            self._epoch += 1
                            self.clear()
                            continue
                        self.status.connected = False
                        break
                    self.status.last_error = "프레임 읽기 실패"
                    self.status.connected = False
                    break

                if self.kind == "file":
                    pos = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
                    if first_media is None:
                        first_media = pos
                    media_ts = pos
                    if self.realtime_file and src_fps > 0:
                        # 파일을 실시간 속도로 재생해 실시간 조건을 흉내낸다
                        target = last_emit + 1.0 / src_fps
                        sleep = target - time.perf_counter()
                        if sleep > 0:
                            time.sleep(sleep)
                        last_emit = max(target, time.perf_counter())
                else:
                    media_ts = now - t_start

                self._seq += 1
                self.status.frames_received += 1
                self.status.last_frame_ts = now
                self.status._fps_win.append(now)
                if len(self.status._fps_win) >= 2:
                    span = self.status._fps_win[-1] - self.status._fps_win[0]
                    if span > 0:
                        self.status.measured_fps = (len(self.status._fps_win) - 1) / span
                self._push(Frame(self._seq, now, media_ts, img))

            try:
                cap.release()
            except Exception:
                pass
            self.status.connected = False
            if self.kind == "file" and self.status.eof and not self.loop_file:
                break
            if not self._stop.is_set():
                self.status.reconnects += 1
                if self._stop.wait(delay):
                    break
                delay = min(delay * 2, self.reconnect_max_delay)


def resolve_source(spec: str) -> tuple[str, str]:
    """'file:D:\\a.mp4' / 'rtsp://…' / 순수 경로 → (url, kind)."""
    if spec.startswith("file:"):
        return spec[5:], "file"
    if "://" in spec:
        return spec, "rtsp"
    p = Path(spec)
    if p.exists():
        return str(p), "file"
    raise FileNotFoundError(f"입력을 해석할 수 없음: {spec}")
