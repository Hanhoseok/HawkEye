"""알림 전후 영상 저장.

- 항상 순환 버퍼(deque)에 최근 pre_seconds 분량의 프레임을 갖고 있는다.
- 알림이 뜨면 **즉시** 스냅샷만 저장하고 녹화 작업을 큐에 넣는다.
  영상 파일 쓰기를 기다리느라 알림이 늦어지면 안 되므로 쓰기는 별도 스레드가 한다.
- 이후 post_seconds 동안 들어오는 프레임을 같은 작업에 붙인 뒤 mp4 로 쓴다.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class _Job:
    event_id: str
    out_path: Path
    frames: list[np.ndarray]
    deadline: float
    fps: float
    done: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


class ClipRecorder:
    def __init__(self, out_dir: Path, *, pre_seconds: float, post_seconds: float,
                 fps: float, width: int = 960, enabled: bool = True):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.pre = pre_seconds
        self.post = post_seconds
        self.fps = max(1.0, fps)
        self.width = width
        self.enabled = enabled
        self.ring: deque[np.ndarray] = deque(maxlen=max(1, int(pre_seconds * self.fps)))
        self._jobs: list[_Job] = []
        self.errors: list[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._writer_loop, name="recorder", daemon=True)
        self._thread.start()

    def _resize(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if w <= self.width:
            return img
        nh = int(round(h * self.width / w))
        nh -= nh % 2
        return cv2.resize(img, (self.width, max(2, nh)), interpolation=cv2.INTER_AREA)

    def add(self, img: np.ndarray) -> None:
        """샘플링된 프레임을 넣는다(추론에 쓰는 것과 같은 시간 간격)."""
        if not self.enabled:
            return
        small = self._resize(img)
        self.ring.append(small)
        with self._lock:
            for j in self._jobs:
                if not j.done:
                    with j.lock:
                        j.frames.append(small)

    def snapshot(self, img: np.ndarray, event_id: str, quality: int = 85) -> str | None:
        """알림 대표 이미지. 즉시 저장(작고 빠름)."""
        p = self.out_dir / f"{event_id}.jpg"
        ok, buf = cv2.imencode(".jpg", self._resize(img), [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return None
        p.write_bytes(buf.tobytes())
        return str(p)

    def start_clip(self, event_id: str) -> str | None:
        """녹화 예약. 즉시 반환한다(파일은 나중에 생긴다)."""
        if not self.enabled:
            return None
        out = self.out_dir / f"{event_id}.mp4"
        job = _Job(event_id=event_id, out_path=out, frames=list(self.ring),
                   deadline=time.time() + self.post, fps=self.fps)
        with self._lock:
            self._jobs.append(job)
        return str(out)

    def _writer_loop(self) -> None:
        while not self._stop.is_set():
            time.sleep(0.25)
            now = time.time()
            with self._lock:
                ready = [j for j in self._jobs if not j.done and now >= j.deadline]
            for j in ready:
                try:
                    self._write(j)
                except Exception as exc:
                    self.errors.append(f"{j.event_id}: {exc}")
                    logging.getLogger("storeguard.recorder").warning(
                        "사건 %s 영상 저장 실패: %s", j.event_id, exc)
                j.done = True
            with self._lock:
                self._jobs = [j for j in self._jobs if not j.done]

    def _write(self, job: _Job) -> None:
        with job.lock:
            frames = list(job.frames)
        if not frames:
            return
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # 주의: OpenCV 는 **확장자**로 컨테이너를 고른다. 임시 파일이 '.part' 로 끝나면
        # VideoWriter 가 열리지 않아 0바이트 파일만 남는다. 반드시 .mp4 로 끝나게 한다.
        tmp = job.out_path.with_suffix(".part.mp4")
        vw = cv2.VideoWriter(str(tmp), fourcc, job.fps, (w, h))
        if not vw.isOpened():
            raise RuntimeError(f"VideoWriter 열기 실패: {tmp}")
        for f in frames:
            if f.shape[:2] != (h, w):
                f = cv2.resize(f, (w, h))
            vw.write(f)
        vw.release()
        tmp.replace(job.out_path)

    def pending(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs if not j.done)

    def stop(self, flush: bool = True) -> None:
        """종료. flush=True 면 아직 post 구간을 다 못 모은 작업도 지금까지 모인 프레임으로 저장한다.

        (그렇게 하지 않으면 알림 직후 프로세스가 끝날 때 영상이 통째로 사라진다.)
        """
        self._stop.set()
        self._thread.join(timeout=3.0)
        if not flush:
            return
        with self._lock:
            pending = [j for j in self._jobs if not j.done]
        for j in pending:
            try:
                self._write(j)
            except Exception as exc:
                self.errors.append(f"{j.event_id}: {exc}")
            j.done = True
