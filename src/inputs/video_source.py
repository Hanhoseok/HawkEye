"""Phase 0: 영상 입력 계층.

OpenCV 로 영상을 프레임 단위로 읽으면서 frame index 와 ISO8601 timestamp 를 유지한다.
근거: 02 기능 명세 F1
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from ..core.types import Frame

# POS_MSEC 가 이 값을 넘으면 코덱이 쓰레기값을 준 것으로 본다(일주일 분량).
_MAX_PTS_MS = 7 * 24 * 3600 * 1000.0


class VideoSource:
    """mp4 파일 또는 카메라 인덱스를 프레임 스트림으로 바꾼다.

    with 문으로 쓰면 capture 가 확실히 닫힌다.

        with VideoSource("a.mp4") as src:
            for frame in src:
                ...
    """

    def __init__(
        self,
        source: str | int,
        start_time: str | None = None,
        start_frame: int = 0,
        max_frames: int | None = None,
        stride: int = 1,
        scale: float = 1.0,
        scale_mode: str = "shrink",
        blackout: str | None = None,
    ) -> None:
        if isinstance(source, str) and not source.isdigit():
            if not Path(source).exists():
                raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {source}")
        elif isinstance(source, str):
            source = int(source)

        self.source = source
        self.start_frame = max(0, start_frame)
        self.max_frames = max_frames
        self.stride = max(1, stride)
        self.scale = scale if scale and scale > 0 else 1.0
        self.scale_mode = scale_mode
        self._blackout = self._parse_blackout(blackout)
        # 한 번이라도 POS_MSEC 가 이상하면 이후로는 계속 fps 기반으로 계산한다.
        # 중간에 방식이 오락가락하면 timestamp 가 튀기 때문이다.
        self._use_pts = True
        self._start_time = (
            datetime.fromisoformat(start_time) if start_time else datetime.now().astimezone()
        )

        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"영상을 열 수 없습니다: {source}")

        self.fps: float = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        raw_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.scale != 1.0 and self.scale_mode == "resize":
            self.width, self.height = int(raw_w * self.scale), int(raw_h * self.scale)
        else:
            self.width, self.height = raw_w, raw_h
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.start_frame:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None  # type: ignore[assignment]

    @staticmethod
    def _parse_blackout(spec: str | None) -> tuple[int, int] | None:
        """"START:END" 를 프레임 구간으로 바꾼다."""
        if not spec:
            return None
        start, _, end = spec.partition(":")
        return int(start), int(end)

    def _apply_scale(self, image):
        """카메라가 멀어진 상황을 흉내낸다.

        resize: 프레임 자체를 축소 -> 디테일만 손실
        shrink: 프레임 크기는 유지하고 내용만 축소 -> 사람의 픽셀 크기까지 감소
        INTER_AREA 는 축소 시 정보 손실 특성이 실제 원거리 촬영에 가깝다.
        """
        h, w = image.shape[:2]
        small = cv2.resize(
            image,
            (max(1, int(w * self.scale)), max(1, int(h * self.scale))),
            interpolation=cv2.INTER_AREA,
        )
        if self.scale_mode == "resize":
            return small
        canvas = np.zeros_like(image)
        sh, sw = small.shape[:2]
        y0, x0 = (h - sh) // 2, (w - sw) // 2
        canvas[y0 : y0 + sh, x0 : x0 + sw] = small
        return canvas

    def __iter__(self) -> Iterator[Frame]:
        # frame.index 는 원본 영상의 절대 프레임 번호를 유지한다.
        # 실험 구간을 옮겨도 로그의 frame 번호가 원본과 그대로 대응해야 하기 때문이다.
        read_index = self.start_frame
        emitted = 0
        while True:
            ok, image = self._cap.read()
            if not ok:
                break

            if (read_index - self.start_frame) % self.stride == 0:
                if self.scale != 1.0:
                    image = self._apply_scale(image)
                if self._blackout and self._blackout[0] <= read_index < self._blackout[1]:
                    # 모두가 사라진 긴 공백을 만든다. tracker 는 track 을 버릴 수밖에 없고,
                    # 이후 같은 사람을 다시 이어붙일 수 있는지는 신원 레지스트리에 달린다.
                    image = np.zeros_like(image)
                pts_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
                expected_ms = read_index * 1000.0 / self.fps
                # 일부 컨테이너/코덱(MPEG2 등 실제 CCTV 녹화본)은 POS_MSEC 로
                # 0 이나 터무니없이 큰 값을 돌려준다. 그대로 쓰면 timestamp 계산이 깨진다.
                if self._use_pts and not (
                    0.0 <= pts_ms <= _MAX_PTS_MS and pts_ms <= expected_ms * 4 + 60_000.0
                ):
                    self._use_pts = False
                if not self._use_pts or (pts_ms == 0.0 and read_index > 0):
                    pts_ms = expected_ms
                timestamp = (self._start_time + timedelta(milliseconds=pts_ms)).isoformat()
                yield Frame(index=read_index, timestamp=timestamp, pts_ms=pts_ms, image=image)
                emitted += 1
                if self.max_frames is not None and emitted >= self.max_frames:
                    break

            read_index += 1
