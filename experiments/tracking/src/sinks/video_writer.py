"""결과 영상 저장. 근거: 02 기능 명세 F9(일부)"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2


class VideoWriter:
    """첫 프레임 크기에 맞춰 지연 초기화되는 mp4 writer."""

    def __init__(self, path: str | Path, fps: float, fourcc: str = "mp4v") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps if fps and fps > 0 else 30.0
        self._fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self._writer: cv2.VideoWriter | None = None

    def write(self, image: Any) -> None:
        if self._writer is None:
            height, width = image.shape[:2]
            self._writer = cv2.VideoWriter(
                str(self.path), self._fourcc, self.fps, (width, height)
            )
            if not self._writer.isOpened():
                raise RuntimeError(f"결과 영상을 열 수 없습니다: {self.path}")
        self._writer.write(image)

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
