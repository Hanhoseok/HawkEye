"""장면마다의 실제 시각 — '장면 번호 ÷ fps' 대신.

3차 촬영 angle1 휴대폰은 표시는 60fps 인데 장면을 빠뜨린다 (39편 중 31편, 장면 간격 16~233ms).
표시 60fps 로 세면 끝에서 0.5~10.4초, OpenCV 가 주는 평균 fps 로 세면 끝은 맞지만 중간이 최대 2.5초 어긋난다
(장면이 몰려서 빠짐 — C05 2.2초, N08-1 2.5초; docs/multiview-progress.md 후속 16).
OpenCV 의 CAP_PROP_POS_MSEC (read/grab 직후) 가 파일의 장면 시각(pts)과 같아서 그것을 쓴다.
단 **첫 장면만** 엉뚱한 값이 나온다 (3차 angle0 T01: -133.6ms, ffprobe 는 0) → `first_ms` 로 고친다.
첫 장면을 기준으로 빼면 그 영상 전체가 밀리므로 빼지 않는다 (파일 시각 그대로 — 소리 시각과 같은 축).
"""

from __future__ import annotations

import bisect
from pathlib import Path
from typing import Sequence

import numpy as np


def read_times(path: Path) -> np.ndarray:
    """영상의 장면별 시각 (ms, 첫 장면 기준). 순서대로 grab 만 한다."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    out = []
    while cap.grab():
        out.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    if len(out) > 1:
        out[0] = first_ms(out[0], out[1], fps)
    elif out:
        out[0] = max(out[0], 0.0)
    return np.asarray(out, dtype=float)


def first_ms(raw0: float, raw1: float, fps: float) -> float:
    """첫 장면 시각: OpenCV 값이 음수이거나 둘째 장면 이상이면 둘째 장면 - 1/fps (0 아래로는 안 감)."""
    if 0.0 <= raw0 < raw1:
        return raw0
    return max(0.0, raw1 - 1000.0 / fps)


def index_at(times: Sequence[float], ms: float) -> int:
    """ms 에 가장 가까운 장면 번호. 범위 밖은 첫·마지막 장면으로 자른다 (화면에서 범위 밖을 검게 할지는 호출부가 정한다)."""
    n = len(times)
    if n == 0:
        return 0
    i = bisect.bisect_left(times, ms)
    if i <= 0:
        return 0
    if i >= n:
        return n - 1
    return i if times[i] - ms < ms - times[i - 1] else i - 1


def end_ms(times: Sequence[float], fps: float) -> float:
    """영상이 끝나는 시각 = 마지막 장면 시각 + 마지막 간격 (장면이 하나면 1/fps)."""
    if len(times) == 0:
        return 0.0
    gap = times[-1] - times[-2] if len(times) > 1 else 1000.0 / fps
    return float(times[-1] + gap)


def drift_s(times: Sequence[float], fps: float) -> float:
    """'번호 ÷ fps' 로 셌을 때 끝에서 어긋나는 정도 (초)."""
    if len(times) == 0:
        return 0.0
    return float((times[-1] - (len(times) - 1) * 1000.0 / fps) / 1000.0)
