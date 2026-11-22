"""영상의 소리를 16kHz 모노 배열로 읽는다.

먼저 ffmpeg 를 쓰고, 없으면 OpenCV 의 Windows 소리 읽기(MSMF)를 쓴다. MSMF 는 한글 경로를 못 열어
영문 경로로 잠깐 잇거나(하드 링크) 복사해서 읽는다. 둘 다 안 되면 빈 배열.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SR = 16000


def read_audio(path: str | Path, sr: int = SR) -> np.ndarray:
    path = Path(path)
    got = _ffmpeg(path, sr)
    if got is None:
        got = _msmf(path, sr)
    return got if got is not None else np.zeros(0)


def _ffmpeg(path: Path, sr: int) -> np.ndarray | None:
    try:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr),
                              "-f", "s16le", "-"], capture_output=True).stdout
    except OSError:
        return None
    return np.frombuffer(raw[: len(raw) // 2 * 2], np.int16).astype(np.float64)


def _msmf(path: Path, sr: int) -> np.ndarray | None:
    import cv2

    if not hasattr(cv2, "CAP_PROP_AUDIO_STREAM") or not path.exists():
        return None
    tmp_dir = None
    src = path
    if not str(path).isascii():
        tmp_dir = Path(tempfile.mkdtemp(prefix="hawkeye_audio_"))
        src = tmp_dir / f"audio{path.suffix}"
        try:
            os.link(path, src)
        except OSError:
            shutil.copyfile(path, src)
    try:
        cap = cv2.VideoCapture(str(src), cv2.CAP_MSMF,
                               [cv2.CAP_PROP_AUDIO_STREAM, 0, cv2.CAP_PROP_VIDEO_STREAM, -1,
                                cv2.CAP_PROP_AUDIO_DATA_DEPTH, cv2.CV_16S, cv2.CAP_PROP_AUDIO_SAMPLES_PER_SECOND, sr])
        if not cap.isOpened():
            return None
        base = int(cap.get(cv2.CAP_PROP_AUDIO_BASE_INDEX))
        channels = max(1, int(cap.get(cv2.CAP_PROP_AUDIO_TOTAL_CHANNELS)))
        chunks = []
        while cap.grab():
            parts = []
            for ch in range(channels):
                ok, data = cap.retrieve(None, base + ch)
                if ok and data is not None and data.size:
                    parts.append(np.asarray(data, dtype=np.float64).reshape(-1))
            if parts:
                n = min(len(p) for p in parts)
                chunks.append(np.mean([p[:n] for p in parts], axis=0))
        cap.release()
        return np.concatenate(chunks) if chunks else np.zeros(0)
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def tone_score(p: np.ndarray, f: np.ndarray, hz: float, tol_hz: float = 40.0) -> np.ndarray:
    """장면마다 그 주파수의 순음 점수 = min(둘레 주파수 대비, 그 영상 평소 대비). p = STFT 세기, f = 주파수."""
    band = (f >= hz - tol_hz) & (f <= hz + tol_hz)
    near = (f >= hz * 0.7) & (f <= hz * 1.4) & ~((f >= hz - 3 * tol_hz) & (f <= hz + 3 * tol_hz))
    top = p[band].max(axis=0)
    return np.minimum(top / (np.median(p[near], axis=0) + 1e-9), top / (np.median(top) + 1e-9))


def chimes(x: np.ndarray, sr: int = SR, low_hz: float = 984.0, high_hz: float = 1312.0,
           min_low: float = 1000.0, min_high: float = 300.0, max_gap_s: float = 0.5) -> list[float]:
    """결제 완료 소리(낮은 음 → 높은 음 두 순음)가 시작된 때 (초). 4차 촬영 계산대: 약 984Hz → 1312Hz.

    낮은 음 점수가 min_low 를 넘고 max_gap_s 안에 높은 음 점수가 min_high 를 넘어야 한다.
    4차 42쌍에서 결제 소리는 낮은 음 5,000 넘음 · 높은 음 1,000 넘음, 다른 소리는 낮은 음 110 아래였다 (두 카메라 모두).
    """
    if len(x) < sr // 2:
        return []
    from scipy.signal import stft

    f, t, z = stft(x, fs=sr, nperseg=1024, noverlap=768)
    p = np.abs(z) ** 2
    low, high = tone_score(p, f, low_hz), tone_score(p, f, high_hz)
    found: list[float] = []
    for i in np.flatnonzero(low >= min_low):
        if found and t[i] - found[-1] <= 1.0:
            continue
        after = (t > t[i]) & (t <= t[i] + max_gap_s)
        if after.any() and high[after].max() >= min_high:
            found.append(float(t[i]))
    return found
