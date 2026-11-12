"""두 카메라 녹음의 시차 (GCC-PHAT) — 발 위치 시차(floor_sync)와 맞춰 볼 두 번째 독립 신호.

정의는 src/inputs/sync.py 와 같다: 공통 시각 = angle0 시각 + offset_ms (기준 angle1).

두 휴대폰 마이크가 멀고 매장이 조용해 피크가 약하다(다음 후보의 1.1~2.4배). 300~4000Hz 만 쓰면
저주파 웅웅거림·고주파 잡음이 빠져 39쌍 중 38쌍이 ±170ms 안에 모였지만, 눈으로 본 N06 은 틀렸다(+1ms, 실제 +0.7~1.0초).
그래서 혼자 믿지 않고 발 위치와 맞을 때만 쓴다 (floor_sync.combine).
"""

from __future__ import annotations

import numpy as np

BAND_HZ = (300.0, 4000.0)
MIN_PEAK = 1.2
"""소리 후보로 쓸 최소 피크 비 (가장 높은 상관 / ±50ms 밖 최대)."""
SILENT_RMS = 1e-3
"""정규화한 소리(최대 1)의 RMS 가 이보다 작으면 무음."""


def gcc_phat(a: np.ndarray, b: np.ndarray, sr: int, max_lag_s: float = 3.0,
             band: tuple[float, float] = BAND_HZ) -> tuple[float, float]:
    """a = angle0 소리, b = angle1 소리 → (offset_ms, 피크 비). 무음이면 (nan, 0)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0 or silent(a) or silent(b):
        return float("nan"), 0.0
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    x = np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n))
    x /= np.abs(x) + 1e-12
    x[(freqs < band[0]) | (freqs > band[1])] = 0
    c = np.fft.irfft(x, n)
    m = int(max_lag_s * sr)
    c = np.concatenate([c[-m:], c[:m + 1]])
    lags = np.arange(-m, m + 1) / sr
    k = int(np.argmax(c))
    far = np.abs(lags - lags[k]) > 0.05
    peak = float(c[k] / np.max(c[far])) if far.any() and np.max(c[far]) > 0 else 0.0
    # c[k] 가 크다 = a 의 사건이 b 보다 lags[k] 초 늦게 보인다 → angle0 시각 - lag = angle1 시각
    return float(-lags[k] * 1000.0), peak


def silent(x: np.ndarray) -> bool:
    top = float(np.max(np.abs(x))) if len(x) else 0.0
    return top == 0.0 or float(np.sqrt(np.mean((x / top) ** 2))) < SILENT_RMS


def usable(offset_ms: float | None, peak: float) -> bool:
    return offset_ms is not None and np.isfinite(offset_ms) and peak >= MIN_PEAK
