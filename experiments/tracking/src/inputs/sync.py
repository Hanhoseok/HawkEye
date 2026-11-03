"""두 카메라 녹화본의 시차를 기록·추정한다 (영상 라이브러리 없이 — 표와 신호 계산만).

두 각도 영상은 초당 장면 수(30/60)도, 녹화 시작 시각도 다르다 (limitations B-10).
기준 카메라(전체 카메라 angle1)의 재생 시각을 공통 시간축으로 삼고, 다른 카메라는
`offset_ms` 만큼 밀어서 같은 축에 놓는다:

    공통 시각 = 그 카메라의 재생 시각 + offset_ms
    offset_ms = (기준 카메라에서 그 사건이 보인 시각) − (이 카메라에서 같은 사건이 보인 시각)

사람 신호로 자동 추정한 값은 약하고 신호마다 답이 달랐다(design §2-3) → 제안값으로만 쓰고,
사람이 두 화면을 보며 확정한 값을 표(`labels/<촬영>/sync.csv`)에 남긴다.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

FIELDS = ("pair", "video", "reference", "offset_ms", "method", "checked_by", "note")


@dataclass
class SyncRow:
    pair: str
    """짝 이름 (예: N1)."""
    video: str
    """밀어 맞출 영상 (예: 1(angle0)/20261028_N1.mp4)."""
    reference: str
    """기준 영상 (예: 1(angle1)/N01.mp4)."""
    offset_ms: float
    method: str = "manual"
    """manual (사람이 두 화면을 보고 맞춤) / suggest (자동 추정, 확인 전)."""
    checked_by: str = ""
    note: str = ""


def offset_from_events(event_ms_in_video: float, event_ms_in_reference: float) -> float:
    """같은 사건(박수, 문 열림)이 각 영상에서 보인 재생 시각으로 시차를 구한다."""
    return event_ms_in_reference - event_ms_in_video


class SyncTable:
    """짝별 시차 표. 같은 짝을 다시 저장하면 덮어쓴다."""

    def __init__(self, rows: list[SyncRow] | None = None) -> None:
        self.rows: dict[str, SyncRow] = {r.pair: r for r in rows or []}

    @classmethod
    def load(cls, path: str | Path) -> "SyncTable":
        p = Path(path)
        if not p.exists():
            return cls()
        with p.open(encoding="utf-8", newline="") as f:
            rows = [SyncRow(**{**r, "offset_ms": float(r["offset_ms"])}) for r in csv.DictReader(f)]
        return cls(rows)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            for pair in sorted(self.rows):
                writer.writerow(asdict(self.rows[pair]))

    def upsert(self, row: SyncRow) -> None:
        self.rows[row.pair] = row

    def offset_ms(self, pair: str) -> float | None:
        row = self.rows.get(pair)
        return row.offset_ms if row else None


def best_lag(signal: np.ndarray, reference: np.ndarray, hz: float, max_lag_s: float = 3.0) -> tuple[float, float]:
    """두 신호(같은 간격 hz)의 상관이 가장 큰 시차를 찾는다.

    돌려주는 값: (offset_ms, 상관). offset_ms 는 위 정의와 같다 — signal 쪽 시각 + offset_ms = reference 쪽 시각.
    """
    n = min(len(signal), len(reference))
    a = np.asarray(signal[:n], dtype=float)
    b = np.asarray(reference[:n], dtype=float)
    a, b = a - a.mean(), b - b.mean()
    best = (0, -1.0)
    limit = int(round(max_lag_s * hz))
    for lag in range(-limit, limit + 1):
        # reference 의 k 번째 = signal 의 k - lag 번째 → signal 시각 + lag = reference 시각
        if lag >= 0:
            x, y = a[: n - lag], b[lag:]
        else:
            x, y = a[-lag:], b[: n + lag]
        den = float(np.linalg.norm(x) * np.linalg.norm(y))
        corr = float(x @ y / den) if den else 0.0
        if corr > best[1]:
            best = (lag, corr)
    return best[0] * 1000.0 / hz, best[1]
