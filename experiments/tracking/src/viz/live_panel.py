"""실시간 상황판 — 영상 옆에 손님별 상태와 최근 사건을 띄운다 (`--panel`, `--show` 면 자동).

    ┌──────────── 영상 ────────────┬──── 상황판 ────┐
    │ 사람 상자·구역·선반           │ 12.3초  정답 구간 │
    │                              │ 손님 2  물건 1 · 결제 0  출구 앞  WARNING │
    │                              │ 손님 3  물건 0 · 결제 0  매장 안          │
    │                              │ 최근 사건                               │
    │                              │  75.2s 손님 2 WARNING ...               │
    └──────────────────────────────┴─────────────────────────────────────────┘

글자는 한글이라 OpenCV 대신 PIL(맑은 고딕)로 그린다. 글꼴이 없으면 기본 글꼴로 대신한다.
--truth 로 정답 구간(초)을 주면, 그 구간 안에서 상단에 '정답 구간'을 띄운다 — 검증하면서 맞춰 볼 수 있게.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import numpy as np

from ..core.types import RiskLevel

WIDTH = 380
MIN_HEIGHT = 480
COLORS = {  # RGB (PIL)
    RiskLevel.HIGH_RISK: (235, 70, 70), RiskLevel.REVIEW: (245, 160, 60),
    RiskLevel.WARNING: (240, 210, 70), RiskLevel.CLEAR: (110, 210, 120),
}
FONT_PATHS = ("C:/Windows/Fonts/malgun.ttf", "/System/Library/Fonts/AppleSDGothicNeo.ttc",
              "/usr/share/fonts/truetype/nanum/NanumGothic.ttf")


def _font(size: int):
    from PIL import ImageFont
    for p in FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def parse_truth(text: str | None) -> list[tuple[float, float]]:
    """'12.5-30;55-60' -> [(12.5, 30.0), (55.0, 60.0)] (초)."""
    spans = []
    for part in (text or "").replace(",", ";").split(";"):
        if "-" in part:
            a, b = part.split("-", 1)
            spans.append((float(a), float(b)))
    return spans


class LivePanel:
    def __init__(self, title: str = "", truth: list[tuple[float, float]] | None = None, log_lines: int = 9) -> None:
        self.title = title
        self.truth = truth or []
        self.log: deque = deque(maxlen=log_lines)
        self.verdict: dict[int, RiskLevel] = {}
        self._fonts = (_font(17), _font(14), _font(13))

    def note(self, time_sec: float, text: str, color=(220, 220, 220)) -> None:
        self.log.appendleft((f"{time_sec:6.1f}s  {text}", color))

    def add_events(self, events) -> None:
        for e in events:
            self.verdict[e.person_id] = e.level
            self.note(e.time_sec, f"손님 {e.person_id} {e.level.value} · 물건 {e.taken} 결제 {e.paid}",
                      COLORS.get(e.level, (220, 220, 220)))

    def render(self, canvas: np.ndarray, frame, risk=None) -> np.ndarray:
        from PIL import Image, ImageDraw

        h, w = canvas.shape[:2]
        if h < MIN_HEIGHT:   # 작은 영상(UCF 320x240)은 키워서 보여 준다
            s = MIN_HEIGHT / h
            canvas = cv2.resize(canvas, (int(w * s), MIN_HEIGHT), interpolation=cv2.INTER_LINEAR)
            h, w = canvas.shape[:2]
        panel = Image.new("RGB", (WIDTH, h), (24, 26, 30))
        d = ImageDraw.Draw(panel)
        big, mid, small = self._fonts
        t = frame.pts_ms / 1000.0
        y = 8
        d.text((10, y), self.title[:32], font=mid, fill=(170, 170, 170))
        y += 22
        d.text((10, y), f"{t:6.1f}초", font=big, fill=(255, 255, 255))
        inside = any(a <= t <= b for a, b in self.truth)
        if self.truth:
            d.rounded_rectangle((120, y, WIDTH - 10, y + 24), 5,
                                fill=(150, 40, 40) if inside else (50, 54, 60))
            d.text((130, y + 3), "정답: 절도 구간" if inside else "정답: 절도 구간 아님", font=mid, fill=(255, 255, 255))
        y += 36
        d.text((10, y), "손님", font=mid, fill=(150, 200, 255))
        y += 22
        if risk is not None:
            now = frame.pts_ms
            shown = 0
            for c in sorted(risk.book.customers.values(), key=lambda c: c.person_id):
                recent = now - c.last_seen_ms < 10_000
                if not (recent or c.departed or c.taken):
                    continue
                if shown >= 7:
                    break
                if c.departed:
                    where = "나감"
                elif now - c.last_seen_ms > 1000:
                    where = "안 보임"
                elif c.exit_since_ms is not None:
                    where = "출구 앞"
                elif c.checkout_since_ms is not None:
                    where = "계산대"
                elif c.outside_since_ms is not None:
                    where = "문 밖"
                else:
                    where = "매장 안"
                level = self.verdict.get(c.person_id)
                color = COLORS.get(level, (230, 230, 230))
                d.text((10, y), f"손님 {c.person_id}", font=mid, fill=color)
                d.text((80, y), f"물건 {c.taken} · 결제 {c.paid_items}", font=mid, fill=(230, 230, 230))
                d.text((210, y), where, font=mid, fill=(180, 180, 180))
                if level is not None:
                    d.text((275, y), level.value, font=small, fill=color)
                y += 21
                shown += 1
            if not shown:
                d.text((10, y), "(아직 없음)", font=small, fill=(120, 120, 120))
                y += 21
        y += 10
        d.line((10, y, WIDTH - 10, y), fill=(60, 64, 70))
        y += 8
        d.text((10, y), "최근 사건", font=mid, fill=(150, 200, 255))
        y += 22
        for text, color in self.log:
            if y > h - 18:
                break
            d.text((10, y), text[:40], font=small, fill=color)
            y += 18
        side = cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)
        return cv2.hconcat([canvas, side])
