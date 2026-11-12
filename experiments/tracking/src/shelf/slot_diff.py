"""칸 그림 비교 — 물건을 몰라도 칸이 비었는지 본다 (설계 M-7 보조, docs/multiview-design.md).

물건 탐지(일반 YOLO)는 흰 카드를 0/5 로 못 봤고(D-14), 붙어 있는 병 묶음에서 하나 빠진 것도 '변화 없음'으로 봤다
(2차 촬영 N1, 2026-11-03). 그러면 손님 장부가 손 뻗기를 '만지기만'으로 지워 절도를 놓친다(D-17).
칸 상자는 사람이 미리 그렸으므로 칸 안 그림이 방문 전후로 바뀌었는지는 물건 종류와 상관없이 볼 수 있다.

    1. 기준 그림: 칸을 아무도 안 가리고 stable_seconds 동안 그대로면 그때 그림을 저장
    2. 가렸다가 다시 stable_seconds 동안 안 가리면 → 방문 전 기준 그림과 비교
    3. 비교: 밝기를 맞춘(평균·편차 정규화) 흐린 회색 그림의 차이가 diff_level 을 넘는 화소 비율.
       카메라가 몇 픽셀 흔들려도 되게 ±shift_px 안에서 가장 잘 맞는 위치로 잰다.
    4. 조명 거르기: 같은 때 다른 칸들도 함께 바뀌었으면(그 칸만 뚜렷하지 않으면) 조명·그림자로 보고 버린다.
    5. 방향: 빈 칸 사진이 있으면 그쪽에 가까워졌는지로, 없으면 윤곽선 양으로 추정 —
       물건이 빠지면 빈 선반 면이 드러나 윤곽선이 준다(줄어듦), 놓으면 는다(늘어남), 애매하면 '바뀜'.

영상 라이브러리(cv2)를 쓰는 관찰 층이다. 판단(⑤)은 결과 문자열만 받는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class SlotChange:
    slot: str
    start_ms: float
    """가리기 시작한 시각."""
    end_ms: float
    """다시 안정된 시각."""
    ratio: float
    """바뀐 화소 비율 0~1."""
    direction: str
    """줄어듦 / 늘어남 / 바뀜."""
    edge_ratio: float
    """지금 윤곽선 양 ÷ 기준 윤곽선 양."""

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


@dataclass
class _Slot:
    name: str
    box: tuple[int, int, int, int]
    reference: np.ndarray | None = None
    clear_since: float | None = None
    blocked_since: float | None = None
    was_blocked: bool = False
    empty: np.ndarray | None = None


@dataclass
class SlotDiffConfig:
    stable_seconds: float = 0.6
    diff_level: float = 0.6
    """정규화한 밝기 차이(표준편차 단위)가 이보다 크면 '바뀐 화소'."""
    min_ratio: float = 0.08
    """바뀐 화소가 칸의 이 비율 이상이어야 변화."""
    lighting_margin: float = 0.05
    """다른 칸들의 바뀐 비율(중앙값)보다 이만큼 더 커야 그 칸만의 변화로 본다."""
    shift_px: int = 4
    edge_drop: float = 0.85
    edge_rise: float = 1.15
    block_margin: float = 0.15
    """칸 상자를 이 비율만큼 넓혀 사람 상자와 겹치면 '가림' (손·팔이 칸 위에 걸치는 경우)."""


def _prep(patch: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32), (7, 7), 0)
    return (g - g.mean()) / (g.std() + 1e-6)


def _edges(patch: np.ndarray) -> float:
    g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return float((cv2.Canny(g, 40, 120) > 0).mean())


def changed_ratio(ref: np.ndarray, cur: np.ndarray, level: float, shift: int) -> float:
    """두 칸 그림(같은 크기 BGR)에서 바뀐 화소 비율. ±shift 안에서 가장 작은 값."""
    a, b = _prep(ref), _prep(cur)
    h, w = a.shape
    best = 1.0
    for dy in range(-shift, shift + 1, 2):
        for dx in range(-shift, shift + 1, 2):
            ya, yb = max(0, dy), max(0, -dy)
            xa, xb = max(0, dx), max(0, -dx)
            hh, ww = h - abs(dy), w - abs(dx)
            if hh <= 0 or ww <= 0:
                continue
            d = np.abs(a[ya:ya + hh, xa:xa + ww] - b[yb:yb + hh, xb:xb + ww])
            best = min(best, float((d > level).mean()))
    return best


class SlotDiff:
    def __init__(self, slots: list[tuple[str, tuple[float, float, float, float]]],
                 config: SlotDiffConfig | None = None, empty_images: dict[str, np.ndarray] | None = None) -> None:
        """slots: (칸 이름, 픽셀 상자 x1,y1,x2,y2)."""
        self.config = config or SlotDiffConfig()
        self.slots = [_Slot(n, tuple(int(round(v)) for v in b)) for n, b in slots]
        for s in self.slots:
            if empty_images and s.name in empty_images:
                s.empty = empty_images[s.name]
        self.changes: list[SlotChange] = []
        self.episodes: list[dict] = []
        """가림 한 번이 끝날 때마다 한 줄 (평가용 기록 — 판정에는 안 씀). state:
        compared(기준과 비교함, changed 0/1) / no_reference(첫 가림이라 기준 없음) / unfinished(finish 때 아직 가림·안정 전)."""

    def reset(self) -> None:
        for s in self.slots:
            s.reference = None
            s.clear_since = s.blocked_since = None
            s.was_blocked = False
        self.changes = []
        self.episodes = []

    def _episode(self, s: _Slot, end_ms: float, state: str, ratio: float | None = None,
                 baseline: float | None = None, change: SlotChange | None = None) -> None:
        self.episodes.append({"slot": s.name, "start_ms": s.blocked_since if s.blocked_since is not None else end_ms,
                              "end_ms": end_ms, "state": state, "ratio": ratio, "baseline": baseline,
                              "changed": int(change is not None),
                              "direction": change.direction if change else "",
                              "edge_ratio": change.edge_ratio if change else None})

    def finish(self, now_ms: float) -> None:
        """영상 끝: 아직 가렸거나 다시 보인 지 안정 시간이 안 된 칸을 unfinished 로 남긴다."""
        for s in self.slots:
            if s.was_blocked:
                self._episode(s, now_ms, "unfinished")
                s.was_blocked = False
                s.blocked_since = None

    def _blocked(self, slot: _Slot, people: list) -> bool:
        x1, y1, x2, y2 = slot.box
        mx, my = (x2 - x1) * self.config.block_margin, (y2 - y1) * self.config.block_margin
        bx1, by1, bx2, by2 = x1 - mx, y1 - my, x2 + mx, y2 + my
        return any(not (b[2] < bx1 or b[0] > bx2 or b[3] < by1 or b[1] > by2) for b in people)

    def update(self, now_ms: float, image, people_boxes: list) -> list[SlotChange]:
        """이번 장면에서 확정된 칸 변화들."""
        if image is None:
            return []
        cfg = self.config
        stable = cfg.stable_seconds * 1000.0
        ready: list[tuple[_Slot, np.ndarray]] = []
        for s in self.slots:
            x1, y1, x2, y2 = s.box
            patch = image[max(0, y1):y2, max(0, x1):x2]
            if patch.size == 0:
                continue
            if self._blocked(s, people_boxes):
                if not s.was_blocked:
                    s.blocked_since = now_ms
                s.was_blocked = True
                s.clear_since = None
                continue
            if s.clear_since is None:
                s.clear_since = now_ms
            if now_ms - s.clear_since < stable:
                continue
            if s.reference is None:
                if s.was_blocked:
                    self._episode(s, now_ms, "no_reference")
                    s.blocked_since = None
                s.reference = patch.copy()
                s.was_blocked = False
                continue
            if s.was_blocked:
                ready.append((s, patch.copy()))
            else:
                s.reference = patch.copy()   # 아무 일 없을 때는 기준을 천천히 새로 (조명 변화 따라가기)
        if not ready:
            return []
        # 같은 때 다른 칸들의 바뀐 정도 — 조명 변화 거르기
        others = {}
        for s in self.slots:
            if s.reference is None:
                continue
            x1, y1, x2, y2 = s.box
            cur = image[max(0, y1):y2, max(0, x1):x2]
            if cur.shape == s.reference.shape:
                others[s.name] = changed_ratio(s.reference, cur, cfg.diff_level, cfg.shift_px)
        out = []
        for s, cur in ready:
            ratio = others.get(s.name, changed_ratio(s.reference, cur, cfg.diff_level, cfg.shift_px))
            rest = [v for k, v in others.items() if k != s.name]
            baseline = float(np.median(rest)) if rest else 0.0
            change = None
            if ratio >= cfg.min_ratio and ratio - baseline >= cfg.lighting_margin:
                er = _edges(cur) / max(_edges(s.reference), 1e-6)
                if s.empty is not None and s.empty.shape == cur.shape:
                    closer = changed_ratio(s.empty, cur, cfg.diff_level, cfg.shift_px) < \
                        changed_ratio(s.empty, s.reference, cfg.diff_level, cfg.shift_px)
                    direction = "줄어듦" if closer else "늘어남"
                else:
                    direction = "줄어듦" if er <= cfg.edge_drop else ("늘어남" if er >= cfg.edge_rise else "바뀜")
                change = SlotChange(s.name, s.blocked_since or now_ms, now_ms, ratio, direction, er)
                out.append(change)
            self._episode(s, now_ms, "compared", ratio, baseline, change)
            s.reference = cur
            s.was_blocked = False
            s.blocked_since = None
        self.changes.extend(out)
        return out
