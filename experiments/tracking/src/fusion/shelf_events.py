"""⑦-b 선반 사건 묶기 — 선반 카메라가 본 변화·손 접촉과 센서 기록을 전체 카메라의 손님에게 붙인다.

설계 M-7·M-8 (docs/multiview-design.md):
  주인 정하는 순서
    1. 선반 카메라가 '가린 사람 1명'으로 정한 변화 → 그 사람의 짝(⑦-a)
    2. 그 시간 그 선반 칸에 손목이 닿은 사람(손-칸 접촉) → 그 사람의 짝
    3. 그 시간 바닥의 선반 앞 띠에 서 있던 전체 카메라 손님이 한 명뿐이면 그 사람
    4. 여러 명이면 '누구 것인지 모름' (-1) — 지금과 같다
  센서 기록은 같은 시간(±event_tolerance)의 카메라 변화와 묶는다. 개수는 센서를 믿는다.
  센서 레인의 주인 (_lane_owner, 위 순서로 못 정할 때 — sensor_lane_first 면 먼저):
    위쪽(전체) 카메라에서 그 레인에 가장 오래 손이 머문 사람 → 선반 카메라 그 레인 손-칸 접촉 → 위쪽 카메라에서 그 시각 받침을 몸으로 덮은 사람

카메라마다 장면 번호가 다르므로(30/60FPS) 손님 장부로 넘기기 전에 모두 전체 카메라의 장면 번호로 바꾼다.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from ..core.config import FusionConfig
from ..core.types import TakeCandidate
from ..risk.customers import ShelfVisit
from .cross_view import CrossViewLinker


@dataclass(frozen=True)
class TimeAxis:
    """카메라 하나의 장면 번호 ↔ 공통 시각(ms)."""

    fps: float
    offset_ms: float = 0.0
    times: tuple | None = None
    """장면 번호 → 실제 시각(ms). 장면을 빠뜨리는 영상(3차 angle1)용. 없으면 번호 ÷ fps (기존 그대로)."""

    def to_ms(self, frame: int) -> float:
        if self.times:
            return self.times[max(0, min(int(frame), len(self.times) - 1))] + self.offset_ms
        return frame * 1000.0 / self.fps + self.offset_ms

    def to_frame(self, ms: float) -> int:
        if self.times:
            from ..inputs.frame_times import index_at

            return index_at(self.times, ms - self.offset_ms)
        return int(round((ms - self.offset_ms) * self.fps / 1000.0))


@dataclass(frozen=True)
class SensorEvent:
    """선반 센서 기록 하나 (설계 M-8 형식: 칸, 시각, 변화량, 확신도)."""

    time_ms: float
    slot: str
    delta_g: float
    confidence: float = 1.0
    unit: str = "g"
    """delta_g 의 단위. "g" = 무게(칸 무게 sensor_unit_g 로 개수 환산), "count" = 개수 그대로 (3차 레인 거리 센서)."""


def load_sensor_events(path: str | Path) -> list[SensorEvent]:
    """CSV: time_sec, slot, delta_g, confidence (기준 카메라 시간축)."""
    with Path(path).open(encoding="utf-8", newline="") as f:
        return [SensorEvent(float(r["time_sec"]) * 1000.0, r["slot"], float(r["delta_g"]),
                            float(r.get("confidence") or 1.0)) for r in csv.DictReader(f)]


@dataclass
class FusedRecord:
    """shelf_events.jsonl 한 줄 — 어떻게 주인을 정했는지 남긴다."""

    kind: str
    """visit (선반 변화) / take (손 접촉·머묾) / sensor."""
    start_ms: float
    end_ms: float
    owner: int
    how: str
    detail: str = ""

    def to_dict(self) -> dict:
        return {k: (round(v, 1) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


class ShelfEventFuser:
    def __init__(self, config: FusionConfig, linker: CrossViewLinker, detail_axis: TimeAxis,
                 overview_axis: TimeAxis, slot_shelf: dict[str, str] | None = None,
                 sensors: list[SensorEvent] | None = None) -> None:
        self.config = config
        self.linker = linker
        self.detail_axis = detail_axis
        self.overview_axis = overview_axis
        self.slot_shelf = slot_shelf or {}
        """칸 이름 → 그 칸이 속한 선반 이름 (선반 카메라 구역 파일)."""
        self.sensors = sorted(sensors or [], key=lambda s: s.time_ms)
        self.reset()

    def reset(self) -> None:
        self._touches: list[tuple[float, float, int, str]] = []
        """선반 카메라의 손-칸 접촉: (시작 ms, 끝 ms, 선반 카메라 신원, 칸)."""
        self._band: list[tuple[float, int]] = []
        """전체 카메라에서 선반 앞 띠에 서 있던 기록: (ms, 손님)."""
        self._sensor_used: set[int] = set()
        self._slot_changes: list = []
        """선반 카메라의 칸 그림 비교 결과 (SlotChange, 공통 시각)."""
        self._reaches: list[tuple[float, float, int]] = []
        """전체 카메라 팔 뻗기: (시작 ms, 끝 ms, 전체 카메라 손님)."""
        self._ov_lane: list[tuple[float, int, str]] = []
        """위쪽(전체) 카메라에서 손이 레인 안에 찍힌 장면: (ms, 손님, 레인)."""
        self._cover: list[tuple[float, int, float]] = []
        """위쪽(전체) 카메라에서 사람 상자가 받침을 덮은 장면: (ms, 손님, 덮은 비율)."""
        self.records: list[FusedRecord] = []

    # ------------------------------------------------------------------ 입력

    def note_band(self, ms: float, overview_id: int) -> None:
        self._band.append((ms, overview_id))
        if len(self._band) > 20000:
            self._band = self._band[-10000:]

    def note_reach(self, start_ms: float, end_ms: float, overview_id: int) -> None:
        self._reaches.append((start_ms, end_ms, overview_id))
        self.records.append(FusedRecord("reach", start_ms, end_ms, overview_id, "전체 카메라 팔 뻗기",
                                        "손목이 화면의 선반 영역 안"))

    def note_overview_lane(self, ms: float, overview_id: int, lane: str) -> None:
        self._ov_lane.append((ms, overview_id, lane))
        if len(self._ov_lane) > 20000:
            self._ov_lane = self._ov_lane[-10000:]

    def note_tray_cover(self, ms: float, overview_id: int, ratio: float) -> None:
        self._cover.append((ms, overview_id, ratio))
        if len(self._cover) > 20000:
            self._cover = self._cover[-10000:]

    def note_slot_change(self, change) -> None:
        self._slot_changes.append(change)

    def _slot_change_for(self, start: float, end: float, shelf: str) -> tuple[str | None, str]:
        """방문 시간과 겹친 칸 변화 — 줄어듦 > 늘어남 > 바뀜 순으로 하나. (방향, 설명)"""
        tol = self.config.event_tolerance_seconds * 1000.0
        hits = [c for c in self._slot_changes
                if self.slot_shelf.get(c.slot, shelf) == shelf and c.end_ms >= start - tol and c.start_ms <= end + tol]
        for word in ("줄어듦", "늘어남", "바뀜"):
            found = [c for c in hits if c.direction == word]
            if found:
                return word, ", ".join(f"{c.slot} {c.direction}({c.ratio:.2f})" for c in found)
        return None, ""

    def detail_take(self, take: TakeCandidate) -> TakeCandidate | None:
        """선반 카메라의 손 접촉 → 전체 카메라 손님의 집기 동작. 짝을 못 찾으면 None."""
        start, end = self.detail_axis.to_ms(take.start_frame), self.detail_axis.to_ms(take.end_frame)
        self._touches.append((start, end, take.person_id, take.zone))
        owner = self.linker.owner(take.person_id, start, end)
        how = "짝"
        if owner is None:
            reach = self._reach_owners(start, end)
            if len(reach) == 1:
                owner, how = reach.pop(), "전체 카메라 팔 뻗기"
            else:
                owner, how = self._band_owner(start, end), "선반 앞 한 명"
        self.records.append(FusedRecord("take", start, end, owner if owner is not None else -1, how,
                                        f"선반 카메라 {take.person_id} 손-칸 접촉 {take.zone}"))
        if owner is None:
            return None
        return replace(take, person_id=owner, zone=self.config.shelf_zone,
                       start_frame=self.overview_axis.to_frame(start), end_frame=self.overview_axis.to_frame(end))

    def detail_visit(self, visit: ShelfVisit, start_ms: float, end_ms: float) -> ShelfVisit:
        """선반 카메라의 선반 방문 → 전체 카메라 손님의 방문 (센서가 있으면 함께 묶음)."""
        owner, how = None, ""
        sensor = self._sensor_between(start_ms, end_ms, visit.shelf)
        lane_owner, lane_how = self._lane_owner(sensor, start_ms, end_ms, visit.shelf)
        if lane_owner is not None and self.config.sensor_lane_first:
            owner, how = lane_owner, lane_how
        reach = self._reach_owners(start_ms, end_ms)
        if owner is None and len(reach) == 1:
            # 전체 카메라에서 그 시각 선반으로 팔을 뻗은 사람이 한 명 — 카메라 사이 짝(선반 앞 띠 한 줄 좌표, B-14)을
            # 거치지 않는 직접 근거라 먼저 쓴다 (C2: 비슷한 옷 2명이 앞뒤로 서면 띠 위치로는 못 가름)
            owner, how = next(iter(reach)), "전체 카메라 팔 뻗기"
        if owner is None and visit.person_id > 0:
            owner, how = self.linker.owner(visit.person_id, start_ms, end_ms), "가린 사람의 짝"
        if owner is None:
            owners = self._touch_owners(start_ms, end_ms, visit.shelf)
            if len(owners) == 1:
                owner, how = owners.pop(), "손-칸 접촉"
            elif len(owners) > 1:
                both = owners & reach
                if len(both) == 1:
                    owner, how = both.pop(), "손 접촉 여럿 중 전체 카메라 팔 뻗기"
                elif lane_owner is not None:
                    owner, how = lane_owner, f"손 접촉 {len(owners)}명 중 {lane_how}"
                else:
                    how = f"손 접촉 {len(owners)}명 — 모름"
        if owner is None and not how:
            owner = self._band_owner(start_ms, end_ms)
            how = "선반 앞 한 명" if owner is not None else "선반 앞 여러 명 또는 없음 — 모름"
        if owner is None and lane_owner is not None:
            owner, how = lane_owner, lane_how
        removed, added, source, extra = visit.removed, visit.added, visit.source, ""
        slot_change, slot_text = visit.slot_change, ""
        if slot_change is None:
            slot_change, slot_text = self._slot_change_for(start_ms, end_ms, visit.shelf)
            if slot_text:
                extra += f" / 칸 그림: {slot_text}"
        if sensor:
            s_removed, s_added = self._sensor_counts(sensor)
            if (s_removed, s_added) != (removed, added):
                extra = f" / 센서와 다름 (카메라 -{removed}+{added}, 센서 -{s_removed}+{s_added})"
            removed, added, source = s_removed, s_added, "sensor+vision"
        out = replace(visit, person_id=owner if owner is not None else -1, removed=removed, added=added,
                      source=source, zone=self.config.shelf_zone, slot_change=slot_change,
                      start_frame=self.overview_axis.to_frame(start_ms), end_frame=self.overview_axis.to_frame(end_ms),
                      detail=visit.detail + extra)
        self.records.append(FusedRecord("visit", start_ms, end_ms, out.person_id, how, out.detail))
        return out

    def sensor_only(self, now_ms: float) -> list[ShelfVisit]:
        """카메라 방문과 묶이지 않은 채 시간이 지난 센서 기록 → 센서만의 방문."""
        out = []
        tol = self.config.event_tolerance_seconds * 1000.0
        for i, s in enumerate(self.sensors):
            if i in self._sensor_used or s.time_ms > now_ms - 3 * tol:
                continue
            self._sensor_used.add(i)
            shelf = self.slot_shelf.get(s.slot, s.slot)
            owner, how = self._lane_owner([s], s.time_ms - tol, s.time_ms + tol, shelf)
            if owner is None:
                owners = self._touch_owners(s.time_ms - tol, s.time_ms + tol, shelf)
                owner, how = (owners.pop(), "손-칸 접촉") if len(owners) == 1 else (self._band_owner(s.time_ms - tol, s.time_ms + tol), "선반 앞 한 명")
            removed, added = self._sensor_counts([s])
            visit = ShelfVisit(person_id=owner if owner is not None else -1, shelf=self.slot_shelf.get(s.slot, s.slot),
                               start_frame=self.overview_axis.to_frame(s.time_ms - tol),
                               end_frame=self.overview_axis.to_frame(s.time_ms + tol),
                               removed=removed, added=added, zone=self.config.shelf_zone,
                               detail=f"센서 {s.slot} {self._delta_text(s)}", source="sensor")
            self.records.append(FusedRecord("sensor", s.time_ms, s.time_ms, visit.person_id,
                                            how if owner is not None else "모름", visit.detail))
            out.append(visit)
        return out

    # ------------------------------------------------------------------ 내부

    def _touch_owners(self, start: float, end: float, shelf: str, slots: set[str] | None = None) -> set[int]:
        """그 시간 그 선반에 손이 닿은 전체 카메라 손님들. slots 를 주면 그 칸(레인)에 닿은 손님만."""
        tol = self.config.event_tolerance_seconds * 1000.0
        owners = set()
        for s, e, pid, slot in self._touches:
            if self.slot_shelf.get(slot, shelf) != shelf:
                continue
            if slots is not None and slot not in slots:
                continue
            if e >= start - tol and s <= end + tol:
                o = self.linker.owner(pid, s, e)
                if o is not None:
                    owners.add(o)
        return owners

    def _lane_owner(self, sensors: list[SensorEvent], start: float, end: float, shelf: str) -> tuple[int | None, str]:
        """센서가 바뀐 칸(레인)의 주인을 카메라로 가린다. (주인, 이유) — 못 정하면 (None, "").

        순서 (센서 시각 근처, 위쪽 카메라는 sensor_lookback/lookahead 구간):
          1. 위쪽 카메라: 그 레인 안에 손이 들어간 사람이 한 명 (overview_lane_match, floor.yaml lanes_image)
          2. 선반 카메라: 그 레인에 손-칸 접촉한 사람이 한 명 (sensor_lane_match, 칸 이름 = 센서 칸 이름)
          3. 위쪽 카메라: 손은 안 보이지만 그 시각 받침을 몸으로 덮은 사람이 한 명
        """
        if not sensors:
            return None, ""
        lanes = {s.slot for s in sensors}
        names = ",".join(sorted(lanes))
        w0 = min(s.time_ms for s in sensors) - self.config.sensor_lookback_seconds * 1000.0
        w1 = max(s.time_ms for s in sensors) + self.config.sensor_lookahead_seconds * 1000.0
        detail = self._touch_owners(start, end, shelf, slots=lanes) if self.config.sensor_lane_match else set()
        hands = self._overview_lane_owners(lanes, w0, w1) if self.config.overview_lane_match else set()
        if len(hands) == 1:
            owner = next(iter(hands))
            note = f" (선반 카메라 손 접촉 {sorted(detail)})" if detail and detail != {owner} else ""
            return owner, f"위쪽 카메라 센서 레인 {names} 손{note}"
        if len(detail) == 1 and (not hands or detail <= hands):
            return next(iter(detail)), f"센서 레인 {names} 손 접촉"
        if self.config.overview_lane_match:
            cover = self._cover_owners(w0, w1)
            if len(cover) == 1:
                return next(iter(cover)), f"센서 레인 {names} 시각 받침을 가린 사람 (위쪽 카메라)"
        return None, ""

    def _overview_lane_owners(self, lanes: set[str], start: float, end: float) -> set[int]:
        """위쪽 카메라에서 그 시간 가장 오래 연속으로 손이 머문 레인이 센서 레인이고, 그 연속이
        overview_lane_min_frames 장면 이상인 손님들."""
        gap = self.config.overview_lane_gap_seconds * 1000.0
        times: dict[tuple[int, str], list[float]] = {}
        for ms, pid, lane in self._ov_lane:
            if start <= ms <= end:
                times.setdefault((pid, lane), []).append(ms)
        best: dict[int, tuple[int, str]] = {}
        for (pid, lane), ts in times.items():
            ts = sorted(set(ts))
            run = longest = 1
            for a, b in zip(ts, ts[1:]):
                run = run + 1 if b - a <= gap else 1
                longest = max(longest, run)
            if longest > best.get(pid, (0, ""))[0]:
                best[pid] = (longest, lane)
        return {pid for pid, (n, lane) in best.items() if lane in lanes and n >= self.config.overview_lane_min_frames}

    def _cover_owners(self, start: float, end: float) -> set[int]:
        """그 시간 받침을 tray_cover_ratio 이상 덮은 장면이 tray_cover_min_frames 이상인 손님들."""
        n = Counter(pid for ms, pid, r in self._cover if start <= ms <= end and r >= self.config.tray_cover_ratio)
        return {pid for pid, k in n.items() if k >= self.config.tray_cover_min_frames}

    def _reach_owners(self, start: float, end: float) -> set[int]:
        tol = self.config.event_tolerance_seconds * 1000.0
        return {o for s, e, o in self._reaches if e >= start - tol and s <= end + tol}

    def present(self, start: float, end: float) -> set[int]:
        """그 시간 바닥의 선반 앞 띠에 서 있던 전체 카메라 손님들."""
        return {pid for ms, pid in self._band if start <= ms <= end}

    def _band_owner(self, start: float, end: float) -> int | None:
        people = {pid for ms, pid in self._band if start <= ms <= end}
        return people.pop() if len(people) == 1 else None

    def _sensor_between(self, start: float, end: float, shelf: str) -> list[SensorEvent]:
        tol = self.config.event_tolerance_seconds * 1000.0
        hits = []
        for i, s in enumerate(self.sensors):
            if i in self._sensor_used or self.slot_shelf.get(s.slot, s.slot) != shelf:
                continue
            if start - tol <= s.time_ms <= end + tol:
                hits.append(s)
                self._sensor_used.add(i)
        return hits

    def _delta_text(self, s: SensorEvent) -> str:
        """칸 무게(sensor_unit_g)가 있으면 g(무게 변화), 없으면 개수 변화 — 3차 실제 레인 센서는 개수를 delta_g 칸에 넣는다."""
        if s.unit != "count" and self.config.sensor_unit_g.get(s.slot):
            return f"{s.delta_g:+.0f}g"
        return f"{s.delta_g:+.0f}개"

    def _sensor_counts(self, events: list[SensorEvent]) -> tuple[int, int]:
        removed = added = 0
        for s in events:
            if s.unit == "count":
                n = max(1, int(round(abs(s.delta_g))))      # 레인 센서: 한 번에 2개 꺼냄(REMOVE 2) = 2개
            else:
                unit = float(self.config.sensor_unit_g.get(s.slot, 0) or 0)
                n = max(1, int(round(abs(s.delta_g) / unit))) if unit > 0 else 1
            if s.delta_g < 0:
                removed += n
            elif s.delta_g > 0:
                added += n
        return removed, added
