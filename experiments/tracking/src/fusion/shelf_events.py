"""⑦-b 선반 사건 묶기 — 선반 카메라가 본 변화·손 접촉과 센서 기록을 전체 카메라의 손님에게 붙인다.

설계 M-7·M-8 (docs/multiview-design.md):
  주인 정하는 순서
    1. 선반 카메라가 '가린 사람 1명'으로 정한 변화 → 그 사람의 짝(⑦-a)
    2. 그 시간 그 선반 칸에 손목이 닿은 사람(손-칸 접촉) → 그 사람의 짝
    3. 그 시간 바닥의 선반 앞 띠에 서 있던 전체 카메라 손님이 한 명뿐이면 그 사람
    4. 여러 명이면 '누구 것인지 모름' (-1) — 지금과 같다
  센서 기록은 같은 시간(±event_tolerance)의 카메라 변화와 묶는다. 개수는 센서를 믿는다.
  fusion.sensor_kinds (센서 레인 → 상품 종류) 를 주면 **종류별로 합친다**: 센서 레인의 종류는 센서 개수만, 나머지 종류는 카메라 개수.
  (예전에는 방문에 센서 기록이 하나라도 있으면 개수 전부를 센서 것으로 바꿔 센서 없는 종류가 빠졌고, 센서 기록이 방문보다 늦게 오면
  카메라와 센서가 같은 물건을 따로 세었다 — D-24.)

카메라마다 장면 번호가 다르므로(30/60FPS) 손님 장부로 넘기기 전에 모두 전체 카메라의 장면 번호로 바꾼다.
"""

from __future__ import annotations

import csv
import re
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


_KIND = re.compile(r"\d+단 \d+번째 (.+)")


def visit_kinds(visit: ShelfVisit) -> tuple[Counter, Counter]:
    """선반 지도가 그 방문에서 본 (꺼낸 종류별 개수, 놓은 종류별 개수). 기록에 종류가 없으면(옛 기록) 설명 글
    '[선반] 사라짐: 1단 2번째 donggwaru, 1단 4번째 cantata / 생김: … -> 손님 1' 에서 읽는다."""
    if visit.removed_kinds or visit.added_kinds:
        return Counter(visit.removed_kinds), Counter(visit.added_kinds)
    removed, added = Counter(), Counter()
    body = visit.detail.partition("] ")[2] or visit.detail
    for part in body.partition(" -> ")[0].split(" / "):
        for word, into in (("사라짐: ", removed), ("생김: ", added)):
            if part.startswith(word):
                for item in part[len(word):].split(", "):
                    m = _KIND.fullmatch(item.strip())
                    if m:
                        into[m.group(1)] += 1
    return removed, added


@dataclass
class FusedRecord:
    """shelf_events.jsonl 한 줄 — 어떻게 주인을 정했는지 남긴다."""

    kind: str
    """visit (선반 변화) / take (손 접촉) / sensor."""
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
                 sensors: list[SensorEvent] | None = None, overview=None, holders=None) -> None:
        self.config = config
        self.overview = overview
        """전체 카메라 상품 개수 (OverviewCounter, fusion.overview_count). 있으면 방문의 개수를 그쪽 것으로 바꾼다."""
        self.holders = holders
        """전체 카메라에서 상품을 든 사람 (HolderTracker, fusion.overview_owner). 있으면 주인을 못 정한 방문의 주인을 여기서 묻는다."""
        self.hold_wait_ms = 0.0
        """방문이 끝난 뒤 이만큼까지 든 사람을 본다 — 판정이 방문을 넘기기 전에 기다리는 시간 (그때까지는 반드시 본 장면)."""
        self.linker = linker
        self.detail_axis = detail_axis
        self.overview_axis = overview_axis
        self.slot_shelf = slot_shelf or {}
        """칸 이름 → 그 칸이 속한 선반 이름 (선반 카메라 구역 파일)."""
        self.sensors = sorted(sensors or [], key=lambda s: s.time_ms)
        self.reset()

    def reset(self) -> None:
        self.shares: list[ShelfVisit] = []
        self._touches: list[tuple[float, float, int, str]] = []
        """선반 카메라의 손-칸 접촉: (시작 ms, 끝 ms, 선반 카메라 신원, 칸)."""
        self._band: list[tuple[float, int]] = []
        """전체 카메라에서 선반 앞 띠에 서 있던 기록: (ms, 손님)."""
        self._sensor_used: set[int] = set()
        self._slot_changes: list = []
        """선반 카메라의 칸 그림 비교 결과 (SlotChange, 공통 시각)."""
        self._reaches: list[tuple[float, float, int]] = []
        """전체 카메라 팔 뻗기: (시작 ms, 끝 ms, 전체 카메라 손님)."""
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

    def swap_overview(self, a: int, b: int, since_ms: float) -> None:
        """since_ms 부터 적어 둔 '선반 앞에 있던 사람'·'팔을 뻗은 사람'에서 전체 카메라 신원 a·b 를 서로 바꾼다 (fusion.swap_repair)."""
        flip = {a: b, b: a}
        self._band = [(ms, flip.get(pid, pid) if ms >= since_ms else pid) for ms, pid in self._band]
        self._reaches = [(s, e, flip.get(pid, pid) if s >= since_ms else pid) for s, e, pid in self._reaches]

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
        reach = self._reach_owners(start_ms, end_ms)
        if len(reach) == 1:
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
                else:
                    how = f"손 접촉 {len(owners)}명 — 모름"
        if owner is None and not how:
            owner = self._band_owner(start_ms, end_ms)
            how = "선반 앞 한 명" if owner is not None else "선반 앞 여러 명 또는 없음 — 모름"
        removed, added, source, extra = visit.removed, visit.added, visit.source, ""
        detail = visit.detail
        slot_change, slot_text = visit.slot_change, ""
        if slot_change is None and self.config.slot_evidence:
            slot_change, slot_text = self._slot_change_for(start_ms, end_ms, visit.shelf)
            if slot_text:
                extra += f" / 칸 그림: {slot_text}"
        count = self.overview.change(start_ms, end_ms) if self.overview is not None else None
        use = count if self.config.count_source == "overview" else None
        """개수로 쓸 전체 카메라 개수 — fusion.count_source 가 shelf 면 안 쓰고 적어만 둔다 (선반 지도 개수를 쓴다)."""
        cam_removed, cam_added = ((Counter(use.removed_by), Counter(use.added_by)) if use is not None else visit_kinds(visit))
        covered = set(self.config.sensor_kinds.values())
        by_kind: dict[str, int] = {}
        """꺼낸 것의 종류별 개수 — 주인을 못 정했을 때 '그 종류를 먼저 든 사람'을 묻는 데 쓴다."""
        if covered:
            # 종류별로 합친다: 센서 레인의 종류는 센서 개수만, 나머지 종류는 카메라(count_source 가 정한 쪽) 개수
            sensor = self._sensor_between(start_ms, end_ms, visit.shelf, cam_removed, cam_added)
            s_removed, s_added = self._sensor_counts(sensor)
            by_sensor = {-1: Counter(), 1: Counter()}
            for s in sensor:
                by_sensor[-1 if s.delta_g < 0 else 1][self.config.sensor_kinds.get(s.slot)] += sum(self._sensor_counts([s]))
            # 센서 레인의 종류: 센서가 센 만큼은 센서 개수. 카메라가 더 많이 봤으면 sensor_kind_rule 이 정한다 (max = 남는 만큼 카메라 개수로 더 셈)
            more = self.config.sensor_kind_rule == "max"
            rest_removed = {k: n - (by_sensor[-1][k] if k in covered else 0) for k, n in cam_removed.items()
                            if k not in covered or (more and n > by_sensor[-1][k])}
            rest_added = {k: n - (by_sensor[1][k] if k in covered else 0) for k, n in cam_added.items()
                          if k not in covered or (more and n > by_sensor[1][k])}
            removed, added = s_removed + sum(rest_removed.values()), s_added + sum(rest_added.values())
            kinds = Counter(rest_removed)
            for s in sensor:
                if s.delta_g < 0:
                    kinds[self.config.sensor_kinds.get(s.slot, s.slot)] += self._sensor_counts([s])[0]
            by_kind = dict(kinds)
            parts = [f"센서 {s.slot} {self._delta_text(s)}" for s in sensor]
            parts += [f"카메라 {k} -{n}" for k, n in sorted(rest_removed.items())]
            parts += [f"카메라 {k} +{n}" for k, n in sorted(rest_added.items())]
            if sensor:
                source = "sensor+vision"
            elif use is not None:
                source = "overview+vision"
            extra += " / 종류별 합침: " + (", ".join(parts) or "변화 없음")
            left = [f"{k} {'-' if sign < 0 else '+'}{n - by_sensor[sign][k]}" for sign, cam in ((-1, cam_removed), (1, cam_added))
                    for k, n in sorted(cam.items()) if k in covered and n > by_sensor[sign][k]]
            if left:
                extra += f" — 센서 레인인데 카메라만 본 것 {', '.join(left)}" + (" (카메라 개수로 셈)" if more else " (안 셈)")
            if (removed, added) != (visit.removed, visit.added):
                extra += f" (선반 지도 -{visit.removed}+{visit.added})"
        else:
            sensor = self._sensor_between(start_ms, end_ms, visit.shelf)
            if sensor:
                s_removed, s_added = self._sensor_counts(sensor)
                if (s_removed, s_added) != (removed, added):
                    extra += f" / 센서와 다름 (카메라 -{removed}+{added}, 센서 -{s_removed}+{s_added})"
                removed, added, source = s_removed, s_added, "sensor+vision"
        if count is not None:
            # 전체 카메라가 방문 전 · 뒤의 상품 개수를 봤다 — 센서가 없는 방문이면 개수를 그쪽 것으로 (센서가 있으면 적어만 둔다)
            same = (count.removed, count.added) == (visit.removed, visit.added)
            extra += f" / 전체 카메라 개수 {count.text()} (-{count.removed}+{count.added}{'' if same else ', 선반 지도와 다름'})"
            if use is not None and not covered and not sensor:
                removed, added, source = count.removed, count.added, "overview+vision"
            if use is not None and not sensor and not same:
                detail = self._count_detail(detail, count)
        elif self.overview is not None and self.overview.fed:
            extra += " / 전체 카메라 개수 못 봄"
        # 카메라가 종류를 본 방문인가 — 전체 카메라 개수를 쓰는 중이면 그 개수를 본 방문만 (못 봤으면 예전처럼 안 묻는다)
        seen = bool(cam_removed) and (use is not None or self.config.count_source != "overview")
        if not covered and seen:
            by_kind = dict(cam_removed) if (removed, added) == (sum(cam_removed.values()), sum(cam_added.values())) else {}
        held: list[tuple[str, int | None]] = []
        asked = covered or seen
        if owner is None and self.holders is not None and asked:
            # 주인을 못 정했는데 어느 종류가 줄었는지 안다 — 그 종류를 먼저 든 사람이 꺼낸 사람
            names = by_kind if covered else dict(cam_removed)
            held = self.holders.owners(names, start_ms, end_ms + self.hold_wait_ms) if names else []
            who = {p for _, p in held}
            if held and None not in who and len(who) == 1:
                owner, how = who.pop(), f"전체 카메라에서 먼저 든 사람 ({how})"
                held = []
            elif who == {None} or not by_kind:
                # 아무도 든 것을 못 봤다 / 센서 개수와 달라 종류별로 나눌 수 없다 — 모름 그대로
                held = []
        out = replace(visit, person_id=owner if owner is not None else -1, removed=removed, added=added,
                      source=source, zone=self.config.shelf_zone, slot_change=slot_change,
                      start_frame=self.overview_axis.to_frame(start_ms), end_frame=self.overview_axis.to_frame(end_ms),
                      detail=detail + extra)
        if held:
            out, how = self._share(out, held, how, start_ms, end_ms)
        elif owner is not None and self.holders is not None and (by_kind if covered else seen):
            out = replace(out, held_ms=self._held_ms(sorted(by_kind if covered else cam_removed), owner, start_ms, end_ms))
        self.records.append(FusedRecord("visit", start_ms, end_ms, out.person_id, how,
                                        out.detail if not held else detail + extra + " / 든 사람: " + self._held_text(held)))
        return out

    @staticmethod
    def _held_text(held: list[tuple[str, int | None]]) -> str:
        return ", ".join(f"{name} → {'손님 ' + str(p) if p is not None else '모름'}" for name, p in held)

    def _held_ms(self, names: list[str], owner: int, start_ms: float, end_ms: float) -> float | None:
        """그 사람이 이 종류들 가운데 하나라도 든 것으로 처음 본 시각 (못 봤으면 None)."""
        times = [t for name in names
                 for pid, t in self.holders.first_held(name, start_ms, end_ms + self.hold_wait_ms).items() if pid == owner]
        return min(times) if times else None

    def _share(self, out: ShelfVisit, held: list[tuple[str, int | None]], how: str,
               start_ms: float, end_ms: float) -> tuple[ShelfVisit, str]:
        """종류마다 든 사람이 다른 방문 → 사람별 방문으로 나눈다. 첫 사람의 방문을 돌려주고 나머지는 shares 에 쌓는다
        (든 사람을 못 본 상품은 '주인 모름' 방문으로 남는다). 놓은 개수는 첫 사람의 방문에 둔다."""
        head, sep, _ = out.detail.partition("] ")
        tail = out.detail.partition(" / ")[2]
        by: dict[int | None, list[str]] = {}
        for name, p in held:
            by.setdefault(p, []).append(name)
        people = sorted(by, key=lambda p: (p is None, [h[1] for h in held].index(p)))
        parts = []
        for i, p in enumerate(people):
            names = by[p]
            text = (f"{head}{sep}" if sep else "") + f"꺼냄: {', '.join(names)} (전체 카메라에서 먼저 든 사람)"
            text += (f" -> 손님 {p}" if p is not None else "") + (f" / {tail}" if tail else "")
            parts.append(replace(out, person_id=p if p is not None else -1, removed=len(names),
                                 added=out.added if i == 0 else 0, detail=text,
                                 held_ms=self._held_ms(names, p, start_ms, end_ms) if p is not None else None))
        self.shares.extend(parts[1:])
        return parts[0], f"전체 카메라에서 먼저 든 사람 — 사람별로 나눔 ({how})"

    def take_shares(self) -> list[ShelfVisit]:
        """방금 나눈 방문의 나머지 사람 몫 (detail_visit 바로 뒤에 꺼내 간다)."""
        out, self.shares = self.shares, []
        return out

    @staticmethod
    def _count_detail(detail: str, count) -> str:
        """개수를 전체 카메라 것으로 바꾼 방문의 설명: '[선반] 사라짐: … -> 손님 1' → '[선반] 전체 카메라 개수 … — 선반 지도는 사라짐: … -> 손님 1'.
        행동 설명(손님 장부)은 ' -> ' 앞을 쓴다 — 개수와 설명이 어긋나 보이지 않게."""
        head, sep, what = detail.partition("] ")
        if not sep:
            head, what = "", detail
        what, arrow, who = what.partition(" -> ")
        return f"{head}{sep}전체 카메라 개수 {count.text()} — 선반 지도는 {what or '변화 없음'}{arrow}{who}"

    def sensor_only(self, now_ms: float, hold: list[tuple[float, float, str]] | None = None) -> list[ShelfVisit]:
        """카메라 방문과 묶이지 않은 채 시간이 지난 센서 기록 → 센서만의 방문.
        hold = 전체 카메라 개수를 기다리느라 아직 못 넘긴 방문들 (시작 ms, 끝 ms, 선반) — 그 방문이 묶어 갈 센서 기록은 남겨 둔다."""
        out = []
        tol = self.config.event_tolerance_seconds * 1000.0
        for i, s in enumerate(self.sensors):
            if i in self._sensor_used or s.time_ms > now_ms - 3 * tol:
                continue
            if any(start - tol <= s.time_ms <= end + tol and self.slot_shelf.get(s.slot, s.slot) == shelf
                   for start, end, shelf in hold or []):
                continue
            self._sensor_used.add(i)
            early = s.time_ms - tol - self.config.sensor_lag_seconds * 1000.0   # 센서는 꺼낸 뒤 늦게 적는다 — 그만큼 앞까지 본다
            owners = self._touch_owners(early, s.time_ms + tol, self.slot_shelf.get(s.slot, s.slot))
            owner, how = (owners.pop(), "손-칸 접촉") if len(owners) == 1 else (self._band_owner(early, s.time_ms + tol), "선반 앞 한 명")
            removed, added = self._sensor_counts([s])
            visit = ShelfVisit(person_id=owner if owner is not None else -1, shelf=self.slot_shelf.get(s.slot, s.slot),
                               start_frame=self.overview_axis.to_frame(early),
                               end_frame=self.overview_axis.to_frame(s.time_ms + tol),
                               removed=removed, added=added, zone=self.config.shelf_zone,
                               detail=f"센서 {s.slot} {self._delta_text(s)}", source="sensor")
            self.records.append(FusedRecord("sensor", s.time_ms, s.time_ms, visit.person_id,
                                            how if owner is not None else "모름", visit.detail))
            out.append(visit)
        return out

    # ------------------------------------------------------------------ 내부

    def _touch_owners(self, start: float, end: float, shelf: str) -> set[int]:
        tol = self.config.event_tolerance_seconds * 1000.0
        owners = set()
        for s, e, pid, slot in self._touches:
            if self.slot_shelf.get(slot, shelf) != shelf:
                continue
            if e >= start - tol and s <= end + tol:
                o = self.linker.owner(pid, s, e)
                if o is not None:
                    owners.add(o)
        return owners

    def _reach_owners(self, start: float, end: float) -> set[int]:
        tol = self.config.event_tolerance_seconds * 1000.0
        return {o for s, e, o in self._reaches if e >= start - tol and s <= end + tol}

    def present(self, start: float, end: float) -> set[int]:
        """그 시간 바닥의 선반 앞 띠에 서 있던 전체 카메라 손님들."""
        return {pid for ms, pid in self._band if start <= ms <= end}

    def _band_owner(self, start: float, end: float) -> int | None:
        people = {pid for ms, pid in self._band if start <= ms <= end}
        return people.pop() if len(people) == 1 else None

    def _sensor_between(self, start: float, end: float, shelf: str, cam_removed: Counter | None = None,
                        cam_added: Counter | None = None) -> list[SensorEvent]:
        """그 방문에 묶을 센서 기록. 방문 시간(±허용 시간) 안의 것 전부 + (cam_* 를 줬을 때) 방문이 끝나고 sensor_lag_seconds 안에
        온 것 가운데 **카메라가 그 방문에서 그 종류가 바뀐 것을 봤고 아직 센서 기록으로 설명 안 된** 것."""
        tol = self.config.event_tolerance_seconds * 1000.0
        hits = []
        for i, s in enumerate(self.sensors):
            if i in self._sensor_used or self.slot_shelf.get(s.slot, s.slot) != shelf:
                continue
            if start - tol <= s.time_ms <= end + tol:
                hits.append(s)
                self._sensor_used.add(i)
        lag = self.config.sensor_lag_seconds * 1000.0
        if lag <= 0 or cam_removed is None:
            return hits
        need = {-1: Counter(cam_removed), 1: Counter(cam_added or {})}
        for s in hits:
            sign = -1 if s.delta_g < 0 else 1
            need[sign][self.config.sensor_kinds.get(s.slot)] -= sum(self._sensor_counts([s]))
        for i, s in enumerate(self.sensors):
            if i in self._sensor_used or self.slot_shelf.get(s.slot, s.slot) != shelf or s.delta_g == 0:
                continue
            kind, sign = self.config.sensor_kinds.get(s.slot), (-1 if s.delta_g < 0 else 1)
            if end + tol < s.time_ms <= end + tol + lag and kind is not None and need[sign][kind] > 0:
                hits.append(s)
                self._sensor_used.add(i)
                need[sign][kind] -= sum(self._sensor_counts([s]))
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
