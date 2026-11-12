"""카메라 2대의 관찰을 합쳐 손님 장부·위험 판정(⑤⑥)에 넘긴다 — 실시간 파이프라인과 기록 재실행이 같이 쓴다.

    선반 카메라 장면  → detail_step   : 사람 위치(띠 축)·외형을 교차 화면 연결에, 손 접촉·선반 방문은 대기열에
    전체 카메라 장면  → overview_step : 바닥 위치를 연결에, 짝을 갱신하고, 대기열에서 주인이 정해진 사건을 꺼내
                                        전체 카메라 손님의 기록으로 위험 판정에 넣는다

판단(⑤⑥)은 전체 카메라의 장면 시계로 돈다. 선반 카메라의 사건은 짝이 그 시간을 덮을 때까지(조각 2개) 기다렸다 넘긴다.
영상·모델 라이브러리를 모른다 — 저장된 기록만으로 다시 돌릴 수 있다 (scripts/dual_eval.py judge).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import numpy as np

from ..core.config import FusionConfig, InteractionConfig
from ..core.types import Frame, IdentityObservation, RiskEvent, TakeCandidate
from ..interaction.detector import InteractionDetector
from ..risk.customers import ShelfVisit
from ..zones.floor import FloorMap
from ..zones.zone_map import ZoneMap
from .cross_view import CrossViewLinker
from .shelf_events import SensorEvent, ShelfEventFuser, TimeAxis


class DualJudge:
    def __init__(
        self,
        config: FusionConfig,
        risk,
        floor: FloorMap,
        band_map: tuple[float, float],
        detail_axis: TimeAxis,
        overview_axis: TimeAxis,
        detail_width: int,
        overview_size: tuple[int, int],
        slot_shelf: dict[str, str] | None = None,
        sensors: list[SensorEvent] | None = None,
        shelf_image: ZoneMap | None = None,
        detail_point=None,
    ) -> None:
        self.config = config
        self.risk = risk
        self.floor = floor
        self.band_map = band_map
        """선반 카메라 사람 상자 가운데 x(화면 비율) → 띠 위치: band = a·x + b. 정상 영상으로 맞춘다."""
        self.detail_width = detail_width
        self.detail_point = detail_point
        """선반 카메라 사람 상자 → band_map 입력 x (화면 비율). 없으면 상자 가운데 x ÷ 화면 폭 (2차). 3차는 카메라 움직임을 되돌린 좌표."""
        self.overview_size = overview_size
        self.linker = CrossViewLinker(config)
        self.fuser = ShelfEventFuser(config, self.linker, detail_axis, overview_axis, slot_shelf, sensors)
        self.shelf_image = shelf_image
        self.reach = None
        """전체 카메라 팔 뻗기 (손목이 화면의 선반 영역 안). 손목 기록만 있으면 되므로 기록 재실행에서도 같이 돈다."""
        if config.overview_reach_seconds > 0 and shelf_image:
            shelf_image.resolve(*overview_size)
            self.reach = InteractionDetector(
                InteractionConfig(signal="hand", dwell_seconds=config.overview_reach_seconds,
                                  hand_conf=config.overview_reach_conf), shelf_image)
        self.reset()

    def reset(self) -> None:
        self.linker.reset()
        self.fuser.reset()
        self.risk.reset()
        if self.reach is not None:
            self.reach.reset()
        self._pending_takes: list[tuple[float, TakeCandidate]] = []
        self._pending_visits: list[tuple[float, ShelfVisit, float, float]] = []
        self.events: list[RiskEvent] = []
        self.unowned_takes = 0

    # ------------------------------------------------------------------

    def detail_band(self, bbox) -> float:
        a, b = self.band_map
        if self.detail_point is not None:
            return a * self.detail_point(bbox, self.detail_width) + b
        return a * ((bbox[0] + bbox[2]) / 2.0 / self.detail_width) + b

    def detail_step(
        self,
        pts_ms: float,
        identities: list[IdentityObservation],
        takes: list[TakeCandidate],
        visits: list[tuple[ShelfVisit, float, float]],
        appearance: Callable[[int], np.ndarray | None] | None = None,
        resolve: Callable[[int], int] | None = None,
        slot_changes: list | None = None,
    ) -> None:
        for sc in slot_changes or []:
            self.fuser.note_slot_change(sc)
        for obs in identities:
            if obs.person_id <= 0:
                continue
            pid = resolve(obs.person_id) if resolve else obs.person_id
            self.linker.observe("detail", pid, pts_ms, self.detail_band(obs.bbox),
                                appearance(pid) if appearance else None)
        for t in takes:
            pid = resolve(t.person_id) if resolve and t.person_id > 0 else t.person_id
            self._pending_takes.append((self.fuser.detail_axis.to_ms(t.end_frame), replace(t, person_id=pid)))
        for visit, start_ms, end_ms in visits:
            self._pending_visits.append((end_ms, visit, start_ms, end_ms))

    def overview_step(
        self,
        frame: Frame,
        identities: list[IdentityObservation],
        takes: list[TakeCandidate],
        merges: list[dict],
        appearance: Callable[[int], np.ndarray | None] | None = None,
        resolve: Callable[[int], int] | None = None,
    ):
        w, h = self.overview_size
        for obs in identities:
            if obs.person_id <= 0:
                continue
            pid = resolve(obs.person_id) if resolve else obs.person_id
            point = self.floor.foot_on_floor(obs.bbox, w, h)
            band = self.floor.band_position(point)   # 띠는 band_map 을 맞춘 상자 기준 그대로
            foot = self.floor.foot_on_floor(obs.bbox, w, h, obs.keypoints) if self.floor.foot_mode != "bbox" else point
            in_band = any(z.name == self.config.shelf_zone for z in self.floor.zones_at(foot))
            if band is not None and not (-0.3 <= band <= 1.3):
                band = None
            self.linker.observe("overview", pid, frame.pts_ms, band, appearance(pid) if appearance else None)
            if in_band:
                self.fuser.note_band(frame.pts_ms, pid)
        self.linker.update(frame.pts_ms)
        reach_takes = (self._reach_takes(self.reach.update(frame, self._standing(identities)), resolve)
                       if self.reach else [])
        # 손 접촉도 방문과 같이 기다렸다 넘긴다 — 그 시간을 덮는 짝 기록이 쌓인 뒤에 주인을 정해야
        # 손 접촉과 방문이 같은 손님으로 해석된다 (2차 C1: 일찍 정하면 손 접촉과 방문의 주인이 갈렸다)
        delay = self.config.release_delay_seconds * 1000.0
        fused_takes, fused_visits = self._release(frame.pts_ms - delay, frame.pts_ms - self.VISIT_DELAY_MS)
        fused_visits += self.fuser.sensor_only(frame.pts_ms)
        if not self.config.overview_reach_takes:
            reach_takes = []   # 주인 정하기 근거로만 (note_reach 는 위에서 이미 적음)
        events, records = self.risk.update(frame, identities, list(takes) + reach_takes + fused_takes, merges,
                                           fused_visits)
        self.events.extend(events)
        return events, records

    def finish(self, frame: Frame, resolve: Callable[[int], int] | None = None) -> list[RiskEvent]:
        reach_takes = self._reach_takes(self.reach.flush(), resolve) if self.reach else []
        fused_takes, fused_visits = self._release(float("inf"), float("inf"))
        if self.config.overview_reach_takes:
            fused_takes = reach_takes + fused_takes
        fused_visits += self.fuser.sensor_only(float("inf"))
        if fused_takes or fused_visits:
            events, _ = self.risk.update(frame, [], fused_takes, [], fused_visits)
            self.events.extend(events)
        done = self.risk.finish(frame, self.overview_size)
        self.events.extend(done)
        return done

    def _standing(self, identities: list[IdentityObservation]) -> list[IdentityObservation]:
        """발(상자 아래 가운데)이 화면의 선반 영역 밖인 사람만 — 선반 위 상품 포장의 사람 그림에 관절이 잡힌
        가짜 사람을 뺀다 (2차 N8-2 18.5초: 사람이 없는데 선반 안에 손목 둘)."""
        from ..zones.geometry import contains_point

        polys = [self.shelf_image.polygon_of(z.name) for z in self.shelf_image.zones]
        out = []
        for o in identities:
            fx, fy = (o.bbox[0] + o.bbox[2]) / 2.0, o.bbox[3]
            if not any(p and contains_point(p, fx, fy) for p in polys):
                out.append(o)
        return out

    def _reach_takes(self, reaches: list[TakeCandidate], resolve) -> list[TakeCandidate]:
        """전체 카메라 팔 뻗기 → 주인 정하기 근거로 적는다. overview_reach_takes 면 그 손님의 손 뻗기로도 넘긴다."""
        out = []
        axis = self.fuser.overview_axis
        for t in reaches:
            pid = resolve(t.person_id) if resolve else t.person_id
            self.fuser.note_reach(axis.to_ms(t.start_frame), axis.to_ms(t.end_frame), pid)
            out.append(replace(t, person_id=pid, zone=self.config.shelf_zone))
        return out

    VISIT_DELAY_MS = 1500.0
    """선반 방문은 끝나고 이만큼 기다렸다 넘긴다 — 칸 그림 비교 결과(같은 무렵 확정)를 함께 붙이려고."""

    def _release(self, until_ms: float, visits_until_ms: float | None = None) -> tuple[list[TakeCandidate], list[ShelfVisit]]:
        visits_until_ms = until_ms if visits_until_ms is None else visits_until_ms
        takes, keep = [], []
        for end_ms, take in self._pending_takes:
            if end_ms <= until_ms:
                fused = self.fuser.detail_take(take)
                if fused is None:
                    self.unowned_takes += 1
                else:
                    takes.append(fused)
            else:
                keep.append((end_ms, take))
        self._pending_takes = keep
        visits, keepv = [], []
        for item in self._pending_visits:
            end_ms, visit, start_ms, stop_ms = item
            if end_ms <= visits_until_ms:
                fused = self.fuser.detail_visit(visit, start_ms, stop_ms)
                visits.append(fused)
                if self.config.unchanged_clears_all and fused.removed == 0 and fused.added == 0                         and fused.slot_change is None:
                    # 물건 탐지도 칸 그림도 '그대로'인 방문: 그 시간 선반 앞에 있던 다른 손님의 손 뻗기도
                    # 아무것도 안 가져간 것이다 (설계 M-7 — 둘 다 그대로일 때만 집기를 지운다)
                    for other in self.fuser.present(start_ms, stop_ms) - {fused.person_id}:
                        visits.append(replace(fused, person_id=other, detail=fused.detail + " (함께 있던 손님)"))
            else:
                keepv.append(item)
        self._pending_visits = keepv
        return takes, visits
