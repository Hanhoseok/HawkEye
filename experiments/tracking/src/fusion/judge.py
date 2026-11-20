"""카메라 2대의 관찰을 합쳐 손님 장부·위험 판정(⑤⑥)에 넘긴다 — 실시간 파이프라인과 기록 재실행이 같이 쓴다.

    선반 카메라 장면  → detail_step   : 사람 위치(띠 축)·외형을 교차 화면 연결에, 손 접촉·선반 방문은 대기열에
    전체 카메라 장면  → overview_step : 바닥 위치를 연결에, 짝을 갱신하고, 대기열에서 주인이 정해진 사건을 꺼내
                                        전체 카메라 손님의 기록으로 위험 판정에 넣는다

판단(⑤⑥)은 전체 카메라의 장면 시계로 돈다. 선반 카메라의 사건은 짝이 그 시간을 덮을 때까지(조각 2개) 기다렸다 넘긴다.
영상·모델 라이브러리를 모른다 — 저장된 기록만으로 다시 돌릴 수 있다 (scripts/dual_eval.py judge).

매장 신원은 전체 카메라의 화면 신원 그대로다. 단 전체 카메라가 두 사람의 번호를 뒤바꿔 붙인 것을 선반 카메라로 알아챘을 때
(fusion.swap_repair)는 그때부터 두 번호를 서로 바꿔 읽는다 — 짝·손님 장부·판정에는 바꿔 읽은 번호(매장 신원)가 들어간다.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from typing import Callable

import numpy as np

from ..core.config import FusionConfig, InteractionConfig
from ..core.types import Frame, IdentityObservation, RiskEvent, TakeCandidate
from ..interaction.detector import InteractionDetector
from ..risk.customers import ShelfVisit
from ..zones.floor import FloorMap, foot_pixel
from ..zones.zone_map import ZoneMap
from .cross_view import CrossViewLinker
from .overview_count import HolderTracker, OverviewCounter
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
        detail_floor: FloorMap | None = None,
        detail_height: int | None = None,
    ) -> None:
        self.config = config
        self.risk = risk
        self.floor = floor
        self.band_map = band_map
        """선반 카메라 사람 상자 가운데 x(화면 비율) → 띠 위치: band = a·x + b. 정상 영상으로 맞춘다."""
        self.detail_width = detail_width
        self.detail_point = detail_point
        """선반 카메라 사람 상자 → band_map 입력 x (화면 비율). 없으면 상자 가운데 x ÷ 화면 폭 (2차). 3차는 카메라 움직임을 되돌린 좌표."""
        self.detail_floor = detail_floor
        """선반 카메라 바닥 지도 (전체 카메라와 같은 바닥 좌표). 있으면 선반 카메라 사람의 발도 바닥 cm 로 연결에 넘긴다 (3차)."""
        self.detail_height = detail_height or round(detail_width * 9 / 16)
        self.overview_size = overview_size
        self.linker = CrossViewLinker(config)
        self.counter = (OverviewCounter(score=config.overview_count_score, edge_ms=config.event_tolerance_seconds * 1000.0,
                                        late_ms=config.overview_count_wait_seconds * 1000.0)
                        if config.overview_count else None)
        """전체 카메라 상품 개수 (fusion.overview_count). overview_step 에 상품 탐지 결과(products)가 들어와야 센다."""
        self.holders = (HolderTracker(min_frames=config.overview_owner_frames, score=config.overview_count_score)
                        if config.overview_owner and self.counter is not None else None)
        """전체 카메라에서 상품을 든 사람 (fusion.overview_owner). 주인을 못 정한 방문의 주인을 여기서 묻는다."""
        self.fuser = ShelfEventFuser(config, self.linker, detail_axis, overview_axis, slot_shelf, sensors, self.counter,
                                     self.holders)
        self.fuser.hold_wait_ms = self.VISIT_DELAY_MS
        if self.holders is not None and getattr(risk, "book", None) is not None:
            self.holders.resolve = risk.book.resolve   # 든 뒤에 합쳐진 신원은 한 사람으로
        self.shelf_image = shelf_image
        self.reach = None
        """전체 카메라 팔 뻗기 (손목이 화면의 선반 영역 안). 손목 기록만 있으면 되므로 기록 재실행에서도 같이 돈다."""
        if config.overview_reach_seconds > 0 and shelf_image:
            shelf_image.resolve(*overview_size)
            self.reach = InteractionDetector(
                InteractionConfig(signal="hand", dwell_seconds=config.overview_reach_seconds,
                                  hand_conf=config.overview_reach_conf), shelf_image)
        if getattr(risk, "consistency", None) is not None:
            # 신원 확인 값은 화면 신원으로 재 둔 것이다 — 손님 장부가 매장 신원으로 물으면 화면 신원으로 바꿔 묻는다
            check = risk.consistency
            risk.consistency = lambda pid: check(self.screen_id(pid))
        self.reset()

    def reset(self) -> None:
        self.linker.reset()
        self.fuser.reset()
        if self.counter is not None:
            self.counter.reset()
        if self.holders is not None:
            self.holders.reset()
        self.risk.reset()
        if self.reach is not None:
            self.reach.reset()
        self._pending_takes: list[tuple[float, TakeCandidate]] = []
        self._pending_visits: list[tuple[float, ShelfVisit, float, float]] = []
        self._waiting: list[tuple[float, float, str]] = []
        """넘길 때가 됐는데 전체 카메라 개수를 기다리느라 붙들고 있는 방문 (시작 ms, 끝 ms, 선반) — fusion.overview_count 일 때만 생긴다."""
        self.events: list[RiskEvent] = []
        self.unowned_takes = 0
        self._store: dict[int, int] = {}
        """전체 카메라 화면 신원 → 매장 신원. 번호 맞바뀜을 되돌린 신원만 들어 있다 (fusion.swap_repair). 없는 신원은 화면 신원 그대로."""
        self.swaps: list[dict] = []
        """되돌린 번호 맞바뀜: ms(이때부터 바꿔 읽음) · screen(화면 신원 둘) · store(그 둘이 그 전까지 가리키던 매장 신원) · 근거."""
        self._merges: list[dict] = []
        """손님 장부에 넘긴 합침 기록 (매장 신원으로 바꿔 적은 것). 받은 합침 기록과 길이가 같다."""
        self._boxes: dict[str, deque] = {"detail": deque(), "overview": deque()}
        """최근 장면의 사람 상자 (ms, {신원: 상자}) — 번호 맞바뀜 확인에 쓴다."""
        self._closed_seen = 0

    # ------------------------------------------------------------------ 번호 맞바뀜 되돌리기 (fusion.swap_repair)

    def store_id(self, screen_id: int) -> int:
        """전체 카메라 화면 신원(합친 뒤 번호) → 매장 신원."""
        return self._store.get(screen_id, screen_id)

    def screen_id(self, store_id: int) -> int:
        """매장 신원 → 지금 그 사람에게 붙어 있는 전체 카메라 화면 신원."""
        for screen, store in self._store.items():
            if store == store_id:
                return screen
        return store_id

    def final_id(self, person_id: int, resolve: Callable[[int], int] | None = None) -> int:
        """판정에 적힌 번호 → 끝까지 합친 뒤의 매장 신원. resolve = 전체 카메라의 마지막 합침 풀이.
        번호를 바꿔 읽은 적이 있으면 화면 신원의 합침 풀이를 그대로 쓸 수 없어 손님 장부의 합침 기록을 따른다."""
        if self.swaps:
            return self.risk.book.resolve(person_id)
        return resolve(person_id) if resolve else person_id

    def _renamed(self, item, resolve):
        """화면 신원으로 적힌 관찰·집기 후보를 매장 신원으로. 바꿔 읽는 신원이 아니면 그대로 돌려준다."""
        if item.person_id <= 0:
            return item
        store = self._store.get(resolve(item.person_id) if resolve else item.person_id)
        return item if store is None else replace(item, person_id=store)

    def _note_merges(self, merges: list[dict]) -> list[dict]:
        for m in merges[len(self._merges):]:
            if m["from"] in self._store or m["into"] in self._store:
                gone = m["from"]
                m = {**m, "from": self.store_id(m["from"]), "into": self.store_id(m["into"])}
                self._store.pop(gone, None)   # 합쳐져 없어진 화면 신원
            self._merges.append(m)
        return self._merges

    def _note_boxes(self, view: str, pts_ms: float, boxes: dict) -> None:
        if not self.config.swap_repair:
            return
        seen = self._boxes[view]
        seen.append((pts_ms, boxes))
        horizon = pts_ms - self.config.window_seconds * 1000.0 * 4.0
        while seen and seen[0][0] < horizon:
            seen.popleft()

    @staticmethod
    def box_overlap(a, b) -> float:
        """두 상자가 겹친 넓이 ÷ 작은 상자 넓이."""
        w = min(a[2], b[2]) - max(a[0], b[0])
        h = min(a[3], b[3]) - max(a[1], b[1])
        if w <= 0 or h <= 0:
            return 0.0
        small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
        return w * h / small if small > 0 else 0.0

    def _most_overlap(self, view: str, x: int, y: int, start: float, end: float) -> float | None:
        """그 시간 동안 두 사람이 한 장면에 함께 나온 것 중 가장 많이 겹친 정도. 함께 나온 장면이 없으면 None."""
        found = [self.box_overlap(boxes[x], boxes[y]) for ms, boxes in self._boxes[view]
                 if start <= ms <= end and x in boxes and y in boxes]
        return max(found) if found else None

    def _repair_swaps(self, now_ms: float) -> None:
        """방금 닫힌 조각에서 선반 카메라 두 사람의 짝이 서로 맞바뀌었고, 그 무렵 전체 카메라의 두 상자는 겹쳤는데
        선반 카메라의 두 상자는 떨어져 있었으면 — 전체 카메라가 번호를 뒤바꿔 붙인 것이다. 이번 조각 시작부터 두 번호를 바꿔 읽는다."""
        limit = self.config.swap_overlap
        for d1, d2, a, b in self.linker.mutual_swaps():
            (start, _, _), (since, end, _) = list(self.linker.windows)[-2:]
            over = self._most_overlap("overview", a, b, start, end)
            shelf = self._most_overlap("detail", d1, d2, start, end)
            if over is None or shelf is None or over < limit or shelf >= limit:
                continue
            screen_a, screen_b = self.screen_id(a), self.screen_id(b)
            self.linker.swap_overview(a, b, since)
            self.fuser.swap_overview(a, b, since)
            for screen, store in ((screen_a, b), (screen_b, a)):
                if screen == store:
                    self._store.pop(screen, None)
                else:
                    self._store[screen] = store
            self.swaps.append({"ms": since, "found_ms": now_ms, "screen": [screen_a, screen_b], "store": [a, b],
                               "detail": [d1, d2], "overview_overlap": round(over, 3), "detail_overlap": round(shelf, 3)})

    # ------------------------------------------------------------------

    def _link_candidate(self, person_id: int, obs: IdentityObservation, width: int, height: int) -> bool:
        """이 전체 카메라 표본이 그 신원을 두 화면 짝의 후보로 만들어도 되나 (`fusion.link_customers_only`).
        한 조각(0.5초)에 이런 표본이 하나라도 있는 신원만 후보다.

        ① 손님 장부가 손님으로 센 신원이면 된다 (매장 안 바닥에 risk.inside_seconds 이상 — 한 번 세면 그대로라,
           그 뒤 앞사람에게 가려 발이 안 보여도 후보로 남는다).
        ② 아직 아니어도, **지금 발목이 보이고 그 발이 매장 안 바닥 구역(risk.inside_zones)에 있으면** 된다 — 막 들어온 손님이
           1초를 채우기 전에도 짝이 맺히게. 발목이 안 보이는 표본은 여기에 못 든다: 상자 아래 가운데는 발이 아닐 수 있고
           (3차 N09 20.5초: 겹친 두 손님을 덮은 가짜 상자의 아래 가운데가 계산대 구역 안이었다), 반직선은 어디든 지나간다.
        창가 겉옷·가방의 가짜 사람 상자는 매장 안 바닥을 밟지 않아 둘 다 못 넘는다.
        장부는 이 장면의 짝짓기 뒤에 갱신되므로 ① 은 한 장면 늦다 — 조각(0.5초)에 견주면 무시할 만하다."""
        book = self.risk.book
        customer = book.customers.get(person_id)
        if customer is not None and customer.seen_inside:
            return True
        (px, py), ankle = foot_pixel(tuple(obs.bbox), getattr(obs, "keypoints", None), self.floor.ankle_conf)
        x, y = px / width, py / height
        if not ankle or not self.floor.on_floor_side(x, y):
            return False
        return any(z.name in book.config.inside_zones for z in self.floor.zones_at(self.floor.to_floor(x, y)))

    def detail_band(self, bbox) -> float:
        a, b = self.band_map
        if self.detail_point is not None:
            return a * self.detail_point(bbox, self.detail_width) + b
        return a * ((bbox[0] + bbox[2]) / 2.0 / self.detail_width) + b

    @staticmethod
    def hidden_behind(bbox, others) -> bool:
        """상자 아래 가운데가 더 앞에 선 다른 사람 상자 안인가 (`fusion.ankle_hidden_behind`). others = 같은 장면의 다른 사람 상자들.
        '더 앞' = 그 상자의 아래 변이 이 상자 아래 변보다 이 상자 높이의 5% 넘게 아래 (foot_pixel 이 발목을 상자 안으로 치는 여유와 같다)."""
        x1, y1, x2, y2 = bbox[:4]
        cx, pad = (x1 + x2) / 2.0, 0.05 * (y2 - y1)
        return any(o[0] <= cx <= o[2] and o[1] <= y2 and o[3] > y2 + pad for o in others)

    def _link_keypoints(self, obs, identities):
        """두 화면 짝의 발 자리를 잴 때 쓸 관절 — 앞사람에게 아래가 가린 상자면 None (발목을 안 믿고 반직선으로)."""
        keypoints = getattr(obs, "keypoints", None)
        if keypoints is None or not self.config.ankle_hidden_behind:
            return keypoints
        return None if self.hidden_behind(obs.bbox, [o.bbox for o in identities if o is not obs]) else keypoints

    RAY_STEP = 0.02
    """발목이 안 보일 때 반직선 방향을 잴 화면 세로 간격 (화면 높이 비율)."""

    @classmethod
    def floor_point(cls, fm: FloorMap, bbox, keypoints, width: int, height: int):
        """사람의 발을 바닥 cm 로: (점, 방향). 지평선 너머면 (None, None).

        발목이 보이면 두 발목의 가운데, 방향 None. 발목이 안 보이면(발이 화면 아래로 잘렸거나 앞사람·가구에 가림) 상자 아래 가운데와
        '화면 아래쪽'이 바닥에서 가리키는 방향 — 실제 발은 그 반직선 위 어딘가다 (2026-11-13: 3차 전체 카메라는 선반 앞 사람의 발이
        화면 아래로 잘리고, 계산대 사람의 발은 앞사람에게 가린다. 상자 아래를 발로 믿으면 40cm~1m 어긋났다)."""
        (px, py), ankle = foot_pixel(tuple(bbox), keypoints, fm.ankle_conf)
        x, y = px / width, py / height
        if not fm.on_floor_side(x, y):
            return None, None
        point = fm.to_floor(x, y)
        if ankle:
            return point, None
        if not fm.on_floor_side(x, y + cls.RAY_STEP):
            return point, None
        below = fm.to_floor(x, y + cls.RAY_STEP)
        dx, dy = below[0] - point[0], below[1] - point[1]
        norm = (dx * dx + dy * dy) ** 0.5
        return point, ((dx / norm, dy / norm) if norm > 1e-9 else None)

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
        boxes = {}
        for obs in identities:
            if obs.person_id <= 0:
                continue
            pid = resolve(obs.person_id) if resolve else obs.person_id
            floor, ray = (self.floor_point(self.detail_floor, obs.bbox, self._link_keypoints(obs, identities),
                                           self.detail_width, self.detail_height)
                          if self.detail_floor is not None else (None, None))
            self.linker.observe("detail", pid, pts_ms, self.detail_band(obs.bbox),
                                appearance(pid) if appearance else None, floor, ray)
            boxes[pid] = obs.bbox
        self._note_boxes("detail", pts_ms, boxes)
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
        products: list | None = None,
    ):
        """products = 이 장면에서 찾은 상품 상자 [[종류, x1, y1, x2, y2, 점수], …] (fusion.overview_count 일 때만 쓴다)."""
        w, h = self.overview_size
        if self.counter is not None and products is not None:
            self.counter.update(frame.pts_ms, products, [o.bbox for o in identities])
        boxes = {}
        hands = []
        for obs in identities:
            if obs.person_id <= 0:
                continue
            screen = resolve(obs.person_id) if resolve else obs.person_id
            pid = self._store.get(screen, screen)   # 매장 신원 — 번호 맞바뀜을 되돌린 신원만 화면 신원과 다르다
            boxes[pid] = obs.bbox
            hands.append((pid, obs.bbox, getattr(obs, "keypoints", None)))
            point = self.floor.foot_on_floor(obs.bbox, w, h)
            band = self.floor.band_position(point)   # 띠는 band_map 을 맞춘 상자 기준 그대로
            foot = self.floor.foot_on_floor(obs.bbox, w, h, obs.keypoints) if self.floor.foot_mode != "bbox" else point
            in_band = any(z.name == self.config.shelf_zone for z in self.floor.zones_at(foot))
            if band is not None and not (-0.3 <= band <= 1.3):
                band = None
            floor_cm, ray = (self.floor_point(self.floor, obs.bbox, self._link_keypoints(obs, identities), w, h)
                             if self.config.position_mode == "floor" else (None, None))
            # 짝 후보 표시만 다르다 — 표본은 그대로 넣고, 선반 앞에 누가 있었나(note_band)·손님 장부·위험 판정도 그대로 본다
            candidate = not self.config.link_customers_only or self._link_candidate(pid, obs, w, h)
            self.linker.observe("overview", pid, frame.pts_ms, band, appearance(screen) if appearance else None, floor_cm, ray,
                                candidate)
            if in_band:
                self.fuser.note_band(frame.pts_ms, pid)
        if self.holders is not None and products is not None:
            self.holders.update(frame.pts_ms, products, hands, self.counter.region)
        if not self.config.overview_dwell:
            takes = []   # 전체 카메라의 '선반 앞 머묾'은 판정에 안 넣는다 (fusion.overview_dwell)
        self._note_boxes("overview", frame.pts_ms, boxes)
        self.linker.update(frame.pts_ms)
        if self.config.swap_repair and self.linker.closed != self._closed_seen:
            self._closed_seen = self.linker.closed
            self._repair_swaps(frame.pts_ms)
        reach_takes = (self._reach_takes(self.reach.update(frame, self._standing(identities)), resolve)
                       if self.reach else [])
        # 손 접촉도 방문과 같이 기다렸다 넘긴다 — 그 시간을 덮는 짝 기록이 쌓인 뒤에 주인을 정해야
        # 손 접촉과 방문이 같은 손님으로 해석된다 (2차 C1: 일찍 정하면 손 접촉과 방문의 주인이 갈렸다)
        delay = self.config.release_delay_seconds * 1000.0
        fused_takes, fused_visits = self._release(frame.pts_ms - delay, frame.pts_ms - self.VISIT_DELAY_MS)
        fused_visits += self.fuser.sensor_only(frame.pts_ms, self._waiting)
        if not self.config.overview_reach_takes:
            reach_takes = []   # 주인 정하기 근거로만 (note_reach 는 위에서 이미 적음)
        merges = self._note_merges(merges)
        takes = list(takes)
        if self._store:
            # 손님 장부에는 매장 신원으로 (팔 뻗기 감지기는 위에서 화면 신원 그대로 봤다 — 그쪽은 _reach_takes 가 바꾼다)
            identities = [self._renamed(o, resolve) for o in identities]
            takes = [self._renamed(t, resolve) for t in takes]
        events, records = self.risk.update(frame, identities, takes + reach_takes + fused_takes, merges, fused_visits)
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
            pid = self.store_id(resolve(t.person_id) if resolve else t.person_id)
            self.fuser.note_reach(axis.to_ms(t.start_frame), axis.to_ms(t.end_frame), pid)
            out.append(replace(t, person_id=pid, zone=self.config.shelf_zone))
        return out

    def _count_ready(self, end_ms: float, until_ms: float) -> bool:
        """전체 카메라 개수를 쓰는 중이면: 방문 뒤의 개수를 봤거나, 더 기다릴 시간(overview_count_wait_seconds)이 지났나."""
        if self.counter is None or not self.counter.fed or self.counter.ready(end_ms):
            return True
        return end_ms <= until_ms - self.config.overview_count_wait_seconds * 1000.0

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
        self._waiting = []
        for item in self._pending_visits:
            end_ms, visit, start_ms, stop_ms = item
            if end_ms <= visits_until_ms and not self._count_ready(end_ms, visits_until_ms):
                # 평소라면 지금 넘겼을 방문 — 그때 묶였을 센서 기록이 그 사이 '센서만의 방문'으로 나가지 않게 적어 둔다
                self._waiting.append((start_ms, stop_ms, visit.shelf))
                keepv.append(item)
            elif end_ms <= visits_until_ms:
                fused = self.fuser.detail_visit(visit, start_ms, stop_ms)
                visits.append(fused)
                visits += self.fuser.take_shares()   # 든 사람별로 나눈 방문의 나머지 몫 (fusion.overview_owner)
                if self.config.unchanged_clears_all and fused.removed == 0 and fused.added == 0                         and fused.slot_change is None:
                    # 물건 탐지도 칸 그림도 '그대로'인 방문: 그 시간 선반 앞에 있던 다른 손님의 손 뻗기도
                    # 아무것도 안 가져간 것이다 (설계 M-7 — 둘 다 그대로일 때만 집기를 지운다)
                    for other in self.fuser.present(start_ms, stop_ms) - {fused.person_id}:
                        visits.append(replace(fused, person_id=other, detail=fused.detail + " (함께 있던 손님)"))
            else:
                keepv.append(item)
        self._pending_visits = keepv
        return takes, visits
