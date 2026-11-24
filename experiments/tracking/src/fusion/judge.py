"""카메라 2대의 관찰을 합쳐 손님 장부·위험 판정(⑤⑥)에 넘긴다 — 실시간 파이프라인과 기록 재실행이 같이 쓴다.

    선반 카메라 장면  → detail_step   : 사람 위치(띠 축)·외형을 교차 화면 연결에, 손 접촉·선반 방문은 대기열에
    전체 카메라 장면  → overview_step : 바닥 위치를 연결에, 짝을 갱신하고, 대기열에서 주인이 정해진 사건을 꺼내
                                        전체 카메라 손님의 기록으로 위험 판정에 넣는다

판단(⑤⑥)은 전체 카메라의 장면 시계로 돈다. 선반 카메라의 사건은 짝이 그 시간을 덮을 때까지(조각 2개) 기다렸다 넘긴다.
영상·모델 라이브러리를 모른다 — 저장된 기록만으로 다시 돌릴 수 있다 (scripts/dual_eval.py judge).

매장 신원은 전체 카메라의 화면 신원 그대로다. 단 전체 카메라가 두 사람의 번호를 뒤바꿔 붙인 것을 선반 카메라로 알아챘을 때
(fusion.swap_repair)는 그때부터 두 번호를 서로 바꿔 읽는다 — 짝·손님 장부·판정에는 바꿔 읽은 번호(매장 신원)가 들어간다.

두 카메라 모두 '두 사람을 헷갈린 구간'(상자 겹침 · 합쳐짐)을 적는다 (confusion.ConfusionWatch). 전체 카메라가 헷갈린 구간은
선반 카메라로 정한다 (fusion.swap_since_confusion): 맞바뀜을 되돌릴 때 전체 카메라가 본 것은 번호가 바뀐 때부터 바꿔 읽고,
헷갈린 장면에서 본 '손에 든 상품'은 따로 둔다 (후속 68).
합쳐졌던 사람의 상자가 다시 생긴 순간에는 선반 카메라의 앞뒤 순서로 번호가 바뀌었는지 바로 묻는다 (fusion.swap_ask_detail, 후속 69).
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from typing import Callable

import numpy as np

from ..core.config import FusionConfig, InteractionConfig
from ..core.types import Frame, IdentityObservation, RiskEvent, TakeCandidate
from ..interaction.detector import InteractionDetector
from ..risk.customers import UNKNOWN_KIND, HeldItem, ShelfVisit
from ..zones.floor import FloorMap, foot_pixel
from ..zones.zone_map import ZoneMap
from .cross_view import CrossViewLinker
from .confusion import ConfusionWatch
from .overview_count import HOLD_WITHIN_MS, HolderTracker, OverviewCounter
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
        self.visit_delay_ms = self.VISIT_DELAY_MS
        """선반 방문을 끝난 뒤 이만큼 기다렸다 넘긴다. 센서가 늦게 적는 매장(fusion.sensor_lag_seconds)은 그 기록이 올 때까지 더 기다린다."""
        if sensors and config.sensor_lag_seconds > 0:
            self.visit_delay_ms = max(self.VISIT_DELAY_MS,
                                      (config.event_tolerance_seconds + config.sensor_lag_seconds) * 1000.0)
        self.fuser.hold_wait_ms = self.visit_delay_ms
        if self.holders is not None and getattr(risk, "book", None) is not None:
            self.holders.resolve = risk.book.resolve   # 든 뒤에 합쳐진 신원은 한 사람으로
        if self.holders is not None and config.swap_since_confusion:
            self.holders.confused = lambda pid, ms: bool(self.watch["overview"].touching(pid, ms, ms))
        self.handover_on = bool(config.handover and self.holders is not None and getattr(risk, "book", None) is not None)
        if self.handover_on:
            risk.before_judge = self._hand_overs   # 이번 장면의 방문 · 결제를 장부에 넣은 뒤, 판정하기 전에
        self.shelf_image = shelf_image
        self.reach = None
        """전체 카메라 팔 뻗기 (손목이 화면의 선반 영역 안). 손목 기록만 있으면 되므로 기록 재실행에서도 같이 돈다."""
        if config.overview_reach_seconds > 0 and shelf_image:
            shelf_image.resolve(*overview_size)
            self.reach = InteractionDetector(
                InteractionConfig(signal="hand", min_seconds=config.overview_reach_seconds,
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
        self.held: list[dict] = []
        """손에 든 것을 본 기록 (fusion.overview_instant): pid · name · seen_ms(처음 본 때) · ms(확정한 때) · counted(셌나) ·
        why(안 셌으면 까닭) · until_ms(이것을 대신한 선반 방문이 끝난 때 — 아직 안 바뀌었으면 None)."""
        self._settled: list[tuple[float, float]] = []
        self._waiting: list[tuple[float, float, str]] = []
        self._open_visits: list[tuple[float, str]] = []
        """선반 카메라가 알려 준, 아직 안 끝난 방문 (시작 ms — 공통 시각, 선반). fusion.sensor_hold_visits 가 그 사이의 센서 기록을 붙든다."""
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
        self.watch = {view: ConfusionWatch(limit=self.config.swap_overlap) for view in ("detail", "overview")}
        """카메라마다 두 사람을 헷갈린 구간 (겹침 · 합쳐짐). 번호 맞바뀜을 되돌릴 때 '언제부터'를 여기서 묻는다 (fusion.swap_since_confusion)."""
        self.flips: list[dict] = []
        """두 화면 짝이 서로 뒤집혔지만 전체 카메라 번호 맞바뀜으로 안 본 것 — 까닭과 함께 (재 보려고 적어 둔다)."""
        self.hand_checks: list[dict] = []
        """자기 것이 아닌 상품을 든 순간마다 따져 본 것 (fusion.handover, 장면마다 새로 쓴다): ms · pid(든 손님) · name ·
        say("건넴" · "직접 집음" · "기다림" · "줄 사람 없음" · "못 가름") · giver · giver_frames(건넨 손님 손에서 그 종류를 본 장면 수) · why."""
        self._touch: dict[tuple[int, int], list[float]] = {}
        """두 손님의 상자가 닿은 장면의 시각들 (전체 카메라, 작은 번호 먼저) — 건넴은 닿을 만큼 가까워야 한다."""
        self._hand_key = None
        self.asks: list[dict] = []
        """전체 카메라에서 합쳐졌던 상자가 다시 생긴 순간마다 선반 카메라에 물은 것 (fusion.swap_ask_detail): ms · store(사라졌던 번호,
        남아 있던 번호) · say("바뀜" · "그대로" · "못 물음") · why · gap_cm(사라졌던 번호의 사람이 앞이면 +) · samples."""

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
        self.watch[view].update(pts_ms, boxes)
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
                why = ("함께 나온 장면 없음" if over is None or shelf is None
                       else "두 카메라 모두 겹침" if over >= limit else "전체 카메라가 안 겹침")
                self.flips.append({"ms": since, "found_ms": now_ms, "store": [a, b], "detail": [d1, d2], "why": why,
                                   "overview_overlap": None if over is None else round(over, 3),
                                   "detail_overlap": None if shelf is None else round(shelf, 3)})
                continue
            seen_since = since
            """전체 카메라가 본 것(자리 · 선반 앞에 있던 사람 · 팔 뻗기 · 든 상품)을 바꿔 읽기 시작할 때."""
            episode = self.watch["overview"].find(a, b, start, end) if self.config.swap_since_confusion else None
            if episode is not None and episode.switch_ms is not None:
                seen_since = min(since, episode.switch_ms)
            screens, moved = self._read_swapped(a, b, since, seen_since, now_ms)
            self.swaps.append({"ms": since, "found_ms": now_ms, "screen": screens, "store": [a, b],
                               "detail": [d1, d2], "overview_overlap": round(over, 3), "detail_overlap": round(shelf, 3),
                               "seen_ms": seen_since, "held_moved": moved, "how": "pair",
                               "episode": None if episode is None else episode.to_dict()})

    def _read_swapped(self, a: int, b: int, since_ms: float, seen_ms: float, now_ms: float) -> tuple[list[int], int]:
        """매장 신원 a · b 를 서로 바꿔 읽기 시작한다 — 짝 기록은 since_ms 부터, 전체 카메라가 본 것은 seen_ms 부터.
        (그 두 사람의 화면 신원, 주인을 옮긴 '든 것' 수)를 돌려준다."""
        screen_a, screen_b = self.screen_id(a), self.screen_id(b)
        self.linker.swap_overview(a, b, since_ms, seen_ms)
        self.fuser.swap_overview(a, b, seen_ms)
        moved = self._swap_held(a, b, seen_ms, now_ms) if seen_ms < since_ms else 0
        self.watch["overview"].rename(a, b)
        for screen, store in ((screen_a, b), (screen_b, a)):
            if screen == store:
                self._store.pop(screen, None)
            else:
                self._store[screen] = store
        return [screen_a, screen_b], moved

    # ------------------------------------------------------------------ 상자가 다시 생긴 순간에 선반 카메라에 묻기 (fusion.swap_ask_detail)

    def _toward_camera(self, point: tuple[float, float]) -> tuple[float, float] | None:
        """그 바닥 점에서 전체 카메라 쪽을 가리키는 방향 (단위 벡터) — 화면에서 아래쪽이 카메라에 가깝다."""
        if not self.floor.image_side_ok(*point):
            return None
        x, y = self.floor.to_image(*point)
        if not self.floor.on_floor_side(x, y + self.RAY_STEP):
            return None
        below = self.floor.to_floor(x, y + self.RAY_STEP)
        dx, dy = below[0] - point[0], below[1] - point[1]
        norm = (dx * dx + dy * dy) ** 0.5
        return (dx / norm, dy / norm) if norm > 1e-9 else None

    def _depth_order(self, hidden: int, kept: int, episode, now_ms: float) -> dict:
        """사라졌던 번호(hidden)와 남아 있던 번호(kept)의 사람 가운데 누가 전체 카메라 쪽에 가까웠나 — 선반 카메라의 발 자리로.
        say: "바뀜"(사라졌던 번호의 사람이 앞 — 남은 상자는 앞사람을 따라가므로 번호가 바뀐 것) · "그대로" · "못 물음"."""
        window = self.config.window_seconds * 1000.0
        ask = {"ms": now_ms, "gone_ms": episode.gone_ms, "store": [hidden, kept], "say": "못 물음", "why": "",
               "detail": None, "gap_cm": None, "samples": 0}
        if self.detail_floor is None:
            return {**ask, "why": "선반 카메라 바닥 지도 없음"}
        start = episode.start_ms
        before = {d: self.linker.owner(d, start - 2.0 * window, start) for d in self.linker.detail_ids(start - window)}
        mine = [d for d, o in before.items() if o == hidden]
        other = [d for d, o in before.items() if o == kept]
        if len(mine) != 1 or len(other) != 1:
            return {**ask, "why": "헷갈리기 전에 선반 카메라 짝이 한 사람씩이 아님"}
        d1, d2 = mine[0], other[0]
        ask["detail"] = [d1, d2]
        since = episode.gone_ms - window
        if self.watch["detail"].find(d1, d2, since, now_ms, 0.0) is not None:   # 발 자리를 재는 바로 그 시간에
            return {**ask, "why": "선반 카메라도 두 사람을 헷갈림"}
        feet1, feet2 = self.linker.detail_feet(d1, since, now_ms), self.linker.detail_feet(d2, since, now_ms)
        gaps = []
        for ms in sorted(set(feet1) & set(feet2)):
            p, q = feet1[ms], feet2[ms]
            toward = self._toward_camera(((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0))
            if toward is not None:
                gaps.append((p[0] - q[0]) * toward[0] + (p[1] - q[1]) * toward[1])
        ask["samples"] = len(gaps)
        if len(gaps) < self.config.swap_ask_min_samples:
            return {**ask, "why": "두 사람의 발목이 함께 보인 선반 카메라 장면이 모자람"}
        ask["gap_cm"] = round(sum(gaps) / len(gaps), 1)
        if min(gaps) < 0.0 < max(gaps):
            return {**ask, "why": "앞뒤 순서가 장면마다 다름"}
        if abs(ask["gap_cm"]) < self.config.swap_ask_min_gap_cm:
            return {**ask, "why": "앞뒤 차이가 작음"}
        return {**ask, "say": "바뀜" if ask["gap_cm"] > 0 else "그대로"}

    def _ask_detail(self, now_ms: float) -> dict[int, int]:
        """방금 장면에서 상자가 다시 생긴 사람마다 선반 카메라에 묻고, '바뀜'이면 지금부터 두 번호를 바꿔 읽는다.
        바꿔 읽기 시작한 두 번호 {a: b, b: a} 를 돌려준다 (없으면 빈 사전) — 이 장면에서 이미 모은 것을 부른 쪽이 고쳐 쓴다.
        한 장면에 하나만 바꾼다 (세 사람이 한꺼번에 얽힌 장면은 본 적이 없다)."""
        for hidden, kept, episode in list(self.watch["overview"].returned):
            ask = self._depth_order(hidden, kept, episode, now_ms)
            self.asks.append(ask)
            if ask["say"] != "바뀜":
                continue
            screens, moved = self._read_swapped(hidden, kept, now_ms, episode.gone_ms, now_ms)
            self.swaps.append({"ms": now_ms, "found_ms": now_ms, "screen": screens, "store": [hidden, kept],
                               "detail": ask["detail"], "overview_overlap": None, "detail_overlap": None,
                               "seen_ms": episode.gone_ms, "held_moved": moved, "how": "ask",
                               "gap_cm": ask["gap_cm"], "samples": ask["samples"], "episode": episode.to_dict()})
            return {hidden: kept, kept: hidden}
        return {}

    def _swap_held(self, a: int, b: int, since_ms: float, now_ms: float) -> int:
        """since_ms 부터 전체 카메라가 본 '손에 든 상품'의 주인을 a · b 사이에서 서로 바꾼다: 든 사람 기록, 이미 센 것(손님 장부), 적어 둔 줄."""
        if self.holders is None:
            return 0
        self.holders.swap(a, b, since_ms)
        flip = {a: b, b: a}
        moved = 0
        for row in self.held:
            if row["seen_ms"] >= since_ms and row["pid"] in flip:
                row["pid"] = flip[row["pid"]]
                self.holders.mark_told(row["pid"], row["name"])   # 옮겨 받은 사람에게 같은 종류를 또 세지 않게
                moved += 1
        seen: set[tuple[int, str]] = set()
        for row in sorted(self.held, key=lambda r: r["seen_ms"]):
            key = (row["pid"], row["name"])
            if row["counted"] and row["until_ms"] is None and key in seen:
                row["counted"], row["why"] = False, "번호를 바꿔 읽어 옮겨 옴 — 이미 센 종류"
            elif row["counted"] and row["until_ms"] is None:
                seen.add(key)
        book = getattr(self.risk, "book", None)
        if book is not None and hasattr(book, "swap_held"):
            book.swap_held(a, b, since_ms, now_ms)
        return moved

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
        open_visits: list[tuple[float, str]] | None = None,
    ) -> None:
        """open_visits = 지금 선반이 가려져 아직 안 끝난 방문 (시작 ms — 공통 시각, 선반 이름). None 이면 모름 (안 붙든다)."""
        self._open_visits = list(open_visits or [])
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
        merges: list[dict],
        appearance: Callable[[int], np.ndarray | None] | None = None,
        resolve: Callable[[int], int] | None = None,
        products: list | None = None,
    ):
        """products = 이 장면에서 찾은 상품 상자 [[종류, x1, y1, x2, y2, 점수], …] (fusion.overview_count 일 때만 쓴다).

        전체 카메라의 '선반 앞 머묾'(선반 구역에 느리게 3초)은 여기 들어오지 않는다 — 상품 · 관절 인식 전의 임시 근거라 지웠다 (후속 58).
        이 카메라가 판정에 넣는 것: 사람의 위치 · 팔 뻗기(주인 근거) · 상품 개수 · 상품을 든 사람."""
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
        self._note_boxes("overview", frame.pts_ms, boxes)   # 든 사람을 적기 전에 — 이 장면이 헷갈린 장면인지 먼저 안다
        if self.handover_on:
            self._note_touch(frame.pts_ms, boxes)
        if self.config.swap_ask_detail:
            flip = self._ask_detail(frame.pts_ms)
            if flip:   # 이 장면의 자리 표본 · 선반 앞 기록은 _read_swapped 가 고쳤다 — 아직 안 넘긴 손 자리만 여기서 고친다
                hands = [(flip.get(pid, pid), bbox, kp) for pid, bbox, kp in hands]
        if self.holders is not None and products is not None:
            self.holders.update(frame.pts_ms, products, hands, self.counter.region)
            if self.config.overview_instant:
                self._count_held(frame)
        self.linker.update(frame.pts_ms)
        if self.config.swap_repair and self.linker.closed != self._closed_seen:
            self._closed_seen = self.linker.closed
            self._repair_swaps(frame.pts_ms)
        reach_takes = (self._reach_takes(self.reach.update(frame, self._standing(identities)), resolve)
                       if self.reach else [])
        # 손 접촉도 방문과 같이 기다렸다 넘긴다 — 그 시간을 덮는 짝 기록이 쌓인 뒤에 주인을 정해야
        # 손 접촉과 방문이 같은 손님으로 해석된다 (2차 C1: 일찍 정하면 손 접촉과 방문의 주인이 갈렸다)
        delay = self.config.release_delay_seconds * 1000.0
        fused_takes, fused_visits = self._release(frame.pts_ms - delay, frame.pts_ms - self.visit_delay_ms)
        self._settle_held()
        fused_visits += self.fuser.sensor_only(frame.pts_ms, self._sensor_hold())
        if not self.config.overview_reach_takes:
            reach_takes = []   # 주인 정하기 근거로만 (note_reach 는 위에서 이미 적음)
        merges = self._note_merges(merges)
        if self._store:
            # 손님 장부에는 매장 신원으로 (팔 뻗기 감지기는 위에서 화면 신원 그대로 봤다 — 그쪽은 _reach_takes 가 바꾼다)
            identities = [self._renamed(o, resolve) for o in identities]
        events, records = self.risk.update(frame, identities, reach_takes + fused_takes, merges, fused_visits)
        self.events.extend(events)
        return events, records

    def finish(self, frame: Frame, resolve: Callable[[int], int] | None = None) -> list[RiskEvent]:
        reach_takes = self._reach_takes(self.reach.flush(), resolve) if self.reach else []
        fused_takes, fused_visits = self._release(float("inf"), float("inf"))
        self._settle_held()
        if self.config.overview_reach_takes:
            fused_takes = reach_takes + fused_takes
        fused_visits += self.fuser.sensor_only(float("inf"))
        if fused_takes or fused_visits:
            events, _ = self.risk.update(frame, [], fused_takes, [], fused_visits)
            self.events.extend(events)
        if self.handover_on:
            self._hand_overs(frame)   # 남은 방문을 모두 넘긴 뒤의 장부로 한 번 더
        done = self.risk.finish(frame, self.overview_size)
        self.events.extend(done)
        return done

    # ------------------------------------------------------------------ 건넴 (fusion.handover)

    TOUCH_PAD = 0.15
    """두 사람의 상자가 '닿았다' — 상자를 폭의 이만큼씩 넓혀 겹치면. 맞춰 정한 값이 아니다."""

    def _note_touch(self, pts_ms: float, boxes: dict) -> None:
        ids = sorted(boxes)
        for i, a in enumerate(ids):
            ax = (boxes[a][2] - boxes[a][0]) * self.TOUCH_PAD
            for b in ids[i + 1:]:
                bx = (boxes[b][2] - boxes[b][0]) * self.TOUCH_PAD
                if (min(boxes[a][2] + ax, boxes[b][2] + bx) > max(boxes[a][0] - ax, boxes[b][0] - bx)
                        and min(boxes[a][3], boxes[b][3]) > max(boxes[a][1], boxes[b][1])):
                    self._touch.setdefault((a, b), []).append(pts_ms)

    def _touched(self, a: int, b: int, start_ms: float, end_ms: float) -> bool:
        return any(start_ms <= t <= end_ms for t in self._touch.get((min(a, b), max(a, b)), ()))

    def _hand_overs(self, frame: Frame) -> None:
        """손님끼리 건넨 것을 찾아 손님 장부의 건넴 기록을 맞춘다 (fusion.handover — 규칙은 설정의 설명).

        처음부터 다시 따진다: 든 구간을 시간순으로 보며, 그때까지 집은 것(선반 방문 · 든 것) + 앞선 건넴으로 가진 것이 없는 손님이
        든 것을 건넴으로 본다. 장부에 이미 있는 건넴은 그대로 두고(건넬 때의 결제 상태를 지킨다), 새로 생긴 것은 넣고, 없어진 것은 지운다."""
        book = self.risk.book
        key = (len(self.holders.samples), book.changes, len(self._pending_visits), len(self._open_visits))
        if key == self._hand_key:
            return
        axis = self.fuser.overview_axis
        slack = self.config.event_tolerance_seconds * 1000.0
        near = self.config.handover_near_seconds * 1000.0
        waiting = [s for s, _ in self._open_visits] + [s for _, _, s, _ in self._pending_visits]
        wanted: list[tuple[float, int, int, str, bool]] = []   # (ms, 건넨 손님, 받은 손님, 종류, 직접 집음)
        checks = []

        def have(pid: int, at: int) -> dict[str, int]:
            customer = book.customers.get(pid)
            out = dict(customer.picked_kinds(at)) if customer is not None else {}
            for _, giver, receiver, name, _ in wanted:
                if giver == pid:
                    k = name if out.get(name, 0) > 0 else UNKNOWN_KIND
                    out[k] = out.get(k, 0) - 1
                if receiver == pid:
                    out[name] = out.get(name, 0) + 1
            return {k: n for k, n in out.items() if n > 0}

        for start, end, pid, name in self.holders.holds():
            pid = book.resolve(pid)
            at = axis.to_frame(start)
            mine = have(pid, at)
            if mine.get(name) or mine.get(UNKNOWN_KIND) or (mine and self.config.handover_own_any):
                continue   # 자기 것을 들었다
            row = {"ms": round(start, 1), "pid": pid, "name": name, "giver": None, "why": ""}
            checks.append(row)
            if any(s <= start + slack for s in waiting):
                row["say"] = "기다림"   # 이 순간을 덮을 수 있는 선반 방문이 아직 안 끝났다 — 직접 집었을 수 있다
                continue
            cands, why = [], []
            for q, other in book.customers.items():
                if q == pid:
                    continue
                theirs = have(q, at)
                if not (theirs.get(name) or theirs.get(UNKNOWN_KIND)):
                    continue
                if not (other.first_seen_ms <= start <= other.last_seen_ms + 1000.0):
                    why.append(f"손님 {q} 는 그때 매장에 없었음")
                elif not self._touched(pid, q, start - near, start + near):
                    why.append(f"손님 {q} 와 닿지 않았음")
                elif self.holders.count(q, name, start, min(end, start + HOLD_WITHIN_MS)) >= self.holders.min_frames:
                    why.append(f"손님 {q} 도 그때 들고 있었음")
                else:
                    cands.append(q)
            if len(cands) > 1:   # 그 종류를 가장 늦게까지 들고 있던 손님
                seen = {q: self.holders.last_seen(q, name, start) for q in cands}
                best = sorted((t, q) for q, t in seen.items() if t is not None)
                cands = [best[-1][1]] if best and (len(best) == 1 or best[-1][0] > best[-2][0]) else cands
            if len(cands) == 1:
                # 건넨 손님 손에서 그 종류를 본 적이 없으면 건넨 것이 아니다 — 받은 손님이 직접 집었고 주인을 잘못 정했던 것 (4차 C12)
                seen = self.holders.count(cands[0], name, 0.0, start - 0.1)
                direct = seen < self.config.handover_giver_frames
                row["say"], row["giver"], row["giver_frames"] = ("직접 집음" if direct else "건넴"), cands[0], seen
                wanted.append((start, cands[0], pid, name, direct))
            else:
                row["say"] = "못 가름" if cands else "줄 사람 없음"
                row["why"] = (f"줄 수 있던 손님이 {len(cands)}명" if cands else "; ".join(why))
        self.hand_checks = checks

        old = {(h.ms, h.giver, h.receiver, h.name, h.direct): h for h in book.handovers}
        for k, h in old.items():
            if k not in wanted:
                book.drop_handover(h)
        for k in wanted:
            if k not in old:
                ms, giver, receiver, name, direct = k
                book.hand_over(giver, receiver, name, ms, axis.to_frame(ms), frame.pts_ms, direct)
        self._hand_key = (len(self.holders.samples), book.changes, len(self._pending_visits), len(self._open_visits))

    def _sensor_hold(self) -> list[tuple[float, float, str]]:
        """센서 기록을 붙들어 둘 방문들 (시작 ms, 끝 ms, 선반): 전체 카메라 개수를 기다리는 방문 + (fusion.sensor_hold_visits)
        끝났지만 아직 안 넘긴 방문 · 아직 안 끝난 방문 (끝 = 끝 모름)."""
        lag = self.config.sensor_lag_seconds * 1000.0   # 방문이 끝난 뒤 늦게 온 기록도 그 방문이 먼저 볼 수 있게
        hold = [(start_ms, stop_ms + lag, shelf) for start_ms, stop_ms, shelf in self._waiting]
        if self.config.sensor_hold_visits:
            hold += [(start_ms, stop_ms + lag, visit.shelf) for _, visit, start_ms, stop_ms in self._pending_visits]
            hold += [(start_ms, float("inf"), shelf) for start_ms, shelf in self._open_visits]
        return hold

    def _count_held(self, frame: Frame) -> None:
        """전체 카메라가 방금 '손에 들었다'고 확정한 상품을 그 손님의 물건으로 먼저 센다 (fusion.overview_instant).

        선반 방문은 선반을 가린 사람이 비켜야 끝난다 — 그때까지 개수를 기다리지 않는다. 손에서 처음 보기 조금 전에 선반 앞에 있던
        사람만 센다 (들고 들어온 손님 · 계산대에서 건네받은 일행은 안 센다)."""
        book = getattr(self.risk, "book", None)
        if book is None:
            return
        near = self.config.overview_instant_near_seconds * 1000.0
        axis = self.fuser.overview_axis
        for pid, name, seen_ms, ms in self.holders.fresh():
            at_shelf = pid in {book.resolve(p) for p in self.fuser.present(seen_ms - near, ms)}
            self.held.append({"pid": pid, "name": name, "seen_ms": seen_ms, "ms": ms, "counted": at_shelf,
                              "why": "" if at_shelf else "선반 앞에 있지 않았음", "until_ms": None})
            if at_shelf:
                book.add_held(HeldItem(pid, name, axis.to_frame(seen_ms - near), frame.index, seen_ms), frame.pts_ms)
            elif self.config.hold_per_run:
                self.holders.drop(pid, name)   # 선반에서 시작한 든 구간이 아니다 — 버린다. 끊겼다 다시 들면 또 따진다

    def _settle_held(self) -> None:
        """방금 넘긴 선반 방문이 덮는 '든 것'을 지운다 — 그 방문의 개수 · 주인이 대신한다 (방문을 장부에 넣기 바로 전에 부른다)."""
        settled, self._settled = self._settled, []
        book = getattr(self.risk, "book", None)
        if book is None or not self.held:
            return
        slack = self.config.event_tolerance_seconds * 1000.0   # 방문의 시작(선반 카메라가 본 가림)과 전체 카메라 시각의 어긋남 (3차 C02: 0.3초 앞)
        for start_ms, end_ms in settled:
            gone = book.drop_held(start_ms - slack, end_ms + self.visit_delay_ms)
            for h in gone:
                for row in self.held:
                    if row["until_ms"] is None and row["counted"] and row["seen_ms"] == h.seen_ms and row["name"] == h.name:
                        row["until_ms"] = end_ms

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
        if self.counter is None or not self.counter.fed or self.counter.ready(end_ms) or self.config.count_source != "overview":
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
                self._settled.append((start_ms, stop_ms))
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
