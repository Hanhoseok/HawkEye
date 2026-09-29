"""손님 상태 — 손님마다 '집은 것 / 결제한 것 / 계산하지 않은 것'을 들고 다닌다.

    손님 3 이 선반 앞에서 멈춤         -> TAKE 후보 1건    taken 1, paid 0 -> unpaid 1
    계산대 구역에 서 있을 때 POS 결제 1개 -> 손님 3 에 연결  taken 1, paid 1 -> unpaid 0

지금 시스템은 **어떤 상품을 집었는지 모른다**(docs/phase5-take-return-survey.md).
그래서 상품 목록이 아니라 개수로만 센다. 결제도 품목 수로만 맞춘다.

개수도 정확하지 않다. 한 번 멈춰 서서 두 선반에 걸치면 TAKE 후보가 두 건 나온다
(store-aisle 실측: 손님 1 의 9.8~13.6초 오른쪽 선반, 11.5~22.0초 가운데 진열대).
그래서 **시간이 겹치는 후보는 한 번의 집기 행동**으로 센다. 그래도 과다 계수는 남으므로
위험 판정은 개수 비교에 크게 기대지 않는다(engine.py).

신원이 나중에 합쳐지면(계층 2 의 late merge) 두 손님의 기록도 하나로 합친다.
합쳐지기 전 번호로 뒤늦게 도착하는 TAKE 후보도 합쳐진 손님에게 간다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.config import RiskConfig
from ..core.types import BBox, IdentityObservation, TakeCandidate, ZoneType
from ..zones.zone_map import ZoneMap
from .payments import PaymentEvent, PaymentRecord


@dataclass
class Customer:
    person_id: int
    first_seen_ms: float
    last_seen_ms: float
    takes: list[TakeCandidate] = field(default_factory=list)
    paid_items: int = 0

    # 계산대 구역
    checkout_since_ms: float | None = None
    """지금 계산대 안에 있다면 들어온 시각."""
    last_checkout_ms: float | None = None
    """마지막으로 계산대 안에 있었던 시각."""

    # 출구 구역
    exit_zone: str | None = None
    exit_since_ms: float | None = None
    """지금 출구 안에 있다면 들어온 시각."""
    exit_left_ms: float | None = None
    """출구를 벗어난 시각."""
    exit_judged: bool = False
    """이번 출구 방문에 대해 이미 판정했는가."""

    last_bbox: BBox | None = None

    @property
    def taken(self) -> int:
        """집기 행동 수. 시간이 겹치는 TAKE 후보는 하나로 묶는다."""
        acts, current_end = 0, None
        for c in sorted(self.takes, key=lambda c: c.start_frame):
            if current_end is None or c.start_frame > current_end:
                acts += 1
                current_end = c.end_frame
            else:
                current_end = max(current_end, c.end_frame)
        return acts

    @property
    def unpaid(self) -> int:
        """계산하지 않은 물건 수(추정). 결제가 더 많으면 0 (TAKE 를 놓쳤을 수 있다)."""
        return max(0, self.taken - self.paid_items)


class CustomerBook:
    """매장 안 손님들의 상태 장부."""

    def __init__(self, config: RiskConfig, zone_map: ZoneMap) -> None:
        self.config = config
        self.zone_map = zone_map
        self.reset()

    def reset(self) -> None:
        self.customers: dict[int, Customer] = {}
        self._alias: dict[int, int] = {}
        self.payments: list[PaymentRecord] = []

    # ------------------------------------------------------------------ 신원

    def resolve(self, person_id: int) -> int:
        while person_id in self._alias:
            person_id = self._alias[person_id]
        return person_id

    def _get(self, person_id: int, now_ms: float) -> Customer:
        pid = self.resolve(person_id)
        if pid not in self.customers:
            self.customers[pid] = Customer(pid, first_seen_ms=now_ms, last_seen_ms=now_ms)
        return self.customers[pid]

    def merge(self, from_id: int, into_id: int) -> None:
        """계층 2 가 두 신원을 합쳤다. 기록도 합친다."""
        src_id, dst_id = self.resolve(from_id), self.resolve(into_id)
        if src_id == dst_id:
            return
        self._alias[src_id] = dst_id
        src = self.customers.pop(src_id, None)
        if src is None:
            return
        dst = self.customers.get(dst_id)
        if dst is None:
            src.person_id = dst_id
            self.customers[dst_id] = src
            return
        dst.takes = sorted(dst.takes + src.takes, key=lambda c: c.start_frame)
        dst.paid_items += src.paid_items
        dst.first_seen_ms = min(dst.first_seen_ms, src.first_seen_ms)
        if src.last_seen_ms >= dst.last_seen_ms:
            # 지금 보이는 쪽은 합쳐지기 전 번호(src)다. 위치·구역 상태는 그쪽을 따른다.
            dst.last_seen_ms = src.last_seen_ms
            dst.last_bbox = src.last_bbox
            dst.checkout_since_ms = src.checkout_since_ms
            dst.exit_zone, dst.exit_since_ms = src.exit_zone, src.exit_since_ms
            dst.exit_left_ms, dst.exit_judged = src.exit_left_ms, src.exit_judged
        for t in (src.last_checkout_ms, dst.last_checkout_ms):
            if t is not None and (dst.last_checkout_ms is None or t > dst.last_checkout_ms):
                dst.last_checkout_ms = t

    # ------------------------------------------------------------------ 기록

    def add_take(self, candidate: TakeCandidate, now_ms: float) -> Customer:
        customer = self._get(candidate.person_id, now_ms)
        customer.takes.append(candidate)
        return customer

    def observe(self, now_ms: float, identities: list[IdentityObservation]) -> list[Customer]:
        """이번 프레임에 보인 손님들의 위치·구역 상태를 갱신한다. 보인 손님 목록을 돌려준다."""
        seen: dict[int, Customer] = {}
        for obs in identities:
            if obs.person_id <= 0:
                continue
            customer = self._get(obs.person_id, now_ms)
            customer.last_seen_ms = now_ms
            customer.last_bbox = obs.bbox

            hits = self.zone_map.evaluate(obs.bbox) if self.zone_map else []
            in_checkout = any(h.zone_type == ZoneType.CHECKOUT and h.contains_foot for h in hits)
            exit_hit = next((h for h in hits if h.zone_type == ZoneType.EXIT and h.contains_foot), None)

            if in_checkout:
                if customer.checkout_since_ms is None:
                    customer.checkout_since_ms = now_ms
                customer.last_checkout_ms = now_ms
            else:
                customer.checkout_since_ms = None

            if exit_hit is not None:
                if customer.exit_since_ms is None:
                    customer.exit_since_ms = now_ms
                    customer.exit_zone = exit_hit.zone
                    rearm_ms = self.config.exit_rearm_seconds * 1000.0
                    if customer.exit_left_ms is not None and now_ms - customer.exit_left_ms >= rearm_ms:
                        customer.exit_judged = False
                customer.exit_left_ms = None
            else:
                if customer.exit_since_ms is not None:
                    customer.exit_left_ms = now_ms
                customer.exit_since_ms = None
            seen[customer.person_id] = customer
        return list(seen.values())

    def pay(self, event: PaymentEvent, now_ms: float) -> PaymentRecord:
        """결제 한 건을 손님에게 연결한다."""
        if event.person_id is not None:
            customer = self._get(event.person_id, now_ms)
            customer.paid_items += event.items
            record = PaymentRecord(event.time_sec, event.items, customer.person_id, "given", 1)
            self.payments.append(record)
            return record

        grace_ms = self.config.checkout_grace_seconds * 1000.0
        nearby = [
            c for c in self.customers.values()
            if c.last_checkout_ms is not None and now_ms - c.last_checkout_ms <= grace_ms
        ]
        if not nearby:
            record = PaymentRecord(event.time_sec, event.items, None, "unmatched", 0)
            self.payments.append(record)
            return record

        # 지금 계산대에 서 있는 사람을 우선하고, 그중 가장 오래 서 있던 사람을 결제자로 본다.
        # 계산하는 사람은 계산대 앞에 머무르고, 뒤에 선 사람이나 지나가는 사람은 잠깐 들른다.
        def rank(c: Customer):
            standing = c.checkout_since_ms is not None
            return (standing, now_ms - (c.checkout_since_ms or now_ms), c.last_checkout_ms)

        payer = max(nearby, key=rank)
        payer.paid_items += event.items
        record = PaymentRecord(event.time_sec, event.items, payer.person_id, "checkout", len(nearby))
        self.payments.append(record)
        return record
