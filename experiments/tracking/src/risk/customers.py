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

## 선반 확인 (docs/shelf-map.md §손님 기록 연결)

선반 지도를 켜면, 손님이 지켜보는 선반에 다녀간 기록(ShelfVisit)이 들어온다.
"그 사이 자리 2개가 비었다" / "1개가 다시 찼다" / "아무것도 안 바뀌었다(구경)".
**같은 시간의 집기 동작은 선반 기록으로 대신한다** — 동작은 '집었을 것 같다', 선반은 '실제로 줄었다'이므로.

    손님 3: 집기 동작 1번(10~14초), 선반 방문 10~15초 '자리 2개 비었음'  -> 물건 2개 (한 번에 두 개)
    손님 4: 집기 동작 1번(20~25초), 선반 방문 20~26초 '변화 없음'        -> 물건 0개 (구경만 함)
    손님 5: 집기 동작 1번(지켜보지 않는 선반), 선반 방문 '1개 다시 참'     -> 물건 0개 (다른 데서 집어 놓고 감)

선반 앞에 두 사람이 함께 있었거나 사람이 안 보인 변화는 누구 것인지 몰라 손님에게 붙이지 않는다.
그때는 지금처럼 집기 동작 수를 쓴다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..core.config import RiskConfig
from ..core.types import BBox, IdentityObservation, TakeCandidate, ZoneType
from ..zones.zone_map import ZoneMap
from .payments import PaymentEvent, PaymentRecord


@dataclass(frozen=True)
class ShelfVisit:
    """손님 한 명이 지켜보는 선반에 다녀간 결과. 선반 지도(src/shelf)가 만들고 여기서는 개수만 쓴다."""

    person_id: int
    shelf: str
    start_frame: int
    end_frame: int
    removed: int
    """그 사이 비게 된 자리 수 (가져감)."""
    added: int
    """그 사이 다시 찬 자리 수 (되돌려놓음)."""
    zone: str | None = None
    """이 선반에 해당하는 SHELF 구역 이름. 주면 그 구역의 집기 동작만 대신한다."""
    detail: str = ""
    """사람이 읽는 설명 (예: '사라짐: 1단 2번째 bottle')."""

    def to_dict(self) -> dict:
        return asdict(self)

    def covers(self, take: TakeCandidate) -> bool:
        """이 방문이 그 집기 동작을 대신하는가 — 같은 시간, 같은 선반."""
        if self.zone is not None and take.zone != self.zone:
            return False
        return take.start_frame <= self.end_frame and take.end_frame >= self.start_frame


@dataclass
class Customer:
    person_id: int
    first_seen_ms: float
    last_seen_ms: float
    takes: list[TakeCandidate] = field(default_factory=list)
    shelf_visits: list[ShelfVisit] = field(default_factory=list)
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
    """이번 출구 방문에서 '다가섬'을 이미 처리했는가 (WARNING 을 냈거나, 낼 필요가 없었거나)."""
    departed: bool = False
    """'나갔다'고 확정하고 최종 판정했는가. 매장 안에서 다시 보이면 되돌린다."""
    last_exit_ms: float | None = None
    """마지막으로 출구 또는 문 밖 구역에 있었던 시각. 출구를 막 지나 사라진 경우를 잡는다."""
    outside_since_ms: float | None = None
    """지금 문 밖 구역에 보인다면 그 시작 시각."""
    inside_checks: list[tuple[float, float]] = field(default_factory=list)
    """(시각, 신원 확인 값) — 매장 안에 있을 때만 잰다. 판정에는 마지막 몇 초 중 가장 높은 값을 쓴다.

    나가기 직전 값을 쓰지 않는 이유 (Shoplifting039 실측): 밝은 유리문 쪽으로 걸어가면 역광 때문에
    같은 사람인데도 값이 0.87 -> 0.68 로 꾸준히 떨어졌다. 조명 변화는 문에 다가가는 몇 초 동안의
    일시적 하락이고, 번호 뒤바뀜은 그 뒤로 계속 낮다. 그래서 최근 몇 초 중 최고값으로 판단한다.
    """
    seen_inside: bool = False
    """매장 안(문 밖 구역이 아닌 곳)에서 보인 적이 있는가. 문 밖 행인은 판정하지 않는다."""

    last_bbox: BBox | None = None

    @staticmethod
    def _acts(takes: list[TakeCandidate]) -> int:
        """집기 행동 수. 시간이 겹치는 TAKE 후보는 하나로 묶는다."""
        acts, current_end = 0, None
        for c in sorted(takes, key=lambda c: c.start_frame):
            if current_end is None or c.start_frame > current_end:
                acts += 1
                current_end = c.end_frame
            else:
                current_end = max(current_end, c.end_frame)
        return acts

    @property
    def uncovered_takes(self) -> list[TakeCandidate]:
        """선반 기록이 대신하지 않은 집기 동작 (지켜보지 않는 선반, 또는 선반 기록이 없는 시간)."""
        return [t for t in self.takes if not any(v.covers(t) for v in self.shelf_visits)]

    @property
    def shelf_net(self) -> int:
        """선반에서 확인한 순 개수 (가져감 - 되돌려놓음)."""
        return sum(v.removed - v.added for v in self.shelf_visits)

    @property
    def taken(self) -> int:
        """가져간 물건 수(추정) = 선반에서 확인한 개수 + 선반 기록이 없는 집기 동작 수.

        선반 지도가 꺼져 있으면 집기 동작 수 그대로다.
        """
        return max(0, self._acts(self.uncovered_takes) + self.shelf_net)

    def count_detail(self) -> str:
        """물건 수의 근거 (판정 문구용). 선반 기록이 없으면 빈 문자열."""
        if not self.shelf_visits:
            return ""
        removed = sum(v.removed for v in self.shelf_visits)
        added = sum(v.added for v in self.shelf_visits)
        parts = [f"선반 확인 {removed}개 가져감"]
        if added:
            parts.append(f"{added}개 되돌려놓음")
        browse = sum(1 for v in self.shelf_visits if not v.removed and not v.added)
        if browse:
            parts.append(f"구경만 {browse}번")
        acts = self._acts(self.uncovered_takes)
        if acts:
            parts.append(f"선반 밖 집기 동작 {acts}번")
        return ", ".join(parts)

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
        self.reappeared: list[int] = []
        self.customers: dict[int, Customer] = {}
        self._alias: dict[int, int] = {}
        self.payments: list[PaymentRecord] = []
        self.shelf_unassigned: int = 0
        """누구 것인지 정하지 못한 선반 변화 수 (두 사람이 함께 있었거나 사람이 안 보임)."""

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
        dst.shelf_visits = sorted(dst.shelf_visits + src.shelf_visits, key=lambda v: v.start_frame)
        dst.paid_items += src.paid_items
        dst.first_seen_ms = min(dst.first_seen_ms, src.first_seen_ms)
        if src.last_seen_ms >= dst.last_seen_ms:
            # 지금 보이는 쪽은 합쳐지기 전 번호(src)다. 위치·구역 상태는 그쪽을 따른다.
            dst.last_seen_ms = src.last_seen_ms
            dst.last_bbox = src.last_bbox
            dst.checkout_since_ms = src.checkout_since_ms
            dst.exit_zone, dst.exit_since_ms = src.exit_zone, src.exit_since_ms
            dst.exit_left_ms, dst.exit_judged = src.exit_left_ms, src.exit_judged
            dst.departed = src.departed
            dst.last_exit_ms, dst.outside_since_ms = src.last_exit_ms, src.outside_since_ms
        dst.seen_inside = dst.seen_inside or src.seen_inside
        for t in (src.last_checkout_ms, dst.last_checkout_ms):
            if t is not None and (dst.last_checkout_ms is None or t > dst.last_checkout_ms):
                dst.last_checkout_ms = t

    # ------------------------------------------------------------------ 기록

    def add_take(self, candidate: TakeCandidate, now_ms: float) -> Customer:
        customer = self._get(candidate.person_id, now_ms)
        customer.takes.append(candidate)
        return customer

    def add_shelf_visit(self, visit: ShelfVisit, now_ms: float) -> Customer | None:
        """선반 방문 결과를 그 손님에게 붙인다. 누구 것인지 모르는(번호 0 이하) 변화는 세기만 한다."""
        if visit.person_id <= 0:
            if visit.removed or visit.added:
                self.shelf_unassigned += 1
            return None
        customer = self._get(visit.person_id, now_ms)
        customer.shelf_visits.append(visit)
        return customer

    def observe(self, now_ms: float, identities: list[IdentityObservation]) -> list[Customer]:
        """이번 프레임에 보인 손님들의 위치·구역 상태를 갱신한다. 보인 손님 목록을 돌려준다."""
        seen: dict[int, Customer] = {}
        self.reappeared: list[int] = []
        for obs in identities:
            if obs.person_id <= 0:
                continue
            customer = self._get(obs.person_id, now_ms)
            hits = self.zone_map.evaluate(obs.bbox) if self.zone_map else []
            outside = any(h.zone_type == ZoneType.OUTSIDE and h.contains_foot for h in hits)
            if customer.departed and not outside:
                # '나갔다'고 확정했는데 다시 보인다 — 나간 게 아니라 탐지가 끊겼던 것이다.
                # 판정을 되돌려 다음 출구 방문 때 다시 판정한다.
                customer.departed = False
                customer.exit_judged = False
                customer.exit_since_ms = None
                customer.exit_left_ms = None
                self.reappeared.append(customer.person_id)
            customer.last_seen_ms = now_ms
            customer.last_bbox = obs.bbox

            if outside:
                if customer.outside_since_ms is None:
                    customer.outside_since_ms = now_ms
                customer.last_exit_ms = now_ms
            else:
                customer.outside_since_ms = None
                customer.seen_inside = True
            in_checkout = any(h.zone_type == ZoneType.CHECKOUT and h.contains_foot for h in hits)
            exit_hit = next((h for h in hits if h.zone_type == ZoneType.EXIT and h.contains_foot), None)

            if in_checkout:
                if customer.checkout_since_ms is None:
                    customer.checkout_since_ms = now_ms
                customer.last_checkout_ms = now_ms
            else:
                customer.checkout_since_ms = None

            if exit_hit is not None:
                customer.last_exit_ms = now_ms
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
