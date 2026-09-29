"""위험 판정 — 계산하지 않은 물건을 가진 채 출구에 들어선 손님을 찾는다.

한 프레임마다 이 순서로 처리한다.

    1. 신원 합침 반영       계층 2 가 두 신원을 합쳤으면 장부도 합친다
    2. TAKE 후보 반영       계층 3 이 끝낸 후보를 해당 손님의 '집은 것'에 더한다
    3. 위치·구역 갱신       누가 계산대에, 누가 출구에 서 있는가
    4. 결제 반영           지금까지 도착한 결제를 계산대의 손님에게 연결한다
    5. 출구 판정           출구에 0.5초 이상 들어선 손님을 판정한다 (아래 표)

| 집기 행동 | 결제 | 판정 |
|---|---|---|
| 0 | - | CLEAR |
| 1 이상 | 0 | **HIGH_RISK** |
| 1 이상 | 행동 수보다 적음 | REVIEW |
| 1 이상 | 행동 수 이상 | CLEAR |

REVIEW 를 따로 두는 이유: 시스템은 집은 물건 수를 정확히 세지 못한다(겹치는 후보는 묶지만
한 번 멈춰 여러 번 손을 뻗거나, 구경만 한 것을 집기로 보기도 한다). "2개 집고 1개만 결제"는
실제 절도 방식이지만, 탐지가 과하게 센 것일 수도 있다. 이런 경우 경보 대신 관리자 확인으로 돌린다.
"결제가 전혀 없음"은 개수를 몰라도 판단할 수 있으므로 HIGH_RISK 로 둔다.

결제(4)를 위치 갱신(3) 다음에 두는 이유: 결제 순간 계산대에 누가 서 있는지 알아야 하기 때문이다.
출구 판정(5)을 맨 뒤에 두는 이유: 같은 프레임에 들어온 결제와 TAKE 를 모두 반영한 뒤 판단해야 한다.

이 계층은 tracker 도, detector 도 모른다. IdentityObservation · TakeCandidate · 결제만 받는다.
"""

from __future__ import annotations

from ..core.config import RiskConfig
from ..core.types import Frame, IdentityObservation, RiskEvent, RiskLevel, TakeCandidate
from ..zones.zone_map import ZoneMap
from .customers import Customer, CustomerBook
from .payments import PaymentFeed, PaymentRecord


class RiskEngine:
    def __init__(self, config: RiskConfig, zone_map: ZoneMap, payments: PaymentFeed | None = None) -> None:
        self.config = config
        self.book = CustomerBook(config, zone_map)
        self.payments = payments or PaymentFeed.empty()
        self.reset()

    def reset(self) -> None:
        self.book.reset()
        self.payments.reset()
        self.events: list[RiskEvent] = []
        self._merges_seen = 0

    def update(
        self,
        frame: Frame,
        identities: list[IdentityObservation],
        takes: list[TakeCandidate] = (),
        merges: list[dict] = (),
    ) -> tuple[list[RiskEvent], list[PaymentRecord]]:
        """이번 프레임에 나온 위험 판정과 결제 연결 결과.

        merges 는 계층 2 의 누적 합침 기록 전체(registry.merges)를 그대로 넘기면 된다.
        이미 반영한 것은 건너뛴다.
        """
        now = frame.pts_ms
        for m in merges[self._merges_seen:]:
            self.book.merge(m["from"], m["into"])
        self._merges_seen = len(merges)

        for c in takes:
            self.book.add_take(c, now)

        visible = self.book.observe(now, identities)

        records = [self.book.pay(e, now) for e in self.payments.due(now)]

        events = []
        min_ms = self.config.exit_min_seconds * 1000.0
        for customer in visible:
            if customer.exit_since_ms is None or customer.exit_judged:
                continue
            if now - customer.exit_since_ms < min_ms:
                continue
            customer.exit_judged = True
            events.append(self._judge(customer, frame))
        self.events.extend(events)
        return events, records

    def add_late_takes(self, takes: list[TakeCandidate], now_ms: float) -> None:
        """영상이 끝날 때 계층 3 이 마저 내보낸 후보. 판정은 없고 장부에만 더한다."""
        for c in takes:
            self.book.add_take(c, now_ms)

    @staticmethod
    def _judge(customer: Customer, frame: Frame) -> RiskEvent:
        unpaid = customer.unpaid
        if unpaid > 0 and customer.paid_items == 0:
            level = RiskLevel.HIGH_RISK
            reason = f"집기 행동 {customer.taken}번, 결제 없음"
        elif unpaid > 0:
            level = RiskLevel.REVIEW
            reason = f"결제가 집기 행동보다 적음 (집음 {customer.taken} / 결제 {customer.paid_items}) — 확인 필요"
        else:
            level = RiskLevel.CLEAR
            reason = (
                "집은 물건 없음" if customer.taken == 0
                else f"모두 결제함 (집음 {customer.taken} / 결제 {customer.paid_items})"
            )
        return RiskEvent(
            person_id=customer.person_id,
            level=level,
            zone=customer.exit_zone or "",
            frame=frame.index,
            timestamp=frame.timestamp,
            time_sec=round(frame.pts_ms / 1000.0, 3),
            bbox=customer.last_bbox or (0.0, 0.0, 0.0, 0.0),
            taken=customer.taken,
            paid=customer.paid_items,
            unpaid=unpaid,
            take_frames=tuple((c.start_frame, c.end_frame) for c in customer.takes),
            reason=reason,
        )

    def summary(self) -> str:
        """영상이 끝났을 때의 장부. 출구에 가지 않은 손님도 보여준다."""
        lines = []
        count = {level: sum(1 for e in self.events if e.level == level) for level in RiskLevel}
        lines.append(
            f"위험 판정       : HIGH_RISK {count[RiskLevel.HIGH_RISK]}건 / "
            f"REVIEW {count[RiskLevel.REVIEW]}건 / CLEAR {count[RiskLevel.CLEAR]}건"
        )
        unmatched = sum(1 for p in self.book.payments if p.person_id is None)
        if self.book.payments:
            lines.append(f"결제            : {len(self.book.payments)}건 (손님 연결 실패 {unmatched}건)")
        open_unpaid = [c for c in self.book.customers.values() if c.unpaid > 0]
        if open_unpaid:
            ids = ", ".join(f"{c.person_id}({c.unpaid}개)" for c in open_unpaid)
            lines.append(f"  └ 영상 끝 기준 미결제 손님: {ids}")
        return "\n".join(lines)
