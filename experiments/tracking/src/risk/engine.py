"""위험 판정 — 계산하지 않은 물건을 가진 채 나가는 손님을 찾는다.

한 프레임마다 이 순서로 처리한다.

    1. 신원 합침 반영       계층 2 가 두 신원을 합쳤으면 장부도 합친다
    2. TAKE 후보 반영       계층 3 이 끝낸 후보를 해당 손님의 '집은 것'에 더한다
       선반 방문 반영       선반 지도가 확인한 '비었다/다시 찼다/그대로'를 손님에게 붙인다 (customers.py)
    3. 위치·구역 갱신       누가 계산대에, 누가 출구에 서 있는가
    4. 결제 반영           지금까지 도착한 결제를 계산대의 손님에게 연결한다
    5. 출구에 다가섬        출구에 0.5초 이상 들어섰고, 이대로 나가면 HIGH_RISK 인 손님 -> WARNING
    6. 나감 확정 + 최종 판정  다음 중 하나면 나간 것으로 확정하고 아래 표로 판정한다
                           (a) 매장 안에 있던 사람이 문 밖(OUTSIDE) 구역에서 0.5초 이상 보임 — 가장 확실한 증거
                           (b) 출구·문 밖 구역에 있다가(또는 그곳을 떠난 지 2초 안에) 사라져 3초간 안 나타남
                               — 문 밖이 안 보이는 매장, 화면 가장자리 출구용

| 가져간 물건 | 결제 | 최종 판정 |
|---|---|---|
| 0 | - | CLEAR |
| 1 이상 | 0 | **HIGH_RISK** (단, 신원 뒤바뀜이 의심되면 REVIEW) |
| 1 이상 | 물건 수보다 적음 | REVIEW |
| 1 이상 | 물건 수 이상 | CLEAR |

'가져간 물건'은 선반 확인 개수 + 선반 기록이 없는 집기 동작 수다(선반 지도가 꺼져 있으면 집기 동작 수).

## 경보를 두 단계로 나눈 이유 (UCF-Crime Shoplifting047)

처음에는 출구에 들어선 순간 HIGH_RISK 를 확정했다. 혼잡한 문 앞에서 두 번 틀렸다.
  - 집기 기록을 가진 추적 번호가 문가에 선 다른 사람에게 옮겨 붙었다 -> 그 사람에게 경보
  - 인파에 가려 잠깐 사라진 것을 '나갔다'로 봤다 -> 곧 다시 나타났지만 경보는 이미 울림
그래서 다가서면 WARNING(직원이 먼저 알 수 있게), 실제로 사라져 돌아오지 않아야 HIGH_RISK 로 확정한다.

## 판정 직전 신원 확인

HIGH_RISK 를 확정하기 직전, 그 신원의 최근 1.5초 모습과 그 이전 모습을 비교한다(계층 2 가 모아 둔 사진).
단, 매장 안에 있던 마지막 5초 중 가장 높은 값을 쓴다 — 밝은 문 쪽으로 다가가면 역광으로 같은 사람도
값이 꾸준히 떨어진다(Shoplifting039: 0.87 -> 0.68). 조명 변화는 일시적이고 뒤바뀜은 지속적이다.
덜 닮았으면 추적 번호가 다른 사람에게 옮겨 붙었을 수 있으므로 HIGH_RISK 대신 REVIEW 로 돌린다.
무고한 손님에게 경보를 울리는 것이 가장 피해야 할 오류이기 때문이다.

## REVIEW 를 따로 두는 이유

시스템은 집은 물건 수를 정확히 세지 못한다. "2개 집고 1개만 결제"는 실제 절도 방식이지만
탐지가 과하게 센 것일 수도 있다. 이런 경우 경보 대신 관리자 확인으로 돌린다.
"결제가 전혀 없음"은 개수를 몰라도 판단할 수 있으므로 HIGH_RISK 로 둔다.

이 계층은 tracker 도, detector 도 모른다. IdentityObservation · TakeCandidate · 결제만 받는다.
신원 확인 값은 함수(consistency)로 받는다. 없으면 확인하지 않는다.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from ..core.config import RiskConfig
from ..core.types import Frame, IdentityObservation, RiskEvent, RiskLevel, TakeCandidate
from ..zones.zone_map import ZoneMap
from .customers import KIND_NAMES, Customer, CustomerBook, ShelfAction, ShelfVisit, classify_visit
from .payments import PaymentFeed, PaymentRecord


class RiskEngine:
    def __init__(
        self,
        config: RiskConfig,
        zone_map: ZoneMap,
        payments: PaymentFeed | None = None,
        consistency: Callable[[int], float | None] | None = None,
    ) -> None:
        """consistency: person_id -> 최근 모습과 이전 모습의 유사도 (IdentityRegistry.recent_consistency)."""
        self.config = config
        self.book = CustomerBook(config, zone_map)
        self.payments = payments or PaymentFeed.empty()
        self.consistency = consistency
        self.reset()

    def reset(self) -> None:
        self.book.reset()
        self.payments.reset()
        self.events: list[RiskEvent] = []
        self.new_actions: list[ShelfAction] = []
        """이번 프레임에 끝난 선반 방문의 행동 판정 (집음 / 놓음 / 만지기만 ...). 화면 표시용.
        최종 목록은 손님마다 Customer.actions — 방문이 끝난 뒤 늦게 도착하는 손 뻗기까지 반영된다."""
        self.retracted: list[tuple[int, int]] = []
        """(손님, 프레임): '나갔다'고 확정했는데 다시 나타난 경우. 이른 판정이 있었다는 기록."""
        self._merges_seen = 0

    def update(
        self,
        frame: Frame,
        identities: list[IdentityObservation],
        takes: list[TakeCandidate] = (),
        merges: list[dict] = (),
        shelf_visits: list[ShelfVisit] = (),
    ) -> tuple[list[RiskEvent], list[PaymentRecord]]:
        """이번 프레임에 나온 판정과 결제 연결 결과.

        merges 는 계층 2 의 누적 합침 기록 전체(registry.merges)를 그대로 넘기면 된다.
        이미 반영한 것은 건너뛴다.
        """
        now = frame.pts_ms
        for m in merges[self._merges_seen:]:
            self.book.merge(m["from"], m["into"])
        self._merges_seen = len(merges)

        for c in takes:
            self.book.add_take(c, now)
        self.new_actions = []
        for v in shelf_visits:
            customer = self.book.add_shelf_visit(v, now)
            if customer is not None:
                self.new_actions.append(classify_visit(replace(v, person_id=customer.person_id), customer.takes))

        visible = self.book.observe(now, identities)
        self.retracted += [(pid, frame.index) for pid in self.book.reappeared]

        # 신원 확인 값은 매장 안에 있을 때만 잰다 (출구 쪽은 역광 등으로 모습이 크게 바뀐다).
        # 출구를 막 지난 직후(문턱)도 매장 안으로 치지 않는다.
        if self.consistency is not None:
            near_ms = self.config.exit_rearm_seconds * 1000.0
            for customer in visible:
                if customer.exit_since_ms is not None or customer.outside_since_ms is not None:
                    continue
                if customer.last_exit_ms is not None and now - customer.last_exit_ms < near_ms:
                    continue
                value = self.consistency(customer.person_id)
                if value is not None:
                    customer.inside_checks.append((now, value))
                    customer.inside_checks = customer.inside_checks[-600:]

        records = [self.book.pay(e, now) for e in self.payments.due(now)]

        events = []

        # 5) 출구에 다가섬 — 이대로 나가면 HIGH_RISK 인 손님에게만 WARNING
        min_ms = self.config.exit_min_seconds * 1000.0
        for customer in visible:
            if customer.exit_since_ms is None or customer.exit_judged:
                continue
            if now - customer.exit_since_ms < min_ms:
                continue
            customer.exit_judged = True
            verdict = self._judge(customer, frame)
            if verdict.level == RiskLevel.HIGH_RISK:
                events.append(replace(
                    verdict,
                    level=RiskLevel.WARNING,
                    reason=verdict.reason + " — 출구에 다가섬. 이대로 나가면 HIGH_RISK",
                ))

        # 6-a) 나감 확정 — 매장 안에 있던 사람이 문 밖 구역에 보임
        for customer in visible:
            if customer.departed or not customer.seen_inside or customer.outside_since_ms is None:
                continue
            if now - customer.outside_since_ms < min_ms:
                continue
            customer.departed = True
            events.append(self._check_identity(customer, self._judge(customer, frame, how="문 밖으로 나감")))

        # 6-b) 나감 확정 — 출구 근처에서 사라져 한동안 다시 나타나지 않음
        confirm_ms = self.config.exit_confirm_seconds * 1000.0
        near_ms = self.config.exit_rearm_seconds * 1000.0
        seen = {c.person_id for c in visible}
        for customer in self.book.customers.values():
            if customer.person_id in seen or customer.departed or not customer.seen_inside:
                continue
            if customer.last_exit_ms is None or customer.last_seen_ms - customer.last_exit_ms > near_ms:
                continue  # 출구 근처가 아닌 곳에서 사라졌다 — 가려짐이나 탐지 실패
            if now - customer.last_seen_ms < confirm_ms:
                continue
            customer.departed = True
            events.append(self._check_identity(customer, self._judge(customer, frame, how="출구 쪽에서 사라짐")))

        self.events.extend(events)
        return events, records

    def add_late_takes(self, takes: list[TakeCandidate], now_ms: float) -> None:
        """영상이 끝날 때 계층 3 이 마저 내보낸 후보. 판정은 없고 장부에만 더한다."""
        for c in takes:
            self.book.add_take(c, now_ms)

    def _check_identity(self, customer: Customer, event: RiskEvent) -> RiskEvent:
        """HIGH_RISK 확정 직전, 추적 번호가 다른 사람에게 옮겨 붙지 않았는지 확인한다.

        값은 나가는 순간이 아니라, 매장 안에 있던 마지막 몇 초(swap_window_seconds) 중 가장 높은 값을 쓴다.
        대가: 나가기 직전 그 몇 초 안에 번호가 뒤바뀐 경우는 이 확인으로 잡지 못한다
        (경보 두 단계가 일부 막는다 — 뒤바뀐 사람이 실제로 나가야 HIGH_RISK 가 확정된다).
        """
        if event.level != RiskLevel.HIGH_RISK or self.consistency is None:
            return event
        value = None
        if customer.inside_checks:
            last_t = customer.inside_checks[-1][0]
            window_ms = self.config.swap_window_seconds * 1000.0
            value = max(v for t, v in customer.inside_checks if t >= last_t - window_ms)
        if value is None:
            return replace(event, reason=event.reason + " (신원 확인: 기록 부족으로 못 함)")
        if value < self.config.swap_threshold:
            return replace(
                event,
                level=RiskLevel.REVIEW,
                identity_check=round(value, 3),
                reason=(
                    event.reason + f" — 그러나 최근 모습이 이전과 달라(유사도 {value:.2f}) "
                    "다른 사람과 뒤바뀌었을 수 있음. 확인 필요"
                ),
            )
        return replace(event, identity_check=round(value, 3))

    @staticmethod
    def _judge(customer: Customer, frame: Frame, how: str = "") -> RiskEvent:
        unpaid = customer.unpaid
        detail = customer.count_detail()
        if not detail:
            # 선반 지도가 없으면 지금까지처럼 집기 행동 수로 말한다.
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
        else:
            if unpaid > 0 and customer.paid_items == 0:
                level = RiskLevel.HIGH_RISK
                reason = f"물건 {customer.taken}개, 결제 없음"
            elif unpaid > 0:
                level = RiskLevel.REVIEW
                reason = f"결제가 물건보다 적음 (물건 {customer.taken} / 결제 {customer.paid_items}) — 확인 필요"
            else:
                level = RiskLevel.CLEAR
                reason = (
                    "가져간 물건 없음" if customer.taken == 0
                    else f"모두 결제함 (물건 {customer.taken} / 결제 {customer.paid_items})"
                )
            reason += f" [{detail}]"
        if how:
            reason += f" — {how}"
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

    def all_actions(self) -> list[ShelfAction]:
        """모든 손님의 최종 행동 목록 (영상 끝에 기록용)."""
        return sorted((a for c in self.book.customers.values() for a in c.actions),
                      key=lambda a: (a.start_frame, a.person_id))

    def summary(self) -> str:
        """영상이 끝났을 때의 장부. 출구에 가지 않은 손님도 보여준다."""
        lines = []
        count = {level: sum(1 for e in self.events if e.level == level) for level in RiskLevel}
        lines.append(
            f"위험 판정       : HIGH_RISK {count[RiskLevel.HIGH_RISK]}건 / "
            f"REVIEW {count[RiskLevel.REVIEW]}건 / CLEAR {count[RiskLevel.CLEAR]}건 "
            f"(출구 접근 WARNING {count[RiskLevel.WARNING]}건)"
        )
        if self.retracted:
            lines.append(f"  └ '나감' 확정 뒤 다시 나타남: {len(self.retracted)}건 (이른 판정, 되돌림)")
        acts = [a for c in self.book.customers.values() for a in c.actions if a.kind != "LOOK"]
        if any(c.shelf_visits for c in self.book.customers.values()):
            counts = ", ".join(
                f"{KIND_NAMES[k]} {sum(1 for a in acts if a.kind == k)}건"
                for k in KIND_NAMES if k != "LOOK" and any(a.kind == k for a in acts)
            )
            lines.append(f"행동 판정       : {counts or '없음'}")
        visits = [v for c in self.book.customers.values() for v in c.shelf_visits]
        if visits or self.book.shelf_unassigned:
            changed = sum(1 for v in visits if v.removed or v.added)
            lines.append(
                f"선반 확인       : 손님에게 연결한 변화 {changed}건, 변화 없는 방문 {len(visits) - changed}건"
                + (f", 누구 것인지 못 정한 변화 {self.book.shelf_unassigned}건" if self.book.shelf_unassigned else "")
            )
        unmatched = sum(1 for p in self.book.payments if p.person_id is None)
        if self.book.payments:
            lines.append(f"결제            : {len(self.book.payments)}건 (손님 연결 실패 {unmatched}건)")
        open_unpaid = [c for c in self.book.customers.values() if c.unpaid > 0]
        if open_unpaid:
            ids = ", ".join(f"{c.person_id}({c.unpaid}개)" for c in open_unpaid)
            lines.append(f"  └ 영상 끝 기준 미결제 손님: {ids}")
        return "\n".join(lines)
