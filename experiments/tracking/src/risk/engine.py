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
        self.before_judge: Callable[[Frame], None] | None = None
        """이번 장면의 기록(방문 · 결제)을 장부에 넣은 뒤, 판정하기 바로 전에 부른다 — 두 카메라 판정이 건넴을 장부에 맞춘다
        (fusion.handover). 없으면 아무 일도 없다."""
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
        self._handover_people: set[int] = set()
        """건넴에 얽힌 적이 있는 손님 (건넴이 나중에 무효가 돼도 남긴다 — 그때도 다시 판정해야 한다)."""

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
        if self.config.payment_mode == "dwell":
            records += self.book.dwell_payments(now, visible)
        if self.before_judge is not None:
            self.before_judge(frame)

        events = []

        # 5) 출구에 다가섬 — 이대로 나가면 HIGH_RISK 인 손님에게만 WARNING
        min_ms = self.config.exit_min_seconds * 1000.0
        for customer in visible:
            if customer.exit_since_ms is None or customer.exit_judged or not customer.seen_inside:
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
            events.append(self._confirm_exit(customer, frame, "문 밖으로 나감"))

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
            events.append(self._confirm_exit(customer, frame, "출구 쪽에서 사라짐"))

        # 6-c) 보류한 나감 판정 — 보류 시간 동안 돌아오지 않았으면 그때 다시 판정해 내보낸다
        hold_ms = self.config.exit_hold_seconds * 1000.0
        for customer in self.book.customers.values():
            if customer.held_since_ms is None or not customer.departed:   # 문 밖에 보여도 됨 — 매장 안으로 돌아오면 장부가 되돌린다
                continue
            if now - customer.held_since_ms < hold_ms:
                continue
            events.append(self._release_hold(customer, frame))

        self._rejudge_handovers(frame, events)
        self.events.extend(events)
        return events, records

    def _rejudge_handovers(self, frame: Frame, events: list[RiskEvent]) -> None:
        """건넴에 얽힌 손님 가운데 **이미 나간** 손님을 다시 판정해, 마지막 판정과 달라졌으면 새 판정을 낸다 (events 에 더한다).

        - 건넨 손님이 먼저(또는 같은 장면에) 나가 '통과'였는데, 받은 손님이 결제 없이 나갔다 → 건넨 손님 확인 필요.
        - 받은 손님이 나간 뒤에야 건넴이 확인됐다 → 받은 손님을 다시 판정한다.
        같은 장면에 낸 판정이면 그 판정을 바꿔 넣는다 (한 손님에게 한 장면에 판정 둘을 내지 않는다).
        나감 보류 중인 손님은 보류가 풀릴 때 판정하므로 건드리지 않는다. 건넴이 없으면 아무 일도 없다."""
        for h in self.book.handovers:
            self._handover_people.update((h.giver, h.receiver))
        if not self._handover_people:
            return
        for pid in sorted({self.book.resolve(p) for p in self._handover_people}):
            customer = self.book.customers.get(pid)
            if customer is None or not customer.departed or customer.held_since_ms is not None:
                continue
            last = next((e for e in reversed(self.events + events)
                         if e.person_id == pid and e.level != RiskLevel.WARNING), None)
            if last is None:
                continue
            now = self._check_identity(customer, self._judge(customer, frame, how="건넴을 반영해 다시 판정"))
            if (now.level, now.taken, now.unpaid) == (last.level, last.taken, last.unpaid):
                continue
            if any(e is last for e in events):
                events[next(i for i, e in enumerate(events) if e is last)] = replace(
                    now, reason=now.reason.replace(" — 건넴을 반영해 다시 판정", last.reason[last.reason.rfind(" — "):]
                                                   if " — " in last.reason else ""))
            else:
                events.append(now)

    def finish(self, frame: Frame, frame_size: tuple[int, int] | None = None) -> list[RiskEvent]:
        """영상이 끝났다 — 출구 쪽에서 사라졌지만 확정 시간(exit_confirm_seconds)을 다 못 채운 손님을 나간 것으로 확정한다.

        녹화 파일은 끝나면 더 이상 아무도 돌아올 수 없다. 직접 찍은 짧은 영상(source/N01 등)은 손님이 나가고
        1초 남짓 뒤에 끝나서, 3초 기다리는 규칙(6-b)으로는 판정이 하나도 나오지 않았다.
        출구가 아닌 곳에서 사라진 손님은 판정하지 않는다.
        마지막 장면에도 보이는 손님은 **출구 구역에서 화면 끝에 걸쳐 있을 때만** 나가는 중으로 본다(frame_size 필요) —
        정상 영상 N05·N06 이 손님이 화면 오른쪽 끝에 반쯤 걸린 채 끝나서(마지막 관측이 끝나기 0.03~0.17초 전) 판정이 없었다.
        출구 구역 안이라도 화면 가운데 서 있으면 아직 안 나간 것이다.
        실시간 카메라에는 '끝'이 없으므로 쓰이지 않는다.
        """
        near_ms = self.config.exit_rearm_seconds * 1000.0
        events = []
        for customer in self.book.customers.values():
            if customer.held_since_ms is not None and customer.departed:
                events.append(self._release_hold(customer, frame, " (영상 끝)"))   # 녹화본은 더 돌아올 수 없다
                continue
            if customer.departed or not customer.seen_inside or customer.last_exit_ms is None:
                continue
            if customer.last_seen_ms - customer.last_exit_ms > near_ms:
                continue  # 출구 근처가 아닌 곳에서 사라졌다
            if frame.pts_ms - customer.last_seen_ms < 300.0:
                # 마지막 장면에도 보인다 — 방금(2초 안) 출구 구역에 있었고 지금 화면 끝에 걸쳐 있으면 나가는 중.
                # 나가는 마지막 몇 장면은 머리·어깨만 잡혀 상자 아래쪽(발)이 출구 띠 밖으로 올라간다 (정상 영상 N06).
                at_edge = frame_size and customer.last_bbox and self._at_edge(customer.last_bbox, frame_size)
                if self.config.exit_end_at_edge_only and not at_edge:
                    continue
            customer.departed = True
            events.append(self._check_identity(customer, self._judge(customer, frame, how="출구 쪽에서 사라짐 (영상 끝)")))
        self._rejudge_handovers(frame, events)
        self.events.extend(events)
        return events

    def given_unpaid(self, customer: Customer) -> list[int]:
        """이 손님이 결제 안 된 물건을 건넸고, 그것을 받은 손님이 결제가 모자란 채 나갔다 — 그 받은 손님들.
        받은 손님이 모자란 것이 건네받은 그 물건인지는 가르지 못한다 (받은 손님의 것 전체로 본다)."""
        out = []
        for h in customer.gave:
            other = self.book.customers.get(self.book.resolve(h.receiver))
            if h.paid or other is None or not other.departed or other.person_id in out:
                continue
            shares = self.config.share_surplus_payments and self.config.payment_mode == "feed"
            self.book.cover_from_surplus(other, givers_only=not shares)
            if other.unpaid > 0:
                out.append(other.person_id)
        return out

    @staticmethod
    def _at_edge(bbox, frame_size, margin: float = 0.02) -> bool:
        w, h = frame_size
        x1, y1, x2, y2 = bbox
        return x1 <= margin * w or x2 >= (1 - margin) * w   # 화면 왼쪽·오른쪽 끝에 걸침

    def add_late_takes(self, takes: list[TakeCandidate], now_ms: float) -> None:
        """영상이 끝날 때 계층 3 이 마저 내보낸 후보. 판정은 없고 장부에만 더한다."""
        for c in takes:
            self.book.add_take(c, now_ms)

    def _confirm_exit(self, customer: Customer, frame: Frame, how: str) -> RiskEvent:
        """나감 확정의 최종 판정. HIGH_RISK 이고 exit_hold_seconds > 0 이면 보류하고 WARNING 을 대신 낸다."""
        event = self._check_identity(customer, self._judge(customer, frame, how=how))
        hold = self.config.exit_hold_seconds
        if hold <= 0 or event.level != RiskLevel.HIGH_RISK:
            return event
        customer.held_since_ms, customer.held_how = frame.pts_ms, how
        return replace(event, level=RiskLevel.WARNING,
                       reason=event.reason + f" — {hold:g}초 안에 돌아오지 않으면 HIGH_RISK")

    def _release_hold(self, customer: Customer, frame: Frame, suffix: str = "") -> RiskEvent:
        """보류를 끝내고 지금까지의 기록(보류 중 도착한 결제 포함)으로 다시 판정한다."""
        how = customer.held_how + suffix
        customer.held_since_ms, customer.held_how = None, ""
        return self._check_identity(customer, self._judge(customer, frame, how=how))

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

    def _judge(self, customer: Customer, frame: Frame, how: str = "") -> RiskEvent:
        # 남는 결제로 메우기는 결제 기록이 있을 때만 — 계산대 체류로 어림한 결제의 '남음'은 개수가 나중에 고쳐져 생긴 것이라 뜻이 없다
        shares = self.config.share_surplus_payments and self.config.payment_mode == "feed"
        if shares:
            covered = self.book.cover_from_surplus(customer)
        elif customer.got:
            covered = self.book.cover_from_surplus(customer, givers_only=True)   # 건넨 손님이 낸 결제는 건넨 물건을 메운다 (후속 70)
        else:
            covered = 0
        unpaid = customer.unpaid
        detail = customer.count_detail()
        short = RiskLevel.HIGH_RISK if self.config.partial_payment_alert else RiskLevel.REVIEW
        """결제가 물건보다 적을 때의 판정 — 기본은 확인 필요, risk.partial_payment_alert 면 경보."""
        paid = customer.paid + covered
        paid_text = f"{customer.paid}" + (f" + 다른 손님의 남는 결제 {covered}" if covered else "")
        if not detail:
            # 선반 지도가 없으면 지금까지처럼 집기 행동 수로 말한다.
            if unpaid > 0 and paid == 0:
                level = RiskLevel.HIGH_RISK
                reason = f"집기 행동 {customer.taken}번, 결제 없음"
            elif unpaid > 0:
                level = RiskLevel.REVIEW     # 집기 행동 수는 물건 수가 아니다 — 개수를 모르니 경보로 올리지 않는다
                reason = f"결제가 집기 행동보다 적음 (집음 {customer.taken} / 결제 {paid_text}) — 확인 필요"
            else:
                level = RiskLevel.CLEAR
                reason = (
                    "집은 물건 없음" if customer.taken == 0
                    else f"모두 결제함 (집음 {customer.taken} / 결제 {paid_text})"
                )
        else:
            if unpaid > 0 and paid == 0:
                level = RiskLevel.HIGH_RISK
                reason = f"물건 {customer.taken}개, 결제 없음"
            elif unpaid > 0:
                level = short
                reason = (f"결제가 물건보다 적음 (물건 {customer.taken} / 결제 {paid_text})"
                          + ("" if short == RiskLevel.HIGH_RISK else " — 확인 필요"))
            else:
                level = RiskLevel.CLEAR
                reason = (
                    "가져간 물건 없음" if customer.taken == 0
                    else f"모두 결제함 (물건 {customer.taken} / 결제 {paid_text})"
                )
            reason += f" [{detail}]"
        if level == RiskLevel.CLEAR and customer.gave:
            # 건넨 물건이 결제 없이 나갔다 — 든 사람이 경보, 건넨 사람은 확인 필요 (사용자 결정 2026-11-23, 후속 70)
            took = self.given_unpaid(customer)
            if took:
                level = RiskLevel.REVIEW
                reason += (" — 건넨 물건이 결제 없이 나감 (받은 사람: "
                           + ", ".join(f"손님 {p}" for p in took) + ") — 확인 필요")
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
            paid=customer.paid,
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
