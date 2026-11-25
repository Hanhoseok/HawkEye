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

## 행동 판정 — 집음 / 놓음 / 만짐 (ShelfAction)

손 뻗기(집기 동작)는 '언제 누가 선반에 손을 댔나', 선반 방문은 '그 사이 무엇이 바뀌었나'를 안다. 둘을 합쳐 행동을 정한다.

| 선반 방문 결과 | 그 시간 손 뻗기 | 행동 |
|---|---|---|
| 자리가 비었음 | 있음/없음 | **집음** (TAKE, 비운 개수만큼) |
| 자리가 다시 참 | 있음/없음 | **놓음** (RETURN) |
| 비기도 하고 차기도 함 | 있음/없음 | **자리 바뀜** (MOVE, 순변화만 셈) |
| 그대로 | 있음 | **만지기만** (TOUCH) |
| 그대로 | 없음 | 구경 (LOOK, 개수 변화 없음) |
| 선반 기록 없음 | 있음 | **확인 못 함** (UNVERIFIED, 지금처럼 1개로 셈) |

근거(evidence)도 함께 남긴다: 선반+동작 / 선반만(손 뻗기가 안 잡힘) / 동작만(선반 기록 없음).
한계: 한 번의 방문 안에서 들었다 놓은 것은 순변화로만 보인다(가려진 동안은 셀 수 없다).

## 건넴 — 물건 단위 장부 (후속 70, 한계 D-13)

집은 물건마다 '지금 든 사람'과 '결제됨'이 따라다닌다고 본다. 손님끼리 건네면 (Handover):

    든 사람만 바뀐다           건넨 손님 -1 (건넴), 받은 손님 +1 (건네받음)
    결제됨은 물건과 함께 간다    건넨 손님이 그때 가진 것을 모두 결제했으면 결제 1개도 받은 손님에게 옮긴다
    나갈 때                   결제 안 된 물건을 **든** 손님이 경보. 그 물건을 건넨 손님은 확인 필요 (engine.py)

직접 집음 (Handover.direct): 물건이 한 손님에게 적혀 있는데 다른 손님 손에서 보였고, **적혀 있던 손님 손에서는 한 번도 못 봤다**.
건넨 것이 아니다 — 함께 선반 앞에 있던 두 사람 가운데 집은 사람을 잘못 정했던 것이다 (4차 C12: 장면으로 확인). 물건은 든 손님의
것으로 옮기지만 '건넴'이라 부르지 않고, 결제도 따라가지 않고, 적혀 있던 손님에게 확인 필요도 붙이지 않는다.

물건마다 줄을 따로 두지는 않는다 — 집은 개수(선반 근거)는 그대로 두고, 건넴 기록을 시간순으로 다시 따라가 손님별 순변화를 낸다
(CustomerBook.rebalance). 선반 근거가 나중에 고쳐져 건넨 손님에게 그 물건이 없게 되면 그 건넴은 무효(void)다.
건넴이 없으면 모든 수가 예전과 같다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

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
    slot_change: str | None = None
    """칸 그림 비교 결과 (설계 M-7 보조): '줄어듦' / '늘어남' / '바뀜'(방향 모름) / None(칸 그대로 또는 비교 안 함).
    물건 탐지가 못 보는 물건도 칸이 비었는지는 알 수 있다."""
    source: str = "vision"
    """근거 출처: vision (카메라) / sensor (선반 센서만) / sensor+vision (둘 다, 개수는 센서) /
    overview+vision (방문은 선반 카메라, 개수는 전체 카메라가 센 상품 — fusion.overview_count)."""
    removed_kinds: tuple = ()
    """꺼낸 상품의 종류 (선반 지도가 본 것, 개수만큼 되풀이 — 예: ("cantata", "donggwaru")). 센서 레인이 있는 종류와 없는 종류를
    가를 때 쓴다 (fusion.sensor_kinds). 옛 기록에는 없다 — 그때는 detail 글에서 읽는다."""
    added_kinds: tuple = ()
    """놓은 상품의 종류."""
    held_ms: float | None = None
    """이 방문에서 꺼낸 상품이 이 손님 손에 들린 것을 전체 카메라가 처음 본 시각 (fusion.overview_owner 일 때만 채운다).
    방문은 선반을 가린 사람이 모두 비켜야 끝나 판정에 늦게 들어온다 — 그 사이 계산대에 다녀온 손님의 결제를 맞춰 보는 데 쓴다."""

    def to_dict(self) -> dict:
        return asdict(self)

    def covers(self, take: TakeCandidate) -> bool:
        """이 방문이 그 집기 동작을 대신하는가 — 같은 시간, 같은 선반."""
        if self.zone is not None and take.zone != self.zone:
            return False
        return take.start_frame <= self.end_frame and take.end_frame >= self.start_frame


@dataclass(frozen=True)
class HeldItem:
    """전체 카메라가 본 '손에 든 상품' 하나 — 선반 방문이 끝나 개수가 확정되기 전에 먼저 세는 것 (fusion.overview_instant).

    그 시간을 덮는 선반 방문이 장부에 들어오면 지워지고 방문의 개수 · 주인으로 바뀐다 (CustomerBook.drop_held)."""

    person_id: int
    name: str
    """상품 종류."""
    start_frame: int
    """이 장면부터의 손 뻗기는 이 '든 것'이 대신한다 (손에서 처음 본 때의 조금 앞)."""
    end_frame: int
    """든 것으로 확정한 장면 — 이때부터 개수에 들어간다."""
    seen_ms: float
    """손에서 처음 본 시각. 선반 방문의 시간과 견주는 데 쓴다."""
    item: int | None = None
    """물건 번호 (fusion.item_tracks — 물건 단위로 센 것). 없으면 '사람 · 종류마다 1개'로 센 것이다."""

    def covers(self, take: TakeCandidate) -> bool:
        return take.start_frame <= self.end_frame and take.end_frame >= self.start_frame


KIND_NAMES = {
    "TAKE": "집음", "RETURN": "놓음", "MOVE": "자리 바뀜",
    "TOUCH": "만지기만", "LOOK": "구경", "UNVERIFIED": "확인 못 한 집기 동작", "HELD": "든 것을 봄",
    "GIVE": "건넴", "GET": "건네받음", "MINE": "직접 집음", "THEIRS": "다른 손님이 집음",
}

UNKNOWN_KIND = "?"
"""종류를 모르는 물건 (선반 기록에 종류가 없거나, 선반 기록이 없는 집기 동작)."""


@dataclass
class Handover:
    """손님끼리 물건 하나를 건넸다 — 전체 카메라가 받은 손님 손에서 본 것 (fusion.handover)."""

    ms: float
    """받은 손님 손에서 처음 본 시각."""
    frame: int
    giver: int
    receiver: int
    name: str
    """상품 종류."""
    paid: bool = False
    """건넬 때 이미 결제된 물건이었나 — 건넨 손님이 그때 가진 것을 모두 결제한 상태였다. 그러면 결제 1개가 물건을 따라간다."""
    direct: bool = False
    """건넨 것이 아니라 받은 손님이 **직접 집은 것** — giver 는 그 물건이 (선반 방문의 주인 정하기로) 적혀 있던 손님일 뿐이다.
    그 손님 손에서 그 종류를 본 적이 없으면 이렇게 본다. 개수는 똑같이 옮기고, 결제 · 확인 필요는 따라가지 않는다."""
    void: bool = False
    """지금 장부로는 성립하지 않는다 (건넨 손님에게 그 물건이 없다 · 두 사람이 한 사람으로 합쳐졌다). rebalance 가 매번 다시 정한다."""

    def to_dict(self) -> dict:
        return {**asdict(self), "ms": round(self.ms, 1)}


@dataclass(frozen=True)
class ShelfAction:
    """손님 한 명의 행동 하나 — 집음 / 놓음 / 자리 바뀜 / 만지기만 / 구경 / 확인 못 함."""

    person_id: int
    kind: str
    count: int
    """가진 물건 수의 변화. 집음 +n, 놓음 -n, 자리 바뀜은 순변화, 만지기만·구경 0, 확인 못 함 +1, 든 것을 봄 +1,
    건넴 -1, 건네받음 +1, 직접 집음 +1, 다른 손님이 집음 -1."""
    start_frame: int
    end_frame: int
    evidence: str
    """선반+동작 / 선반만 / 동작만."""
    shelf: str = ""
    reach_frames: tuple[tuple[int, int], ...] = ()
    """근거가 된 손 뻗기(TAKE 후보)들의 (시작, 끝) 프레임."""
    detail: str = ""

    @property
    def name(self) -> str:
        return KIND_NAMES.get(self.kind, self.kind)

    def describe(self) -> str:
        amount = f" {abs(self.count)}개" if self.kind in ("TAKE", "RETURN") else ""
        where = f" — {self.detail}" if self.detail else ""
        return f"손님 {self.person_id}: {self.name}{amount}{where} ({self.evidence})"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["name"] = self.name
        return d


def classify_visit(visit: ShelfVisit, takes: list[TakeCandidate]) -> ShelfAction:
    """선반 방문 하나를 행동으로. 같은 시간·같은 선반의 손 뻗기를 근거로 붙인다."""
    covered = [t for t in takes if visit.covers(t)]
    count = visit.removed - visit.added
    shelf_word = {"sensor": "센서", "sensor+vision": "센서+선반", "overview+vision": "전체+선반"}.get(visit.source, "선반")
    if visit.removed and visit.added:
        kind = "MOVE"
    elif visit.removed:
        kind = "TAKE"
    elif visit.added:
        kind = "RETURN"
    elif visit.slot_change == "줄어듦":
        # 물건 탐지는 '변화 없음'인데 칸 그림은 비었다 — 모델이 못 보는 물건 (설계 M-7 보조 표)
        kind, count, shelf_word = "TAKE", 1, "칸 변화"
    elif visit.slot_change == "늘어남":
        kind, count, shelf_word = "RETURN", -1, "칸 변화"
    elif visit.slot_change == "바뀜" and covered:
        # 칸은 바뀌었는데 방향을 모른다 — 손 뻗기를 지우지 않고 '확인 못 함'(+1)으로 남긴다 (D-17)
        kind, count, shelf_word = "UNVERIFIED", 1, "칸 변화"
    else:
        kind = "TOUCH" if covered else "LOOK"
    detail = visit.detail.split("] ", 1)[-1].split(" -> ", 1)[0] if visit.detail else ""
    if detail == "변화 없음":
        detail = ""
    return ShelfAction(
        person_id=visit.person_id, kind=kind, count=count,
        start_frame=visit.start_frame, end_frame=visit.end_frame,
        evidence=f"{shelf_word}+동작" if covered else f"{shelf_word}만", shelf=visit.shelf,
        reach_frames=tuple((t.start_frame, t.end_frame) for t in covered), detail=detail,
    )


def _group_by_overlap(takes: list[TakeCandidate]) -> list[list[TakeCandidate]]:
    """시간이 겹치는 TAKE 후보를 한 번의 집기 행동으로 묶는다."""
    groups: list[list[TakeCandidate]] = []
    current_end = None
    for c in sorted(takes, key=lambda c: c.start_frame):
        if current_end is None or c.start_frame > current_end:
            groups.append([c])
            current_end = c.end_frame
        else:
            groups[-1].append(c)
            current_end = max(current_end, c.end_frame)
    return groups


@dataclass
class Customer:
    person_id: int
    first_seen_ms: float
    last_seen_ms: float
    takes: list[TakeCandidate] = field(default_factory=list)
    shelf_visits: list[ShelfVisit] = field(default_factory=list)
    held: list[HeldItem] = field(default_factory=list)
    """전체 카메라가 손에 든 것을 봐서 먼저 센 상품 (아직 선반 방문으로 안 바뀐 것)."""
    paid_items: int = 0
    covered_items: int = 0
    """다른 손님의 남는 결제 · 주인을 못 찾은 결제로 메운 물건 수 (risk.share_surplus_payments, CustomerBook.cover_from_surplus)."""
    gave: list[Handover] = field(default_factory=list)
    """이 손님이 건넨 것 (성립한 것만). CustomerBook.rebalance 가 채운다."""
    got: list[Handover] = field(default_factory=list)
    """이 손님이 건네받은 것 (성립한 것만)."""
    paid_moved: int = 0
    """건넴과 함께 옮겨 온(+) · 옮겨 간(-) 결제 수. paid 에 들어간다."""
    kinds: dict = field(default_factory=dict)
    """지금 든 물건의 종류별 개수 (건넴 반영). 건넴이 하나라도 있을 때만 rebalance 가 채운다."""

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
    held_since_ms: float | None = None
    """나감 판정(HIGH_RISK)을 보류하기 시작한 시각 (risk.exit_hold_seconds). 다시 보이면 버린다."""
    held_how: str = ""
    """보류한 나감의 방식 ("문 밖으로 나감" 등) — 보류가 끝나 내보낼 때 판정 이유에 붙인다."""
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
    dwell_paid_items: int = 0
    """payment_mode="dwell": 계산대 체류로 결제한 물건 수 (취소되면 0). paid_items 에 포함돼 있다."""
    stood_until_ms: float | None = None
    """payment_mode="dwell": 계산대에 dwell_pay_seconds 이상 서 있던 마지막 시각 (늦게 들어온 선반 기록의 결제를 맞춰 보는 데 쓴다)."""
    at_shelf_overlap: float = 0.0
    """지금 SHELF 구역과 몸이 겹친 비율 (가장 큰 값)."""
    seen_inside: bool = False
    """매장 안에서 보인 적이 있는가. 문 밖 행인·문 앞에만 있던 사람은 판정하지 않는다.
    매장 안 = 문 밖 구역이 아닌 곳 (기본), 또는 config.inside_zones 에 inside_seconds 이상 (카메라 2대)."""
    inside_since_ms: float | None = None
    """지금 inside_zones 안에 연속으로 있다면 들어온 시각."""

    last_bbox: BBox | None = None

    @staticmethod
    def _acts(takes: list[TakeCandidate]) -> int:
        """집기 행동 수. 시간이 겹치는 TAKE 후보는 하나로 묶는다."""
        return len(_group_by_overlap(takes))

    @property
    def uncovered_takes(self) -> list[TakeCandidate]:
        """선반 기록이 대신하지 않은 집기 동작 (지켜보지 않는 선반, 또는 선반 기록이 없는 시간)."""
        return [t for t in self.takes
                if not any(v.covers(t) for v in self.shelf_visits) and not any(h.covers(t) for h in self.held)]

    @property
    def shelf_net(self) -> int:
        """선반에서 확인한 순 개수 (가져감 - 되돌려놓음)."""
        return sum(v.removed - v.added for v in self.shelf_visits)

    @property
    def actions(self) -> list[ShelfAction]:
        """이 손님의 행동 목록 (시간순). 선반 방문마다 하나 + 선반 기록이 없는 집기 행동마다 하나 + 건넴 · 건네받음마다 하나."""
        acts = self.own_actions
        for h in self.gave:
            acts.append(ShelfAction(person_id=self.person_id, kind="THEIRS" if h.direct else "GIVE", count=-1,
                                    start_frame=h.frame, end_frame=h.frame, evidence="전체 카메라",
                                    detail=(f"{h.name} — 손님 {h.receiver} 손에서 봄" if h.direct else f"{h.name} → 손님 {h.receiver}")))
        for h in self.got:
            acts.append(ShelfAction(person_id=self.person_id, kind="MINE" if h.direct else "GET", count=1,
                                    start_frame=h.frame, end_frame=h.frame, evidence="전체 카메라",
                                    detail=(f"{h.name} — 손에 든 것을 봄 (손님 {h.giver} 에게 적혀 있던 것)" if h.direct
                                            else f"{h.name} ← 손님 {h.giver}")))
        return sorted(acts, key=lambda a: a.start_frame)

    @property
    def own_actions(self) -> list[ShelfAction]:
        """이 손님이 선반에서 한 일 (건넴 · 건네받음을 뺀 것)."""
        acts = [classify_visit(v, self.takes) for v in self.shelf_visits]
        for group in _group_by_overlap(self.uncovered_takes):
            acts.append(ShelfAction(
                person_id=self.person_id, kind="UNVERIFIED", count=1,
                start_frame=min(t.start_frame for t in group), end_frame=max(t.end_frame for t in group),
                evidence="동작만", shelf=group[0].zone,
                reach_frames=tuple((t.start_frame, t.end_frame) for t in group),
            ))
        for h in self.held:
            acts.append(ShelfAction(person_id=self.person_id, kind="HELD", count=1, start_frame=h.end_frame, end_frame=h.end_frame,
                                    evidence="전체 카메라", detail=h.name))
        return sorted(acts, key=lambda a: a.start_frame)

    @property
    def taken(self) -> int:
        """지금 가진 물건 수(추정) = 행동들의 개수 변화 합.

        = 선반에서 확인한 개수(집음 - 놓음) + 선반 기록이 없는 집기 동작 수 + 건네받은 수 - 건넨 수.
        선반 지도가 꺼져 있으면 집기 동작 수 그대로다.
        """
        return max(0, sum(a.count for a in self.actions))

    @property
    def paid(self) -> int:
        """결제된 물건 수 = 이 손님에게 붙은 결제 + 건넴과 함께 옮겨 온 결제 - 옮겨 간 결제."""
        return max(0, self.paid_items + self.paid_moved)

    def picked_kinds(self, until_frame: int | None = None) -> dict[str, int]:
        """선반에서 집어 든 물건의 종류별 개수 (건넴 반영 전). 종류를 모르면 UNKNOWN_KIND.
        until_frame 을 주면 그 장면까지 시작한 선반 방문 · 든 것 · 집기 동작만 센다."""
        out: dict[str, int] = {}

        def add(name: str, n: int) -> None:
            out[name] = out.get(name, 0) + n

        if until_frame is not None:
            part = replace(self, shelf_visits=[v for v in self.shelf_visits if v.start_frame <= until_frame],
                           held=[h for h in self.held if h.start_frame <= until_frame],
                           takes=[t for t in self.takes if t.start_frame <= until_frame])
            return part.picked_kinds()
        for v in self.shelf_visits:
            if v.removed_kinds or v.added_kinds:
                for name in v.removed_kinds:
                    add(name, 1)
                for name in v.added_kinds:
                    add(name, -1)
                extra = (v.removed - len(v.removed_kinds)) - (v.added - len(v.added_kinds))
                if extra:
                    add(UNKNOWN_KIND, extra)   # 센서 · 전체 카메라 개수로 고친 방문 — 종류 목록보다 개수가 많거나 적다
            else:
                act = classify_visit(v, self.takes)
                if act.count:
                    add(UNKNOWN_KIND, act.count)
        for h in self.held:
            add(h.name, 1)
        groups = len(_group_by_overlap(self.uncovered_takes))
        if groups:
            add(UNKNOWN_KIND, groups)
        minus = sum(-n for n in out.values() if n < 0)   # 놓은 것이 종류가 다르게 적혔으면 모르는 종류에서 뺀다
        out = {name: n for name, n in out.items() if n > 0}
        for name in sorted(out, key=lambda k: k != UNKNOWN_KIND):
            take = min(minus, out[name])
            out[name] -= take
            minus -= take
        return {name: n for name, n in out.items() if n > 0}

    def count_detail(self) -> str:
        """물건 수의 근거 (판정 문구용). 선반 기록이 없으면 빈 문자열."""
        if not self.shelf_visits and not self.held and not self.gave and not self.got:
            return ""
        acts = self.actions
        parts = []
        took = sum(a.count for a in acts if a.kind == "TAKE")
        if took:
            parts.append(f"집음 {took}개")
        held = sum(a.count for a in acts if a.kind == "HELD")
        if held:
            parts.append(f"든 것을 봄 {held}개")
        for direct, items, text in ((True, self.got, "직접 집음 {n}개 ({who} 에게 적혀 있던 것)"), (False, self.got, "건네받음 {n}개 ({who} 에게서)"),
                                    (True, self.gave, "다른 손님이 집음 {n}개 ({who})"), (False, self.gave, "건넴 {n}개 ({who} 에게)")):
            some = [h for h in items if h.direct == direct]
            if some:
                others = sorted({h.giver if items is self.got else h.receiver for h in some})
                parts.append(text.format(n=len(some), who=", ".join(f"손님 {p}" for p in others)))
        put = -sum(a.count for a in acts if a.kind == "RETURN")
        if put:
            parts.append(f"놓음 {put}개")
        moved = [a for a in acts if a.kind == "MOVE"]
        if moved:
            parts.append(f"자리 바뀜 {len(moved)}번(순변화 {sum(a.count for a in moved):+d})")
        touch = sum(1 for a in acts if a.kind == "TOUCH")
        if touch:
            parts.append(f"만지기만 {touch}번")
        look = sum(1 for a in acts if a.kind == "LOOK")
        if look:
            parts.append(f"구경 {look}번")
        unverified = sum(1 for a in acts if a.kind == "UNVERIFIED")
        if unverified:
            parts.append(f"선반으로 확인 못 한 집기 동작 {unverified}번")
        return ", ".join(parts) or "선반 변화 없음"

    @property
    def unpaid(self) -> int:
        """계산하지 않은 물건 수(추정). 결제가 더 많으면 0 (TAKE 를 놓쳤을 수 있다). 남의 남는 결제로 메운 것은 뺀다."""
        return max(0, self.taken - self.paid - self.covered_items)


COVER_RECENT_MS = 500.0
"""다른 손님의 마지막 상자를 '방금 본 것'으로 치는 시간 (risk.return_cover_ignore)."""


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
        self.handovers: list[Handover] = []
        """손님끼리 건넨 기록 (시간순으로 다시 따라간다 — rebalance)."""
        self.changes = 0
        """집은 개수 · 건넴이 바뀐 횟수 (건넴을 다시 따질 때가 됐는지 보는 데 쓴다)."""
        self._had_handovers = False
        self.unassigned_visits: list[ShelfVisit] = []

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
        dst.held = sorted(dst.held + src.held, key=lambda h: h.end_frame)
        dst.paid_items += src.paid_items
        dst.dwell_paid_items += src.dwell_paid_items
        dst.first_seen_ms = min(dst.first_seen_ms, src.first_seen_ms)
        holds = [(c.held_since_ms, c.held_how) for c in (dst, src) if c.held_since_ms is not None]
        if src.last_seen_ms >= dst.last_seen_ms:
            # 지금 보이는 쪽은 합쳐지기 전 번호(src)다. 위치·구역 상태는 그쪽을 따른다.
            dst.last_seen_ms = src.last_seen_ms
            dst.last_bbox = src.last_bbox
            dst.checkout_since_ms = src.checkout_since_ms
            dst.exit_zone, dst.exit_since_ms = src.exit_zone, src.exit_since_ms
            dst.exit_left_ms, dst.exit_judged = src.exit_left_ms, src.exit_judged
            dst.departed = src.departed
            dst.last_exit_ms, dst.outside_since_ms = src.last_exit_ms, src.outside_since_ms
        # 보류한 나감 판정은 어느 쪽 것이든 남긴다 — 매장 안으로 돌아온 게 확인(departed 해제)됐을 때만 버린다
        dst.held_since_ms, dst.held_how = min(holds) if holds and dst.departed else (None, "")
        dst.seen_inside = dst.seen_inside or src.seen_inside
        if src.stood_until_ms is not None and (dst.stood_until_ms is None or src.stood_until_ms > dst.stood_until_ms):
            dst.stood_until_ms = src.stood_until_ms
        for t in (src.last_checkout_ms, dst.last_checkout_ms):
            if t is not None and (dst.last_checkout_ms is None or t > dst.last_checkout_ms):
                dst.last_checkout_ms = t
        self.rebalance()

    # ------------------------------------------------------------------ 건넴

    def hand_over(self, giver: int, receiver: int, name: str, ms: float, frame: int, now_ms: float,
                  direct: bool = False) -> Handover | None:
        """giver 가 receiver 에게 name 하나를 건넸다. 건넨 손님에게 그 물건이 없으면 적지 않는다 (None).
        direct = 건넨 것이 아니라 receiver 가 직접 집은 것 (giver 에게 잘못 적혀 있었다) — 결제는 따라가지 않는다.

        결제됨은 물건과 함께 간다: 건넨 손님이 지금 가진 것을 모두 결제한 상태면(결제 ≥ 가진 수 ≥ 1) 그 물건은 결제된 것 —
        결제 1개가 받은 손님에게 옮겨 간다. 결제 안 한 것이 하나라도 있으면 결제 안 된 물건을 건넨 것으로 본다
        (어느 것을 건넸는지는 모른다. 결제 안 된 쪽으로 봐야 받은 손님 · 건넨 손님 둘 다 확인하게 된다)."""
        g, r = self._get(giver, now_ms), self._get(receiver, now_ms)
        if g is r:
            return None
        item = Handover(ms, frame, g.person_id, r.person_id, name, False if direct else self._paid_then(g, ms, frame), direct)
        self.handovers.append(item)
        self._had_handovers = True
        self.rebalance()
        if item.void:
            self.handovers.remove(item)
            self.rebalance()
            return None
        # payment_mode="dwell": 받은 손님이 그 물건을 든 뒤(ms) 계산대에 섰다 떠났는데, 건넴 · 직접 집음이 그 뒤에야 확인됐다 — 그 물건도
        # 결제한 것이다. 늦게 들어온 선반 방문과 같은 규칙 (_late_dwell_payment). 4차 C13 센서 모드: 손님 1 이 5.5초부터 들고 계산했는데
        # 그 물건은 방문이 끝난 뒤에야 손님 1 의 것으로 옮겨졌다
        if (self.config.payment_mode == "dwell" and not item.paid and r.stood_until_ms is not None
                and r.stood_until_ms >= ms and r.unpaid > 0):
            r.paid_items += 1
            r.dwell_paid_items += 1
            self.payments.append(PaymentRecord(round(now_ms / 1000.0, 3), 1, r.person_id, "dwell", 1))
        return item

    def _paid_then(self, giver: Customer, ms: float, frame: int) -> bool:
        """건넨 순간 건넨 손님이 가진 것을 모두 결제한 상태였나. 건넴은 일어난 뒤 몇 초 늦게 확인된다 — 그 사이의 결제는 치지 않는다.
        결제 기록이 있을 때(feed)는 그 순간까지의 결제 기록으로, 계산대 체류로 어림할 때(dwell)는 지금의 결제 수로 본다
        (체류 결제는 서 있는 동안 늘어나는 것이 기록에 다 남지 않는다)."""
        had = (sum(giver.picked_kinds(frame).values())
               + sum(1 for h in giver.got if h.ms < ms) - sum(1 for h in giver.gave if h.ms < ms))
        if self.config.payment_mode == "dwell":
            paid = giver.paid
        else:
            paid = sum(p.items for p in self.payments
                       if p.person_id is not None and self.resolve(p.person_id) == giver.person_id and p.time_sec * 1000.0 <= ms)
            paid += sum(1 for h in giver.got if h.ms < ms and h.paid) - sum(1 for h in giver.gave if h.ms < ms and h.paid)
        return had >= 1 and paid >= had

    def drop_handover(self, item: Handover) -> None:
        """건넴 하나를 지운다 — 다시 따져 보니 건넴이 아니었다 (받은 손님이 자기 것을 집은 것으로 확인됨 등)."""
        self.handovers = [h for h in self.handovers if h is not item]
        self.rebalance()

    def rebalance(self) -> None:
        """건넴 기록을 시간순으로 다시 따라가 손님별 건넴 · 건네받음 · 옮겨 간 결제를 정한다. 집은 개수(선반 근거)가 바뀔 때마다 부른다.
        건넴이 한 번도 없었으면 아무것도 안 한다."""
        self.changes += 1
        if not self._had_handovers:
            return
        have = {pid: dict(c.picked_kinds()) for pid, c in self.customers.items()}
        for c in self.customers.values():
            c.gave, c.got, c.paid_moved = [], [], 0
        for h in sorted(self.handovers, key=lambda h: h.ms):
            h.giver, h.receiver = self.resolve(h.giver), self.resolve(h.receiver)
            mine = have.get(h.giver)
            if h.giver == h.receiver or mine is None or h.receiver not in have:
                h.void = True
                continue
            key = h.name if mine.get(h.name, 0) > 0 else (UNKNOWN_KIND if mine.get(UNKNOWN_KIND, 0) > 0 else None)
            if key is None:
                h.void = True
                continue
            h.void = False
            mine[key] -= 1
            have[h.receiver][h.name] = have[h.receiver].get(h.name, 0) + 1
            g, r = self.customers[h.giver], self.customers[h.receiver]
            g.gave.append(h)
            r.got.append(h)
            if h.paid:
                g.paid_moved -= 1
                r.paid_moved += 1
        for pid, c in self.customers.items():
            c.kinds = {name: n for name, n in have[pid].items() if n > 0}

    # ------------------------------------------------------------------ 기록

    def add_take(self, candidate: TakeCandidate, now_ms: float) -> Customer:
        customer = self._get(candidate.person_id, now_ms)
        customer.takes.append(candidate)
        self.rebalance()
        return customer

    def add_held(self, item: HeldItem, now_ms: float) -> Customer:
        """전체 카메라가 그 손님 손에 든 상품을 봤다 — 선반 방문을 기다리지 않고 먼저 센다 (fusion.overview_instant)."""
        customer = self._get(item.person_id, now_ms)
        customer.held.append(item)
        self.rebalance()
        return customer

    def swap_held(self, a: int, b: int, since_ms: float, now_ms: float) -> int:
        """since_ms 부터 손에서 본 '든 것'을 손님 a · b 사이에서 서로 옮긴다 — 그때부터 두 사람의 번호가 뒤바뀌어 있었다
        (fusion.swap_since_confusion). 옮긴 개수를 돌려준다."""
        ca, cb = self.customers.get(self.resolve(a)), self.customers.get(self.resolve(b))
        from_a = [h for h in ca.held if h.seen_ms >= since_ms] if ca else []
        from_b = [h for h in cb.held if h.seen_ms >= since_ms] if cb else []
        if not from_a and not from_b:
            return 0
        ca, cb = ca or self._get(a, now_ms), cb or self._get(b, now_ms)
        ca.held = [h for h in ca.held if h.seen_ms < since_ms] + [replace(h, person_id=ca.person_id) for h in from_b]
        cb.held = [h for h in cb.held if h.seen_ms < since_ms] + [replace(h, person_id=cb.person_id) for h in from_a]
        for c in (ca, cb):
            # 한 손님의 '든 것'은 종류마다 하나다 (HolderTracker.fresh) — 옮겨 받아 같은 종류가 둘이 되면 먼저 본 것만 남긴다
            # 물건 단위로 센 것(item 이 있는 것)은 물건마다 따로다 — 같은 종류 둘도 그대로 둔다
            first: dict[str, HeldItem] = {}
            for h in sorted(c.held, key=lambda h: h.seen_ms):
                if h.item is None:
                    first.setdefault(h.name, h)
            c.held = [h for h in c.held if h.item is not None or first[h.name] is h]
        self.rebalance()
        return len(from_a) + len(from_b)

    def rename_held(self, item: int, name: str) -> bool:
        """물건 단위로 센 '든 것'의 종류를 고친다 — 그 물건이 다른 종류로 더 많이 읽혔다 (fusion.item_tracks). 개수는 그대로다."""
        for c in self.customers.values():
            for k, h in enumerate(c.held):
                if h.item == item:
                    c.held[k] = replace(h, name=name)
                    self.rebalance()
                    return True
        return False

    def remove_held(self, item: int) -> bool:
        """물건 단위로 센 '든 것'을 지운다 — 그 물건을 선반에 도로 놓았다 (fusion.item_tracks)."""
        for c in self.customers.values():
            keep = [h for h in c.held if h.item != item]
            if len(keep) != len(c.held):
                c.held = keep
                self.rebalance()
                return True
        return False

    def drop_held(self, start_ms: float, until_ms: float) -> list[HeldItem]:
        """[start_ms, until_ms] 에 손에서 처음 본 '든 것'을 모든 손님에게서 지운다 — 그 시간을 덮는 선반 방문이 대신한다."""
        gone = []
        for c in self.customers.values():
            keep = [h for h in c.held if not start_ms <= h.seen_ms <= until_ms]
            gone += [h for h in c.held if start_ms <= h.seen_ms <= until_ms]
            c.held = keep
        self.rebalance()
        return gone

    def add_shelf_visit(self, visit: ShelfVisit, now_ms: float) -> Customer | None:
        """선반 방문 결과를 그 손님에게 붙인다. 누구 것인지 모르는(번호 0 이하) 변화는 세기만 한다."""
        if visit.person_id <= 0:
            if visit.removed or visit.added:
                self.shelf_unassigned += 1
                self.unassigned_visits.append(visit)
            return None
        customer = self._get(visit.person_id, now_ms)
        customer.shelf_visits.append(visit)
        self.rebalance()
        self._late_dwell_payment(customer, visit, now_ms)
        return customer

    def _late_dwell_payment(self, c: Customer, visit: ShelfVisit, now_ms: float) -> None:
        """payment_mode="dwell": 상품을 손에 든 뒤(held_ms) 계산대에 섰다 떠났는데 선반 기록이 그 뒤에야 들어온 경우 — 그 물건도 결제한 것.

        계산대 체류 결제는 '서 있는 동안 가진 것'만 센다. 두 사람이 함께 선반을 가리면 방문이 늦게 끝나, 먼저 계산하고 떠난 손님의
        물건이 계산 뒤에 장부에 들어온다 (3차 C19). 든 시각을 모르는 방문(held_ms 없음)은 지금처럼 둔다.
        """
        net = visit.removed - visit.added
        if (self.config.payment_mode != "dwell" or visit.held_ms is None or net <= 0
                or c.stood_until_ms is None or c.stood_until_ms < visit.held_ms):
            return
        # 아직 서 있는 손님이어도 여기서 센다 — dwell_payments 는 이미 센 만큼(dwell_paid_items)은 다시 세지 않는다.
        # 이미 결제된 만큼은 뺀다: 손에 든 것을 먼저 세어 둔 물건(HeldItem)은 계산대에서 이미 결제됐을 수 있다 (fusion.overview_instant)
        net = min(net, c.unpaid)
        if net <= 0:
            return
        c.paid_items += net
        c.dwell_paid_items += net
        self.payments.append(PaymentRecord(round(now_ms / 1000.0, 3), net, c.person_id, "dwell", 1))

    def _covers_another(self, obs: IdentityObservation, identities: list[IdentityObservation], now_ms: float) -> bool:
        """이 상자가 다른 손님의 상자를 거의 다 덮는가 — 그 손님에게 겹쳐 생긴 상자다 (risk.return_cover_ignore).

        다른 손님의 상자 = 이번 장면의 것 + 방금(COVER_RECENT_MS 안) 본 마지막 것. 겹친 상자가 생기는 장면에서는
        원래 상자가 한 장씩 빠진다 (4차 L1: 두 번째 장에는 겹친 상자만 있었다).
        """
        need = self.config.return_cover_ignore
        if need <= 0:
            return False
        me = self.resolve(obs.person_id)
        boxes = [o.bbox for o in identities if o.person_id > 0 and self.resolve(o.person_id) != me]
        boxes += [c.last_bbox for c in self.customers.values()
                  if c.person_id != me and not c.departed and c.last_bbox and now_ms - c.last_seen_ms <= COVER_RECENT_MS]
        x0, y0, x1, y1 = obs.bbox
        for a0, b0, a1, b1 in boxes:
            area = (a1 - a0) * (b1 - b0)
            inside = max(0.0, min(x1, a1) - max(x0, a0)) * max(0.0, min(y1, b1) - max(y0, b0))
            if area > 0 and inside / area >= need:
                return True
        return False

    def observe(self, now_ms: float, identities: list[IdentityObservation]) -> list[Customer]:
        """이번 프레임에 보인 손님들의 위치·구역 상태를 갱신한다. 보인 손님 목록을 돌려준다."""
        seen: dict[int, Customer] = {}
        self.reappeared: list[int] = []
        for obs in identities:
            if obs.person_id <= 0:
                continue
            customer = self._get(obs.person_id, now_ms)
            hits = self.zone_map.evaluate(obs.bbox, getattr(obs, "keypoints", None)) if self.zone_map else []
            outside = any(h.zone_type == ZoneType.OUTSIDE and h.contains_foot for h in hits)
            if customer.departed and not outside and self._covers_another(obs, identities, now_ms):
                continue   # 다른 손님에게 겹쳐 생긴 상자에 나간 손님 번호가 잘못 붙었다 — 돌아온 게 아니다 (4차 L1)
            if customer.departed and not outside:
                # '나갔다'고 확정했는데 다시 보인다 — 나간 게 아니라 탐지가 끊겼던 것이다.
                # 판정을 되돌려 다음 출구 방문 때 다시 판정한다.
                customer.departed = False
                customer.exit_judged = False
                customer.exit_since_ms = None
                customer.exit_left_ms = None
                customer.held_since_ms, customer.held_how = None, ""   # 보류하던 나감 판정도 버린다
                self.reappeared.append(customer.person_id)
            customer.last_seen_ms = now_ms
            customer.last_bbox = obs.bbox

            if outside:
                if customer.outside_since_ms is None:
                    customer.outside_since_ms = now_ms
                customer.last_exit_ms = now_ms
            else:
                customer.outside_since_ms = None
                if not self.config.inside_zones:
                    customer.seen_inside = True
            if self.config.inside_zones and not customer.seen_inside:
                inside = any(h.zone in self.config.inside_zones and h.contains_foot for h in hits)
                if inside and self.config.inside_excludes_exit:
                    inside = not any(h.zone_type == ZoneType.EXIT and h.contains_foot for h in hits)
                if not inside:
                    customer.inside_since_ms = None
                else:
                    if customer.inside_since_ms is None:
                        customer.inside_since_ms = now_ms
                    if now_ms - customer.inside_since_ms >= self.config.inside_seconds * 1000.0:
                        customer.seen_inside = True
            in_checkout = any(h.zone_type == ZoneType.CHECKOUT and h.contains_foot for h in hits)
            customer.at_shelf_overlap = max((h.overlap for h in hits if h.zone_type == ZoneType.SHELF), default=0.0)
            exit_hit = next((h for h in hits if h.zone_type == ZoneType.EXIT and h.contains_foot), None)

            grace_ms = self.config.zone_grace_seconds * 1000.0
            if in_checkout:
                if customer.checkout_since_ms is None:
                    customer.checkout_since_ms = now_ms
                customer.last_checkout_ms = now_ms
            elif not (grace_ms > 0 and customer.last_checkout_ms is not None
                      and now_ms - customer.last_checkout_ms <= grace_ms):
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
            elif (grace_ms > 0 and customer.exit_since_ms is not None and customer.last_exit_ms is not None
                  and now_ms - customer.last_exit_ms <= grace_ms):
                pass   # 잠깐 벗어남 — 출구에 머문 시간을 이어서 센다 (zone_grace_seconds)
            else:
                if customer.exit_since_ms is not None:
                    customer.exit_left_ms = now_ms
                customer.exit_since_ms = None
            seen[customer.person_id] = customer
        return list(seen.values())

    def dwell_payments(self, now_ms: float, visible: list[Customer]) -> list[PaymentRecord]:
        """payment_mode="dwell": 계산대에 서 있으면 결제, 결제 뒤 상품 쪽으로 가면 취소."""
        pay_ms = self.config.dwell_pay_seconds * 1000.0
        records = []
        for c in visible:
            if c.checkout_since_ms is not None and now_ms - c.checkout_since_ms >= pay_ms:
                c.stood_until_ms = now_ms
                # 서 있는 동안 늘어난 것(늦게 확정된 선반 기록 등)까지 결제에 포함한다
                # 건넨 물건도 함께 결제한다 — 골라서 일행에게 들려 주고 계산대에 선 손님 (4차 N09, 후속 70)
                owed = max(0, c.taken + self.given_unpaid(c) - (c.paid - c.dwell_paid_items))
                if owed > c.dwell_paid_items:
                    first = c.dwell_paid_items == 0
                    c.paid_items += owed - c.dwell_paid_items
                    c.dwell_paid_items = owed
                    if first:
                        records.append(PaymentRecord(round(now_ms / 1000.0, 3), owed, c.person_id, "dwell", 1))
            elif c.dwell_paid_items and c.at_shelf_overlap >= self.config.cancel_shelf_overlap:
                records.append(PaymentRecord(round(now_ms / 1000.0, 3), -c.dwell_paid_items, c.person_id, "cancel", 1))
                c.paid_items -= c.dwell_paid_items
                c.dwell_paid_items = 0
        self.payments.extend(records)
        return records

    def given_unpaid(self, customer: Customer) -> int:
        """이 손님이 건넨 물건 가운데 받은 손님이 아직 결제하지 않은 것의 수 (돌려받은 것은 뺀다).
        받은 손님마다: (결제 안 된 채 건넨 수 - 돌려받은 수) 와 받은 손님의 모자란 결제 가운데 작은 쪽."""
        if not customer.gave:
            return 0
        net: dict[int, int] = {}
        for h in customer.gave:
            if not h.paid and not h.direct:
                net[h.receiver] = net.get(h.receiver, 0) + 1
        for h in customer.got:
            if h.giver in net and not h.direct:
                net[h.giver] -= 1
        total = 0
        for pid, n in net.items():
            other = self.customers.get(self.resolve(pid))
            if other is not None and n > 0:
                total += min(n, max(0, other.taken - other.paid))
        return total

    def cover_from_surplus(self, customer: Customer, givers_only: bool = False) -> int:
        """남는 결제로 이 손님의 모자란 결제를 메운다 (risk.share_surplus_payments). 메운 개수(이 손님의 covered_items)를 돌려준다.

        남는 결제 = 다른 손님이 자기 물건보다 많이 결제한 만큼 + 결제 순간 계산대에 아무도 없어 주인을 못 찾은 결제.
        결제 로그는 '몇 개를 결제했다'만 알려 주고 누가 했는지는 계산대에 선 사람으로 어림한다 — 일행이 대신 결제하거나(N9),
        결제한 사람과 집은 사람이 다르게 붙으면 한 사람은 결제가 남고 다른 사람은 모자란다. 매장 전체로는 맞는 결제다.
        부를 때마다 다시 센다 (그 사이 결제 · 물건이 달라졌을 수 있다). 다른 손님이 이미 메워 간 만큼은 빼고 남은 것만 쓴다."""
        customer.covered_items = 0
        need = customer.unpaid
        if need <= 0:
            return 0
        if givers_only:
            # 건넨 손님의 남는 결제만 — 건넨 손님이 그 물건 값을 냈다 (남는 결제를 매장 전체로 나누지 않을 때 · 계산대 체류로 어림할 때)
            givers = {self.resolve(h.giver) for h in customer.got if not h.paid and not h.direct}
            spare = sum(max(0, c.paid - c.taken) for pid, c in self.customers.items() if pid in givers and c is not customer)
            spare -= sum(c.covered_items for c in self.customers.values() if c is not customer)
            customer.covered_items = max(0, min(need, spare))
            return customer.covered_items
        spare = sum(max(0, c.paid - c.taken) for c in self.customers.values() if c is not customer)
        spare += sum(p.items for p in self.payments if p.method == "unmatched")
        spare -= sum(c.covered_items for c in self.customers.values() if c is not customer)
        customer.covered_items = max(0, min(need, spare))
        return customer.covered_items

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
