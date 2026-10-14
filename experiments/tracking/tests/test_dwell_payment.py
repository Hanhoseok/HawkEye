"""결제 기록 없는 매장 — 계산대에 서 있으면 결제, 다시 상품 쪽으로 가면 취소."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import RiskConfig  # noqa: E402
from src.core.types import Zone, ZoneType  # noqa: E402
from src.risk.engine import RiskEngine  # noqa: E402
from src.risk.payments import PaymentFeed  # noqa: E402
from src.zones.zone_map import ZoneMap  # noqa: E402
from test_risk import AISLE, CHECKOUT, EXIT, C, H, at, levels, nobody, person, take  # noqa: E402

SHELF_X = 20.0   # 선반 앞 (왼쪽 위 선반 구역과 겹침)


def zones() -> ZoneMap:
    zm = ZoneMap([
        Zone("shelf", ZoneType.SHELF, ((0, 0), (120, 0), (120, 400), (0, 400))),
        Zone("counter", ZoneType.CHECKOUT, ((200, 250), (400, 250), (400, 320), (200, 320))),
        Zone("door", ZoneType.EXIT, ((200, 340), (400, 340), (400, 400), (200, 400))),
    ])
    zm.resolve(400, 400)
    return zm


def engine() -> RiskEngine:
    return RiskEngine(RiskConfig(payment_mode="dwell"), zones(), PaymentFeed.empty())


def stand(eng, foot_y, x, start, seconds):
    out = []
    for i in range(int(round(seconds * 10))):
        ev, _ = eng.update(at(start + i * 0.1), [person(1, foot_y, x=x)])
        out += ev
    return out


def leave(eng, start):
    return stand(eng, EXIT, 250.0, start, 1.0) + nobody(eng, start + 1.0, 4.0)


def test_standing_at_counter_pays():
    eng = engine()
    eng.update(at(1.0), [person(1, AISLE, x=SHELF_X)], takes=[take(1)])
    stand(eng, CHECKOUT, 250.0, 2.0, 2.0)
    assert levels(leave(eng, 5.0)) == [C]
    assert [p.method for p in eng.book.payments] == ["dwell"]


def test_brief_pass_by_counter_is_not_payment():
    """계산대 앞을 1초 스쳐 지나가면 결제가 아니다."""
    eng = engine()
    eng.update(at(1.0), [person(1, AISLE, x=SHELF_X)], takes=[take(1)])
    stand(eng, CHECKOUT, 250.0, 2.0, 1.0)
    assert levels(leave(eng, 4.0))[-1] == H


def test_going_back_to_shelf_cancels_payment():
    """계산대에 섰다가 다시 상품 쪽으로 가면 취소 — 그대로 나가면 HIGH_RISK."""
    eng = engine()
    eng.update(at(1.0), [person(1, AISLE, x=SHELF_X)], takes=[take(1)])
    stand(eng, CHECKOUT, 250.0, 2.0, 2.0)
    stand(eng, AISLE, SHELF_X, 4.5, 1.0)
    assert levels(leave(eng, 6.0))[-1] == H
    assert [p.method for p in eng.book.payments] == ["dwell", "cancel"]


def test_cancel_then_pay_again():
    """취소한 뒤 다시 계산대에 서면 다시 결제."""
    eng = engine()
    eng.update(at(1.0), [person(1, AISLE, x=SHELF_X)], takes=[take(1)])
    stand(eng, CHECKOUT, 250.0, 2.0, 2.0)
    stand(eng, AISLE, SHELF_X, 4.5, 1.0)
    stand(eng, CHECKOUT, 250.0, 6.0, 2.0)
    assert levels(leave(eng, 8.5)) == [C]


def test_feed_mode_ignores_dwell():
    """기본(결제 기록 모드)에서는 계산대에 서 있기만 해서는 결제가 아니다."""
    eng = RiskEngine(RiskConfig(), zones(), PaymentFeed.empty())
    eng.update(at(1.0), [person(1, AISLE, x=SHELF_X)], takes=[take(1)])
    stand(eng, CHECKOUT, 250.0, 2.0, 2.0)
    assert levels(leave(eng, 5.0))[-1] == H
