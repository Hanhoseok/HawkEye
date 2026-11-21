"""실제 결제 로그(POS)를 영상별 결제 기록으로 나눈다 — 4차 촬영부터.

    python scripts/pos_payments.py --shoot 4                       # labels/store01_shoot4/payments.csv · orders.csv
    python scripts/pos_payments.py --shoot 4 --log <payment-log.csv>

결제 로그 (source/sensor_payment_log/payment-log.csv, 품목마다 한 줄):
    주문번호,결제시각,계산대ID,품목순번,바코드,상품명,단가,수량,품목금액,주문총액,주문수량합

영상 시각으로 옮기기:
  1. 영상 시작 = mp4 기록 시각(녹화 끝) - 길이 (전체 카메라 angle1 = 공통 시각). 결제 시각 - 영상 시작 = 로그로 본 영상 시각 (초 단위).
  2. 영상 소리에서 결제 완료 소리(약 984Hz → 1312Hz 두 음, src/inputs/audio_read.chimes)를 찾아 주문과 차례로 짝짓는다.
     짝이 지어지면 결제 시각 = 그 소리가 난 때 (16ms 칸). 4차에서는 소리가 로그 시각보다 2.3~2.5초 늦게 난다 (기기 시계 차이 + 처리 시간).
  3. 소리를 못 찾은 주문은 로그 시각 + 그 촬영의 가운데 늦음으로 적고 note 에 남긴다.

결과:
  payments.csv  pair,attempt,time_sec,items,result,checked_by,note   (scripts/label_payments.py 와 같은 모양 — 판정이 읽는다)
                결제가 없는 영상은 result=none 한 줄. checked_by = pos-log.
  orders.csv    pair,order,attempt,time_sec,log_time,barcode,product,lane,quantity,unit_price   (품목별 — 종류까지 맞춰 볼 때 쓸 재료)

영상 밖의 결제(촬영 사이에 한 시험 결제 등)는 어느 영상에도 넣지 않고 화면에만 알린다.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import shoots  # noqa: E402
from split_sensor_events import KST, read_mvhd  # noqa: E402
from src.inputs import audio_read  # noqa: E402

CHECKED_BY = "pos-log"
LOG_NAME = "payment-log.csv"
PRODUCT_LANE = {"칸타타": "L1", "동과루": "L2", "애플사이다": "L3"}
"""상품 이름에 이 낱말이 들어 있으면 그 레인 (4차 진열)."""
PAD_BEFORE, PAD_AFTER = 1.0, 3.0
"""영상 구간을 이만큼 넓혀 주문을 넣는다 (시계가 초 단위)."""
LAG_RANGE = (0.0, 6.0)
"""결제 소리 - 로그 시각 이 이 안이어야 짝으로 본다 (초)."""
PAY_FIELDS = ("pair", "attempt", "time_sec", "items", "result", "checked_by", "note")
ORDER_FIELDS = ("pair", "order", "attempt", "time_sec", "log_time", "barcode", "product", "lane", "quantity", "unit_price")


@dataclass
class Order:
    number: str
    time: dt.datetime
    lines: list[dict] = field(default_factory=list)

    @property
    def items(self) -> int:
        return sum(int(line["수량"]) for line in self.lines)

    def text(self) -> str:
        return ", ".join(f"{line['상품명']}×{line['수량']}" for line in self.lines)


def load_orders(path: Path) -> list[Order]:
    orders: dict[str, Order] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            when = dt.datetime.strptime(r["결제시각"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)
            orders.setdefault(r["주문번호"], Order(r["주문번호"], when)).lines.append(r)
    return sorted(orders.values(), key=lambda o: o.time)


def lane_of(product: str) -> str:
    return next((lane for word, lane in PRODUCT_LANE.items() if word in product), "")


def match_chimes(log_secs: list[float], chimes: list[float]) -> list[float | None]:
    """주문(로그로 본 영상 시각)마다 짝지은 결제 소리 시각. 시간 순서를 지키며, 늦음이 LAG_RANGE 안인 것만."""
    out: list[float | None] = []
    k = 0
    for t in log_secs:
        while k < len(chimes) and chimes[k] - t < LAG_RANGE[0]:
            k += 1
        if k < len(chimes) and chimes[k] - t <= LAG_RANGE[1]:
            out.append(chimes[k])
            k += 1
        else:
            out.append(None)
    return out


def split(shoot: shoots.Shoot, log: Path) -> tuple[list[dict], list[dict], list[Order], list[float]]:
    orders = load_orders(log)
    used: set[str] = set()
    found: list[tuple[str, list[tuple[Order, float, float | None]], float]] = []
    lags: list[float] = []
    for pair in shoot.pairs():
        _a0, a1 = shoot.video(pair)
        end, duration = read_mvhd(a1)
        start = end.astimezone(KST) - dt.timedelta(seconds=duration)
        inside = [o for o in orders if -PAD_BEFORE <= (o.time - start).total_seconds() <= duration + PAD_AFTER
                  and o.number not in used]
        used |= {o.number for o in inside}
        secs = [(o.time - start).total_seconds() for o in inside]
        heard = match_chimes(secs, audio_read.chimes(audio_read.read_audio(a1))) if inside else []
        lags += [h - s for h, s in zip(heard, secs) if h is not None]
        found.append((pair, list(zip(inside, secs, heard)), duration))
        print(f"  {pair:6s} " + ("; ".join(f"#{o.number} {o.text()} 로그 {s:.0f}s → " + (f"소리 {h:.2f}s" if h is not None else "소리 없음")
                                           for o, s, h in found[-1][1]) or "결제 없음"), flush=True)
    lag = float(np.median(lags)) if lags else 0.0
    pays, items = [], []
    for pair, rows, duration in found:
        if not rows:
            pays.append({"pair": pair, "attempt": 0, "time_sec": "", "items": 0, "result": "none", "checked_by": CHECKED_BY,
                         "note": "결제 로그에 이 영상 시간의 주문 없음"})
        for n, (o, s, h) in enumerate(rows, 1):
            t = h if h is not None else min(max(s + lag, 0.0), duration)
            note = f"주문 #{o.number} {o.text()}; 로그 {o.time:%H:%M:%S} (영상 {s:.0f}s); " + (
                f"결제 소리 {h:.2f}s" if h is not None else f"결제 소리 못 찾음 — 로그 시각 + {lag:.1f}s")
            pays.append({"pair": pair, "attempt": n, "time_sec": f"{t:.2f}", "items": o.items, "result": "paid",
                         "checked_by": CHECKED_BY, "note": note})
            for line in o.lines:
                items.append({"pair": pair, "order": o.number, "attempt": n, "time_sec": f"{t:.2f}", "log_time": f"{o.time:%H:%M:%S}",
                              "barcode": line["바코드"], "product": line["상품명"], "lane": lane_of(line["상품명"]),
                              "quantity": line["수량"], "unit_price": line["단가"]})
    return pays, items, [o for o in orders if o.number not in used], lags


def write(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shoot", type=int, choices=sorted(shoots.SHOOTS), default=shoots.LATEST)
    ap.add_argument("--log", type=Path, help=f"결제 로그 (기본 source/sensor_payment_log/{LOG_NAME})")
    args = ap.parse_args()
    shoot = shoots.get(args.shoot)
    log = args.log or shoots.SOURCE / "sensor_payment_log" / LOG_NAME
    pays, items, outside, lags = split(shoot, log)
    write(shoot.labels / "payments.csv", PAY_FIELDS, pays)
    write(shoot.labels / "orders.csv", ORDER_FIELDS, items)
    paid = [p for p in pays if p["result"] == "paid"]
    print(f"결제 {len(paid)}건 ({len({p['pair'] for p in paid})}편), 결제 없는 영상 {sum(p['result'] == 'none' for p in pays)}편 "
          f"→ {shoot.labels / 'payments.csv'}")
    if lags:
        print(f"결제 소리로 맞춘 주문 {len(lags)}/{len(paid)}건 — 소리가 로그보다 늦은 시간: 가운데 {np.median(lags):.2f}s "
              f"(가장 작음 {min(lags):.2f}s, 가장 큼 {max(lags):.2f}s)")
    if outside:
        print("영상 밖의 결제 (어느 영상에도 안 넣음): " + ", ".join(f"#{o.number} {o.time:%H:%M:%S} {o.text()}" for o in outside))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
