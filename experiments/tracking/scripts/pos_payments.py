"""4차 촬영(2026-11-19) 실제 결제 기록 → 영상별 결제 라벨.

3차까지는 결제 기록이 없어 계산대 체류(dwell)나 바코드 삑 소리 초안(beep_payments.py)을 썼다.
4차는 결제 서버(experiments/phone-scan-checkout)가 남긴 `source/3(log)/payment-log.csv` 가 있다 —
주문마다 결제 시각(초 단위 실제 시각)과 품목·수량. 이것을 영상 시각으로 옮겨 `label_payments.py` 와 같은 형식으로 쓴다.

    python scripts/pos_payments.py                      # → labels/store01_shoot4/payments.csv, pos_orders.csv
    python scripts/pos_payments.py --clock-offset 1.5   # 결제 서버 시계가 휴대폰보다 1.5초 빠를 때 (shoot4_setup.py pay-clock 로 확인)

시각 옮기기 (split_sensor_events.py 와 같은 방식):
    영상 시작 = mp4 기록 시각(촬영이 끝난 시각) − 영상 길이,   주문의 영상 시각 = 결제 시각 − 영상 시작 − clock_offset
결제 시각 축은 전체 카메라(angle1) 재생 시각 = 공통 시각 (src/risk/payments.py) — 카메라 시차는 필요 없다.

배정 규칙:
- 주문 하나 = 결제 한 건 (items = 그 주문의 수량 합, result=paid, checked_by=pos-log).
- 영상 시각이 **[0, 마지막 처리 장면 시각]** 안인 주문만 그 영상에 붙인다. 판정은 마지막 처리 장면까지만 돌고 그 뒤 결제는
  소비하지 않으므로(dual_eval.judge_pair → RiskEngine.update), 영상 끝 뒤 주문을 paid 로 적으면 적혀만 있고 판정에는 안 들어간다.
- 그 밖의 주문(영상 사이·영상 끝 뒤)은 붙이지 않고 pos_orders.csv 에 사유와 함께 남긴다. 끝에서 2초 안이면 붙이되 '확인 필요'.
- 주문이 없는 영상은 `attempt 0 · result none` 한 줄 (dual_eval --payments 가 '라벨 없음'으로 멈추지 않게).
- 사람이 확인한 줄(checked_by 가 pos-log 가 아닌 것)이 있는 짝은 덮어쓰지 않는다.

정밀도: 결제 시각과 mp4 기록 시각이 모두 초 단위이고 두 기기 시계 차이는 재지 않았다 (±1~2초).
결제를 손님에게 붙이는 규칙은 '그 순간 계산대 구역에 있거나 3초 안에 있었던 손님'(risk.checkout_grace_seconds)이라
결제가 늦게 찍히는 쪽은 견디지만 이르게 찍히면 못 붙는다 → 첫 판정 전에 shoot4_setup.py pay-clock 로 시계 차이를 확인한다.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

SOURCE = ROOT.parent.parent / "source"
LOG = SOURCE / "3(log)" / "payment-log.csv"
VIDEOS = SOURCE / "3(angle1)"
LABELS = ROOT / "labels" / "store01_shoot4"
TIMES = ROOT / "outputs" / "dual_eval_shoot4"
"""실제 장면 시각 캐시 (<짝>/times_angle1.json) — dual_eval --shoot 4 가 쓰는 것과 같은 파일."""
STRIDE = 4
"""전체 카메라 처리 간격 (dual_eval.STRIDE["angle1"])."""
NEAR_END = 2.0
"""마지막 처리 장면에서 이 초 안에 든 결제는 붙이되 확인 필요로 표시."""
CHECKED_BY = "pos-log"
KST = dt.timezone(dt.timedelta(hours=9))
ORDER_FIELDS = ["order", "paid_at", "items", "lines", "pair", "time_sec", "status", "note"]


@dataclass
class Order:
    number: int
    paid_at: dt.datetime
    items: int = 0
    lines: list[str] = field(default_factory=list)


@dataclass
class Span:
    """한 영상의 실제 시각 구간."""
    pair: str
    start: dt.datetime
    """영상 0초의 실제 시각 (휴대폰 시계)."""
    last_sec: float
    """마지막 처리 장면 시각 — 이 뒤의 결제는 판정이 소비하지 않는다."""
    length: float


def read_orders(path: Path) -> list[Order]:
    """payment-log.csv 를 주문 단위로 묶는다. 품목 줄마다 수량을 더한다 (주문수량합 열과 다르면 품목 합을 쓴다)."""
    orders: dict[int, Order] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            number = int(r["주문번호"])
            when = dt.datetime.strptime(r["결제시각"].strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)
            o = orders.setdefault(number, Order(number, when))
            qty = int(r["수량"])
            o.items += qty
            o.lines.append(f"{r['상품명'].strip()}×{qty}")
    return sorted(orders.values(), key=lambda o: (o.paid_at, o.number))


def last_processed_sec(times_ms: list[float], stride: int = STRIDE) -> float:
    """stride 장면마다 하나씩 처리할 때 마지막으로 처리되는 장면의 시각(초)."""
    if not times_ms:
        return 0.0
    return times_ms[((len(times_ms) - 1) // stride) * stride] / 1000.0


def video_spans(folder: Path = VIDEOS, times_root: Path = TIMES) -> list[Span]:
    from shoot3_eval import cached_times
    from split_sensor_events import read_mvhd

    spans = []
    for path in sorted(folder.glob("*.mp4")):
        if path.stem == "area_setting":
            continue
        end, length = read_mvhd(path)
        times = cached_times(path, times_root / path.stem / "times_angle1.json")
        spans.append(Span(path.stem, end - dt.timedelta(seconds=length), last_processed_sec(times), length))
    return spans


def assign(orders: list[Order], spans: list[Span], clock_offset: float = 0.0) -> list[dict]:
    """주문마다 배정 결과 한 줄. status: assigned / near-end(붙였지만 확인 필요) / after-end / between / outside."""
    if not spans:
        return []
    first = min(s.start for s in spans)
    last = max(s.start + dt.timedelta(seconds=s.length) for s in spans)
    rows = []
    for o in orders:
        row = {"order": o.number, "paid_at": o.paid_at.strftime("%H:%M:%S"), "items": o.items, "lines": " + ".join(o.lines),
               "pair": "", "time_sec": "", "status": "", "note": ""}
        hit = None
        for s in spans:
            t = (o.paid_at - s.start).total_seconds() - clock_offset
            if 0.0 <= t <= s.last_sec:
                hit = (s, t)
                break
        if hit is not None:
            s, t = hit
            near = s.last_sec - t <= NEAR_END
            row.update(pair=s.pair, time_sec=f"{t:.2f}", status="near-end" if near else "assigned",
                       note=f"영상 끝 {s.last_sec - t:.1f}초 전 — 확인 필요" if near else "")
        elif o.paid_at < first - dt.timedelta(minutes=5) or o.paid_at > last + dt.timedelta(minutes=5):
            row.update(status="outside", note="촬영 시간 밖")
        else:
            late = [(s, (o.paid_at - s.start).total_seconds() - clock_offset) for s in spans]
            late = [(s, t) for s, t in late if s.last_sec < t <= s.length + 10.0]
            if late:
                s, t = late[0]
                row.update(status="after-end", time_sec=f"{t:.2f}",
                           note=f"{s.pair} 마지막 처리 장면({s.last_sec:.1f}초) 뒤 {t - s.last_sec:.1f}초 — 붙이지 않음, 확인 필요")
            else:
                row.update(status="between", note="영상 사이 — 어느 영상에도 안 듦")
        rows.append(row)
    return rows


def to_labels(assigned: list[dict], pairs: list[str]) -> dict:
    """배정 결과 → label_payments.PaymentLabel 목록 (짝마다). 주문 없는 짝은 none 한 줄."""
    from label_payments import PaymentLabel, renumber

    out = {p: [] for p in pairs}
    for r in assigned:
        if r["status"] not in ("assigned", "near-end"):
            continue
        note = f"주문 {r['order']} ({r['paid_at']}) {r['lines']}" + (f"; {r['note']}" if r["note"] else "")
        out[r["pair"]].append(PaymentLabel(r["pair"], 0, float(r["time_sec"]), int(r["items"]), "paid", CHECKED_BY, note))
    for pair, rows in out.items():
        out[pair] = renumber(rows) if rows else [PaymentLabel(pair, 0, None, 0, "none", CHECKED_BY, "결제 기록에 주문 없음")]
    return out


def merge(existing: dict, new: dict) -> tuple[dict, list[str]]:
    """사람이 확인한 짝(checked_by 가 pos-log 가 아닌 줄이 있는 짝)은 그대로 두고 나머지만 새 값으로. (합친 것, 그대로 둔 짝)."""
    kept = [p for p, rows in existing.items() if any(r.checked_by != CHECKED_BY for r in rows)]
    out = dict(new)
    for p in kept:
        out[p] = existing[p]
    for p in existing:   # 이번 영상 목록에 없는 짝의 줄도 지우지 않는다
        out.setdefault(p, existing[p])
    return out, sorted(kept)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log", type=Path, default=LOG)
    ap.add_argument("--videos", type=Path, default=VIDEOS, help="전체 카메라(angle1) 영상 폴더")
    ap.add_argument("--out", type=Path, default=LABELS)
    ap.add_argument("--clock-offset", type=float, default=0.0, metavar="초", help="결제 서버 시계 − 휴대폰 시계 (기본 0)")
    args = ap.parse_args(argv)
    from label_payments import load_labels, save_labels

    orders = read_orders(args.log)
    spans = video_spans(args.videos)
    rows = assign(orders, spans, args.clock_offset)
    labels, kept = merge(load_labels(args.out / "payments.csv"), to_labels(rows, [s.pair for s in spans]))
    save_labels(labels, args.out / "payments.csv")
    with (args.out / "pos_orders.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ORDER_FIELDS)
        w.writeheader()
        w.writerows(rows)
    count = {k: sum(r["status"] == k for r in rows) for k in ("assigned", "near-end", "after-end", "between", "outside")}
    print(f"주문 {len(rows)}건: 배정 {count['assigned'] + count['near-end']} (그중 끝 무렵 {count['near-end']}) · "
          f"영상 끝 뒤 {count['after-end']} · 영상 사이 {count['between']} · 촬영 시간 밖 {count['outside']}  (시계 차이 {args.clock_offset:+.1f}초)")
    by_pair: dict[str, list[dict]] = {}
    for r in rows:
        if r["pair"]:
            by_pair.setdefault(r["pair"], []).append(r)
    for s in spans:
        got = by_pair.get(s.pair, [])
        text = ", ".join(f"{r['time_sec']}초 {r['items']}개" + (" (끝 무렵)" if r["status"] == "near-end" else "") for r in got)
        print(f"  {s.pair:6s} {s.start.astimezone(KST):%H:%M:%S} 길이 {s.length:5.1f}초  {text or '(결제 없음)'}")
    for r in rows:
        if r["status"] in ("after-end", "between"):
            print(f"  ? 주문 {r['order']} {r['paid_at']} {r['items']}개 {r['lines']} — {r['note']}")
    if kept:
        print(f"사람이 확인한 짝은 그대로 둠: {', '.join(kept)}")
    print("주의: 결제 서버 시계와 휴대폰 시계 차이는 재지 않았다 (±1~2초) — 첫 판정 전에 shoot4_setup.py pay-clock")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
