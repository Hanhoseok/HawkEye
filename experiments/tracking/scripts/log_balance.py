"""센서 로그와 결제 로그만으로 영상마다 '꺼낸 것 - 결제한 것'을 센다 — 카메라 없이 보는 정답 검산 (4차부터).

    python scripts/log_balance.py --shoot 4            # 화면에 표 + labels/store01_shoot4/log_balance.csv

보는 것 (레인 센서를 믿는 레인만 — 기본 L1 · L2):
  - 센서: 영상 안의 REMOVE - ADD (종류별). 영상 밖 기록(in_video=0, 촬영 뒤 채워 넣기)은 안 센다.
  - 결제: labels/<회차>/orders.csv 의 종류별 수량.
  - 남음 = 꺼냄 - 결제 (0 아래는 0). 남음이 있으면 '로그로는 결제 안 된 물건이 있다'.
센서를 못 믿는 레인(L3 애플사이다)은 결제 수량만 적는다 — 꺼낸 개수는 카메라만 안다.

정답표(truth.csv)와 어긋나는 영상을 함께 알린다: 정답이 CLEAR 인데 남음이 있거나, HIGH_RISK 인데 남음이 없고 믿는 레인만 다룬 영상.
센서가 꺼냄을 놓치거나(3차 T05) 영상이 끝난 뒤 적으면(4차 S1) 남음이 실제보다 적게 나온다 — 검산이지 정답이 아니다.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import shoots  # noqa: E402

FIELDS = ("pair", "taken", "paid", "unpaid", "other_paid", "late_sensor", "expected", "check")


def sensor_net(path: Path, lanes: tuple[str, ...]) -> tuple[Counter, int]:
    """(영상 안에서 종류별로 꺼낸 개수 - 되돌린 개수, 영상이 끝난 뒤의 REMOVE 수)."""
    net, late = Counter(), 0
    if not path.exists():
        return net, late
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["event"] not in ("REMOVE", "ADD") or r["lane"] not in lanes:
                continue
            if str(r.get("in_video", "1")).strip() != "1":
                late += r["event"] == "REMOVE"
                continue
            net[r["lane"]] -= int(float(r["delta"]))
    return net, late


def balance(labels: Path, pairs: list[str], lanes: tuple[str, ...]) -> list[dict]:
    paid: dict[str, Counter] = {}
    orders = labels / "orders.csv"
    if orders.exists():
        with orders.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                paid.setdefault(r["pair"], Counter())[r["lane"] or "?"] += int(r["quantity"])
    truth = {}
    if (labels / "truth.csv").exists():
        with (labels / "truth.csv").open(encoding="utf-8", newline="") as f:
            truth = {r["pair"]: r["expected"] for r in csv.DictReader(f)}
    rows = []
    for pair in pairs:
        net, late = sensor_net(labels / "sensor" / f"{pair}.csv", lanes)
        mine = paid.get(pair, Counter())
        unpaid = {lane: max(0, net[lane]) - mine[lane] for lane in lanes if max(0, net[lane]) > mine[lane]}
        other = {lane: n for lane, n in mine.items() if lane not in lanes}
        expected = truth.get(pair, "")
        check = ""
        if expected == "CLEAR" and unpaid:
            check = "정답은 CLEAR 인데 로그로는 결제 안 된 물건이 있음"
        elif expected == "HIGH_RISK" and not unpaid and not late:
            check = "로그로는 남는 물건 없음 — 센서 없는 레인의 물건이거나 센서가 놓침 (카메라로 확인)"
        elif not expected and unpaid:
            check = "정답 없음 · 로그로는 결제 안 된 물건이 있음"
        text = lambda c: " ".join(f"{lane}×{n}" for lane, n in sorted(c.items()) if n) or "-"   # noqa: E731
        rows.append({"pair": pair, "taken": text({lane: max(0, net[lane]) for lane in lanes}),
                     "paid": text({lane: mine[lane] for lane in lanes}), "unpaid": text(unpaid), "other_paid": text(other),
                     "late_sensor": late, "expected": expected, "check": check})
    return rows


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shoot", type=int, choices=sorted(shoots.SHOOTS), default=shoots.LATEST)
    ap.add_argument("--lanes", default="L1,L2", help="센서를 믿는 레인 (기본 L1,L2)")
    args = ap.parse_args()
    shoot = shoots.get(args.shoot)
    lanes = tuple(x.strip() for x in args.lanes.split(","))
    rows = balance(shoot.labels, shoot.pairs(), lanes)
    out = shoot.labels / "log_balance.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{'짝':6s} {'꺼냄(센서)':14s} {'결제':14s} {'남음':10s} {'그 밖 결제':10s} 정답        확인")
    for r in rows:
        print(f"{r['pair']:6s} {r['taken']:14s} {r['paid']:14s} {r['unpaid']:10s} {r['other_paid']:10s} {r['expected'] or '?':10s}  {r['check']}"
              + (f" (영상 끝난 뒤 센서 꺼냄 {r['late_sensor']}건)" if r["late_sensor"] else ""))
    flagged = [r["pair"] for r in rows if r["check"]]
    print(f"확인할 영상 {len(flagged)}편: {', '.join(flagged) or '-'}  → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
