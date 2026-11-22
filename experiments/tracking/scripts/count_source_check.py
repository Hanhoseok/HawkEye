"""선반 방문의 개수를 어느 카메라에서 가져올지 고른다 — 정상(N) 영상에서 선반 지도와 전체 카메라 개수를 정답과 견준다.

    python scripts/count_source_check.py --shoot 4                # 정상(N) 영상만 (기준을 고를 때)
    python scripts/count_source_check.py --shoot 4 --only "*"     # 모든 영상 (고른 뒤 확인)
    python scripts/count_source_check.py --shoot 4 --out <폴더>   # 다른 기록 폴더 (선반 지도 설정을 바꿔 다시 돌린 것)

카메라 자리가 바뀌면 어느 카메라가 상품을 잘 세는지가 바뀐다 (scripts/shoots.py 의 count_source):
  3차 — 선반 카메라가 푸셔 줄을 앞에서 봐 뒤 상품이 겹쳐 보였고, 전체 카메라가 옆에서 다 봤다 → overview
  4차 — 전체 카메라가 칸타타 2개 중 1개만 보고 종류를 섞어 봤고, 선반 지도가 맞았다 → shelf

영상마다 종류별 '꺼낸 개수 - 놓은 개수'(영상 전체)를 세 가지로 구해 견준다:
  - 정답: 4차 = 센서 로그(L1 · L2, 영상 안) + 결제 로그의 애플사이다 수량 (정상 영상은 꺼낸 것을 다 결제한다 — 센서가 못 믿는 L3 는 이렇게만 안다)
          3차 = 사람이 찍은 labels/store01_shoot3/takes.csv
  - 선반 지도: 선반 카메라 방문 기록(shelf_visits.jsonl)의 종류별 합
  - 전체 카메라: 판정과 같은 세기(src/fusion/overview_count.py)를 방문마다 돌린 합 — 못 센 방문은 선반 지도 것 (판정이 하는 대로)
정상 영상만으로 고른다 (기준값 규칙). 영상 전체의 합만 본다 — 방문별 · 사람별은 안 본다.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dual_eval  # noqa: E402
import product_overview_items as poi  # noqa: E402
import shoots  # noqa: E402
from src.fusion.shelf_events import visit_kinds  # noqa: E402
from src.risk.customers import ShelfVisit  # noqa: E402

KINDS = tuple(shoots.LANE_KIND.values())


def truth_from_logs(labels: Path, pair: str) -> Counter:
    """4차: 센서(L1 · L2) 영상 안 순 꺼냄 + 결제 로그의 L3 수량."""
    net: Counter = Counter()
    path = labels / "sensor" / f"{pair}.csv"
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if r["event"] in ("REMOVE", "ADD") and r["lane"] in ("L1", "L2") and str(r.get("in_video", "1")).strip() == "1":
                    net[shoots.LANE_KIND[r["lane"]]] -= int(float(r["delta"]))
    with (labels / "orders.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["pair"] == pair and r["lane"] == "L3":
                net[shoots.LANE_KIND["L3"]] += int(r["quantity"])
    return net


def truth_from_takes(labels: Path, pair: str) -> Counter:
    """3차: 사람이 찍은 꺼냄 · 되돌림."""
    net: Counter = Counter()
    with (labels / "takes.csv").open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["pair"] == pair and (r.get("time_sec") or "").strip() and r["lane"] in shoots.LANE_KIND:
                net[shoots.LANE_KIND[r["lane"]]] += int(r.get("qty") or 1) * (1 if r["action"] == "take" else -1)
    return net


def camera_counts(folder: Path, offset: float) -> tuple[Counter, Counter, int, int]:
    """(선반 지도 순 꺼냄, 전체 카메라 순 꺼냄 — 못 센 방문은 선반 지도, 방문 수, 전체 카메라가 센 방문 수)."""
    raw = [json.loads(line) for line in (folder / "angle0" / "shelf_visits.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    times = json.loads((folder / f"times_{poi.CAMERA}.json").read_text(encoding="utf-8"))
    people: dict[int, list] = {}
    for line in (folder / poi.CAMERA / "identities.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            o = json.loads(line)
            people.setdefault(o["frame"], []).append(o["bbox"])
    rows = [json.loads(line) for line in (folder / poi.CAMERA / poi.NAME).read_text(encoding="utf-8").splitlines() if line.strip()]
    frames = [(times[r["frame"]], r["items"], people.get(r["frame"], [])) for r in rows]
    visits = [(v["start_ms"] + offset, v["end_ms"] + offset) for v in raw]
    changes = poi.visit_changes(frames, visits, 3000.0, poi.SHOW, poi.EDGE_MS)
    shelf, over, counted = Counter(), Counter(), 0
    for v, c in zip(raw, changes):
        removed, added = visit_kinds(ShelfVisit(**{k: x for k, x in v.items() if k not in ("start_ms", "end_ms")}))
        mine = Counter({k: removed[k] - added[k] for k in set(removed) | set(added)})
        shelf.update(mine)
        if c is None:
            over.update(mine)
        else:
            counted += 1
            over.update({k: c.removed_by.get(k, 0) - c.added_by.get(k, 0) for k in set(c.removed_by) | set(c.added_by)})
    return shelf, over, len(raw), counted


def text(c: Counter) -> str:
    return " ".join(f"{k[:4]} {c[k]:+d}" for k in KINDS if c[k]) or "0"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shoot", type=int, choices=sorted(shoots.SHOOTS), default=shoots.LATEST)
    ap.add_argument("--only", default="N*", help="짝 이름 패턴 (기본 N* — 정상 영상으로 고른다)")
    ap.add_argument("--out", help="기록 폴더 이름 (outputs/ 아래, 기본 = 그 회차의 기본 기록)")
    args = ap.parse_args()
    dual_eval.set_profile(args.shoot)
    if args.out:
        dual_eval.OUT = ROOT / "outputs" / args.out
    labels = shoots.get(args.shoot).labels
    truth_of = truth_from_logs if (labels / "orders.csv").exists() else truth_from_takes
    table = dual_eval.SyncTable.load(labels / "sync.csv")
    pats = [p.strip() for p in args.only.split(",")]
    ok = {"shelf": 0, "overview": 0}
    n = visits = counted = 0
    for folder in sorted(p for p in dual_eval.OUT.iterdir() if (p / poi.CAMERA / poi.NAME).exists()):
        pair = folder.name
        if not any(fnmatch.fnmatch(pair, p) for p in pats):
            continue
        want = +truth_of(labels, pair)
        shelf, over, nv, nc = camera_counts(folder, table.offset_ms(pair) or 0.0)
        shelf, over = Counter({k: v for k, v in shelf.items() if v}), Counter({k: v for k, v in over.items() if v})
        n, visits, counted = n + 1, visits + nv, counted + nc
        ok["shelf"] += shelf == want
        ok["overview"] += over == want
        print(f"{pair:6s} 정답 {text(want):28s} 선반 지도 {text(shelf):28s}{'O' if shelf == want else 'X'}   "
              f"전체 카메라 {text(over):28s}{'O' if over == want else 'X'} (방문 {nv} 중 {nc} 셈)")
    print(f"영상 {n}편 — 선반 지도 맞음 {ok['shelf']} · 전체 카메라 맞음 {ok['overview']} (전체 카메라가 센 방문 {counted}/{visits})")
    print(f"지금 설정: {args.shoot}차 count_source = {shoots.get(args.shoot).count_source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
