"""선반 지도가 센 것을 사람 정답(labels/store01_shoot3/takes.csv)과 견준다 — 일반 YOLO 때와 상품 모델을 끼웠을 때.

    python scripts/product_shelf_report.py dual_eval_shoot3 dual_eval_shoot3_prod          # 실행 폴더들 (outputs/ 아래)
    python scripts/product_shelf_report.py dual_eval_shoot3 dual_eval_shoot3_prod --judge  # 판정 점수도 기준(baselines/)과 견준다

영상 한 편마다:
  - 개수: 선반 지도의 (사라짐 - 생김) 합 = 정답의 (꺼냄 - 되돌림) 합 이면 맞음.
  - 종류: 사라진 것의 종류별 개수가 정답 꺼냄의 레인별 개수와 같으면 맞음 (L1 칸타타 · L2 동과루 · L3 애플사이다).
          일반 YOLO 는 우리 종류를 모르므로 이 칸은 상품 모델에서만 뜻이 있다.
  - 주인: 변화가 있는 방문 가운데 누구 것인지 정해진 것.

정답의 시각은 쓰지 않는다 (영상 전체의 합만 견준다). 그래서 '꺼냈다 되돌리고 다른 것을 꺼냄'처럼 합이 같은 틀림은 못 잡는다.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

LANE_CLASS = {"L1": "cantata", "L2": "donggwaru", "L3": "applecider"}
CLASSES = tuple(LANE_CLASS.values())
SLOT = re.compile(r"\d+단 \d+번째 ([^,/]+?)(?=,|$| /| ->)")


def parse_detail(detail: str) -> tuple[list[str], list[str]]:
    """방문 기록의 설명 → (사라진 것의 이름들, 생긴 것의 이름들).
    '[shelf] 사라짐: 1단 1번째 donggwaru, 1단 3번째 cantata / 생김: 1단 2번째 cantata -> 손님 3'"""
    body = detail.split("] ", 1)[-1].rsplit(" -> ", 1)[0]
    removed, added = [], []
    for part in body.split(" / "):
        names = [m.strip() for m in SLOT.findall(part)]
        if part.startswith("사라짐"):
            removed += names
        elif part.startswith("생김"):
            added += names
    return removed, added


def truth_of(rows: list[dict]) -> dict[str, dict]:
    """takes.csv → 짝 → {take: 종류별 개수, ret: 종류별 개수}. qty 가 비면 1."""
    out: dict[str, dict] = {}
    for r in rows:
        side = out.setdefault(r["pair"], {"take": Counter(), "ret": Counter()})
        side["take" if r["action"] == "take" else "ret"][LANE_CLASS.get(r["lane"], r["lane"])] += int(r.get("qty") or 1)
    return out


def seen_of(visits: list[dict]) -> dict:
    """한 영상의 선반 방문 기록 → {removed, added: 종류별 개수, changed: 변화 있는 방문 수, owned: 그중 주인이 정해진 수}."""
    out = {"removed": Counter(), "added": Counter(), "changed": 0, "owned": 0}
    for v in visits:
        if v.get("source", "vision") != "vision":
            continue
        removed, added = parse_detail(v.get("detail") or "")
        if len(removed) != v["removed"] or len(added) != v["added"]:     # 설명을 못 읽었으면 개수만 믿는다
            removed, added = ["?"] * v["removed"], ["?"] * v["added"]
        out["removed"].update(removed)
        out["added"].update(added)
        if v["removed"] or v["added"]:
            out["changed"] += 1
            out["owned"] += v["person_id"] > 0
    return out


def compare(truth: dict, seen: dict) -> dict:
    """한 영상: 정답과 선반 지도. net = 꺼냄 - 되돌림."""
    t_take, t_ret = sum(truth["take"].values()), sum(truth["ret"].values())
    s_rem, s_add = sum(seen["removed"].values()), sum(seen["added"].values())
    return {"truth_net": t_take - t_ret, "seen_net": s_rem - s_add, "truth_take": t_take, "seen_removed": s_rem,
            "truth_ret": t_ret, "seen_added": s_add,
            "count_ok": t_take - t_ret == s_rem - s_add,
            "class_ok": all(truth["take"][c] - truth["ret"][c] == seen["removed"][c] - seen["added"][c] for c in CLASSES)
                        and not (set(seen["removed"]) | set(seen["added"])) - set(CLASSES),
            "changed": seen["changed"], "owned": seen["owned"]}


def read_visits(folder: Path, pair: str) -> list[dict] | None:
    path = folder / pair / "angle0" / "shelf_visits.jsonl"
    if not (folder / pair / ".done").exists():
        return None
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def report(folder: Path, truth: dict[str, dict]) -> dict:
    rows = {}
    for pair in sorted(truth):
        visits = read_visits(folder, pair)
        if visits is not None:
            rows[pair] = compare(truth[pair], seen_of(visits))
    n = len(rows)
    return {"pairs": n, "rows": rows,
            "count_ok": sum(r["count_ok"] for r in rows.values()), "class_ok": sum(r["class_ok"] for r in rows.values()),
            "truth_take": sum(r["truth_take"] for r in rows.values()), "seen_removed": sum(r["seen_removed"] for r in rows.values()),
            "truth_ret": sum(r["truth_ret"] for r in rows.values()), "seen_added": sum(r["seen_added"] for r in rows.values()),
            "abs_error": sum(abs(r["truth_net"] - r["seen_net"]) for r in rows.values()),
            "changed": sum(r["changed"] for r in rows.values()), "owned": sum(r["owned"] for r in rows.values())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folders", nargs="+", help="실행 폴더 이름 (outputs/ 아래)")
    ap.add_argument("--judge", action="store_true", help="판정 점수도 기준(baselines/)과 견준다 (3차 네 방식)")
    ap.add_argument("--rows", action="store_true", help="영상별 줄도 찍는다")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    with (ROOT / "labels" / "store01_shoot3" / "takes.csv").open(encoding="utf-8-sig", newline="") as f:
        truth = truth_of(list(csv.DictReader(f)))
    results = {}
    for name in args.folders:
        r = results[name] = report(ROOT / "outputs" / name, truth)
        print(f"{name}: 영상 {r['pairs']}편 (정답이 있는 것)")
        print(f"  개수 맞음 {r['count_ok']}/{r['pairs']} · 종류까지 맞음 {r['class_ok']}/{r['pairs']} · 개수 오차 합 {r['abs_error']}")
        print(f"  사라짐 {r['seen_removed']} (정답 꺼냄 {r['truth_take']}) · 생김 {r['seen_added']} (정답 되돌림 {r['truth_ret']})")
        print(f"  변화 있는 방문 {r['changed']}건 중 주인이 정해진 것 {r['owned']}건")
    if args.rows:
        names = list(results)
        print("\n짝      정답(꺼냄-되돌림)  " + "  ".join(f"{n[-22:]:>22s}" for n in names))
        for pair in sorted(truth):
            cells = []
            for n in names:
                r = results[n]["rows"].get(pair)
                cells.append(f"{'없음':>22s}" if r is None else
                             f"{r['seen_removed']:>3d}-{r['seen_added']:<2d} {'개수O' if r['count_ok'] else '개수X'} "
                             f"{'종류O' if r['class_ok'] else '종류X'} 주인 {r['owned']}/{r['changed']}")
            t = truth[pair]
            print(f"{pair:6s} {sum(t['take'].values()):>3d}-{sum(t['ret'].values()):<2d}              " + "  ".join(cells))
    if args.judge:
        import regress

        for name in args.folders:
            print(f"\n{name}: 판정 (기준 baselines/ 와 견줌)")
            for mode in regress.MODES[3]:
                result = regress.run_mode(3, mode, ROOT / "outputs" / name, [], False)
                base = regress.load_baseline(3, mode)
                if result["failed"] or result["skipped"]:
                    print(f"  {regress.MODE_NAMES[mode]}: 실패 {[f['pair'] for f in result['failed']]} · 기록 없음 {result['skipped']}")
                print("  " + regress.describe(3, mode, regress.compare(base, result["rows"])).replace("\n", "\n  "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
