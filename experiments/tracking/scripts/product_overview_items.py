"""전체 카메라(angle1) 영상에서도 상품을 찾아 기록에 남긴다 — 분석 도구가 전체 카메라 화면에 상품 상자를 그린다.

    python scripts/product_overview_items.py --model training/products/handoff/products-model-v2/yolov8n_1280_v2.pt \
        --from dual_eval_shoot3 --out dual_eval_shoot3_prod

상품 탐지 모델은 두 카메라의 장면으로 학습했다 (labels/products_shoot3). 선반 지도는 선반 카메라(angle0)만 쓰므로
전체 카메라에서 찾은 것은 **보기만** 한다 — 판정에는 안 들어간다.

- --from 의 저장분(cache/angle1.pkl.gz)에서 처리한 장면 번호를 읽고, 그 장면 전부에서 찾는다 (3차는 4장에 1장 = 초당 15장).
- 결과는 <out>/<짝>/angle1/product_items.jsonl — 선반 카메라의 shelf_items.jsonl 과 같은 모양
  ({"frame": 번호, "items": [[종류, x1, y1, x2, y2, 점수], …]}).
- 이미 만든 짝은 건너뛴다 (장면 수와 탐지 설정이 같을 때). 다시 만들려면 --force.

    python scripts/product_overview_items.py --report --out dual_eval_shoot3_prod      # 전체 카메라가 본 것을 센다

--report 가 세는 것 (사람 정답 labels/store01_shoot3/takes.csv 와 견줌, 점수 0.25 이상인 상자만):
  - 처음 1초에 선반에서 본 상품 수가 진열(칸타타 2 · 동과루 3 · 애플사이다 1)과 같은 영상.
  - 꺼낸 뒤 10초 안에 **선반 밖**에서 그 종류의 상자를 본 꺼냄 — 들고 가는 상품이 전체 카메라에 보였나.
  - 첫 꺼냄 전에(아무도 상품을 들지 않았을 때) 선반 밖에 상자가 나온 장면 — 잘못 찾은 것.
  '선반 밖' = 처음 1초에 본 상자들을 감싼 네모(40픽셀 여유)의 밖. 처음 1초에 엉뚱한 데서 나온 상자는 뺀다
  (상자 가운데들의 가운데값에서 200픽셀 넘게 떨어진 것). 시각은 takes.csv(선반 카메라 시각)에 시차를 더해 맞춘다.
  --score 로 세는 점수 기준을 바꾼다 (기본 0.25 = 화면에 그리는 기준).

    python scripts/product_overview_items.py --count --out dual_eval_shoot3_prod       # 전체 카메라의 개수로 꺼냄을 센다

--count 가 하는 것: 선반 자리 안의 상자를 종류별로 세고, 영상의 **처음 개수 - 끝 개수**를 정답(꺼냄 - 되돌림)과 견준다.
선반 지도(선반 카메라)가 같은 영상에서 맞았는지도 나란히 보인다. 푸셔 줄은 선반 카메라가 앞에서 봐서 뒤 상품이 겹쳐 보이는데,
전체 카메라는 옆에서 봐서 나란히 보인다 (docs/shelf-map.md).
  - 같은 종류의 상자가 많이 겹치면(IoU 0.4 이상, --merge) 하나로 친다 — 한 캔에 상자가 둘 나오는 일이 있다.
  - '개수'는 0.6초 넘게 그대로인 것만 본다 (사람이 지나가며 잠깐 가린 것을 거른다). 처음 · 끝은 그런 구간의 첫 · 마지막.
  - 영상 전체의 차이만 본다 — 언제 · 누가는 모른다. 영상이 끝날 때 사람이 선반을 가리고 있으면 틀린다.

모델 · 영상 · 기록은 저장소에 없다 (outputs/ · handoff/ · source/).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

CAMERA = "angle1"
NAME = "product_items.jsonl"
SHOW = 0.25
"""세는 것 · 화면에 그리는 것은 이 점수 이상 (scripts/review_export.py 의 _item_rows 와 같다)."""
STOCK = {"cantata": 2, "donggwaru": 3, "applecider": 1}
"""3차 촬영에서 영상이 시작할 때의 진열 (L1 칸타타 · L2 동과루 · L3 애플사이다)."""
LANE_CLASS = {"L1": "cantata", "L2": "donggwaru", "L3": "applecider"}
START_MS = 1000.0
PAD = 40.0
NEAR = 200.0
"""처음 1초의 상자 가운데, 가운데값에서 이 거리 안에 있는 것만 선반의 상자로 본다 (잘못 찾은 상자 하나가 선반 자리를 화면 전체로 키우지 않게)."""
AFTER_MS = 10000.0
MERGE = 0.4
"""같은 종류의 두 상자가 이만큼 겹치면 한 상품으로 친다 (탐지기 자체의 겹침 제거는 0.7 이라 덜 겹친 쌍이 남는다)."""
STABLE_MS = 600.0
"""개수가 이 시간 넘게 그대로여야 '그때의 개수'로 본다 (선반 지도의 stable_seconds 와 같은 값)."""


def row(frame: int, detections: list) -> dict:
    """한 장면의 탐지 → 기록 한 줄. 좌표는 0.1 픽셀, 점수는 소수 셋째 자리."""
    return {"frame": frame, "items": [[d.class_name, *(round(float(v), 1) for v in d.bbox), round(float(d.score), 3)]
                                      for d in detections]}


def settings(model: str, imgsz: int, conf: float) -> dict:
    return {"model_path": model, "imgsz": imgsz, "conf": conf}


def made(folder: Path, want: dict, frames: int) -> bool:
    """이미 같은 설정으로 그 장면 수만큼 만들어 두었나."""
    meta, path = folder / "product_items.meta.json", folder / NAME
    if not (meta.exists() and path.exists()):
        return False
    try:
        have = json.loads(meta.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return have.get("settings") == want and have.get("frames") == frames


def write(folder: Path, rows: list[dict], want: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / NAME).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (folder / "product_items.meta.json").write_text(   # 기록을 다 쓴 뒤에 적는다 — 중간에 멈추면 다음에 다시 만든다
        json.dumps({"settings": want, "frames": len(rows), "boxes": sum(len(r["items"]) for r in rows)}, ensure_ascii=False),
        encoding="utf-8")


# ---------------------------------------------------------------------- 세기 (--report)

def shown(items: list, score: float = SHOW) -> list:
    return [it for it in items if it[5] >= score]


def start_rows(rows: list[dict], times: list[float]) -> list[dict]:
    """처음 1초의 장면들 (아직 아무도 선반에 닿지 않은 때)."""
    t0 = times[rows[0]["frame"]] if rows else 0.0
    return [r for r in rows if times[r["frame"]] - t0 <= START_MS]


def _center(b: list) -> tuple[float, float]:
    return (b[1] + b[3]) / 2, (b[2] + b[4]) / 2


def shelf_region(rows: list[dict], times: list[float]) -> tuple[float, float, float, float] | None:
    """처음 1초에 선반에서 본 상자들을 감싼 네모 + 여유 = 선반 자리. 하나도 못 봤으면 None.
    기준 점수(SHOW)로 정한다 — 세는 점수를 바꿔도 선반 자리는 같다."""
    boxes = [it for r in start_rows(rows, times) for it in shown(r["items"])]
    if not boxes:
        return None
    xs, ys = sorted(_center(b)[0] for b in boxes), sorted(_center(b)[1] for b in boxes)
    mx, my = xs[len(xs) // 2], ys[len(ys) // 2]
    boxes = [b for b in boxes if abs(_center(b)[0] - mx) <= NEAR and abs(_center(b)[1] - my) <= NEAR]
    return (min(b[1] for b in boxes) - PAD, min(b[2] for b in boxes) - PAD,
            max(b[3] for b in boxes) + PAD, max(b[4] for b in boxes) + PAD)


def away(items: list, region, score: float = SHOW) -> list:
    """선반 자리 밖에 있는 상자 (가운데 점으로 본다)."""
    if region is None:
        return []
    x1, y1, x2, y2 = region
    return [it for it in shown(items, score) if not (x1 <= _center(it)[0] <= x2 and y1 <= _center(it)[1] <= y2)]


def start_count(rows: list[dict], times: list[float], score: float = SHOW) -> dict[str, int]:
    """처음 1초에 종류별로 본 개수 (장면들의 가운데값)."""
    per = [[sum(it[0] == name for it in shown(r["items"], score)) for r in start_rows(rows, times)] for name in STOCK]
    return {name: sorted(c)[len(c) // 2] if c else 0 for name, c in zip(STOCK, per)}


def seen_after(rows: list[dict], times: list[float], region, t_ms: float, name: str | None, score: float = SHOW) -> int:
    """t_ms 부터 10초 안에 선반 밖에서 그 종류(None 이면 아무 종류)를 본 장면 수."""
    return sum(1 for r in rows if t_ms <= times[r["frame"]] <= t_ms + AFTER_MS
               and any(name is None or it[0] == name for it in away(r["items"], region, score)))


def iou(a: list, b: list) -> float:
    w = min(a[3], b[3]) - max(a[1], b[1])
    h = min(a[4], b[4]) - max(a[2], b[2])
    inter = max(0.0, w) * max(0.0, h)
    union = (a[3] - a[1]) * (a[4] - a[2]) + (b[3] - b[1]) * (b[4] - b[2]) - inter
    return inter / union if union > 0 else 0.0


def merge(items: list, limit: float = MERGE) -> list:
    """같은 종류끼리 limit 이상 겹친 상자는 점수 높은 것 하나만 남긴다."""
    keep: list = []
    for it in sorted(items, key=lambda i: -i[5]):
        if all(k[0] != it[0] or iou(k, it) < limit for k in keep):
            keep.append(it)
    return keep


def counts(items: list, region, score: float = SHOW, limit: float = MERGE) -> tuple[int, ...]:
    """선반 자리 안의 종류별 개수 (STOCK 의 순서: 칸타타 · 동과루 · 애플사이다)."""
    if region is None:
        return tuple(0 for _ in STOCK)
    x1, y1, x2, y2 = region
    inside = merge([it for it in shown(items, score) if x1 <= _center(it)[0] <= x2 and y1 <= _center(it)[1] <= y2], limit)
    return tuple(sum(it[0] == name for it in inside) for name in STOCK)


def stable_states(rows: list[dict], times: list[float], region, score: float = SHOW, limit: float = MERGE) -> list[tuple]:
    """개수가 STABLE_MS 넘게 그대로였던 구간들의 개수 (차례대로, 이웃한 같은 것은 하나로)."""
    out: list[tuple] = []
    run, start, last = None, 0.0, 0.0
    for r in [*rows, None]:
        v = counts(r["items"], region, score, limit) if r is not None else None
        if v != run:
            if run is not None and last - start >= STABLE_MS and (not out or out[-1] != run):
                out.append(run)
            run, start = v, times[r["frame"]] if r is not None else 0.0
        if r is not None:
            last = times[r["frame"]]
    return out


def count_report(out: str, score: float = SHOW, limit: float = MERGE) -> dict:
    """영상마다 전체 카메라의 (처음 개수 - 끝 개수)를 정답(꺼냄 - 되돌림)과 견준다. 선반 지도의 결과도 함께."""
    import csv

    import product_shelf_report as shelf

    with (ROOT / "labels" / "store01_shoot3" / "takes.csv").open(encoding="utf-8-sig", newline="") as f:
        truth = shelf.truth_of(list(csv.DictReader(f)))
    shelf_rows = shelf.report(ROOT / "outputs" / out, truth)["rows"]
    total = {"pairs": 0, "ok": 0, "stock_ok": 0, "shelf_ok": 0, "by_class": dict.fromkeys(STOCK, 0), "wrong": [], "fixed": [], "lines": []}
    for pair in sorted(truth):
        folder = ROOT / "outputs" / out / pair
        path = folder / CAMERA / NAME
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        times = json.loads((folder / f"times_{CAMERA}.json").read_text(encoding="utf-8"))
        states = stable_states(rows, times, shelf_region(rows, times), score, limit)
        first, last = (states[0], states[-1]) if states else ((0,) * len(STOCK),) * 2
        want = tuple(truth[pair]["take"][k] - truth[pair]["ret"][k] for k in STOCK)
        got = tuple(a - b for a, b in zip(first, last))
        good, map_good = bool(states) and want == got, bool(shelf_rows.get(pair, {}).get("class_ok"))
        total["pairs"] += 1
        total["ok"] += good
        total["stock_ok"] += first == tuple(STOCK.values())
        total["shelf_ok"] += map_good
        for k, w, g in zip(STOCK, want, got):
            total["by_class"][k] += bool(states) and w == g
        if not good:
            total["wrong"].append(pair)
        elif not map_good:
            total["fixed"].append(pair)
        total["lines"].append(f"{pair:5s} 정답 {want}  전체 카메라 {got} {'O' if good else 'X'}  선반 지도 {'O' if map_good else 'X'}  개수 {' → '.join(str(v) for v in states)}")
    return total


def report(out: str, score: float = SHOW) -> dict:
    import csv

    from src.inputs.sync import SyncTable

    labels = ROOT / "labels" / "store01_shoot3"
    table = SyncTable.load(labels / "sync.csv")
    takes: dict[str, list] = {}
    with (labels / "takes.csv").open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("time_sec") or "").strip():
                takes.setdefault(r["pair"], []).append((float(r["time_sec"]) * 1000.0, r["lane"], r["action"]))
    total = {"pairs": 0, "stock_ok": 0, "takes": 0, "same": 0, "any": 0, "before_frames": 0, "before_seen": 0, "no_region": []}
    lines = []
    for folder in sorted((ROOT / "outputs" / out).iterdir()):
        path = folder / CAMERA / NAME
        if not path.exists():
            continue
        pair = folder.name
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        times = json.loads((folder / f"times_{CAMERA}.json").read_text(encoding="utf-8"))
        region = shelf_region(rows, times)
        count = start_count(rows, times, score)
        offset = table.offset_ms(pair) or 0.0
        mine = sorted((t + offset, lane, action) for t, lane, action in takes.get(pair, []))       # 전체 카메라 시각으로
        took = [(t, LANE_CLASS.get(lane)) for t, lane, action in mine if action == "take"]
        same = sum(seen_after(rows, times, region, t, name, score) > 0 for t, name in took)
        anyk = sum(seen_after(rows, times, region, t, None, score) > 0 for t, name in took)
        first = mine[0][0] - 500.0 if mine else float("inf")
        before = [r for r in rows if times[r["frame"]] < first]
        wrong = sum(1 for r in before if away(r["items"], region, score))
        total["pairs"] += 1
        total["stock_ok"] += count == STOCK
        total["takes"] += len(took)
        total["same"] += same
        total["any"] += anyk
        total["before_frames"] += len(before)
        total["before_seen"] += wrong
        if region is None:
            total["no_region"].append(pair)
        lines.append(f"{pair:5s} 처음 {count['cantata']}·{count['donggwaru']}·{count['applecider']} {'O' if count == STOCK else 'X'}"
                     f"  꺼냄 {len(took)} 중 밖에서 봄 {same} (종류 무관 {anyk})  첫 꺼냄 전 밖 상자 {wrong}/{len(before)}장")
    total["lines"] = lines
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", help="상품 탐지 모델 (.pt)")
    ap.add_argument("--report", action="store_true", help="찾지 않고, 이미 남긴 기록을 사람 정답과 견줘 센다")
    ap.add_argument("--from", dest="source", default="dual_eval_shoot3", help="저장분이 있는 실행 폴더 (outputs/ 아래) — 장면 번호를 읽는다")
    ap.add_argument("--out", required=True, help="기록을 쓸 실행 폴더 (outputs/ 아래)")
    ap.add_argument("--only", help="짝 이름 패턴 (예: C0*)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.1, help="이 점수 이상을 기록한다 (화면에는 0.25 이상만 그린다)")
    ap.add_argument("--force", action="store_true", help="이미 만든 짝도 다시 만든다")
    ap.add_argument("--count", action="store_true", help="찾지 않고, 전체 카메라의 종류별 개수(처음 - 끝)를 정답과 견준다")
    ap.add_argument("--score", type=float, default=SHOW, help="--report · --count 가 세는 점수 기준 (기본 0.25)")
    ap.add_argument("--merge", type=float, default=MERGE, help="--count: 같은 종류의 상자가 이만큼 겹치면 하나로 (기본 0.4, 1 이면 안 합침)")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.count:
        r = count_report(args.out, args.score, args.merge)
        print("\n".join(r["lines"]))
        print(f"\n{args.out}: 영상 {r['pairs']}편 · 점수 {args.score} 이상 · 겹침 {args.merge} 이상은 하나로 · 개수는 (칸타타, 동과루, 애플사이다)")
        print(f"  처음 - 끝 개수가 정답(꺼냄 - 되돌림)과 종류까지 같은 영상: 전체 카메라 {r['ok']}/{r['pairs']} · 선반 지도 {r['shelf_ok']}/{r['pairs']}")
        print("  종류별로 맞은 영상: " + " · ".join(f"{k} {v}" for k, v in r["by_class"].items()))
        print(f"  처음 개수가 진열과 같은 영상 {r['stock_ok']}/{r['pairs']}")
        print(f"  전체 카메라가 틀린 영상: {', '.join(r['wrong']) or '없음'} · 선반 지도는 틀리고 전체 카메라는 맞은 영상: {', '.join(r['fixed']) or '없음'}")
        return 0
    if args.report:
        r = report(args.out, args.score)
        print("\n".join(r["lines"]))
        print(f"\n{args.out}: 영상 {r['pairs']}편 (전체 카메라 상품 기록이 있는 것) · 점수 {args.score} 이상")
        print(f"  처음 1초에 본 개수가 진열(칸타타 2 · 동과루 3 · 애플사이다 1)과 같은 영상 {r['stock_ok']}/{r['pairs']}")
        print(f"  꺼냄 {r['takes']}번 중 10초 안에 선반 밖에서 그 종류를 본 것 {r['same']} (종류 무관 {r['any']})")
        print(f"  첫 꺼냄 전 장면 {r['before_frames']}장 중 선반 밖에 상자가 나온 것 {r['before_seen']}장")
        if r["no_region"]:
            print(f"  처음 1초에 상자를 하나도 못 본 영상: {', '.join(r['no_region'])}")
        return 0
    if not args.model:
        ap.error("--model 이 필요하다 (--report 가 아닐 때)")

    import dual_eval
    import sync_pair
    from product_shelf_cache import detect_all
    from src.core.config import AppConfig
    from src.replay.cache import CameraCache

    dual_eval.set_profile(3)
    base = AppConfig.load(ROOT / "config.yaml")
    want = settings(args.model, args.imgsz, args.conf)
    detector = None
    for pair in dual_eval.pairs(args.only):
        cache_file = ROOT / "outputs" / args.source / pair / "cache" / f"{CAMERA}.pkl.gz"
        folder = ROOT / "outputs" / args.out / pair / CAMERA
        if not cache_file.exists() or not folder.parent.exists():
            print(f"{pair}: 저장분이나 기록 폴더 없음 (건너뜀)", flush=True)
            continue
        cache = CameraCache.load(cache_file)
        if not args.force and made(folder, want, len(cache.frames)):
            print(f"{pair}: 이미 있음", flush=True)
            continue
        if detector is None:
            from src.detectors.yolo_detector import build_item_detector

            detector = build_item_detector(args.model, args.imgsz, args.conf, base.detector.device, [])
        started = time.time()
        _video, ref = sync_pair.pair_files(pair, 3)
        found = detect_all(Path(ref), cache, detector, dual_eval.STRIDE[CAMERA])
        rows = [row(frame, found[frame]) for frame in sorted(found)]
        write(folder, rows, want)
        print(f"{pair}: 장면 {len(rows)}장 · 상품 상자 {sum(len(r['items']) for r in rows)}개 · {time.time() - started:.0f}초", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
