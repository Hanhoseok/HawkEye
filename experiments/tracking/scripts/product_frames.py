"""3차 촬영 영상에서 상품 탐지 학습용 장면을 고르고 뽑는다 (상품 3종: 칸타타 · 동과루 · 애플사이다).

    python scripts/product_frames.py plan               # 어느 장면을 쓸지 정해 labels/products_shoot3/frames.csv 에 적는다 (그림은 안 만듦)
    python scripts/product_frames.py extract            # 묶음 1 의 장면을 그림으로 → outputs/products_shoot3/images/
    python scripts/product_frames.py extract --batch 0  # 후보 전부 (약 6천 장 — 용량 큼)
    python scripts/product_frames.py sheet              # 묶음 1 을 한눈에 보는 모아보기 → outputs/products_shoot3/sheets/
    python scripts/product_frames.py export             # 다 찍은 라벨을 YOLO 학습용으로 → outputs/products_shoot3/ (data.yaml · labels/ · 목록)

라벨은 분석 도구의 '상품 라벨' 화면에서 찍는다 (workbench/products.py → labels/products_shoot3/boxes/<그림 이름>.json).

고르는 방법 (전부 표와 장면 시각으로 정해져 다시 돌려도 같다):
  - base : 영상마다 1초에 한 장.
  - take : 사람 정답(labels/store01_shoot3/takes.csv — 꺼냄 · 되돌림)의 시각 -0.5 ~ +2초, 0.25초 간격. 손에 든 상품이 보이는 때다.
  - scan : 바코드 삑 소리(beeps.csv, kind=scan) 시각 ±0.5초, 0.25초 간격. 계산대에서 상품을 든 때다.
  옆 장면끼리는 거의 같아서 전부 찍지 않는다. **묶음 1**(먼저 라벨을 찍을 것) = take 는 0 · +0.75 · +1.5초, scan 은 그 시각 한 장,
  base 는 영상 · 카메라마다 2장(사람이 보인 장면 먼저). 나머지 후보는 모델이 틀리는 곳을 보고 묶음 2 로 고른다.

나누기: 학습 · 검증 · 시험은 **영상 단위**다 (split 칸). 같은 영상의 두 카메라는 같은 쪽 — 같은 순간을 다른 각도로 본 것이라서.
  세트(N · T · C)별로 이름순 5편마다 시험 1 · 검증 1.

저장소에는 표(frames.csv · classes.yaml)와 라벨만 올린다. 그림은 얼굴이 나오므로 outputs/ 에만 둔다 (언제든 이 스크립트로 다시 뽑힌다).
시각: takes.csv 는 선반 카메라(angle0) 시각이라 전체 카메라(angle1)는 시차(sync.csv)를 더한다. beeps.csv 는 카메라별 시각 그대로.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.inputs.frame_times import index_at  # noqa: E402

LABELS3 = ROOT / "labels" / "store01_shoot3"
DATASET = ROOT / "labels" / "products_shoot3"
"""저장소에 올리는 것: 장면 표(frames.csv) · 종류(classes.yaml) · 라벨."""
OUT = ROOT / "outputs" / "products_shoot3"
"""올리지 않는 것: 뽑은 그림 · 모아보기 · 장면 시각 저장분."""
RECORDS = ROOT / "outputs" / "dual_eval_shoot3"
"""판정용 실행 기록 — 있으면 장면 시각 저장분과 사람 수를 여기서 읽는다 (없어도 된다)."""
CAMERAS = ("angle0", "angle1")

EVERY_MS = 1000.0
TAKE_POOL_MS = tuple(range(-500, 2001, 250))
TAKE_FIRST_MS = (0, 750, 1500)
SCAN_POOL_MS = tuple(range(-500, 501, 250))
SCAN_FIRST_MS = (0,)
BASE_FIRST = 2
"""묶음 1 에 넣는 base 장면 수 (영상 · 카메라마다)."""
COLUMNS = ("image", "pair", "camera", "frame", "time_sec", "reason", "lane", "action", "people", "split", "batch")
PRIORITY = {"take": 0, "scan": 1, "base": 2}


# ---------------------------------------------------------------------- 고르기 (영상을 안 읽는 부분)

def base_frames(times: list[float], every_ms: float = EVERY_MS) -> list[int]:
    """every_ms 마다 그 시각 이후의 첫 장면."""
    out, due = [], times[0] if times else 0.0
    for i, t in enumerate(times):
        if t >= due:
            out.append(i)
            due += every_ms * (int((t - due) // every_ms) + 1)
    return out


def near_frames(times: list[float], at_ms: float, offsets_ms) -> list[int]:
    """at_ms + 각 offset 에 가장 가까운 장면들 (영상 밖 시각은 뺀다, 같은 장면은 한 번만)."""
    out = []
    for off in offsets_ms:
        ms = at_ms + off
        if not times or ms < times[0] - 100.0 or ms > times[-1] + 100.0:
            continue
        i = index_at(times, ms)
        if i not in out:
            out.append(i)
    return out


def split_of(pairs: list[str]) -> dict[str, str]:
    """영상 → train · val · test. 세트(이름 첫 글자)별로 이름순 5편마다 시험 1(셋째) · 검증 1(다섯째)."""
    out = {}
    for kind in sorted({p[0] for p in pairs}):
        for i, pair in enumerate(sorted(p for p in pairs if p[0] == kind)):
            out[pair] = "test" if i % 5 == 2 else ("val" if i % 5 == 4 else "train")
    return out


def spread(items: list, count: int) -> list:
    """items 에서 고르게 count 개 (양 끝은 피한다)."""
    if len(items) <= count:
        return list(items)
    return [items[int((k + 1) * len(items) / (count + 1))] for k in range(count)]


def choose(times: list[float], takes: list[tuple[float, str, str]], scans: list[float],
           people: dict[int, int] | None = None) -> dict[int, dict]:
    """한 영상 · 한 카메라에서 쓸 장면: {장면 번호: {reason, lane, action, batch}}.
    takes = [(ms, 레인, 행동)], scans = [ms] — 모두 이 카메라의 시각. people = {장면 번호: 사람 수} (없으면 None)."""
    rows: dict[int, dict] = {}

    def put(i: int, reason: str, first: bool, lane: str = "", action: str = "") -> None:
        old = rows.get(i)
        if old is None or PRIORITY[reason] < PRIORITY[old["reason"]]:
            rows[i] = {"reason": reason, "lane": lane, "action": action, "batch": 1 if first or (old and old["batch"]) else 0}
        elif first:
            old["batch"] = 1

    base = base_frames(times)
    for i in base:
        put(i, "base", False)
    for ms, lane, action in takes:
        first = set(near_frames(times, ms, TAKE_FIRST_MS))
        for i in near_frames(times, ms, TAKE_POOL_MS):
            put(i, "take", i in first, lane, action)
    for ms in scans:
        first = set(near_frames(times, ms, SCAN_FIRST_MS))
        for i in near_frames(times, ms, SCAN_POOL_MS):
            put(i, "scan", i in first)
    plain = [i for i in base if rows[i]["reason"] == "base"]
    with_people = [i for i in plain if people and people.get(i, 0) > 0]
    for i in spread(with_people if len(with_people) >= BASE_FIRST else plain, BASE_FIRST):
        rows[i]["batch"] = 1
    return rows


# ---------------------------------------------------------------------- 표 읽기

def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_takes(path: Path | None = None) -> dict[str, list[tuple[float, str, str]]]:
    """짝 → [(angle0 시각 ms, 레인, 행동)]."""
    out: dict[str, list] = {}
    for r in _rows(path or LABELS3 / "takes.csv"):
        out.setdefault(r["pair"], []).append((float(r["time_sec"]) * 1000.0, r["lane"], r["action"]))
    return out


def load_scans(path: Path | None = None) -> dict[tuple[str, str], list[float]]:
    """(짝, 카메라) → [그 카메라 시각 ms]. 바코드 삑 소리 중 scan 만."""
    out: dict[tuple[str, str], list[float]] = {}
    for r in _rows(path or LABELS3 / "beeps.csv"):
        if r["kind"] == "scan":
            out.setdefault((r["pair"], r["camera"]), []).append(float(r["time_sec"]) * 1000.0)
    return out


def people_by_frame(pair: str, camera: str) -> dict[int, int] | None:
    """실행 기록의 장면별 사람 상자 수. 기록이 없으면 None."""
    path = RECORDS / pair / camera / "identities.jsonl"
    if not path.exists():
        return None
    out: dict[int, int] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["frame"]] = out.get(r["frame"], 0) + 1
    return out


def frame_times(pair: str, camera: str, video: Path) -> list[float]:
    import shoot3_eval

    cache = RECORDS / pair / f"times_{camera}.json"
    if not cache.exists():
        cache = OUT / "times" / f"{pair}_{camera}.json"
    return shoot3_eval.cached_times(video, cache)


def image_name(pair: str, camera: str, frame: int) -> str:
    return f"{pair}_{camera}_{frame:05d}.jpg"


# ---------------------------------------------------------------------- 명령

def plan(only: set[str] | None = None) -> list[dict]:
    import sync_pair
    from src.inputs.sync import SyncTable

    pairs = sync_pair.all_pairs(3)
    split = split_of(pairs)
    takes, scans = load_takes(), load_scans()
    table = SyncTable.load(LABELS3 / "sync.csv")
    out = []
    for pair in pairs:
        if only and pair not in only:
            continue
        video, ref = sync_pair.pair_files(pair, 3)
        offset = table.offset_ms(pair) or 0.0
        for camera, path in (("angle0", video), ("angle1", ref)):
            times = frame_times(pair, camera, path)
            shift = offset if camera == "angle1" else 0.0
            people = people_by_frame(pair, camera)
            near = None
            if people:
                # 기록은 몇 장면마다 한 번이다 — 가장 가까운 기록 장면의 사람 수를 쓴다
                recorded = sorted(people)
                near = {i: people[min(recorded, key=lambda f: abs(f - i))] for i in base_frames(times)}
            chosen = choose(times, [(ms + shift, lane, action) for ms, lane, action in takes.get(pair, [])],
                            scans.get((pair, camera), []), near)
            for i in sorted(chosen):
                c = chosen[i]
                out.append({"image": image_name(pair, camera, i), "pair": pair, "camera": camera, "frame": i,
                            "time_sec": f"{times[i] / 1000.0:.2f}", "reason": c["reason"], "lane": c["lane"],
                            "action": c["action"], "people": "" if near is None or i not in near else near[i],
                            "split": split[pair], "batch": c["batch"]})
    return out


def write_plan(rows: list[dict], path: Path | None = None) -> Path:
    path = path or DATASET / "frames.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return path


def read_plan(path: Path | None = None) -> list[dict]:
    path = path or DATASET / "frames.csv"
    if not path.exists():
        raise SystemExit(f"{path} 없음 — 먼저 plan")
    return _rows(path)


def summary(rows: list[dict]) -> str:
    def count(pred) -> int:
        return sum(1 for r in rows if pred(r))
    first = [r for r in rows if str(r["batch"]) == "1"]
    lines = [f"후보 {len(rows)}장 (base {count(lambda r: r['reason'] == 'base')} · take {count(lambda r: r['reason'] == 'take')} · "
             f"scan {count(lambda r: r['reason'] == 'scan')}), 영상 {len({r['pair'] for r in rows})}편 × 카메라 2",
             f"묶음 1: {len(first)}장 (base {sum(r['reason'] == 'base' for r in first)} · take {sum(r['reason'] == 'take' for r in first)} · "
             f"scan {sum(r['reason'] == 'scan' for r in first)})"]
    for name in ("train", "val", "test"):
        part = [r for r in first if r["split"] == name]
        lines.append(f"  {name:5s} 영상 {len({r['pair'] for r in rows if r['split'] == name}):2d}편 · 묶음 1 {len(part)}장")
    return "\n".join(lines)


def extract(rows: list[dict], batch: int, only: set[str] | None = None) -> int:
    """고른 장면을 그림으로. 3차 영상은 위치 이동이 안 돼 처음부터 차례로 읽는다 (안 쓰는 장면은 풀지 않고 넘김)."""
    import cv2
    import sync_pair

    want: dict[tuple[str, str], dict[int, str]] = {}
    for r in rows:
        if (batch == 0 or str(r["batch"]) == str(batch)) and (not only or r["pair"] in only):
            want.setdefault((r["pair"], r["camera"]), {})[int(r["frame"])] = r["image"]
    images = OUT / "images"
    images.mkdir(parents=True, exist_ok=True)
    made = 0
    for (pair, camera), frames in sorted(want.items()):
        todo = {i: n for i, n in frames.items() if not (images / n).exists()}
        if not todo:
            continue
        video, ref = sync_pair.pair_files(pair, 3)
        cap = cv2.VideoCapture(str(video if camera == "angle0" else ref))
        last, i = max(todo), 0
        while i <= last and cap.grab():
            if i in todo:
                ok, image = cap.retrieve()
                if ok:
                    # 한글 경로에서도 쓰이게 imencode → 파일
                    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    (images / todo[i]).write_bytes(buf.tobytes())
                    made += 1
            i += 1
        cap.release()
        print(f"  {pair} {camera}: {len(todo)}장", flush=True)
    return made


def sheets(rows: list[dict], batch: int, cols: int = 6, per: int = 36, width: int = 320) -> list[Path]:
    """뽑은 그림을 작은 그림으로 모아 본다 (이름 · 고른 까닭을 위에 적음)."""
    import cv2
    import numpy as np

    picked = [r for r in rows if batch == 0 or str(r["batch"]) == str(batch)]
    out_dir = OUT / "sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for page in range(0, len(picked), per):
        tiles = []
        for r in picked[page:page + per]:
            path = OUT / "images" / r["image"]
            if not path.exists():
                continue
            image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            h = round(image.shape[0] * width / image.shape[1])
            tile = cv2.resize(image, (width, h), interpolation=cv2.INTER_AREA)
            text = f"{r['pair']} {r['camera'][-1]} {r['time_sec']}s {r['reason']} {r['lane']}"
            cv2.rectangle(tile, (0, 0), (width, 16), (0, 0, 0), -1)
            cv2.putText(tile, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        if not tiles:
            continue
        blank = np.zeros_like(tiles[0])
        grid = [np.hstack((tiles[k:k + cols] + [blank] * cols)[:cols]) for k in range(0, len(tiles), cols)]
        path = out_dir / f"batch{batch}_{page // per + 1:02d}.jpg"
        ok, buf = cv2.imencode(".jpg", np.vstack(grid), [cv2.IMWRITE_JPEG_QUALITY, 85])
        path.write_bytes(buf.tobytes())
        made.append(path)
    return made


# ---------------------------------------------------------------------- 라벨 → YOLO

def yolo_lines(boxes: list[dict], width: float, height: float) -> list[str]:
    """상자(그림 픽셀) → YOLO 라벨 줄 'cls cx cy w h' (0~1). 손에 든 표시(held)는 YOLO 형식에 자리가 없어 안 실린다."""
    out = []
    for b in boxes:
        x1, x2 = sorted((float(b["x1"]), float(b["x2"])))
        y1, y2 = sorted((float(b["y1"]), float(b["y2"])))
        out.append(f"{int(b['cls'])} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                   f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
    return out


def read_names(path: Path | None = None) -> dict[int, str]:
    import yaml

    raw = yaml.safe_load((path or DATASET / "classes.yaml").read_text(encoding="utf-8")) or {}
    return {int(i): name for i, name in (raw.get("names") or {}).items()}


def export(rows: list[dict], dataset: Path | None = None, out: Path | None = None) -> dict:
    """다 찍은(done) 장면만 YOLO 학습용으로 낸다. 찍는 중 · 뺀(skip) 장면 · 그림이 없는 장면은 안 들어간다.

    그림은 복사하지 않는다 — out/images/ 의 그림 옆 out/labels/ 에 라벨을 쓰고, 나눔별 목록(train.txt …)에 그림 경로를 적는다.
    (YOLO 는 그림 경로의 /images/ 를 /labels/ 로 바꿔 라벨을 찾는다.)
    """
    dataset, out = dataset or DATASET, out or OUT
    names = read_names(dataset / "classes.yaml")
    labels = out / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    for old in labels.glob("*.txt"):          # 지난번에 낸 것 — 그 뒤 라벨을 지웠을 수 있다
        old.unlink()
    lists: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    stats = {"frames": 0, "boxes": 0, "held": 0, "empty": 0, "doing": 0, "skip": 0, "no_image": 0,
             "by_class": {name: 0 for name in names.values()}, "by_split": {}}
    for r in rows:
        path = dataset / "boxes" / (Path(r["image"]).stem + ".json")
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("done"):
            stats["doing"] += 1
            continue
        if data.get("skip"):
            stats["skip"] += 1
            continue
        if not (out / "images" / r["image"]).exists():
            stats["no_image"] += 1
            continue
        boxes = data.get("boxes") or []
        (labels / (Path(r["image"]).stem + ".txt")).write_text(
            "".join(line + "\n" for line in yolo_lines(boxes, data["width"], data["height"])), encoding="utf-8", newline="\n")
        lists[r["split"]].append(f"./images/{r['image']}")
        stats["frames"] += 1
        stats["boxes"] += len(boxes)
        stats["held"] += sum(1 for b in boxes if b.get("held"))
        stats["empty"] += not boxes
        for b in boxes:
            stats["by_class"][names[int(b["cls"])]] += 1
    for split, items in lists.items():
        (out / f"{split}.txt").write_text("".join(i + "\n" for i in items), encoding="utf-8", newline="\n")
        stats["by_split"][split] = len(items)
    yaml_lines = [f"path: {out.as_posix()}", "train: train.txt", "val: val.txt", "test: test.txt", "names:"]
    yaml_lines += [f"  {i}: {name}" for i, name in sorted(names.items())]
    (out / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8", newline="\n")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=("plan", "extract", "sheet", "export"))
    ap.add_argument("--only", help="짝 이름 (쉼표)")
    ap.add_argument("--batch", type=int, default=1, help="extract · sheet: 묶음 번호 (0 = 후보 전부)")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    if args.what == "plan":
        rows = plan(only)
        if only:
            print("(--only: 표는 안 씀)")
        else:
            print(f"적음: {write_plan(rows).relative_to(ROOT)}")
        print(summary(rows))
        return 0
    rows = read_plan()
    if args.what == "export":
        st = export(rows)
        print(f"낸 장면 {st['frames']}장 (학습 {st['by_split']['train']} · 검증 {st['by_split']['val']} · 시험 {st['by_split']['test']}), "
              f"상자 {st['boxes']}개 (손에 든 것 {st['held']}), 상품 없는 장면 {st['empty']}장")
        print("  종류별: " + " · ".join(f"{k} {v}" for k, v in st["by_class"].items()))
        print(f"  안 낸 것: 찍는 중 {st['doing']}장 · 뺀 것 {st['skip']}장 · 그림 없음 {st['no_image']}장")
        for split in ("train", "val"):
            if not st["by_split"][split]:
                print(f"  ! {split} 이 비었다 — 이대로는 학습이 안 돈다")
        print(f"→ {(OUT / 'data.yaml').relative_to(ROOT)}")
    elif args.what == "extract":
        print(f"새로 뽑은 그림 {extract(rows, args.batch, only)}장 → {(OUT / 'images').relative_to(ROOT)}")
    else:
        made = sheets(rows, args.batch)
        print(f"모아보기 {len(made)}장 → {(OUT / 'sheets').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
