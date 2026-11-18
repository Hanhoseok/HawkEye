"""Roboflow 로 찍은 상품 데이터셋(COCO 형식)을 우리 YOLO 학습용 모양으로 바꾼다.

    python scripts/product_coco.py                       # source/labeling_object → outputs/products_rf/
    python scripts/product_coco.py --src <폴더> --out <폴더>
    python scripts/product_coco.py --publish             # 바꿔 둔 것에서 라벨 · 목록만 training/products/dataset/ 으로 (그림 빼고)

Roboflow 가 내보낸 것을 그대로 학습에 넣으면 안 되는 까닭 네 가지를 여기서 고친다 (limitations D-9):

  1. 나눔이 장면 단위다 — 같은 영상의 옆 장면(거의 같은 그림)이 학습 · 검증 · 시험에 흩어져 점수가 부푼다.
     → 우리 표(labels/products_shoot3/frames.csv)의 **영상 단위 나눔**으로 다시 나눈다. Roboflow 의 나눔은 버린다.
  2. 종류 번호가 다르다 → 이름으로 맞춰 우리 classes.yaml 의 번호로 옮긴다.
  3. COCO 형식이다 → YOLO 형식('cls cx cy w h', 0~1).
  4. 파일 이름에 카메라가 없다 (두 카메라의 같은 이름 영상을 함께 올려서) → 그림을 3차 영상의 첫 장면과 견줘
     카메라를 가르고 이름에 넣는다: <영상>_<카메라>_rf<순번>.jpg  (순번은 Roboflow 가 뽑은 차례이지 영상의 장면 번호가 아니다)

나온 것 (outputs/products_rf/ — 그림에 얼굴이 있어 저장소에 올리지 않는다):
    images/ · labels/            그림(원본에 하드링크, 안 되면 복사)과 YOLO 라벨
    train.txt · val.txt · test.txt, data.yaml                      전체
    val_angle0.txt … test_angle1.txt, data_angle0.yaml · data_angle1.yaml   카메라별 점수를 낼 때
    frames.csv                   그림마다 원본 파일 · 영상 · 카메라 · 순번 · 나눔 · Roboflow 의 나눔 · 상자 수

카메라 가르기는 **지금 카메라 자리**에 맞춘 것이 아니다 — 그 촬영의 영상 자체를 기준으로 쓴다 (다시 촬영하면 그 영상이 기준이 된다).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import product_frames as pf  # noqa: E402

SRC = ROOT.parents[1] / "source" / "labeling_object"
OUT = ROOT / "outputs" / "products_rf"
SPLITS = {"train": "train", "valid": "val", "val": "val", "test": "test"}
"""Roboflow 폴더 이름 → 우리 이름 (표의 '원래 나눔' 칸에만 쓴다)."""
ALIAS = {"applesoda": "applecider"}
"""Roboflow 에 적힌 종류 이름(소문자 · 글자와 숫자만) → 우리 classes.yaml 의 이름."""
NAME = re.compile(r"^(?P<video>.+?)_mp4-(?P<index>\d+)_jpg\.rf\.")
MIN_SIDE = 2.0
"""그림 안으로 자른 뒤 이보다 작은 상자(픽셀)는 버린다."""
MIN_MARGIN = 0.2
"""두 카메라 기준과의 닮음 차이가 이보다 작으면 '모름' — 넣지 않고 알린다. (3차 1,194장에서 가장 작은 차이는 0.45 였다)"""
COLUMNS = ("image", "source", "video", "camera", "index", "split", "roboflow_split", "boxes")
PUBLISHED = COLUMNS + ("bytes", "sha1")
KIT = ROOT / "training" / "products" / "dataset"
"""저장소에 올리는 학습 폴더의 데이터 자리 (그림 빼고)."""


# ---------------------------------------------------------------------- 영상을 읽지 않는 부분

def parse_name(file_name: str) -> tuple[str, int] | None:
    """'C01_mp4-0005_jpg.rf.<글자>.jpg' → ('C01', 5). 모양이 다르면 None."""
    m = NAME.match(file_name)
    return (m.group("video"), int(m.group("index"))) if m else None


def key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def class_map(categories: list[dict], names: dict[int, str]) -> dict[int, int]:
    """COCO 종류 번호 → 우리 번호. 이름으로 맞춘다 ('apple soda' = applecider). 못 맞춘 종류는 빠진다."""
    ours = {key(name): i for i, name in names.items()}
    out = {}
    for c in categories:
        k = key(c["name"])
        k = ALIAS.get(k, k)
        if k in ours:
            out[int(c["id"])] = ours[k]
    return out


def to_boxes(annotations: list[dict], cmap: dict[int, int], width: float, height: float) -> tuple[list[dict], dict]:
    """COCO 상자 [x, y, w, h] → 그림 안으로 자른 {"cls","x1","y1","x2","y2"}. 버린 것은 까닭별로 센다."""
    boxes, dropped = [], {"unknown_class": 0, "tiny": 0}
    for a in annotations:
        cls = cmap.get(int(a["category_id"]))
        if cls is None:
            dropped["unknown_class"] += 1
            continue
        x, y, w, h = (float(v) for v in a["bbox"])
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(float(width), x + w), min(float(height), y + h)
        if x2 - x1 < MIN_SIDE or y2 - y1 < MIN_SIDE:
            dropped["tiny"] += 1
            continue
        boxes.append({"cls": cls, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    boxes.sort(key=lambda b: (b["x1"], b["y1"], b["cls"]))
    return boxes, dropped


def side_of(video: str, plan_split: dict[str, str], unknown: str) -> str | None:
    """영상이 들어갈 쪽. 우리 표에 없는 영상은 unknown ('train' 또는 'skip' → None)."""
    if video in plan_split:
        return plan_split[video]
    return None if unknown == "skip" else unknown


def out_name(video: str, camera: str, index: int) -> str:
    return f"{video}_{camera}_rf{index:04d}.jpg"


def pick_camera(small, refs: dict[str, list]) -> tuple[str, float]:
    """줄인 그림과 카메라별 기준 그림의 닮음(−1~1)을 견줘 (더 닮은 카메라, 두 카메라의 차이)."""
    score = {cam: max(float((small * r).mean()) for r in rs) for cam, rs in refs.items() if rs}
    ranked = sorted(score, key=score.get, reverse=True)
    best = ranked[0]
    return best, (score[best] - score[ranked[1]]) if len(ranked) > 1 else 1.0


# ---------------------------------------------------------------------- 그림 · 영상

def small_of(image):
    """그림 → 64×36 회색, 평균 0 · 크기 1 로 맞춘 것 (밝기가 달라도 견줄 수 있게)."""
    import cv2
    import numpy as np

    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    g = cv2.resize(g, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (g - g.mean()) / (g.std() + 1e-6)


def read_small(path: Path):
    import cv2
    import numpy as np

    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    return None if image is None else small_of(image)


def load_refs(pairs: list[str], count: int = 5) -> dict[str, list]:
    """카메라별 기준 그림 — 3차 영상 몇 편의 첫 장면."""
    import cv2
    import sync_pair

    refs: dict[str, list] = {"angle0": [], "angle1": []}
    for pair in pf.spread(pairs, count):
        for camera, path in zip(("angle0", "angle1"), sync_pair.pair_files(pair, 3)):
            cap = cv2.VideoCapture(str(path))
            ok, frame = cap.read()
            cap.release()
            if ok:
                refs[camera].append(small_of(frame))
    return refs


def place(src: Path, dst: Path) -> str:
    """그림을 dst 에 둔다 — 하드링크(자리를 더 안 씀), 안 되면 복사."""
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


# ---------------------------------------------------------------------- 바꾸기

def read_coco(src: Path) -> tuple[list[dict], dict]:
    """Roboflow 폴더(train · valid · test 마다 _annotations.coco.json) → 그림 목록과 통계."""
    items, stats = [], {"listed": 0, "not_listed": 0, "bad_name": 0, "missing_file": 0}
    for folder in sorted(p for p in src.iterdir() if p.is_dir() and (p / "_annotations.coco.json").exists()):
        data = json.loads((folder / "_annotations.coco.json").read_text(encoding="utf-8"))
        by_image: dict[int, list[dict]] = {}
        for a in data.get("annotations") or []:
            by_image.setdefault(int(a["image_id"]), []).append(a)
        listed = set()
        for im in data.get("images") or []:
            listed.add(im["file_name"])
            stats["listed"] += 1
            parsed = parse_name(im["file_name"])
            if parsed is None:
                stats["bad_name"] += 1
                continue
            if not (folder / im["file_name"]).exists():
                stats["missing_file"] += 1
                continue
            items.append({"path": folder / im["file_name"], "video": parsed[0], "index": parsed[1],
                          "roboflow_split": SPLITS.get(folder.name, folder.name), "width": im["width"], "height": im["height"],
                          "annotations": by_image.get(int(im["id"]), []), "categories": data.get("categories") or []})
        stats["not_listed"] += sum(1 for p in folder.glob("*.jpg") if p.name not in listed)
    return items, stats


def convert(src: Path, out: Path, plan_split: dict[str, str], names: dict[int, str], camera_of,
            unknown: str = "train") -> dict:
    """camera_of(path) → (카메라, 여유). 여유가 MIN_MARGIN 미만이면 넣지 않는다."""
    items, stats = read_coco(src)
    stats.update({"images": 0, "boxes": 0, "empty": 0, "unsure_camera": 0, "skipped_video": 0, "clash": 0,
                  "unknown_class": 0, "tiny": 0, "placed": {}, "moved_side": 0, "unknown_videos": {},
                  "by_class": {name: 0 for name in names.values()}, "by_split": {}, "by_camera": {}, "min_margin": None})
    images, labels = out / "images", out / "labels"
    for folder, pattern in ((images, "*.jpg"), (labels, "*.txt")):
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob(pattern):          # 지난번에 바꾼 것
            old.unlink()
    rows, seen = [], set()
    for it in sorted(items, key=lambda t: (t["video"], t["index"], t["path"].name)):
        side = side_of(it["video"], plan_split, unknown)
        if it["video"] not in plan_split:
            stats["unknown_videos"][it["video"]] = stats["unknown_videos"].get(it["video"], 0) + 1
        if side is None:
            stats["skipped_video"] += 1
            continue
        camera, margin = camera_of(it["path"])
        if stats["min_margin"] is None or margin < stats["min_margin"]:
            stats["min_margin"] = round(margin, 3)
        if margin < MIN_MARGIN:
            stats["unsure_camera"] += 1
            continue
        name = out_name(it["video"], camera, it["index"])
        if name in seen:                           # 같은 영상 · 카메라 · 순번이 두 번 — 뒤의 것은 넣지 않는다
            stats["clash"] += 1
            continue
        seen.add(name)
        boxes, dropped = to_boxes(it["annotations"], class_map(it["categories"], names), it["width"], it["height"])
        for k, v in dropped.items():
            stats[k] += v
        how = place(it["path"], images / name)
        stats["placed"][how] = stats["placed"].get(how, 0) + 1
        (labels / (Path(name).stem + ".txt")).write_text(
            "".join(line + "\n" for line in pf.yolo_lines(boxes, it["width"], it["height"])), encoding="utf-8", newline="\n")
        rows.append({"image": name, "source": f"{it['path'].parent.name}/{it['path'].name}", "video": it["video"], "camera": camera,
                     "index": it["index"], "split": side, "roboflow_split": it["roboflow_split"], "boxes": len(boxes)})
        stats["images"] += 1
        stats["boxes"] += len(boxes)
        stats["empty"] += not boxes
        stats["moved_side"] += side != it["roboflow_split"]
        stats["by_split"][side] = stats["by_split"].get(side, 0) + 1
        stats["by_camera"][f"{camera}/{side}"] = stats["by_camera"].get(f"{camera}/{side}", 0) + 1
        for b in boxes:
            stats["by_class"][names[b["cls"]]] += 1
    write_lists(out, rows, names)
    return stats


def write_lists(out: Path, rows: list[dict], names: dict[int, str], columns=COLUMNS, portable: bool = False) -> None:
    """나눔별 목록 · data.yaml · frames.csv. portable = data.yaml 에 이 컴퓨터의 경로를 적지 않는다
    (path 줄이 없으면 ultralytics 는 data.yaml 이 있는 폴더를 기준으로 읽는다 — 다른 컴퓨터로 옮길 수 있다)."""
    def put(name: str, part: list[dict]) -> None:
        (out / name).write_text("".join(f"./images/{r['image']}\n" for r in part), encoding="utf-8", newline="\n")

    def yaml(name: str, suffix: str) -> None:
        lines = [] if portable else [f"path: {out.as_posix()}"]
        lines += [f"train: train{suffix}.txt", f"val: val{suffix}.txt", f"test: test{suffix}.txt", "names:"]
        lines += [f"  {i}: {n}" for i, n in sorted(names.items())]
        (out / name).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    for side in ("train", "val", "test"):
        put(f"{side}.txt", [r for r in rows if r["split"] == side])
        for camera in pf.CAMERAS:
            put(f"{side}_{camera}.txt", [r for r in rows if r["split"] == side and r["camera"] == camera])
    yaml("data.yaml", "")
    for camera in pf.CAMERAS:
        yaml(f"data_{camera}.yaml", f"_{camera}")
    with (out / "frames.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def publish(out: Path, dest: Path, names: dict[int, str]) -> dict:
    """바꾼 것 가운데 **얼굴이 없는 것만** 저장소의 학습 폴더(dest)로 낸다 — 라벨 · 목록 · data.yaml · frames.csv.

    그림은 내지 않는다. 대신 frames.csv 에 그림마다 크기(bytes)와 sha1 을 적어, 다른 컴퓨터에서 가져다 놓은 그림이
    같은 것인지 확인할 수 있게 한다 (training/products/check.py).
    """
    import hashlib

    with (out / "frames.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    labels = dest / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    for old in labels.glob("*.txt"):
        old.unlink()
    for r in rows:
        stem = Path(r["image"]).stem
        shutil.copyfile(out / "labels" / f"{stem}.txt", labels / f"{stem}.txt")
        raw = (out / "images" / r["image"]).read_bytes()
        r["bytes"], r["sha1"] = len(raw), hashlib.sha1(raw).hexdigest()
    write_lists(dest, rows, names, columns=PUBLISHED, portable=True)
    return {"images": len(rows), "labels": len(list(labels.glob("*.txt")))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, default=SRC, help="Roboflow 가 내보낸 폴더 (train · valid · test)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--unknown", choices=("train", "skip"), default="train",
                    help="우리 표에 없는 영상의 그림을 어디에 둘지 (기본: 학습 — 검증 · 시험 영상이 아니므로 점수를 부풀리지 않는다)")
    ap.add_argument("--publish", nargs="?", type=Path, const=KIT, default=None, metavar="폴더",
                    help=f"바꾸지 않고, 이미 바꿔 둔 --out 에서 라벨 · 목록만 학습 폴더로 낸다 (기본 {KIT.relative_to(ROOT).as_posix()})")
    args = ap.parse_args()
    if args.publish:
        if not (args.out / "frames.csv").exists():
            raise SystemExit(f"{args.out} 에 바꾼 것이 없다 — 먼저 --publish 없이 돌린다")
        p = publish(args.out, args.publish, pf.read_names())
        print(f"라벨 {p['labels']}개 · 목록 · data.yaml · frames.csv → {args.publish} (그림은 내지 않는다)")
        return 0
    if not args.src.exists():
        raise SystemExit(f"{args.src} 없음")
    plan_split: dict[str, str] = {}
    for r in pf.read_plan():
        plan_split[r["pair"]] = r["split"]
    refs = load_refs(sorted(plan_split))
    if not refs["angle0"] or not refs["angle1"]:
        raise SystemExit("3차 영상을 읽지 못해 카메라를 가를 기준이 없다 (source/ 확인)")

    def camera_of(path: Path) -> tuple[str, float]:
        small = read_small(path)
        return ("?", 0.0) if small is None else pick_camera(small, refs)

    s = convert(args.src, args.out, plan_split, pf.read_names(), camera_of, args.unknown)
    print(f"그림 {s['images']}장 · 상자 {s['boxes']}개 (상자 없는 그림 {s['empty']}) → {args.out}")
    print("  나눔  " + " · ".join(f"{k} {s['by_split'].get(k, 0)}" for k in ("train", "val", "test"))
          + f"   (Roboflow 의 나눔에서 옮겨진 것 {s['moved_side']}장)")
    print("  카메라 " + " · ".join(f"{k} {v}" for k, v in sorted(s["by_camera"].items())))
    print("  종류  " + " · ".join(f"{k} {v}" for k, v in s["by_class"].items()))
    print(f"  그림 놓기 {s['placed']} · 카메라 가르기 여유 가장 작은 값 {s['min_margin']}")
    if s["unknown_videos"]:
        print(f"  우리 표에 없는 영상 → {args.unknown}: {s['unknown_videos']}")
    left = {k: s[k] for k in ("not_listed", "bad_name", "missing_file", "unsure_camera", "skipped_video", "clash", "unknown_class", "tiny") if s[k]}
    if left:
        print(f"  ! 안 들어간 것: {left}")
    if not s["by_split"].get("val"):
        print("  ! val 이 비었다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
