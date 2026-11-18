"""기존 학습 데이터(dataset/, 1,194장)에 묶음 2(Roboflow COCO 내보내기)를 더해 dataset_v2/ 를 만든다.

    python -I build_dataset_v2.py <products 폴더> <묶음2 내보내기 폴더> <shoot3 frames.csv> <제외 목록.txt>

- 기존 dataset/ 은 건드리지 않는다 (나눔 · 라벨 그대로 복사).
- 묶음 2 의 나눔은 shoot3 frames.csv 의 영상 단위 나눔을 따른다 (Roboflow 의 나눔은 버린다).
- 종류는 이름으로 맞춘다: apple soda → 2 applecider · cantata → 0 · donggwaru → 1.
- 그림은 하드링크 (같은 드라이브), 안 되면 복사.
- 목록: 전체 · 카메라별, 그리고 점수 비교용 old(기존 1,194장) · new(묶음 2) 따로.
"""
import csv
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

prod, exp, shoot3, excl_file = map(Path, sys.argv[1:5])
old = prod / "dataset"
v2 = prod / "dataset_v2"
CLS = {"cantata": 0, "donggwaru": 1, "apple soda": 2, "applesoda": 2, "applecider": 2}
NAME = re.compile(r"^(?P<stem>.+?)_jpg\.rf\.[^.]+\.jpg$")
SIDES = ("train", "val", "test")
CAMS = ("angle0", "angle1")
W, H = 1920, 1080

if v2.exists():
    raise SystemExit(f"{v2} 가 이미 있다 — 지우고 다시 돌린다")
(v2 / "images").mkdir(parents=True)
(v2 / "labels").mkdir()


def place(src: Path, dst: Path):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


rows = []  # image, origin, video, camera, split, boxes, source
for r in csv.DictReader((old / "frames.csv").open(encoding="utf-8")):
    place(old / "images" / r["image"], v2 / "images" / r["image"])
    shutil.copyfile(old / "labels" / (Path(r["image"]).stem + ".txt"), v2 / "labels" / (Path(r["image"]).stem + ".txt"))
    rows.append({"image": r["image"], "origin": "old", "video": r["video"], "camera": r["camera"], "split": r["split"],
                 "boxes": int(r["boxes"]), "source": r["source"]})

table = {r["image"]: r for r in csv.DictReader(shoot3.open(encoding="utf-8")) if r["batch"] == "2"}
excluded = {line.strip() for line in excl_file.read_text(encoding="utf-8").splitlines() if line.strip()}
stats = Counter()
for side in ("train", "valid", "test"):
    j = json.loads((exp / side / "_annotations.coco.json").read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"].strip().lower() for c in j["categories"]}
    anns = {}
    for a in j["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    for im in j["images"]:
        stem = NAME.match(im["file_name"]).group("stem")
        name = stem + ".jpg"
        if name in excluded:
            stats["excluded"] += 1
            continue
        t = table[name]
        lines = []
        for a in anns.get(im["id"], []):
            cname = cats[a["category_id"]]
            if cname not in CLS:
                stats[f"skip_cat_{cname}"] += 1
                continue
            x, y, w, h = a["bbox"]
            x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
            if x2 - x1 < 2 or y2 - y1 < 2:
                stats["tiny_dropped"] += 1
                continue
            lines.append(f"{CLS[cname]} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        place(exp / side / im["file_name"], v2 / "images" / name)
        (v2 / "labels" / (stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        rows.append({"image": name, "origin": "new", "video": t["pair"], "camera": t["camera"], "split": t["split"],
                     "boxes": len(lines), "source": f"{side}/{im['file_name']}"})
        stats["new_kept"] += 1

# same video must not appear in two splits
vs = {}
for r in rows:
    vs.setdefault(r["video"], set()).add(r["split"])
leak = {v: s for v, s in vs.items() if len(s) > 1}
if leak:
    raise SystemExit(f"영상이 두 쪽에 걸친다: {leak}")


def write_list(name, pick):
    (v2 / f"{name}.txt").write_text("".join(f"./images/{r['image']}\n" for r in rows if pick(r)), encoding="utf-8")


names = "names:\n  0: cantata\n  1: donggwaru\n  2: applecider\n"
for side in SIDES:
    write_list(side, lambda r, s=side: r["split"] == s)
    for cam in CAMS:
        write_list(f"{side}_{cam}", lambda r, s=side, c=cam: r["split"] == s and r["camera"] == c)
    for origin in ("old", "new"):
        write_list(f"{side}_{origin}", lambda r, s=side, o=origin: r["split"] == s and r["origin"] == o)
(v2 / "data.yaml").write_text("train: train.txt\nval: val.txt\ntest: test.txt\n" + names, encoding="utf-8")
for suffix in [f"_{c}" for c in CAMS] + ["_old", "_new"]:
    (v2 / f"data{suffix}.yaml").write_text(
        f"train: train{suffix}.txt\nval: val{suffix}.txt\ntest: test{suffix}.txt\n" + names, encoding="utf-8")
with (v2 / "frames.csv").open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

print(dict(stats))
for origin in ("old", "new", None):
    sel = [r for r in rows if origin is None or r["origin"] == origin]
    print(origin or "total", len(sel), "images,", sum(r["boxes"] for r in sel), "boxes;",
          dict(Counter(r["split"] for r in sel)))
