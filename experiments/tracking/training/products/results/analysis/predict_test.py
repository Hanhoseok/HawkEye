"""Run best.pt on test images, split GT boxes into 'static' (same place in most frames of the video)
and 'moved' (rarely at that place -> likely held / taken), measure recall for each, and save
annotated crops of moved boxes.

usage: python predict_test.py <products_dir> <out_dir> [split=test] [model_name=yolov8n_1280] [--no-images]
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

prod, out = Path(sys.argv[1]), Path(sys.argv[2])
args = [a for a in sys.argv[3:] if not a.startswith("--")]
SPLIT = args[0] if len(args) > 0 else "test"
MODEL = args[1] if len(args) > 1 else "yolov8n_1280"
IMAGES = "--no-images" not in sys.argv
data = prod / "dataset"
out.mkdir(parents=True, exist_ok=True)
(out / "moved").mkdir(exist_ok=True)
(out / "full").mkdir(exist_ok=True)
NAMES = {0: "cantata", 1: "donggwaru", 2: "applecider"}
W, H = 1920, 1080


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0


rows = [r for r in csv.DictReader((data / "frames.csv").open(encoding="utf-8")) if r["split"] == SPLIT]
gt = {}
for r in rows:
    boxes = []
    for line in (data / "labels" / (Path(r["image"]).stem + ".txt")).read_text().splitlines():
        if not line.strip():
            continue
        c, cx, cy, w, h = line.split()
        cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
        boxes.append((int(c), (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
    gt[r["image"]] = boxes

# stability: share of other frames of the same video+camera that have a same-class box at the same place
groups = defaultdict(list)
for r in rows:
    groups[(r["video"], r["camera"])].append(r["image"])
stab = {}
for (v, cam), imgs in groups.items():
    for im in imgs:
        others = [o for o in imgs if o != im]
        for k, (c, b) in enumerate(gt[im]):
            hits = sum(any(c2 == c and iou(b, b2) > 0.3 for c2, b2 in gt[o]) for o in others)
            stab[(im, k)] = hits / len(others) if others else 1.0

model = YOLO(str(prod / "handoff" / MODEL / "best.pt"))
preds = {}
for i in range(0, len(rows), 16):
    batch = [str(data / "images" / r["image"]) for r in rows[i:i + 16]]
    for r, res in zip(rows[i:i + 16], model.predict(batch, imgsz=1280, conf=0.25, device=0, verbose=False)):
        preds[r["image"]] = [(int(c), tuple(map(float, b)), float(s))
                             for c, b, s in zip(res.boxes.cls.tolist(), res.boxes.xyxy.tolist(), res.boxes.conf.tolist())]

stats = defaultdict(lambda: {"n": 0, "found": 0, "found_anycls": 0})
moved = []
for r in rows:
    im = r["image"]
    for k, (c, b) in enumerate(gt[im]):
        kind = "moved" if stab[(im, k)] < 0.15 else "static"
        best = max([(iou(b, pb), pc, ps) for pc, pb, ps in preds[im]], default=(0, None, 0))
        same = max([iou(b, pb) for pc, pb, ps in preds[im] if pc == c], default=0)
        key = f"{kind}_{r['camera']}"
        for kk in (kind, key):
            stats[kk]["n"] += 1
            stats[kk]["found"] += same >= 0.5
            stats[kk]["found_anycls"] += best[0] >= 0.5
        if kind == "moved":
            moved.append({"image": im, "camera": r["camera"], "cls": NAMES[c], "box": [round(x) for x in b],
                          "stability": round(stab[(im, k)], 2), "iou_same_cls": round(same, 2),
                          "best_pred": None if best[1] is None else {"cls": NAMES[best[1]], "iou": round(best[0], 2), "conf": round(best[2], 2)}})

# false positives: predictions not matching any GT box
fp = defaultdict(int)
for r in rows:
    for pc, pb, ps in preds[r["image"]]:
        if not any(c == pc and iou(b, pb) >= 0.5 for c, b in gt[r["image"]]):
            fp[r["camera"]] += 1

summary = {k: {**v, "recall": round(v["found"] / v["n"], 3) if v["n"] else None} for k, v in sorted(stats.items())}
summary["false_positives_by_camera"] = dict(fp)
(out / "summary.json").write_text(json.dumps({"summary": summary, "moved": moved}, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(summary, indent=1))
if not IMAGES:
    sys.exit(0)


def draw(img, im):
    for c, b in gt[im]:
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
    for pc, pb, ps in preds[im]:
        cv2.rectangle(img, (int(pb[0]), int(pb[1])), (int(pb[2]), int(pb[3])), (0, 0, 255), 2)
        cv2.putText(img, f"{NAMES[pc][:4]} {ps:.2f}", (int(pb[0]), int(pb[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return img


# crops around moved boxes: one per image, most "moved" box first
seen = set()
for m in sorted(moved, key=lambda m: m["stability"]):
    if m["image"] in seen:
        continue
    seen.add(m["image"])
    img = draw(cv2.imread(str(data / "images" / m["image"])), m["image"])
    x1, y1, x2, y2 = m["box"]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half = 220
    crop = img[max(0, cy - half):cy + half, max(0, cx - half):cx + half]
    crop = cv2.resize(crop, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    ok = "HIT" if m["iou_same_cls"] >= 0.5 else "MISS"
    cv2.imwrite(str(out / "moved" / f"{ok}_{m['cls']}_{Path(m['image']).stem}.jpg"), crop)

# a few full frames per camera
for cam in ("angle0", "angle1"):
    for im in [r["image"] for r in rows if r["camera"] == cam][::40]:
        cv2.imwrite(str(out / "full" / im), draw(cv2.imread(str(data / "images" / im)), im))
print("moved boxes:", len(moved), "images with moved boxes:", len(seen))
