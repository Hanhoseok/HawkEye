"""받은 상품 탐지 모델을 이 컴퓨터(CPU)에서 확인한다 — '움직인 상품'을 몇 개 찾는지 · 한 장에 몇 초 걸리는지.

    python scripts/product_model_check.py outputs/products_rf training/products/handoff/products-model-v2/yolov8s_1280_v2.pt outputs/products_model_v2

- '움직인 상품' = 같은 영상 · 카메라의 다른 장면에서 그 자리에 거의 없는 상자 (손에 든 것이 대부분이지만 전부는 아니다).
  규칙은 다른 컴퓨터의 training/products/results/analysis/predict_test.py 와 같다: 안정도 < 0.15 · conf 0.25 · 같은 종류 IoU >= 0.5.
- 시간은 한 장씩 넣어 잰다 (해상도 1280, 첫 장 제외).
- 시험(test) 쪽에서 놓친 것 · 찾은 것을 8장씩 모아 그림으로 낸다 (초록 = 정답, 빨강 = 예측) — 얼굴이 나올 수 있어 outputs/ 에만 둔다.
- mAP 는 여기서 내지 않는다: ultralytics 의 val(data=outputs/products_rf/data.yaml, split=val · test, imgsz=1280)로 낸다.
"""
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

NAMES = {0: "cantata", 1: "donggwaru", 2: "applecider"}
W, H = 1920, 1080


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0


def read(path):
    return cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_COLOR)


def main():
    from ultralytics import YOLO

    data, model_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    result = {}
    for split in ("val", "test"):
        rows = [r for r in csv.DictReader((data / "frames.csv").open(encoding="utf-8")) if r["split"] == split]
        gt = {}
        for r in rows:
            boxes = []
            for line in (data / "labels" / (Path(r["image"]).stem + ".txt")).read_text().splitlines():
                if line.strip():
                    c, cx, cy, w, h = line.split()
                    cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
                    boxes.append((int(c), (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
            gt[r["image"]] = boxes
        groups = defaultdict(list)
        for r in rows:
            groups[(r["video"], r["camera"])].append(r["image"])
        stab = {}
        for imgs in groups.values():
            for im in imgs:
                others = [o for o in imgs if o != im]
                for k, (c, b) in enumerate(gt[im]):
                    hits = sum(any(c2 == c and iou(b, b2) > 0.3 for c2, b2 in gt[o]) for o in others)
                    stab[(im, k)] = hits / len(others) if others else 1.0
        preds, times = {}, []
        for n, r in enumerate(rows):
            image = read(data / "images" / r["image"])
            t = time.perf_counter()
            res = model.predict(image, imgsz=1280, conf=0.25, device="cpu", verbose=False)[0]
            if n:                                   # 첫 장은 준비 시간이 섞인다
                times.append(time.perf_counter() - t)
            preds[r["image"]] = [(int(c), tuple(map(float, b)), float(s))
                                 for c, b, s in zip(res.boxes.cls.tolist(), res.boxes.xyxy.tolist(), res.boxes.conf.tolist())]
        stats = defaultdict(lambda: {"n": 0, "found": 0})
        moved = []
        for r in rows:
            im = r["image"]
            for k, (c, b) in enumerate(gt[im]):
                kind = "moved" if stab[(im, k)] < 0.15 else "static"
                same = max([iou(b, pb) for pc, pb, ps in preds[im] if pc == c], default=0)
                for key in (kind, f"{kind}_{r['camera']}"):
                    stats[key]["n"] += 1
                    stats[key]["found"] += same >= 0.5
                if kind == "moved":
                    moved.append({"image": im, "camera": r["camera"], "cls": NAMES[c], "box": [round(x) for x in b], "iou": round(same, 2)})
        times.sort()
        result[split] = {"images": len(rows), "stats": dict(sorted(stats.items())),
                         "sec_per_image": {"median": round(times[len(times) // 2], 3), "mean": round(sum(times) / len(times), 3),
                                           "p90": round(times[int(len(times) * 0.9)], 3)}}
        print(split, json.dumps(result[split], ensure_ascii=False), flush=True)
        if split == "test":                          # 눈으로 볼 것: 놓친 것 · 찾은 것 몇 장씩 (초록 = 정답, 빨강 = 예측)
            for tag, items in (("MISS", [m for m in moved if m["iou"] < 0.5]), ("HIT", [m for m in moved if m["iou"] >= 0.5])):
                tiles, seen = [], set()
                for m in items:
                    if m["image"] in seen or len(tiles) >= 8:
                        continue
                    seen.add(m["image"])
                    img = read(data / "images" / m["image"])
                    for c, b in gt[m["image"]]:
                        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 1)
                    for pc, pb, ps in preds[m["image"]]:
                        cv2.rectangle(img, (int(pb[0]), int(pb[1])), (int(pb[2]), int(pb[3])), (0, 0, 255), 1)
                    x1, y1, x2, y2 = m["box"]
                    cx, cy, half = (x1 + x2) // 2, (y1 + y2) // 2, 150
                    x0, y0 = min(max(0, cx - half), W - 2 * half), min(max(0, cy - half), H - 2 * half)
                    crop = cv2.resize(img[y0:y0 + 2 * half, x0:x0 + 2 * half], (450, 450), interpolation=cv2.INTER_CUBIC)
                    cv2.putText(crop, f"{m['cls']} {m['image'][:16]}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    tiles.append(crop)
                if tiles:
                    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 4)
                    sheet = np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)])
                    (out / f"moved_{tag}.jpg").write_bytes(cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes())
    (out / "moved_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
