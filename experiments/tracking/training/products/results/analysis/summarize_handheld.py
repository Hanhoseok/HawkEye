"""analysis/handheld/<모델>_<데이터>_<나눔>/summary.json 을 모아 손에 든 상품 찾은 비율을 표로 낸다.

    python -I summarize_handheld.py <handheld 폴더> 모델1 모델2 ...

- old  = 기존 1,194장 (dataset)
- new  = 묶음 2 만 (dataset_v2 에서 이름에 '_rf' 가 없는 그림). '움직인 상품' 판정은 같은 영상 · 카메라의 다른 장면과 견준다.
"""
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
models = sys.argv[2:]


def load(m, d, s):
    p = root / f"{m}_{d}_{s}" / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


print(f"{'model':18s} {'split':5s} {'old moved':>14s} {'old static':>16s} {'new moved':>14s}  new moved by class")
for s in ("val", "test"):
    for m in models:
        a, b = load(m, "dataset", s), load(m, "dataset_v2", s)
        if not a or not b:
            print(f"{m:18s} {s:5s}  (없음)")
            continue
        mv, st = a["summary"]["moved"], a["summary"]["static"]
        new = [x for x in b["moved"] if "_rf" not in x["image"]]
        hit = [x for x in new if x["iou_same_cls"] >= 0.5]
        n_c, h_c = Counter(x["cls"] for x in new), Counter(x["cls"] for x in hit)
        print(f"{m:18s} {s:5s} {mv['found']:>4d}/{mv['n']:<4d}{mv['found'] / mv['n']:>5.0%} "
              f"{st['found']:>5d}/{st['n']:<5d}{st['found'] / st['n']:>5.1%} "
              f"{len(hit):>4d}/{len(new):<4d}{len(hit) / max(1, len(new)):>5.0%}  "
              + " ".join(f"{c} {h_c[c]}/{n_c[c]}" for c in sorted(n_c)))
