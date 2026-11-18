import json, sys
from collections import Counter
from pathlib import Path

res, *moved_files = sys.argv[1:]
r = json.loads(Path(res).read_text(encoding="utf-8"))
for key in ("val_all", "test_all", "test_angle0", "test_angle1"):
    print(key)
    for name, s in r["scores"][key]["classes"].items():
        print(f"  {name:11s} map50 {s['map50']:.3f}  map50-95 {s['map50_95']:.3f}  P {s['precision']:.3f}  R {s['recall']:.3f}")
for f in moved_files:
    m = json.loads(Path(f).read_text(encoding="utf-8"))["moved"]
    n, hit = Counter(x["cls"] for x in m), Counter(x["cls"] for x in m if x["iou_same_cls"] >= 0.5)
    wrong = Counter((x["cls"], x["best_pred"]["cls"]) for x in m
                    if x["iou_same_cls"] < 0.5 and x["best_pred"] and x["best_pred"]["iou"] >= 0.5)
    print(f, {c: f"{hit[c]}/{n[c]}" for c in n}, "wrong class:", dict(wrong))
