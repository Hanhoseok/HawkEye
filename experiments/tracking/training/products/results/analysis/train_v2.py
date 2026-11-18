"""dataset_v2 (기존 1,194장 + 묶음 2) 로 추가 학습하고, 옛 모델과 새 모델을 같은 잣대로 견준다.

    python -I train_v2.py <products 폴더> [--base yolov8s_1280] [--name yolov8s_1280_v2] [--epochs 100] [--workers 4]
    python -I train_v2.py <products 폴더> --score-only yolov8s_1280_v2      # 학습 없이 점수만

- 시작 가중치: handoff/<base>/best.pt (처음부터가 아니라 이어서 배움)
- 점수: val · test × (전체 · old = 기존 1,194장 · new = 묶음 2 · 카메라별). 옛 모델 n · s 도 같은 목록으로 낸다.
- 결과: handoff/<name>/ (best.pt · result.json · results.csv · args.yaml)
"""
import argparse
import json
import shutil
import time
from pathlib import Path

from ultralytics import YOLO

if __name__ == "__main__":   # Windows: DataLoader 일꾼이 이 파일을 다시 읽는다
    ap = argparse.ArgumentParser()
    ap.add_argument("prod", type=Path)
    ap.add_argument("--base", default="yolov8s_1280")
    ap.add_argument("--name", default="yolov8s_1280_v2")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--score-only", default="")
    a = ap.parse_args()

    prod = a.prod.resolve()
    data = prod / "dataset_v2"
    runs, handoff = prod / "runs", prod / "handoff"


    def scores_of(m):
        b, names = m.box, m.names
        out = {"precision": round(float(b.mp), 4), "recall": round(float(b.mr), 4),
               "map50": round(float(b.map50), 4), "map50_95": round(float(b.map), 4), "classes": {}}
        for k, c in enumerate(b.ap_class_index):
            out["classes"][names[int(c)]] = {"precision": round(float(b.p[k]), 4), "recall": round(float(b.r[k]), 4),
                                             "map50": round(float(b.ap50[k]), 4), "map50_95": round(float(b.ap[k]), 4)}
        return out


    def score(weights: Path):
        res = {}
        for side in ("val", "test"):
            for which in ("", "_old", "_new", "_angle0", "_angle1"):
                m = YOLO(str(weights)).val(data=str(data / f"data{which}.yaml"), split=side, imgsz=a.imgsz, batch=a.batch,
                                           device=0, workers=a.workers, plots=False, verbose=False, project=str(runs),
                                           name=f"_score_v2_{side}{which}", exist_ok=True)
                res[f"{side}{which or '_all'}"] = scores_of(m)
        return res


    started = time.time()
    if a.score_only:
        save = runs / a.score_only
    else:
        model = YOLO(str(handoff / a.base / "best.pt"))
        model.train(data=str(data / "data.yaml"), imgsz=a.imgsz, epochs=a.epochs, patience=a.patience, batch=a.batch,
                    device=0, workers=a.workers, project=str(runs), name=a.name)
        save = Path(model.trainer.save_dir)
    seconds = round(time.time() - started)

    best = save / "weights" / "best.pt"
    out = handoff / save.name
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best, out / "best.pt")
    for extra in ("results.csv", "args.yaml"):
        if (save / extra).exists():
            shutil.copyfile(save / extra, out / extra)

    result = {"name": save.name, "base": a.base, "data": "dataset_v2", "imgsz": a.imgsz, "batch": a.batch,
              "train_seconds": None if a.score_only else seconds,
              "scores": {save.name: score(best)}}
    for old in ("yolov8n_1280", "yolov8s_1280"):
        if (handoff / old / "best.pt").exists():
            result["scores"][old] = score(handoff / old / "best.pt")
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    print(f"\n학습 {seconds / 60:.0f}분 · 결과 {out}")
    keys = ["val_all", "val_old", "val_new", "test_all", "test_old", "test_new"]
    print(f"{'model':18s} " + " ".join(f"{k:>16s}" for k in keys))
    for model_name, s in result["scores"].items():
        print(f"{model_name:18s} " + " ".join(f"{s[k]['map50']:.3f}/{s[k]['map50_95']:.3f}".rjust(16) for k in keys))
