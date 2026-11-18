"""상품 탐지 모델을 학습하고, 안 쓴 영상으로 점수를 내고, 가져갈 것을 한 폴더에 모은다.

    python train.py                                   # 기본: yolov8n · 해상도 1280 · 최대 100바퀴 (20바퀴 동안 안 나아지면 멈춤)
    python train.py --model yolov8s.pt --name s1280   # 더 큰 모델
    python train.py --resume <이름>                    # 끊긴 학습을 이어서
    python train.py --score <이름>                     # 학습은 하지 않고, 있는 모델의 점수만 다시

하는 일 (차례대로):
  1. 데이터 검사 (check.py 와 같다) — 안 맞으면 시작하지 않는다
  2. 학습 → runs/<이름>/
  3. 가장 좋았던 모델(best.pt)로 검증(val) · 시험(test) 점수를 전체 · 카메라별로 낸다
  4. handoff/<이름>/ 에 모은다: best.pt · result.json · results.csv · args.yaml   ← **이 폴더를 원래 컴퓨터로 가져간다**

이 폴더의 다른 파일과 ultralytics 만 쓴다 (저장소의 다른 코드에 기대지 않는다).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from check import CAMERAS, DATA, HERE, check, read_list, report

RUNS = HERE / "runs"
HANDOFF = HERE / "handoff"


def pick_device(asked: str = "auto") -> str:
    """auto = GPU 가 있으면 첫 번째 GPU, 없으면 CPU."""
    if asked != "auto":
        return asked
    import torch

    return "0" if torch.cuda.is_available() else "cpu"


def scores_of(metrics) -> dict:
    """ultralytics 의 검증 결과 → 적어 둘 숫자. map50 = 상자가 절반 넘게 겹치면 맞은 것으로 친 점수 (0~1)."""
    box, names = metrics.box, metrics.names
    out = {"precision": round(float(box.mp), 4), "recall": round(float(box.mr), 4),
           "map50": round(float(box.map50), 4), "map50_95": round(float(box.map), 4), "classes": {}}
    for k, cls in enumerate(box.ap_class_index):
        out["classes"][names[int(cls)]] = {"precision": round(float(box.p[k]), 4), "recall": round(float(box.r[k]), 4),
                                           "map50": round(float(box.ap50[k]), 4), "map50_95": round(float(box.ap[k]), 4)}
    return out


def epochs_run(results_csv: Path) -> dict:
    """results.csv → 실제로 돈 바퀴 수와 검증 점수가 가장 좋았던 바퀴."""
    if not results_csv.exists():
        return {"run": 0, "best": None}
    with results_csv.open(encoding="utf-8", newline="") as f:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    key = "metrics/mAP50-95(B)"
    best = max(rows, key=lambda r: float(r.get(key) or 0), default=None)
    return {"run": len(rows), "best": int(float(best["epoch"])) if best else None}


def environment(device: str) -> dict:
    import torch
    import ultralytics

    env = {"python": platform.python_version(), "os": platform.platform(), "torch": torch.__version__,
           "ultralytics": ultralytics.__version__, "device": device, "cpu_count": os.cpu_count(), "gpu": None, "git": None}
    if device != "cpu" and torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
    try:
        env["git"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True,
                                    timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return env


def score_all(best: Path, imgsz: int, batch: int, device: str, workers: int) -> dict:
    """검증 · 시험 점수를 전체 · 카메라별로. 목록이 빈 쪽은 건너뛴다."""
    from ultralytics import YOLO

    out = {}
    for side in ("val", "test"):
        for which in ("all",) + CAMERAS:
            suffix = "" if which == "all" else f"_{which}"
            if not read_list(DATA / f"{side}{suffix}.txt"):
                continue
            m = YOLO(str(best)).val(data=str(DATA / f"data{suffix}.yaml"), split=side, imgsz=imgsz, batch=batch, device=device,
                                    workers=workers, plots=False, verbose=False, project=str(RUNS), name=f"_score_{side}{suffix}",
                                    exist_ok=True)
            out[f"{side}_{which}"] = scores_of(m)
    return out


def summary(result: dict) -> str:
    lines = [f"모델 {result['model']} · 해상도 {result['imgsz']} · {result['epochs']['run']}바퀴 돌았고 가장 좋았던 바퀴는 {result['epochs']['best']}"
             f" · 학습 {result['train_seconds'] / 60:.0f}분"]
    for key, s in result["scores"].items():
        side, which = key.split("_", 1)
        name = {"val": "검증", "test": "시험"}[side] + " " + {"all": "전체", "angle0": "선반 카메라", "angle1": "전체 카메라"}[which]
        lines.append(f"  {name:12s} map50 {s['map50']:.3f} · map50-95 {s['map50_95']:.3f} · 정밀도 {s['precision']:.3f} · 재현율 {s['recall']:.3f}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="yolov8n.pt", help="시작 가중치 (없으면 ultralytics 가 내려받는다)")
    ap.add_argument("--imgsz", type=int, default=1280, help="학습 해상도 — 상품이 작아 1280 을 기본으로 둔다")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20, help="이만큼의 바퀴 동안 검증 점수가 안 나아지면 멈춘다")
    ap.add_argument("--batch", type=int, default=0, help="한 번에 넣는 그림 수 (기본: GPU 16 · CPU 8). 메모리가 모자라면 줄인다")
    ap.add_argument("--device", default="auto", help="auto · cpu · 0 (첫 GPU) …")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--name", default="", help="이 학습의 이름 (기본: 모델_해상도)")
    ap.add_argument("--fraction", type=float, default=1.0, help="학습 그림의 일부만 (돌아가는지 볼 때만 쓴다)")
    ap.add_argument("--resume", default="", metavar="이름", help="runs/<이름> 의 끊긴 학습을 이어서")
    ap.add_argument("--score", default="", metavar="이름", help="runs/<이름> 의 best.pt 로 점수만 다시 낸다")
    args = ap.parse_args()

    c = check()
    print(report(c), flush=True)
    if not c["ok"]:
        return 1

    from ultralytics import YOLO

    device = pick_device(args.device)
    batch = args.batch or (8 if device == "cpu" else 16)
    started = time.time()
    if args.score:
        save = RUNS / args.score
    elif args.resume:
        model = YOLO(str(RUNS / args.resume / "weights" / "last.pt"))
        model.train(resume=True)
        save = Path(model.trainer.save_dir)
    else:
        name = args.name or f"{Path(args.model).stem}_{args.imgsz}"
        model = YOLO(args.model)
        model.train(data=str(DATA / "data.yaml"), imgsz=args.imgsz, epochs=args.epochs, patience=args.patience, batch=batch,
                    device=device, workers=args.workers, fraction=args.fraction, project=str(RUNS), name=name)
        save = Path(model.trainer.save_dir)        # 같은 이름이 있으면 ultralytics 가 번호를 붙인다
    seconds = time.time() - started
    if args.score:                                 # 점수만 다시 낼 때는 지난번에 적어 둔 학습 시간을 그대로 둔다
        old = HANDOFF / save.name / "result.json"
        seconds = json.loads(old.read_text(encoding="utf-8")).get("train_seconds", 0) if old.exists() else 0
    best = save / "weights" / "best.pt"
    if not best.exists():
        print(f"{best} 없음 — 학습이 끝나지 않았다")
        return 1

    used = {}
    if (save / "args.yaml").exists():
        import yaml

        used = yaml.safe_load((save / "args.yaml").read_text(encoding="utf-8")) or {}
    imgsz = int(used.get("imgsz") or args.imgsz)
    result = {"name": save.name, "model": str(used.get("model") or args.model), "imgsz": imgsz,
              "epochs": {"asked": int(used.get("epochs") or args.epochs), **epochs_run(save / "results.csv")},
              "patience": used.get("patience"), "batch": used.get("batch") or batch, "fraction": used.get("fraction", args.fraction),
              "train_seconds": round(seconds), "environment": environment(device),
              "dataset": {"images": c["images"], "boxes": c["boxes"], **c["by_split"]},
              "scores": score_all(best, imgsz, batch, device, args.workers)}

    out = HANDOFF / save.name
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best, out / "best.pt")
    for extra in ("results.csv", "args.yaml"):
        if (save / extra).exists():
            shutil.copyfile(save / extra, out / extra)
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("\n" + summary(result))
    print(f"\n가져갈 폴더: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
