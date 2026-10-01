"""학습 스크립트.

특징
  - Kinetics-400 사전학습 백본 전이학습 (백본 lr 축소)
  - AMP(fp16) 기본 사용. GTX 1050 Ti 4GB 에서 사실상 필수
  - 클래스 불균형: WeightedRandomSampler + 손실 클래스 가중
  - 에폭마다 체크포인트 저장, --resume 으로 재개
  - val macro-F1 기준 best 체크포인트 선택
  - 설정/클래스매핑/환경 스냅샷을 체크포인트 옆에 저장(재현용)

PowerShell 예:
    D:\\computervision\\.venv\\Scripts\\python.exe -m storeguard.train --name baseline_r2p1d
    D:\\computervision\\.venv\\Scripts\\python.exe -m storeguard.train --name baseline_r2p1d --resume
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from storeguard.config import Config, env_report
from storeguard.data.dataset import make_datasets
from storeguard.models.factory import build_model, count_params, param_groups
from storeguard.utils import confusion_matrix, get_logger, macro_f1, prf_per_class, set_seed, write_json


def restrict_classes_to_data(cfg: Config, log) -> list[str]:
    """학습 데이터에 실제로 등장하는 클래스만 남긴다.

    왜 필요한가
      설정(`classes.names`)에는 이상행동 3종 + 구매행동 6종이 모두 적혀 있다.
      그런데 구매행동 데이터가 아직 없으면, 그대로 학습할 경우 **한 번도 학습되지 않은
      출력 유닛 6개**가 체크포인트에 남는다. 대시보드는 체크포인트의 클래스 목록을 믿으므로
      "구매행동도 탐지한다"고 잘못 표시된다.
      그래서 학습 클립에 한 번도 나오지 않은 클래스는 아예 빼고 학습한다.

    배경 클래스는 항상 남긴다(음성 클래스가 없으면 모델이 성립하지 않는다).
    """
    from storeguard.data.clips import build_clip_list
    from storeguard.data.dataset import load_index, load_splits

    rows = load_index(cfg)
    splits = load_splits(cfg)
    present = {s.label for s in build_clip_list(rows, splits["train"], cfg)}
    keep = [c for c in cfg.class_names
            if c == "background_unlabeled" or c in present]
    dropped = [c for c in cfg.class_names if c not in keep]
    if dropped:
        log.warning("학습 데이터에 없는 클래스를 제외한다: %s", ", ".join(dropped))
        log.warning("  → 이 클래스들은 모델 출력에 포함되지 않는다. "
                    "해당 데이터를 넣고 다시 학습해야 탐지된다.")
        cfg.set("classes.names", keep)
    return dropped


def _seed_worker(worker_id: int) -> None:
    """Windows spawn 환경에서 워커마다 numpy 시드가 같아지면 증강이 복제된다.

    모듈 최상위에 두어야 pickle 이 된다(지역 함수는 spawn 에서 직렬화 실패).
    """
    base = torch.initial_seed() % (2**31 - 1)
    np.random.seed((base + worker_id) % (2**31 - 1))


def make_loaders(cfg: Config, ds: dict):
    bs = int(cfg["train.batch_size"])
    nw = int(cfg["train.num_workers"])
    imb = str(cfg.get("train.imbalance", "both"))
    train_ds = ds["train"]

    sampler = None
    if imb in ("sampler", "both") and len(train_ds):
        counts = np.bincount(train_ds.labels, minlength=len(cfg.class_names)).astype(np.float64)
        counts[counts == 0] = 1.0
        w = (1.0 / counts)[train_ds.labels]
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                        num_samples=len(train_ds), replacement=True)

    common = dict(num_workers=nw, pin_memory=True, persistent_workers=nw > 0,
                  prefetch_factor=2 if nw > 0 else None,
                  worker_init_fn=_seed_worker if nw > 0 else None)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=sampler is None,
                              sampler=sampler, drop_last=True, **common)
    val_loader = DataLoader(ds["val"], batch_size=int(cfg["eval.batch_size"]),
                            shuffle=False, **common)
    return train_loader, val_loader


def class_weights(cfg: Config, ds) -> torch.Tensor | None:
    """손실 클래스 가중치.

    'weights'         : 전 클래스 역빈도. 배경까지 끌어올려 배경/행동 비율을 크게 바꾼다.
    'action_balanced' : 배경 가중은 1.0 으로 고정하고 **배경을 제외한 모든 행동 클래스끼리**
                        균형을 맞춘다. 실제 매장은 대부분이 배경이므로 배경 비중을 낮추면
                        오경보가 늘어난다. 반면 행동 클래스 사이의 수량 차이는 크다
                        (이상행동은 영상 700여 개, 구매행동은 100여 개). 이쪽만 보정한다.
    """
    mode = str(cfg.get("train.imbalance", "none"))
    if mode not in ("weights", "both", "action_balanced"):
        return None
    counts = np.bincount(ds.labels, minlength=len(cfg.class_names)).astype(np.float64)
    counts[counts == 0] = 1.0
    if mode == "action_balanced":
        bg = cfg.class_to_idx.get("background_unlabeled", -1)
        # 배경을 뺀 **모든** 클래스(이상행동 + 구매행동)를 균형 대상으로 본다.
        act_idx = [i for i in range(len(counts)) if i != bg]
        target = float(np.mean(counts[act_idx]))
        w = np.ones_like(counts)
        for i in act_idx:
            # 극단적인 가중치는 학습을 불안정하게 만든다. 0.2~5배로 자른다.
            w[i] = float(np.clip(target / counts[i], 0.2, 5.0))
    else:
        w = counts.sum() / (len(counts) * counts)
    return torch.as_tensor(w, dtype=torch.float32)


@torch.no_grad()
def evaluate_loader(model, loader, device, n_classes, amp: bool) -> dict:
    model.eval()
    ys, ps, probs = [], [], []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            logit = model(x)
        p = torch.softmax(logit.float(), dim=1).cpu().numpy()
        probs.append(p)
        ps.append(p.argmax(1))
        ys.append(y.numpy())
    if not ys:
        return {"macro_f1": 0.0, "cm": np.zeros((n_classes, n_classes), dtype=np.int64),
                "per_class": [], "n": 0}
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    cm = confusion_matrix(y, p, n_classes)
    return {"macro_f1": macro_f1(cm), "cm": cm, "per_class": prf_per_class(cm),
            "n": int(len(y)), "acc": round(float((y == p).mean()), 4),
            "probs": np.concatenate(probs)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--name", default="baseline")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0, help="스모크 테스트용 스텝 제한")
    ap.add_argument("--keep-all-classes", action="store_true",
                    help="학습 데이터에 없는 클래스도 출력 유닛으로 남긴다(기본은 제외)")
    a = ap.parse_args(argv)

    cfg = Config.load(a.config, a.overrides)
    run_dir = cfg.path("paths.runs") / a.name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger("train", run_dir / "train.log")

    set_seed(int(cfg["project.seed"]), bool(cfg.get("project.deterministic", False)))
    device = torch.device(cfg.device())
    log.info("device=%s", device)
    log.info("env=%s", json.dumps(env_report(), ensure_ascii=False))

    dropped_classes = [] if a.keep_all_classes else restrict_classes_to_data(cfg, log)
    log.info("학습 클래스 %d개: %s", len(cfg.class_names), ", ".join(cfg.class_names))

    ds = make_datasets(cfg)
    for k, d in ds.items():
        log.info("%s: 클립 %d개 / 영상 %d개 / 클래스 %s",
                 k, len(d), len({s.stem for s in d.specs}), d.class_counts())
    if len(ds["train"]) == 0:
        log.error("학습 클립이 0개다. 캐시(data/cache)와 분할(splits.json)을 확인할 것.")
        return 2

    n_classes = len(cfg.class_names)
    model = build_model(str(cfg["model.arch"]), n_classes,
                        pretrained=bool(cfg["model.pretrained"]),
                        dropout=float(cfg["model.dropout"])).to(device)
    log.info("model=%s params=%.1fM pretrained=%s",
             cfg["model.arch"], count_params(model) / 1e6, cfg["model.pretrained"])

    train_loader, val_loader = make_loaders(cfg, ds)
    cw = class_weights(cfg, ds["train"])
    if cw is not None:
        log.info("class weights=%s", [round(float(v), 3) for v in cw])
    crit = nn.CrossEntropyLoss(weight=None if cw is None else cw.to(device),
                               label_smoothing=float(cfg.get("train.label_smoothing", 0.0)))

    opt = torch.optim.AdamW(param_groups(model, float(cfg["train.lr"]),
                                         float(cfg["train.backbone_lr_mult"]),
                                         float(cfg["train.weight_decay"])))
    epochs = int(cfg["train.epochs"])
    steps_per_epoch = max(1, len(train_loader))
    warmup = int(cfg.get("train.warmup_epochs", 0)) * steps_per_epoch
    total = epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    amp = bool(cfg["train.amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    ckpt_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    start_epoch, best_metric, history = 0, -1.0, []
    if a.resume and ckpt_path.exists():
        st = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        start_epoch = st["epoch"] + 1
        best_metric = st.get("best_metric", -1.0)
        history = st.get("history", [])
        log.info("재개: epoch %d 부터, best=%.4f", start_epoch, best_metric)

    cfg.save(run_dir / "config.yaml")
    write_json(run_dir / "classes.json",
               {"names": cfg.class_names, "class_to_idx": cfg.class_to_idx,
                "ko": cfg.get("classes.ko", {}),
                "alerting": cfg.action_classes,
                "excluded_no_data": dropped_classes})
    write_json(run_dir / "env.json", env_report())
    write_json(run_dir / "dataset_stats.json",
               {k: {"clips": len(d), "videos": len({s.stem for s in d.specs}),
                    "class_counts": d.class_counts()} for k, d in ds.items()})

    monitor = str(cfg.get("train.monitor", "macro_f1"))
    patience = int(cfg.get("train.early_stop_patience", 0))
    bad_epochs = 0
    grad_clip = float(cfg.get("train.grad_clip", 0) or 0)

    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        running, seen, correct = 0.0, 0, 0
        for step, (x, y, _) in enumerate(train_loader):
            if a.max_steps and step >= a.max_steps:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=amp):
                out = model(x)
                loss = crit(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()

            running += float(loss.item()) * y.size(0)
            seen += y.size(0)
            correct += int((out.argmax(1) == y).sum().item())
            if step % 20 == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0
                log.info("ep%d %d/%d loss=%.4f acc=%.3f lr=%.2e vram=%.2fGB",
                         epoch, step, steps_per_epoch, running / max(seen, 1),
                         correct / max(seen, 1), sched.get_last_lr()[-1], mem)

        val = evaluate_loader(model, val_loader, device, n_classes, amp)
        rec = {
            "epoch": epoch, "train_loss": round(running / max(seen, 1), 4),
            "train_acc": round(correct / max(seen, 1), 4),
            "val_macro_f1": val["macro_f1"], "val_acc": val.get("acc", 0.0),
            "val_n": val["n"], "sec": round(time.time() - t0, 1),
            "val_per_class": {c: val["per_class"][i] for i, c in enumerate(cfg.class_names)}
            if val["per_class"] else {},
        }
        history.append(rec)
        log.info("epoch %d 완료: %s", epoch, json.dumps(
            {k: rec[k] for k in ("train_loss", "train_acc", "val_macro_f1", "val_acc", "sec")},
            ensure_ascii=False))
        log.info("val confusion\n%s", val["cm"])

        state = {"model": model.state_dict(), "optimizer": opt.state_dict(),
                 "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                 "epoch": epoch, "best_metric": best_metric, "history": history,
                 "config": cfg.raw, "classes": cfg.class_names}
        torch.save(state, ckpt_path)

        cur = rec[f"val_{monitor}"] if f"val_{monitor}" in rec else rec["val_macro_f1"]
        if cur > best_metric:
            best_metric = cur
            state["best_metric"] = best_metric
            # best.pt 에는 옵티마이저 상태를 넣지 않는다. 재개는 last.pt 로만 한다.
            # (AdamW 상태가 모델의 2배라 빼면 파일이 약 1/3 로 줄어든다. Colab 처럼
            #  체크포인트를 원격 저장소에 쓰는 환경에서 차이가 크다.)
            torch.save({k: v for k, v in state.items()
                        if k not in ("optimizer", "scheduler", "scaler")}, best_path)
            write_json(run_dir / "best_val.json",
                       {"epoch": epoch, monitor: cur,
                        "per_class": rec["val_per_class"], "cm": val["cm"].tolist()})
            log.info("best 갱신: %s=%.4f -> %s", monitor, cur, best_path.name)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if patience and bad_epochs >= patience:
                log.info("early stop (%d 에폭 개선 없음)", bad_epochs)
                break
        write_json(run_dir / "history.json", history)

    log.info("학습 종료. best %s=%.4f, 체크포인트=%s", monitor, best_metric, best_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
