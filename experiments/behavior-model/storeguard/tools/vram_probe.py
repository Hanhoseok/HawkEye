"""VRAM 실측 도구.

배치 크기를 키우면서 학습 1스텝(forward+backward)과 추론 1스텝을 실제로 돌려
GTX 1050 Ti 4GB 에서 어디까지 되는지 확인한다. 추측하지 않고 측정한다.

PowerShell 예:
    D:\\computervision\\.venv\\Scripts\\python.exe -m storeguard.tools.vram_probe
    D:\\computervision\\.venv\\Scripts\\python.exe -m storeguard.tools.vram_probe --arch mc3_18 --crop 160
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from storeguard.config import Config, env_report
from storeguard.models.factory import build_model, count_params
from storeguard.utils import write_json


def try_batch(arch: str, bs: int, clip_len: int, crop: int, n_classes: int,
              amp: bool, device: torch.device, train: bool) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # 실제 학습/추론 경로와 같은 조건으로 잰다.
    # (실측) benchmark 를 끄면 배치 1 추론이 441ms, 켜고 워밍업하면 76ms 로 5배 이상 차이난다.
    torch.backends.cudnn.benchmark = True
    try:
        model = build_model(arch, n_classes, pretrained=False).to(device)
        model.train(train)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4) if train else None
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        crit = nn.CrossEntropyLoss()
        x = torch.randn(bs, 3, clip_len, crop, crop, device=device)
        y = torch.randint(0, n_classes, (bs,), device=device)

        times = []
        # cudnn 자동 튜닝 비용이 첫 스텝들에 몰리므로 워밍업을 충분히 준다(측정 왜곡 방지)
        # (실측) 배치 1에서는 워밍업 3회로 부족해 지연이 441ms 로 과대 측정되었다. 6회면 77ms 로 수렴한다.
        n_warmup, n_timed = 6, 6
        for i in range(n_warmup + n_timed):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if train:
                with torch.autocast("cuda", enabled=amp):
                    loss = crit(model(x), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                with torch.no_grad(), torch.autocast("cuda", enabled=amp):
                    model(x)
            torch.cuda.synchronize()
            if i >= n_warmup:
                times.append(time.perf_counter() - t0)
        peak = torch.cuda.max_memory_allocated() / 1024**3
        res = {
            "ok": True, "batch": bs, "peak_gb": round(peak, 3),
            "step_ms": round(1000 * sum(times) / len(times), 1),
            "clips_per_sec": round(bs * len(times) / sum(times), 1),
        }
    except torch.cuda.OutOfMemoryError:
        res = {"ok": False, "batch": bs, "error": "CUDA OOM"}
    except RuntimeError as exc:
        res = {"ok": False, "batch": bs, "error": str(exc)[:200]}
    finally:
        for v in ("model", "opt", "x", "y"):
            if v in dir():
                pass
        torch.cuda.empty_cache()
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--arch", default=None)
    ap.add_argument("--crop", type=int, default=None)
    ap.add_argument("--clip-len", type=int, default=None)
    ap.add_argument("--batches", default="1,2,4,8,12,16")
    ap.add_argument("--no-amp", action="store_true")
    a = ap.parse_args(argv)

    cfg = Config.load(a.config)
    if not torch.cuda.is_available():
        print("CUDA 를 쓸 수 없다. CPU 로는 이 측정이 의미 없다.")
        return 1

    arch = a.arch or str(cfg["model.arch"])
    crop = a.crop or int(cfg["data.train_crop"])
    clip_len = a.clip_len or int(cfg["data.clip_len"])
    amp = not a.no_amp and bool(cfg["train.amp"])
    device = torch.device("cuda")
    n_classes = len(cfg.class_names)

    m = build_model(arch, n_classes, pretrained=False)
    out = {
        "env": env_report(), "arch": arch, "params_M": round(count_params(m) / 1e6, 2),
        "input": f"{clip_len}x{crop}x{crop}", "amp": amp,
        "train_steps": [], "infer_steps": [],
    }
    del m

    for bs in [int(b) for b in a.batches.split(",")]:
        r = try_batch(arch, bs, clip_len, crop, n_classes, amp, device, train=True)
        print("학습", json.dumps(r, ensure_ascii=False))
        out["train_steps"].append(r)
        if not r["ok"]:
            break
    for bs in [int(b) for b in a.batches.split(",")]:
        r = try_batch(arch, bs, clip_len, crop, n_classes, amp, device, train=False)
        print("추론", json.dumps(r, ensure_ascii=False))
        out["infer_steps"].append(r)
        if not r["ok"]:
            break

    ok_train = [r for r in out["train_steps"] if r["ok"]]
    ok_infer = [r for r in out["infer_steps"] if r["ok"]]
    out["권장_train_batch"] = ok_train[-1]["batch"] if ok_train else 0
    out["최대_infer_batch"] = ok_infer[-1]["batch"] if ok_infer else 0
    out["실시간_추론_1클립_지연_ms"] = next(
        (r["step_ms"] for r in out["infer_steps"] if r["ok"] and r["batch"] == 1), None)

    path = cfg.path("paths.reports") / f"vram_probe_{arch}_{clip_len}x{crop}.json"
    write_json(path, out)
    print(json.dumps({k: out[k] for k in
                      ("arch", "params_M", "input", "amp", "권장_train_batch",
                       "최대_infer_batch", "실시간_추론_1클립_지연_ms")},
                     ensure_ascii=False, indent=2))
    print(f"저장: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
