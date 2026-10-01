"""공용 유틸: 시드, 로깅, 지표."""
from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

_RTSP_CRED = re.compile(r"(?P<scheme>\w+://)(?P<user>[^:/@]+)(:(?P<pw>[^@/]*))?@")


def mask_url(url: str) -> str:
    """rtsp://user:pass@host/… → rtsp://***:***@host/…  (로그/응답 노출 금지)"""
    if not url:
        return url
    return _RTSP_CRED.sub(lambda m: f"{m.group('scheme')}***:***@", url)


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def get_logger(name: str = "storeguard", logfile: Path | None = None) -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    if logfile:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    return log


def write_json(path: str | Path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def lap(self) -> float:
        t = time.perf_counter()
        d = t - self.t0
        self.t0 = t
        return d


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def prf_per_class(cm: np.ndarray) -> list[dict]:
    out = []
    for i in range(cm.shape[0]):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out.append({"tp": tp, "fp": fp, "fn": fn, "support": int(cm[i, :].sum()),
                    "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)})
    return out


def macro_f1(cm: np.ndarray) -> float:
    return round(float(np.mean([c["f1"] for c in prf_per_class(cm)])), 4)
