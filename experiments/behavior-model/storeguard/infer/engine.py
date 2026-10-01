"""모델 로딩과 클립 추론.

- 체크포인트 옆의 config.yaml / classes.json 을 읽어 **학습과 동일한 전처리**를 재현한다.
- 가중치가 없으면 로드하지 않고 상태를 MODEL_NOT_LOADED 로 둔다.
  이 경우 어떤 점수도 만들어내지 않는다(가짜 탐지 금지).
"""
from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from storeguard.config import Config


class ModelNotLoaded(RuntimeError):
    pass


class ClipClassifier:
    """(T, H, W, 3) uint8 RGB 클립 → 클래스 확률."""

    def __init__(self, run_dir: Path | None, device: str = "auto", cfg: Config | None = None):
        self.run_dir = Path(run_dir) if run_dir else None
        self.state = "MODEL_NOT_LOADED"
        self.error: str | None = None
        self.cfg = cfg
        self.model = None
        self.device = None
        self.class_names: list[str] = cfg.class_names if cfg else []
        self.checkpoint: dict = {}
        self._lock = threading.Lock()
        self.last_infer_ms: float | None = None
        self.warmup_ms: float | None = None
        self._times: deque[float] = deque(maxlen=50)
        if self.run_dir is not None:
            self.load(device)

    # --- 로딩 ---
    def load(self, device: str = "auto") -> bool:
        import torch

        self.state = "LOADING"
        try:
            if self.run_dir is None or not self.run_dir.exists():
                raise FileNotFoundError(f"run 디렉터리 없음: {self.run_dir}")
            ckpt = self.run_dir / "best.pt"
            if not ckpt.exists():
                ckpt = self.run_dir / "last.pt"
            if not ckpt.exists():
                raise FileNotFoundError(f"체크포인트(best.pt/last.pt) 없음: {self.run_dir}")
            cfg_path = self.run_dir / "config.yaml"
            self.cfg = Config.load(cfg_path) if cfg_path.exists() else self.cfg
            if self.cfg is None:
                raise FileNotFoundError("설정 파일 없음")

            from storeguard.models.factory import build_model

            dev = self.cfg.device() if device == "auto" else device
            self.device = torch.device(dev)
            st = torch.load(ckpt, map_location=self.device, weights_only=False)
            names = st.get("classes") or self.cfg.class_names
            model = build_model(str(self.cfg["model.arch"]), len(names),
                                pretrained=False, dropout=float(self.cfg["model.dropout"]))
            model.load_state_dict(st["model"])
            model.to(self.device).eval()
            self.model = model
            self.warmup_ms = self._warmup(model, names)
            self.class_names = list(names)
            self.checkpoint = {
                "path": str(ckpt), "epoch": st.get("epoch"),
                "best_metric": st.get("best_metric"), "arch": str(self.cfg["model.arch"]),
                "device": str(self.device),
            }
            self.state = "READY"
            self.error = None
            return True
        except Exception as exc:
            self.model = None
            self.state = "MODEL_NOT_LOADED"
            self.error = f"{type(exc).__name__}: {exc}"
            return False

    def _warmup(self, model, names: list[str]) -> float | None:
        """cudnn 알고리즘 자동 선택을 미리 끝낸다.

        (실측) 이 과정을 건너뛰면 배치 1 추론이 450ms 대에 머문다.
        워밍업 후에는 같은 GPU 에서 80ms 수준으로 떨어진다. 실시간 경로에서는 필수다.
        """
        import torch

        if self.device is None or self.device.type != "cuda":
            return None
        try:
            torch.backends.cudnn.benchmark = True
            L = int(self.cfg["data.clip_len"])
            crop = int(self.cfg["data.eval_crop"])
            amp = bool(self.cfg.get("train.amp", True))
            x = torch.zeros(1, 3, L, crop, crop, device=self.device)
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(8):
                    with torch.autocast("cuda", enabled=amp):
                        model(x)
            torch.cuda.synchronize()
            return round((time.perf_counter() - t0) * 1000.0, 1)
        except Exception:
            return None

    # --- 추론 ---
    def predict(self, clip: np.ndarray) -> dict[str, float]:
        """clip: (T, H, W, 3) uint8 RGB, 캐시 해상도(기본 128x171)."""
        if self.state != "READY" or self.model is None:
            raise ModelNotLoaded(self.error or "모델이 로드되지 않았다")
        import torch
        from storeguard.data.dataset import preprocess_clip

        with self._lock:
            t0 = time.perf_counter()
            x = preprocess_clip(clip, self.cfg).unsqueeze(0).to(self.device)
            amp = bool(self.cfg.get("train.amp", True)) and self.device.type == "cuda"
            with torch.no_grad():
                with torch.autocast("cuda", enabled=amp):
                    out = self.model(x)
                p = torch.softmax(out.float(), dim=1)[0].cpu().numpy()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000.0
            self.last_infer_ms = round(dt, 2)
            self._times.append(dt)
        return {c: float(p[i]) for i, c in enumerate(self.class_names)}

    def status(self) -> dict:
        arr = list(self._times)
        return {
            "state": self.state,
            "error": self.error,
            "checkpoint": self.checkpoint,
            "classes": self.class_names,
            "last_infer_ms": self.last_infer_ms,
            "warmup_ms": self.warmup_ms,
            "infer_ms_mean": round(float(np.mean(arr)), 2) if arr else None,
            "infer_ms_p90": round(float(np.percentile(arr, 90)), 2) if arr else None,
            "throughput_clips_per_sec": round(1000.0 / float(np.mean(arr)), 2) if arr else None,
        }
