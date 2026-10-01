"""PyTorch Dataset: 캐시된 프레임(.npy)에서 클립을 읽는다."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from storeguard.config import Config
from storeguard.data.clips import ClipSpec, build_clip_list
from storeguard.data.framestore import FrameStore, available_stems, open_store


def load_index(cfg: Config) -> list[dict]:
    p = cfg.path("paths.index") / "clips.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_splits(cfg: Config) -> dict:
    p = cfg.path("paths.index") / "splits.json"
    return json.loads(p.read_text(encoding="utf-8"))


class ClipDataset(Dataset):
    """(C, T, H, W) float32 텐서와 정수 라벨을 돌려준다."""

    def __init__(self, cfg: Config, stems: list[str], *, train: bool,
                 rows: list[dict] | None = None):
        self.cfg = cfg
        self.train = train
        self.cache = cfg.path("paths.cache")
        self.clip_len = int(cfg["data.clip_len"])
        self.crop = int(cfg["data.train_crop"] if train else cfg["data.eval_crop"])
        self.mean = np.array(cfg["data.mean"], dtype=np.float32)
        self.std = np.array(cfg["data.std"], dtype=np.float32)
        self.class_to_idx = cfg.class_to_idx
        self.hflip = float(cfg.get("train.augment.hflip", 0.0)) if train else 0.0
        self.jitter = float(cfg.get("train.augment.color_jitter", 0.0)) if train else 0.0
        self.temporal_jitter = int(cfg.get("train.augment.temporal_jitter", 0)) if train else 0
        self.random_crop = bool(cfg.get("train.augment.random_crop", True)) and train

        rows = rows if rows is not None else load_index(cfg)
        available = available_stems(self.cache)
        requested = list(stems)
        stems = [s for s in requested if s in available]
        self.missing_cache = [s for s in requested if s not in available]
        if self.missing_cache:
            # 전처리가 끝나기 전에 학습을 시작하면 데이터가 조용히 줄어든다. 반드시 알린다.
            import warnings
            warnings.warn(
                f"프레임 캐시가 없는 영상 {len(self.missing_cache)}/{len(requested)}개를 건너뛴다. "
                f"storeguard.data.preprocess 를 끝까지 돌렸는지 확인할 것. "
                f"예: {self.missing_cache[:3]}", stacklevel=2)
        specs = build_clip_list(rows, stems, cfg)
        # 이 모델이 모르는 클래스의 클립은 버린다.
        # (예: 4클래스로 학습한 예전 체크포인트를 구매행동이 포함된 새 데이터로 평가할 때)
        known = set(self.class_to_idx)
        self.skipped_labels = sorted({s.label for s in specs if s.label not in known})
        if self.skipped_labels:
            import warnings
            warnings.warn(
                f"이 모델의 클래스에 없는 라벨의 클립을 건너뛴다: {', '.join(self.skipped_labels)}",
                stacklevel=2)
        self.specs: list[ClipSpec] = [s for s in specs if s.label in known]
        self.labels = np.array([self.class_to_idx[s.label] for s in self.specs], dtype=np.int64)
        self._stores: dict[str, FrameStore] = {}

    # --- 통계 ---
    def class_counts(self) -> dict[str, int]:
        out = {c: 0 for c in self.cfg.class_names}
        for s in self.specs:
            out[s.label] += 1
        return out

    def __len__(self) -> int:
        return len(self.specs)

    def _store(self, stem: str) -> FrameStore:
        st = self._stores.get(stem)
        if st is None:
            st = open_store(self.cache, stem)
            if len(self._stores) > 48:      # 워커당 열린 파일 수 제한
                self._stores.clear()
            self._stores[stem] = st
        return st

    def __getitem__(self, i: int):
        spec = self.specs[i]
        store = self._store(spec.stem)
        T_total = store.n_frames

        start = spec.start
        if self.temporal_jitter:
            j = int(np.random.randint(-self.temporal_jitter, self.temporal_jitter + 1))
            start = max(0, min(start + j, T_total - 1 - (self.clip_len - 1) * spec.stride))

        idxs = [min(T_total - 1, start + k * spec.stride) for k in range(self.clip_len)]
        clip = store.read(idxs)                                   # (T, H, W, 3) uint8 RGB

        H, W = clip.shape[1], clip.shape[2]
        ch = cw = self.crop
        if self.random_crop:
            y = int(np.random.randint(0, max(1, H - ch + 1)))
            x = int(np.random.randint(0, max(1, W - cw + 1)))
        else:
            y = max(0, (H - ch) // 2)
            x = max(0, (W - cw) // 2)
        clip = clip[:, y:y + ch, x:x + cw, :]

        if self.hflip and np.random.rand() < self.hflip:
            clip = clip[:, :, ::-1, :]

        out = clip.astype(np.float32) / 255.0
        if self.jitter:
            # 밝기/대비만 간단히. 매장 조명 차이를 흉내낸다.
            b = 1.0 + float(np.random.uniform(-self.jitter, self.jitter))
            c = 1.0 + float(np.random.uniform(-self.jitter, self.jitter))
            m = out.mean()
            out = np.clip((out - m) * c + m * b, 0.0, 1.0)
        out = (out - self.mean) / self.std
        tensor = torch.from_numpy(np.ascontiguousarray(out.transpose(3, 0, 1, 2)))  # (C,T,H,W)
        return tensor, int(self.labels[i]), i


def make_datasets(cfg: Config) -> dict[str, ClipDataset]:
    rows = load_index(cfg)
    splits = load_splits(cfg)
    return {
        "train": ClipDataset(cfg, splits["train"], train=True, rows=rows),
        "val": ClipDataset(cfg, splits["val"], train=False, rows=rows),
        "test": ClipDataset(cfg, splits["test"], train=False, rows=rows),
    }


def preprocess_clip(frames: np.ndarray, cfg: Config) -> torch.Tensor:
    """실시간 추론용. frames: (T, H, W, 3) uint8 RGB, 이미 cache 크기로 리사이즈된 상태.

    학습과 동일하게 center crop + 정규화만 한다.
    """
    crop = int(cfg["data.eval_crop"])
    mean = np.array(cfg["data.mean"], dtype=np.float32)
    std = np.array(cfg["data.std"], dtype=np.float32)
    H, W = frames.shape[1], frames.shape[2]
    y = max(0, (H - crop) // 2)
    x = max(0, (W - crop) // 2)
    clip = frames[:, y:y + crop, x:x + crop, :].astype(np.float32) / 255.0
    clip = (clip - mean) / std
    return torch.from_numpy(np.ascontiguousarray(clip.transpose(3, 0, 1, 2)))
