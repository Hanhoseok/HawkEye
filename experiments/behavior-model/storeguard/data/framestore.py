"""프레임 캐시 읽기 계층.

두 가지 저장 형식을 같은 인터페이스로 읽는다.

  .npy   원시 uint8 배열 (T, H, W, 3). 빠르지만 크다. 영상당 약 11.8 MB.
  .jpz   프레임별 JPEG 를 이어 붙인 npz. 영상당 약 2.3 MB (품질 95 기준, 원본의 21%).

왜 .jpz 가 필요한가
  로컬 학습만 할 때는 .npy 가 낫다. 그러나 Colab 등 외부 GPU 로 옮기려면 캐시를 업로드해야 하고,
  .npy 전체는 24 GB 라 구글 드라이브 무료 용량(15 GB)에 들어가지 않는다.
  .jpz 로 바꾸면 약 5 GB 가 되어 업로드 시간과 용량 문제가 함께 해결된다.
  (실측) 16프레임 클립 1개 디코딩에 약 4 ms. 초당 수십 클립을 먹이기에 충분하다.

주의
  .jpz 는 손실 압축이다(품질 95 에서 PSNR 약 31 dB). 같은 형식으로 학습·평가해야 한다.
  .npy 로 학습한 가중치를 .jpz 로 평가하면 약간의 차이가 생길 수 있다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

NPY = ".npy"
JPZ = ".jpz.npz"


class FrameStore:
    """영상 1개의 프레임을 인덱스로 읽는다."""

    __slots__ = ("stem", "path", "kind", "_arr", "_blob", "_off", "_n")

    def __init__(self, path: Path, kind: str):
        self.path = path
        self.kind = kind
        self.stem = path.name[: -len(JPZ)] if kind == "jpz" else path.stem
        self._arr = None
        self._blob = None
        self._off = None
        if kind == "npy":
            self._arr = np.load(path, mmap_mode="r")
            self._n = int(self._arr.shape[0])
        else:
            z = np.load(path)
            self._blob = z["jpeg"]
            self._off = z["off"]
            self._n = int(len(self._off) - 1)

    @property
    def n_frames(self) -> int:
        return self._n

    def read(self, indices: Sequence[int]) -> np.ndarray:
        """(N, H, W, 3) uint8 RGB."""
        if self.kind == "npy":
            return np.asarray(self._arr[list(indices)], dtype=np.uint8)
        import cv2

        out = []
        for i in indices:
            i = int(i)
            b = self._blob[self._off[i]: self._off[i + 1]]
            img = cv2.imdecode(b, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"JPEG 디코딩 실패: {self.path} frame {i}")
            out.append(img[:, :, ::-1])
        return np.ascontiguousarray(np.stack(out, axis=0))


def store_path(cache_dir: Path, stem: str) -> tuple[Path, str] | None:
    """stem 에 해당하는 캐시 파일과 형식. 없으면 None."""
    p = cache_dir / f"{stem}{NPY}"
    if p.exists():
        return p, "npy"
    p = cache_dir / f"{stem}{JPZ}"
    if p.exists():
        return p, "jpz"
    return None


def open_store(cache_dir: Path, stem: str) -> FrameStore:
    found = store_path(cache_dir, stem)
    if found is None:
        raise FileNotFoundError(f"캐시 없음: {cache_dir / stem} ({NPY} 또는 {JPZ})")
    return FrameStore(*found)


def available_stems(cache_dir: Path) -> set[str]:
    """캐시가 있는 stem 집합. 두 형식을 모두 본다."""
    out = {p.stem for p in cache_dir.glob(f"*{NPY}")}
    out |= {p.name[: -len(JPZ)] for p in cache_dir.glob(f"*{JPZ}")}
    return out
