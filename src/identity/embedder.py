"""외형 임베딩 추출기.

레지스트리가 쓰는 유일한 외부 모델 의존성이며, boxmot 은 이 파일 안에서 끝난다.
`embed(image, boxes) -> (N, D) L2 정규화 ndarray` 만 지키면 다른 ReID 백본으로 교체할 수 있다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.config import ReIDConfig


class ReIDEmbedder:
    """boxmot 의 ReID 백본을 감싼 어댑터."""

    def __init__(self, config: ReIDConfig) -> None:
        from boxmot.reid.core import ReID

        self.config = config
        self._model = ReID(
            weights=Path(config.weights), device=config.device, half=config.half
        ).model
        if hasattr(self._model, "warmup"):
            self._model.warmup()

    def embed(self, image, boxes: np.ndarray) -> np.ndarray | None:
        """bbox 를 잘라 외형 특징 벡터를 뽑고 L2 정규화해서 돌려준다.

        정규화해 두면 이후 유사도 계산이 내적 한 번으로 끝난다.
        """
        if boxes is None or len(boxes) == 0:
            return None
        features = self._model.get_features(boxes.astype(np.float32), image)
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features[None, :]
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return features / norms


def build_embedder(config: ReIDConfig):
    """교체 지점. 다른 ReID 백본을 쓰려면 여기만 고친다."""
    if not config.weights:
        return None
    return ReIDEmbedder(config)
