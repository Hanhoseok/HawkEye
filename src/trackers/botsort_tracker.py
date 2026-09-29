"""BoT-SORT + ReID 추적 계층 (ByteTrack 대체 실험용).

ByteTrack 은 움직임(칼만 필터)과 IoU 만으로 매칭하므로, 대상이 1초 이상 가려지면
예측 위치가 어긋나 같은 사람으로 이어붙이지 못한다. (docs/phase3-failure-notes.md)

BoT-SORT 는 여기에 **외형 임베딩(ReID)** 을 더해, 위치가 어긋나도 생김새가 같으면
동일 인물로 판정할 수 있다.

이 파일도 ByteTrack 어댑터와 똑같이 boxmot 자료형을 밖으로 내보내지 않는다.
바깥에서 보면 두 tracker 는 완전히 같은 인터페이스다.
근거: 05 아키텍처 §3~4 (tracker 교체 가능성)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.config import TrackerConfig
from ..core.types import Detection, Frame, TrackObservation


class BotSortTracker:
    """boxmot.BotSort 어댑터."""

    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        self._tracker = self._build()

    def _build(self):
        from boxmot.reid.core import ReID
        from boxmot.trackers.botsort.botsort import BotSort

        reid = self.config.reid
        reid_model = None
        if reid.weights:
            # 가중치가 없으면 boxmot 이 자동으로 내려받는다.
            reid_model = ReID(
                weights=Path(reid.weights), device=reid.device, half=reid.half
            ).model

        tracker = BotSort(
            reid_model=reid_model,
            with_reid=reid_model is not None,
            # ByteTrack 설정과 의미를 맞춰 준다. 두 tracker 를 같은 조건에서 비교하기 위함이다.
            track_high_thresh=self.config.track_activation_threshold,
            new_track_thresh=self.config.track_activation_threshold,
            track_buffer=self.config.lost_track_buffer,
            match_thresh=self.config.minimum_matching_threshold,
            frame_rate=self.config.frame_rate,
            proximity_thresh=reid.proximity_threshold,
            appearance_thresh=reid.appearance_threshold,
            # 고정 CCTV 이므로 카메라 움직임 보정(CMC)은 불필요하고 느리기만 하다.
            cmc_method=None,
        )
        if getattr(tracker, "model", None) is not None:
            tracker.model.warmup()
        return tracker

    def reset(self) -> None:
        self._tracker = self._build()

    def update(self, frame: Frame, detections: list[Detection]) -> list[TrackObservation]:
        dets = self._to_boxmot(detections)
        results = self._tracker.update(dets, frame.image)
        return self._to_observations(frame, results, detections)

    @staticmethod
    def _to_boxmot(detections: list[Detection]) -> np.ndarray:
        """boxmot 입력 형식: (N, 6) = [x1, y1, x2, y2, conf, cls]"""
        if not detections:
            return np.empty((0, 6), dtype=np.float32)
        return np.array(
            [[*d.bbox, d.score, d.class_id] for d in detections], dtype=np.float32
        )

    @staticmethod
    def _to_observations(
        frame: Frame, results, detections: list[Detection]
    ) -> list[TrackObservation]:
        """boxmot 출력 (N, 8) = [x1, y1, x2, y2, id, conf, cls, det_ind] 을 표준형으로."""
        observations: list[TrackObservation] = []
        if results is None or len(results) == 0:
            return observations

        arr = np.asarray(results)
        for row in arr:
            x1, y1, x2, y2, track_id, conf, cls = (float(v) for v in row[:7])
            det_ind = int(row[7]) if arr.shape[1] > 7 else -1
            source = detections[det_ind] if 0 <= det_ind < len(detections) else None
            class_name = source.class_name if source else None
            observations.append(
                TrackObservation(
                    track_id=int(track_id),
                    bbox=(x1, y1, x2, y2),
                    frame=frame.index,
                    timestamp=frame.timestamp,
                    score=conf,
                    class_name=class_name,
                    keypoints=source.keypoints if source else None,
                )
            )
        return observations
