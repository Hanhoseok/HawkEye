"""Phase 2: 사람 추적 계층.

ByteTrack 을 붙여 프레임이 바뀌어도 같은 사람에게 동일한 track_id 를 유지한다.
supervision 의 자료형(sv.Detections)은 이 파일 밖으로 나가지 않으며,
바깥으로는 표준 TrackObservation 만 내보낸다.

따라서 ByteTrack -> BoT-SORT 로 교체하더라도 같은 인터페이스를 구현하면
뒤쪽 로직(TAKE/RETURN, 상태 관리, POS, 위험 판정)은 그대로 유지된다.
근거: 02 기능 명세 F3, 04 인터페이스 §6, 05 아키텍처 §3~4
"""

from __future__ import annotations

import inspect

import numpy as np
import supervision as sv

from ..core.config import TrackerConfig
from ..core.types import Detection, Frame, TrackObservation

# supervision 버전에 따라 생성자 인자 이름이 달라서 구/신 이름을 함께 매핑한다.
_PARAM_ALIASES: dict[str, tuple[str, ...]] = {
    "track_activation_threshold": ("track_activation_threshold", "track_thresh"),
    "lost_track_buffer": ("lost_track_buffer", "track_buffer"),
    "minimum_matching_threshold": ("minimum_matching_threshold", "match_thresh"),
    "frame_rate": ("frame_rate",),
    "minimum_consecutive_frames": ("minimum_consecutive_frames",),
}


class ByteTrackTracker:
    """supervision.ByteTrack 어댑터."""

    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        self._tracker = self._build()

    def _build(self) -> sv.ByteTrack:
        accepted = set(inspect.signature(sv.ByteTrack.__init__).parameters)
        kwargs: dict[str, float | int] = {}
        for key, aliases in _PARAM_ALIASES.items():
            value = getattr(self.config, key)
            for alias in aliases:
                if alias in accepted:
                    kwargs[alias] = value
                    break
        return sv.ByteTrack(**kwargs)

    def reset(self) -> None:
        self._tracker = self._build()

    def update(self, frame: Frame, detections: list[Detection]) -> list[TrackObservation]:
        sv_detections = self._to_supervision(detections)
        tracked = self._tracker.update_with_detections(sv_detections)
        return self._to_observations(frame, tracked, detections)

    @staticmethod
    def _to_supervision(detections: list[Detection]) -> sv.Detections:
        if not detections:
            return sv.Detections.empty()
        return sv.Detections(
            xyxy=np.array([d.bbox for d in detections], dtype=np.float32),
            confidence=np.array([d.score for d in detections], dtype=np.float32),
            class_id=np.array([d.class_id for d in detections], dtype=int),
            data={
                "class_name": np.array([d.class_name for d in detections]),
                # keypoints 를 되찾기 위한 원본 인덱스.
                # supervision 은 임의 배열을 track 결과까지 실어 보낸다.
                "det_index": np.arange(len(detections)),
            },
        )

    @staticmethod
    def _to_observations(
        frame: Frame, tracked: sv.Detections, detections: list[Detection]
    ) -> list[TrackObservation]:
        observations: list[TrackObservation] = []
        class_names = tracked.data.get("class_name") if tracked.data else None
        det_index = tracked.data.get("det_index") if tracked.data else None

        for i in range(len(tracked)):
            track_id = tracked.tracker_id[i] if tracked.tracker_id is not None else None
            if track_id is None:
                continue  # 아직 확정되지 않은 track 은 내보내지 않는다
            x1, y1, x2, y2 = (float(v) for v in tracked.xyxy[i])
            observations.append(
                TrackObservation(
                    track_id=int(track_id),
                    bbox=(x1, y1, x2, y2),
                    frame=frame.index,
                    timestamp=frame.timestamp,
                    score=float(tracked.confidence[i]) if tracked.confidence is not None else None,
                    class_name=str(class_names[i]) if class_names is not None else None,
                    keypoints=(
                        detections[int(det_index[i])].keypoints
                        if det_index is not None and 0 <= int(det_index[i]) < len(detections)
                        else None
                    ),
                )
            )
        return observations
