"""계층 간 경계를 정의하는 인터페이스.

pipeline 은 구체 클래스가 아니라 이 프로토콜에만 의존한다.
따라서 YOLO -> 다른 detector, ByteTrack -> BoT-SORT 로 교체해도 pipeline 은 바뀌지 않는다.
근거: 05 아키텍처 §3 모듈 구성
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import Detection, Frame, TrackObservation


@runtime_checkable
class Detector(Protocol):
    """프레임 하나를 받아 Detection 목록을 돌려준다."""

    def detect(self, frame: Frame) -> list[Detection]: ...


@runtime_checkable
class Tracker(Protocol):
    """Detection 목록에 track_id 를 붙여 TrackObservation 으로 표준화한다.

    tracker 라이브러리의 고유 자료형은 이 경계를 넘어가지 않는다.
    """

    def update(self, frame: Frame, detections: list[Detection]) -> list[TrackObservation]: ...

    def reset(self) -> None:
        """새 영상을 처리하기 전에 내부 상태를 비운다."""
