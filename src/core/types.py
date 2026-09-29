"""파이프라인 전 계층이 공유하는 표준 자료형.

이 모듈은 외부 라이브러리(ultralytics, supervision, cv2 등)를 절대 import 하지 않는다.
detector / tracker 를 교체하더라도 이 자료형만 유지되면 뒤쪽 로직은 그대로 쓸 수 있다.
근거: 04 인터페이스 §6, 05 아키텍처 §4 계층 분리 원칙
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any

# [x1, y1, x2, y2] 픽셀 좌표 (좌상단 원점)
BBox = tuple[float, float, float, float]

# COCO 17 keypoint. 각 항목은 (x, y, conf).
Keypoints = tuple[tuple[float, float, float], ...]

# COCO keypoint 인덱스 (pose 모델이 내보내는 순서)
KP_NOSE = 0
KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER = 5, 6
KP_LEFT_ELBOW, KP_RIGHT_ELBOW = 7, 8
KP_LEFT_WRIST, KP_RIGHT_WRIST = 9, 10
KP_LEFT_HIP, KP_RIGHT_HIP = 11, 12


@dataclass(frozen=True, slots=True)
class Frame:
    """영상에서 읽어낸 한 장의 프레임과 그 시간 정보."""

    index: int
    """0부터 시작하는 프레임 번호."""

    timestamp: str
    """ISO8601 문자열. 영상 시작 시각 + 프레임 재생 위치."""

    pts_ms: float
    """영상 시작 기준 재생 위치(ms). 디버깅·동기화용."""

    image: Any
    """BGR ndarray. 타입 힌트에 numpy 를 노출하지 않기 위해 Any 로 둔다."""


@dataclass(frozen=True, slots=True)
class Detection:
    """한 프레임에서 탐지된 객체 하나. tracker 의 입력."""

    bbox: BBox
    score: float
    class_id: int
    class_name: str
    keypoints: Keypoints | None = None
    """pose detector 를 쓸 때만 채워진다. 일반 detector 면 None."""


@dataclass(frozen=True, slots=True)
class TrackObservation:
    """1차 구현의 표준 출력. 이후 모든 로직의 입력이 되는 계약.

    문서에 고정된 필드는 track_id / bbox / frame / timestamp 네 개이며,
    score 와 class_name 은 디버깅을 위한 선택적 확장 필드다.
    """

    track_id: int
    bbox: BBox
    frame: int
    timestamp: str
    score: float | None = None
    class_name: str | None = None
    keypoints: Keypoints | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class IdentityObservation:
    """매장 단위 신원이 붙은 관측. 계층 2의 표준 출력.

    track_id 는 tracker 가 끊길 때마다 바뀌지만, person_id 는 고객이 매장을 나갈 때까지 유지된다.
    이후 로직(TAKE/RETURN, 상태 관리, POS, 위험 판정)은 track_id 가 아니라 person_id 를 쓴다.
    """

    person_id: int
    """매장 단위 신원. -1 은 아직 배정되지 않음(관측 품질 미달)을 뜻한다."""

    track_id: int
    """어느 track 에서 나온 관측인지. 디버깅과 tracker 평가에 쓴다."""

    bbox: BBox
    frame: int
    timestamp: str
    score: float | None = None
    class_name: str | None = None

    is_new: bool = False
    """이 프레임에서 새로 등록된 신원인지 (= 매장 입장)."""

    matched_similarity: float | None = None
    """기존 신원과 이어붙였을 때의 외형 유사도. 신규 등록이면 None."""

    keypoints: Keypoints | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ZoneType(str, Enum):
    """구역 종류. 용도에 따라 판정 방식이 다르다."""

    SHELF = "SHELF"        # 상품 진열대. 사람이 '안에' 서지 않으므로 겹침 비율로 본다
    EXIT = "EXIT"          # 출구. 발 위치가 안에 들어왔는지로 본다
    CHECKOUT = "CHECKOUT"  # 계산대
    ENTRANCE = "ENTRANCE"  # 입구
    OTHER = "OTHER"


@dataclass(frozen=True, slots=True)
class Zone:
    """매장 안의 한 구역.

    좌표는 normalized=True 면 0~1 상대값, False 면 이미지 픽셀이다.
    상대값을 지원하는 이유: 같은 카메라라도 처리 해상도를 바꾸면
    픽셀 좌표가 전부 어긋나기 때문이다(--scale 실험에서 실제로 겪었다).
    """

    name: str
    type: ZoneType
    polygon: tuple[tuple[float, float], ...]
    normalized: bool = False


@dataclass(frozen=True, slots=True)
class ZoneHit:
    """한 사람과 한 구역의 관계."""

    zone: str
    zone_type: ZoneType
    overlap: float
    """사람 bbox 넓이 중 구역과 겹치는 비율 (0~1). SHELF 근접 판정에 쓴다."""

    contains_foot: bool
    """발 위치(bbox 아래쪽 중앙)가 구역 안에 있는지. EXIT/CHECKOUT 판정에 쓴다."""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["zone_type"] = self.zone_type.value
        return d


@dataclass(frozen=True, slots=True)
class TakeCandidate:
    """상품을 집었을 '가능성이 있는' 구간. 계층 3의 출력.

    이름이 TakeEvent 가 아니라 Candidate 인 이유: 이 판정은 상품이 실제로 옮겨졌는지
    전혀 알지 못한다. 아는 것은 "선반 앞에서 멈춰 있었다"뿐이다.
    상품 개별 추적은 측정 결과 실패했고(docs/phase5-take-return-survey.md),
    ReID 로도 고칠 수 없다. 같은 그릇끼리는 외형이 실제로 동일하기 때문이다.
    """

    person_id: int
    zone: str
    start_frame: int
    end_frame: int
    start_timestamp: str
    end_timestamp: str
    duration_sec: float
    observations: int

    max_overlap: float
    """구간 중 선반과 가장 많이 겹쳤을 때의 비율."""

    min_speed: float
    """구간 중 가장 느렸을 때의 속도. 체구 높이 대비 초당 이동량이라 거리와 무관하다."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
