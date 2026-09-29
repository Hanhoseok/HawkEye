"""config.yaml 로더.

임계값·모델 경로를 코드에 하드코딩하지 않기 위한 계층.
근거: 05 아키텍처 §4
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class VideoConfig:
    source: str = "data/videos/sample.mp4"
    start_time: str | None = None  # ISO8601. None 이면 처리 시작 시각을 사용
    start_frame: int = 0  # 이 프레임부터 처리 (특정 구간만 실험할 때)
    max_frames: int | None = None  # 디버깅용 조기 종료
    stride: int = 1  # N 프레임마다 1장 처리 (1 = 모든 프레임)
    scale: float = 1.0  # 축소 배율. 1.0 미만이면 '카메라가 멀어진 상황'을 흉내낸다
    # resize: 프레임 자체를 줄인다 -> 디테일만 손실. 사람이 화면에서 차지하는 비율은 그대로
    # shrink: 프레임 크기는 유지하고 내용만 줄인다 -> 디테일 + 사람 크기 둘 다 감소 (실제 원거리에 가까움)
    scale_mode: str = "shrink"
    blackout: str | None = None
    """"START:END" 형태. 이 원본 프레임 구간을 검은 화면으로 덮는다.
    tracker 버퍼보다 긴 공백을 인위적으로 만들어 신원 레지스트리를 검증하는 실험용."""


@dataclass
class DetectorConfig:
    name: str = "yolo"
    model_path: str = "yolov8n.pt"
    device: str = "cpu"  # cpu / cuda:0 / intel:gpu(OpenVINO)
    imgsz: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    classes: list[str] = field(default_factory=lambda: ["person"])
    half: bool = False


@dataclass
class ReIDConfig:
    """BoT-SORT 등 외형 기반 tracker 에서만 쓰이는 설정.

    ByteTrack 은 외형 모델이 없으므로 이 값들을 무시한다.
    """

    weights: str = "osnet_x0_25_msmt17.pt"  # 경량 ReID 백본. 없으면 자동 다운로드
    device: str = "cpu"
    half: bool = False
    proximity_threshold: float = 0.5   # 이 IoU 거리 안에 있어야 외형 비교 대상
    appearance_threshold: float = 0.25  # 외형 거리 상한. 작을수록 엄격


@dataclass
class TrackerConfig:
    name: str = "bytetrack"
    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 30
    minimum_matching_threshold: float = 0.8  # IoU 거리(1-IoU) 상한. 클수록 관대
    frame_rate: int = 30
    minimum_consecutive_frames: int = 1
    reid: ReIDConfig = field(default_factory=ReIDConfig)


@dataclass
class ZonesConfig:
    """구역 정의 파일 경로. 좌표는 카메라 설치 위치에 종속이므로 코드가 아니라 파일에 둔다."""

    path: str = "zones.yaml"
    enabled: bool = True


@dataclass
class IdentityConfig:
    """매장 단위 신원 레지스트리(계층 2) 설정.

    tracker 의 lost_track_buffer 는 '초 단위 짧은 공백'을 메우는 장치라
    고객이 매장에 머무는 내내 추적하는 데는 쓸 수 없다. 이 계층이 그 역할을 맡는다.
    """

    enabled: bool = True
    match_threshold: float = 0.65
    """코사인 유사도가 이 값 이상이면 같은 사람으로 본다. 높일수록 보수적."""

    max_templates: int = 5
    """한 사람당 보관할 외형 템플릿 수. 여러 개를 두면 크기·자세 변화를 견딘다."""

    template_refresh_seconds: float = 1.0
    """살아 있는 track 의 템플릿을 몇 초마다 갱신할지."""

    retire_after_seconds: float = 30.0
    """이 시간 동안 '연속으로' 보이지 않으면 매장을 나간 것으로 보고 대조 대상에서 제외한다.

    체류 시간 상한이 아니다. 보일 때마다 시계가 초기화되므로 계속 보이는 고객은 만료되지 않는다.

    단위가 프레임이 아니라 초인 이유: 프레임 번호는 원본 fps 와 stride 에 따라
    같은 값이 다른 시간을 뜻한다. 실제로 900프레임을 30초로 알고 썼으나
    59.9fps 영상에서는 15초였다. Frame.pts_ms 를 쓰면 이 혼동이 없다.
    """

    min_tensor_height: int = 65
    """이 크기(텐서 픽셀) 미만인 관측은 신원 판단에 쓰지 않는다. 근거: docs/scale-limits.md"""


@dataclass
class InteractionConfig:
    """계층 3 · TAKE 후보 판정 설정.

    '선반에 가까움'만으로는 판정할 수 없다는 것이 실측으로 확인됐다.
    전체 영상에서 모든 사람이 선반에 30~90% 접촉했다(docs/zones.md).
    통로를 걷는 것 자체가 선반 옆에 있는 것이기 때문이다.
    그래서 근접에 **멈춤(체류 + 저속)** 을 추가 조건으로 건다.
    """

    enabled: bool = True
    min_overlap: float = 0.15
    """SHELF 구역과 이 비율 이상 겹쳐야 '접촉'으로 본다."""

    max_speed: float = 0.35
    """이보다 느려야 '멈췄다'고 본다. 단위는 **체구 높이 / 초**.

    픽셀 속도를 쓰지 않는 이유: 멀리 있는 사람은 같은 속도로 걸어도 픽셀 이동이 적다.
    사람 크기가 한 화면 안에서 4~10배 변하므로(docs/scale-limits.md)
    픽셀 기준 임계값은 거리에 따라 전혀 다른 뜻이 된다.
    """

    dwell_seconds: float = 1.0
    """접촉+저속 상태가 이 시간 이상 이어져야 후보로 본다."""

    gap_tolerance_seconds: float = 0.5
    """조건이 잠깐 끊겨도 이 시간 안에 회복되면 같은 구간으로 이어 본다."""

    speed_window_seconds: float = 0.4
    """속도를 계산할 때 쓰는 시간 창. 프레임 간 속도는 잡음이 심하다."""

    signal: str = "dwell"
    """무엇을 근거로 후보를 낼지.

    dwell : 몸이 선반과 겹침 + 저속 (pose 불필요)
    hand  : 손목이 선반 구역 안에 들어감 (pose 필요)
    both  : 둘 다 만족
    """

    hand_conf: float = 0.5
    """이 신뢰도 미만인 손목은 무시한다. pose 는 가려진 관절을 (0,0) 으로 내보낸다."""

    raise_min_seconds: float = 0.3
    """signal="stage" 의 2단계. 손이 이 시간 이상 올라가 있어야 한 건으로 센다.

    체류는 원래 긴 동작(수 초)이고 손 올리기는 원래 짧은 동작(0.5초 내외)이다.
    둘에 같은 지속 기준을 강요하면(= AND 방식) 짧은 쪽이 항상 희생된다.
    실제로 AND 로 3초를 요구하자 F1 이 0.29 까지 떨어졌다.
    """

    wrist_zone_max: float = 0.25
    """signal="reach" 전용. 손목-선반 거리가 이 값(체구 대비) 이하면 '닿았다'로 본다.

    **수직 천장 시점에서만 의미가 있다.** 그 화각에서는 이미지가 거의 평면도라
    이 거리가 실제 접촉 여부를 뜻한다.

    MERL 실측(라벨 기준):
        손이 선반 안에  중앙값 0.072 (25~75%: 0.000~0.205)
        선반 보기만    중앙값 0.599 (25~75%: 0.423~0.775)
    비스듬한 각도에서는 쓰지 말 것 — 깊이 모호성 때문에 구분되지 않는다.
    """

    best_raise_only: bool = False
    """체류 구간당 손 올림을 몇 건 내보낼지. True 면 가장 높이 든 것 하나만.

    한 체류 구간 안에서 손은 여러 번 올라간다(구경 중에도). 전부 내보내면
    후보 수만 늘고 정밀도가 무너진다 — 실측에서 19건 -> 35건이 되며 정밀도가 0.63 -> 0.34 로 떨어졌다.
    """

    hand_height_min: float = 0.10
    """손목이 엉덩이보다 이 값(체구 높이 대비) 이상 위에 있어야 '손을 올렸다'고 본다.

    실측: TAKE 중앙값 0.146 / BROWSE 0.036. signal="raise" 일 때만 쓰인다.
    """


@dataclass
class OutputConfig:
    dir: str = "outputs"
    write_video: bool = True
    video_name: str = "tracked.mp4"
    write_observations: bool = True
    observations_name: str = "observations.jsonl"
    write_identities: bool = True
    identities_name: str = "identities.jsonl"
    write_takes: bool = True
    takes_name: str = "take_candidates.jsonl"
    show_window: bool = False
    draw_zones: bool = True
    draw_trail: bool = True
    trail_length: int = 30


@dataclass
class AppConfig:
    video: VideoConfig = field(default_factory=VideoConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    zones: ZonesConfig = field(default_factory=ZonesConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(
            video=VideoConfig(**(raw.get("video") or {})),
            detector=DetectorConfig(**(raw.get("detector") or {})),
            tracker=_build_tracker_config(raw.get("tracker") or {}),
            zones=ZonesConfig(**(raw.get("zones") or {})),
            identity=IdentityConfig(**(raw.get("identity") or {})),
            interaction=InteractionConfig(**(raw.get("interaction") or {})),
            output=OutputConfig(**(raw.get("output") or {})),
        )


def _build_tracker_config(raw: dict[str, Any]) -> TrackerConfig:
    """tracker 설정 안에 중첩된 reid 블록을 따로 조립한다."""
    raw = dict(raw)
    reid = ReIDConfig(**(raw.pop("reid", None) or {}))
    return TrackerConfig(reid=reid, **raw)
