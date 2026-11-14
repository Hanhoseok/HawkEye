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
    live_process_fps: float = 5.0
    """실시간 카메라에서 초당 처리하는 장면 수 어림. 밀린 장면은 버리므로 카메라 fps 대신 이 값으로 선반 지도 확정 창(장면 수)을 정한다.
    녹화본은 쓰지 않는다 (fps ÷ stride). docs/shoot4-demo-plan.md §4-2."""


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

    path: str = "zones/store-aisle.yaml"
    enabled: bool = True
    foot_point: str = "bbox"
    """바닥 구역 판단에 쓰는 발 점 (전체 카메라 바닥 지도, floor.yaml).
    - "bbox": 사람 상자 아래 가운데 (지금까지).
    - "ankle": 관절 모델의 두 발목 가운데 (믿을 만한 발목이 없으면 상자 아래 가운데). 전체 카메라를 yolo-pose 로 돌린 기록에서만 의미가 있다.
    팔을 뻗으면 상자가 넓어져 '아래 가운데'가 실제 발에서 수십 px 옮겨 가 실제 자리에 맞게 그린 구역에서 머문 시간이 안 채워졌다 (B-15, 2026-11-08).
    선반 앞 띠 위치(두 카메라 짝)는 band_map 을 상자 기준으로 맞췄으므로 이 값과 상관없이 상자 아래 가운데를 쓴다."""
    ankle_conf: float = 0.3
    """foot_point="ankle" 일 때 발목 신뢰도 하한."""


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

    template_refresh_seconds: float = 0.5
    """살아 있는 track 에서 몇 초마다 사진을 한 장 더 모을지. 템플릿 갱신과 '나중에 합치기'의 증거가 된다.

    0.5 인 이유: 같은 5초라도 0.5초 간격 10장(88%)이 1초 간격 5장(80%)보다 같은 사람을 잘 잇는다
    (MERL train 실측, scripts/identity_calibration.py). 연속 프레임은 자세가 같아 도움이 안 되고,
    간격을 두면서 장수를 늘려야 잡음이 상쇄된다.
    """

    retire_after_seconds: float = 30.0
    """이 시간 동안 '연속으로' 보이지 않으면 매장을 나간 것으로 보고 대조 대상에서 제외한다.

    체류 시간 상한이 아니다. 보일 때마다 시계가 초기화되므로 계속 보이는 고객은 만료되지 않는다.

    단위가 프레임이 아니라 초인 이유: 프레임 번호는 원본 fps 와 stride 에 따라
    같은 값이 다른 시간을 뜻한다. 실제로 900프레임을 30초로 알고 썼으나
    59.9fps 영상에서는 15초였다. Frame.pts_ms 를 쓰면 이 혼동이 없다.
    """

    min_tensor_height: int = 65
    """이 크기(텐서 픽셀) 미만인 관측은 신원 판단에 쓰지 않는다. 근거: docs/scale-limits.md"""

    # --- 위치·시간 근거 (track stitching) ---
    # 생김새만으로 판단하면 생김새가 약한 화각(천장)에서 한 사람이 둘로 쪼개진다.
    # MERL 실측: 같은 사람끼리 유사도가 0.58~0.72 로 기준(0.65)을 걸쳐, 끊겼다 다시 나타날 때
    # 새 신원으로 등록됐다 (docs/merl-evaluation.md §4-E).
    # 그래서 "방금 근처에서 사라진 사람이 곧 다시 나타났다"면 생김새 기준을 낮춰준다.

    stitch_enabled: bool = True

    stitch_max_seconds: float = 3.0
    """이 시간 안에 다시 나타나야 '같은 사람이 이어서 나온 것'으로 본다."""

    stitch_max_distance: float = 1.0
    """사라진 자리와 나타난 자리 사이 거리 상한. 단위는 체구 높이(bbox 높이) 배수."""

    swap_check_samples: int = 3
    """판정 직전 신원 확인(risk.swap_threshold)에 쓰는 '최근 사진' 장수. 0.5초 간격이면 1.5초."""

    swap_check_min_older: int = 6
    """신원 확인에 필요한 '이전 사진' 최소 장수. 기록이 적으면 이전 평균 자체가 불안정해 확인하지 않는다."""

    stitch_threshold: float = 0.45
    """위치·시간이 맞을 때 적용하는 완화된 생김새 기준.

    0 으로 두지 않는 이유: 사람이 붐빌 때 A 가 사라진 자리에 B 가 곧바로 들어설 수 있다.
    생김새를 전혀 안 보면 그 둘을 합쳐 버린다(오병합). 최소한의 닮음은 요구한다.
    """

    # --- 나중에 합치기 (late merge) ---
    # 새 track 은 사진 한 장으로 즉시 판정한다. 판정을 몇 초씩 보류하면 그동안의 행동을 놓치기 때문이다.
    # 대신 한 장이라 틀리기 쉽다(같은 사람 57% 만 이어줌). 그래서 새로 등록된 신원이 몇 초간
    # 사진을 모으면, '그 직전에 사라진 사람'과 평균끼리 다시 비교해 같으면 합친다.
    # MERL train 실측(오병합 5% 기준): 즉시 1장 57% -> 0.5초 간격 10장 평균 88%.

    late_merge_enabled: bool = True

    edge_margin: float = 0.02
    """bbox 가 화면 가장자리에서 이 비율(화면 크기 대비) 안쪽에 닿아 있으면 즉시 판정하지 않고 새 신원으로 둔다.

    track 이 생기는 순간은 대개 사람이 가장자리로 막 들어오는 순간이라 몸이 잘려 있다.
    people-detection 에서는 머리카락만 보이는 첫 프레임끼리 0.71~0.83 으로 닮아 보여
    옷차림이 전혀 다른 사람들이 한 신원으로 묶였다. 잘린 사진으로 합치지 않고,
    몸이 다 보인 뒤 '나중에 합치기'가 판단하게 한다. 0 이면 끈다.
    """

    tentative_seconds: float = 0.0
    """새 신원을 '임시'로 두었다가, 이 시간 동안 연속으로 '쓸 만한 크기 + 정식 조건'을 채우면 정식 신원으로 올린다. 0 이면 끈다.

    정식 조건은 레지스트리 밖에서 준다(`IdentityRegistry.set_promote_check`). 카메라 2대 전체 카메라는
    '발이 매장 안 바닥(risk.inside_zones)에 있음' — 손님 조건(C-20)과 같은 기준이다. 조건이 없으면 시간만 본다.

    임시 신원도 번호는 그대로 내보내므로 뒤 계층(손님·판정)은 똑같이 동작한다. 다른 점은 둘뿐이다:
      - 새 track 의 대조 상대가 되지 않는다 (되살아난 번호의 원래 신원일 때만 예외).
      - 나중에 합치기의 양쪽 어디에도 들지 않는다.
    외투·창밖 행인·문 앞 조각 같은 가짜 사람이 보관함에 남아 진짜 손님과 맺어지는 것을 막는다 (C-11:
    2차 38쌍 전체 카메라 신원 105개 중 손님 53, 나머지 52개가 외형 사진과 함께 대조 상대로 남아 있었음).
    """

    track_gap_recheck_seconds: float = 0.0
    """track 이 이 시간보다 오래 끊겼다가 같은 번호로 되살아나면, 신원을 새 track 처럼 다시 판정한다. 0 이면 끈다.

    BoT-SORT 는 사라진 track 을 lost_track_buffer(90 장, 초당 15장이면 6초) 동안 기억했다가 외형이 느슨하게
    맞으면(appearance_threshold 0.5) 되살린다. 되살린 상대가 다른 사람이어도 신원 계층은 그 번호의 신원을 그대로
    물려줬다. 2차 촬영 C12: 입장 순간 0.2초 잡힌 조각 번호가 3초 뒤 선반 앞에서 겹친 두 손님 상자에 되살아나
    니트 손님이 그 조각의 신원으로 넘어가 한 사람이 두 손님이 됐다 (C-18, docs/multiview-progress.md).

    다시 판정할 때 그 번호의 원래 신원은 '방금 근처에서 사라진 사람'처럼 완화 기준(stitch_threshold)을 쓴다 —
    가산점 없이는 맞게 이어진 번호까지 쪼갰다(C21). 가장자리에 걸린 동안은 판정을 미룬다.
    """

    edge_sides: tuple[str, ...] = ("left", "top", "right", "bottom")
    """가장자리 규칙을 적용할 변. 화각 때문에 늘 잘리는 변은 빼야 한다.

    2차 촬영 전체 카메라(angle1)는 머리가 위쪽 끝에 잘리는 화각이라 관측의 71% 가 위쪽 가장자리에 닿았고,
    새 신원 81개 중 78개가 '가장자리라 즉시 판정 건너뜀'으로 생겼다 — 입장 순간뿐 아니라 매장 안쪽 전체에서
    재입장·조각이 기존 손님과 대조되지 않았다 (C-18). 선반 카메라(angle0)는 발이 아래쪽 끝에 98% 잘린다.
    늘 잘리는 변은 모든 사진이 똑같이 잘려 있어 서로 비교해도 된다.
    """

    late_merge_samples: tuple[int, ...] = (10, 20, 40)
    """새 신원의 사진 수가 이 값에 도달할 때마다 재확인한다. 0.5초 간격이면 5초·10초·20초.

    매 장마다 확인하지 않는 이유: 확인할 때마다 우연히 기준을 넘을 기회가 생겨 오병합이 누적된다.
    """

    late_merge_threshold: float = 0.875
    """평균끼리의 유사도 기준. 한 장 기준(match_threshold)과 척도가 다르다 — 평균이면 전반적으로 높다.

    MERL train 20명, 0.5초 간격 10장 평균 vs 과거 기록 평균에서 오병합 5% 가 되는 값(0.876).
    """

    late_merge_overlap_seconds: float = 0.0
    """나중에 합치기 상대가 새 신원과 이만큼까지 '함께 보인' 것은 봐준다 (0 = 함께 보였으면 절대 안 합침).
    3차 C14: 손님이 문으로 나가는 0.66초 동안 유리 밖 모습에 새 신원이 먼저 생겨, 돌아온 뒤에도 원래 신원과 못 합쳤다
    (docs/multiview-progress.md 후속 31).
    """

    nested_hold_overlap: float = 0.0
    """새 신원을 만들려는 상자가 이번 장면에 이미 신원이 있는 상자와 포개져 있으면 새 신원을 만들지 않고 보류한다. 0 이면 끈다.

    포개짐 = 겹친 넓이 ÷ 둘 중 작은 상자의 넓이. 이 값 이상이면 포개진 것으로 본다 (한 상자가 다른 상자 안에 통째로 들면 1.0).
    신원 장부는 사람이 아니라 상자(추적 번호)를 센다. 탐지기가 한 사람 위에 상자를 둘 내면(머리 상자 + 큰 상자, 윗몸 상자 + 온몸 상자)
    '한 장면에 같은 신원은 한 번'이라 두 번째 상자는 새 신원이 됐고, 추적기가 그 상자를 이어 가면 손님이 새 신원으로 바뀌었다
    (3차 C03 6.87·8.60초, docs/multiview-progress.md 후속 42 덧붙임 2 · 후속 43).
    기존 신원과 외형이 맞아 이어지는 것은 막지 않는다 — **새로 만드는 것만** 미룬다. 보류한 상자는 신원 -1 로 나가고 다음 장면에 다시 본다.

    잰 결과 (후속 43, 전체 카메라에만 · 꺼 둠): 3차 39쌍은 0.5~0.9 어디서나 판정 그대로, 손님 수 38 → 39/39(C03), 신원 131 → 116~118.
    2차 38쌍은 0.7 이하에서 **C15 가 틀려진다** — 문으로 함께 들어온 두 사람의 상자가 0.81 포개져 한 신원으로 합쳐졌다. 0.9 는 2차도 그대로.
    상자만으로는 '한 사람 위의 두 상자'와 '포개 선 두 사람'을 못 가른다(한계 C-25). 정상(N) 영상은 거의 한 명이라 이 값의 근거가 못 된다.
    """

    min_identity_score: float = 0.0
    """탐지 점수가 이 값 미만인 상자로는 신원을 만들지도, 되살아난 번호를 다시 판정하지도 않고 보류한다. 0 이면 끈다.

    탐지 기준(detector.conf_threshold 0.25)을 겨우 넘는 상자는 사람이 아닌 것(창 유리 그림·옷·가방)이나 가려져 조각난 몸인 때가 많다.
    3차 C03 13.33초: 놓친 손님 번호가 사람 없는 창가 상자(0.26)에 되살아나 외형 0.50 인데 원래 주인 가산점으로 그 손님 신원이 됐다.
    이미 신원을 갖고 이어지는 추적 번호에는 적용하지 않는다(점수가 잠깐 낮아져도 그대로) — 새로 정하거나 다시 정할 때만.

    잰 결과 (후속 43, 꺼 둠): 3차는 대가가 크다 — 정상 12쌍에서 신원이 붙은 줄이 0.4 에서 9%, 0.5 에서 22% 줄고, 0.35 부터 T09·C05 에
    판정이 하나씩 더 나오고 0.4 부터 C14 손님 수가 틀려진다. 2차는 판정·손님 수 그대로에 신원 101 → 84(0.4), 줄은 2% 만 준다.
    카메라 자리에 따라 낮은 점수 상자의 몫이 다르다(3차 전체 카메라는 가려진 진짜 사람도 0.3 안팎으로 잡힌다). C03 의 창가 상자는 위 포개짐 보류만으로도 막혔다.
    """


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
class RiskConfig:
    """손님 상태 · 결제 · 위험 판정 설정 (docs/risk-pipeline.md).

    판정 규칙: 계산하지 않은 물건(TAKE 후보 수 - 결제 품목 수)이 1개 이상인 손님이
    출구(EXIT) 구역에 들어서면 HIGH_RISK. 없으면 CLEAR.
    """

    enabled: bool = True

    payments: str | None = None
    """결제 기록 CSV 경로 (src/risk/payments.py). 없으면 아무도 결제하지 않은 것으로 본다."""

    inside_zones: list[str] = field(default_factory=list)
    """'매장 안'으로 칠 구역 이름 (예: shelf, counter, aisle). 발이 이 구역에 inside_seconds 이상 있어야 손님으로 판정한다.
    비어 있으면 지금까지처럼 '문 밖 구역이 아니면 매장 안'.
    2차 촬영(카메라 2대)에서 문 앞에 서 있는 사람·유리창 밖 행인·입장 순간 외투와 겹친 상자가 모두 손님이 되어
    손님 수가 38쌍 중 26쌍에서 많게 나왔다 (limitations C-20)."""

    inside_seconds: float = 0.0
    """inside_zones 안에 이만큼 연속으로 있어야 '매장 안에 들어온 손님'."""

    inside_excludes_exit: bool = False
    """True 면 발이 출구(EXIT) 구역 안일 때는 inside_zones 에 겹쳐 있어도 '매장 안'으로 치지 않는다.
    3차 촬영 바닥 지도는 통로(aisle)가 문(door) 구역을 덮어, 입구에 서 있기만 한 사람(T05)이 1초 만에 손님이 됐다
    (docs/multiview-progress.md 후속 27·29)."""

    exit_min_seconds: float = 0.5
    """발이 출구 구역 안에 이만큼 연속으로 있어야 '출구에 들어섰다'고 본다.

    출구 옆을 스쳐 지나가는 것까지 판정하지 않기 위해서다.
    """

    exit_confirm_seconds: float = 3.0
    """출구 구역 안에서 마지막으로 보이고 이만큼 다시 나타나지 않으면 '나갔다'고 확정하고 최종 판정한다.

    경보를 두 단계로 나눈 이유 (UCF-Crime Shoplifting047): 출구에 들어선 순간 HIGH_RISK 를 확정하면,
    혼잡한 문 앞에서 추적 번호가 다른 사람에게 옮겨 붙거나 인파에 가려 잠깐 사라진 것만으로 오경보가 났다.
    그래서 출구에 다가서면 WARNING, 실제로 사라진 뒤 이만큼 다시 나타나지 않아야 HIGH_RISK 로 확정한다.

    - 출구가 화면 가장자리여도 동작한다(0.5초 머물지 않고 곧장 사라져도 된다, Shoplifting031).
    - TAKE 후보는 사람이 안 보인 뒤 gap_tolerance_seconds(0.5초) 뒤에 끝나므로 그보다 커야 한다.
    - 탐지가 이보다 길게 끊기면 여전히 '나갔다'로 오인한다(031 에서 3.7초 끊김 실측).
      그 뒤 다시 나타나면 판정을 되돌리고 '이른 판정'으로 기록한다.
    - 대가: HIGH_RISK 가 이만큼 늦게 나온다. 다가선 순간의 WARNING 으로 직원이 먼저 알 수 있다.
    """

    exit_hold_seconds: float = 0.0
    """나감 확정(문 밖으로 나감·출구 쪽에서 사라짐)이 HIGH_RISK 일 때, 이만큼 더 기다렸다가 내보낸다 (0 = 바로).

    그동안 같은 손님이 다시 보이면 보류를 버리고 다음 출구 방문 때 다시 판정한다 — 나갔다 곧 돌아와 결제한 손님(3차 C12·C15, 2차 C12)에게
    HIGH_RISK 가 먼저 나가던 것을 막는다. 보류하는 동안은 WARNING 을 하나 낸다. 보류가 끝나면 그때까지 도착한 결제까지 넣어 다시 판정한다.
    영상이 끝나면(finish) 보류 중인 손님은 바로 확정한다. 대가: HIGH_RISK 가 이만큼 늦다 (docs/multiview-progress.md 후속 33)."""

    zone_grace_seconds: float = 0.0
    """계산대·출구 구역에서 발 점이 이만큼 이하로 잠깐 벗어나면 머문 시간을 0 으로 되돌리지 않는다 (0 = 끔, 한 장면만 벗어나도 다시 셈).
    발 점은 사람 상자 아래 가운데라 팔을 뻗거나 다리가 가려지면 수십 px 흔들린다. 구역을 실제 바닥 자리에 꼭 맞게 그리면
    이 흔들림으로 계산대 1.5초·출구 0.5초가 끝내 안 채워진다 (2026-11-08 구역 편집: N1 계산대 7.1초 서 있었는데 최장 연속 0.9초)."""

    exit_end_at_edge_only: bool = True
    """영상 끝(finish)에 마지막 장면에도 보이는 손님은 **출구 구역에서 화면 끝에 걸쳤을 때만** 나가는 중으로 본다 (기본, 2차 N05·N06).
    False 면 출구 구역 안(최근 exit_rearm_seconds 안에 출구에 있었음)이기만 하면 나가는 중 — 출구가 화면 가운데인 카메라용
    (3차 전체 카메라: 문이 삼각대 옆 벽 뒤, 화면 x≈200px. 영상이 나가는 순간 끝나 화면 끝 조건을 못 넘음, 2026-11-12)."""

    exit_rearm_seconds: float = 2.0
    """출구 구역을 벗어난 지 이만큼 지나야 다음 방문으로 본다.

    구역 경계에서 발 위치가 들락날락하면 같은 방문에 판정이 여러 번 나가는 것을 막는다.
    """

    swap_threshold: float = 0.78
    """HIGH_RISK 확정 직전 신원 확인 기준. 최근 1.5초(사진 3장) 평균과 그 이전 평균의 유사도가 이보다 낮으면
    추적 번호가 다른 사람에게 옮겨 붙었다고 의심해 HIGH_RISK 대신 REVIEW 로 돌린다.

    MERL train 실측 (0.5초 간격 사진, 과거 8장 이상):
        0.75 -> 같은 사람을 뒤바뀜으로 오인 2.7% / 실제 뒤바뀜을 잡음 71%
        0.78 -> 같은 사람을 뒤바뀜으로 오인 5.6% / 실제 뒤바뀜을 잡음 82%   <- 채택
        0.80 -> 같은 사람을 뒤바뀜으로 오인 9.4% / 실제 뒤바뀜을 잡음 87%
    같은 사람을 오인해도 경보가 사라지는 게 아니라 REVIEW(관리자 확인)로 내려갈 뿐이다.
    """

    swap_window_seconds: float = 5.0
    """신원 확인에 쓰는 구간. 매장 안에 있던 마지막 이 시간 동안 잰 값 중 가장 높은 값을 쓴다.

    문에 다가가며 역광으로 잠깐 떨어지는 것(일시적)과 번호 뒤바뀜(지속적)을 가르기 위해서다.
    대가: 나가기 직전 이 시간 안에 뒤바뀐 경우는 잡지 못한다.
    """

    payment_mode: str = "feed"
    """결제를 어떻게 아나.
    - "feed":  결제 기록(CSV, POS)을 받아 계산대에 있던 손님에게 붙인다.
    - "dwell": 결제 기록이 없는 매장 — **계산대에 dwell_pay_seconds 이상 서 있으면 그때까지 가진 것을 모두 결제한 것**으로 본다.
               결제 뒤 다시 상품 쪽(SHELF 구역)으로 가면 **취소**로 보고, 다음 계산대 체류에서 다시 결제한다.
               (직접 찍은 매장 영상의 규칙, 2026-10-14 사용자 확인)"""

    dwell_pay_seconds: float = 1.5
    """payment_mode="dwell": 계산대 구역에 이만큼 연속으로 있으면 결제. 정상 영상의 계산대 체류는 2.4~9초였다."""

    cancel_shelf_overlap: float = 0.5
    """payment_mode="dwell": 결제 뒤 몸이 SHELF 구역과 이 비율 이상 겹치면 '상품 쪽으로 돌아감' = 결제 취소."""

    checkout_grace_seconds: float = 3.0
    """결제 순간 계산대 구역에 없더라도, 이 시간 안에 있었던 손님이면 결제자로 본다.

    POS 기록 시각과 영상 시각이 조금 어긋나거나, 결제 직후 한 걸음 물러나는 경우를 위한 여유.
    """


@dataclass
class AlertConfig:
    """관리자 앱·대시보드 경보 서버로 위험 판정을 보내는 설정 (src/sinks/alert_sink.py).

    서버와 전송 모듈은 app-pos(한호석)가 만든 것을 쓴다. 서버 주소가 없으면 보내지 않는다.
    API 키는 여기 두지 않는다 — 환경변수 HAWKEYE_API_KEY 로만 받는다(설정 파일·터미널 기록에 남지 않게).
    """

    server: str | None = None
    """경보 서버 주소 (예: http://127.0.0.1:8000). 비우면 환경변수 HAWKEYE_SERVER, 그것도 없으면 보내지 않는다."""

    camera_id: str = "cam1"

    client_path: str = "../app-pos/client"
    """app-pos 전송 모듈(hawkeye_client) 위치. 이 폴더(experiments/tracking) 기준."""

    snapshot_seconds: float = 6.0
    """경보 사진용으로 보관할 최근 장면 길이(초). '출구 쪽에서 사라짐' 경보는 사라지고 3초 뒤에 나오므로
    그 손님이 마지막으로 보인 장면을 찾으려면 그보다 길어야 한다."""

    snapshot_max_side: int = 960
    """보관하는 장면의 긴 변 크기. 메모리를 아끼려고 줄여 둔다."""


@dataclass
class ShelfConfig:
    """선반 지도 — 지켜보는 선반에서 물건이 몇 개 줄었는지 세어 손님 기록에 붙인다 (src/shelf, docs/shelf-map.md).

    areas 가 비어 있으면 꺼진다. 영역은 카메라마다 다르므로 보통 명령줄 --shelf 로 준다.
    """

    areas: list[dict] = field(default_factory=list)
    """[{name, box: [x1, y1, x2, y2] (화면 비율 0~1), zone: SHELF 구역 이름(선택)}].
    zone 을 주면 그 구역의 집기 동작만 선반 기록으로 대신한다. 선반이 여럿이면 주는 것이 안전하다."""

    model_path: str = "yolov8n.pt"
    """물건 탐지 모델. 지금은 일반 YOLO(2단계에서 우리 상품 모델로 교체)."""

    imgsz: int = 1280
    """물건 탐지 해상도. 작은 물건용으로 크게 한다. 사람 탐지는 원래 설정(detector.imgsz)대로 따로 한다 —
    1280 으로 사람까지 찾으면 가까운 사람이 조각 상자 여러 개로 잡혔다(Pexels 10566665)."""

    conf: float = 0.05
    """물건 탐지기 최소 점수. 낮게 잡아 '약한 탐지'까지 받는다 — 지도에는 map_score 이상만 넣고,
    약한 탐지는 사라짐·생김을 확인하는 데만 쓴다 (ShelfMonitor 설명)."""

    map_score: float = 0.15
    """선반 지도에 넣는 최소 점수 (예전의 탐지 기준과 같은 값)."""

    weak_presence: float = 0.3
    """사라짐·생김 확인: 그 자리에 약한 탐지라도 장면의 이 비율 이상 보이면 '계속 있었다'로 본다."""

    stable_seconds: float = 0.6
    """선반 지도를 확정하는 데 필요한 연속 관찰 시간. 1초로는 떠난 뒤 확정이 늦었다(Pexels 10566665)."""

    log_items: bool = False
    """물건 탐지 결과를 shelf_items.jsonl 로 남긴다 (평가용 — 영상 없이 선반 판정을 다시 돌림)."""

    idle_every: int = 3
    """아무 일 없이 지도가 확정돼 있을 때는 이 장면 수마다 한 번만 물건을 찾는다 (선반을 가린 동안은 아예 안 찾음).
    누가 다녀간 뒤 비교할 때는 매 장면 찾는다. 1 이면 예전처럼 매 장면."""

    ignore: list[str] = field(default_factory=lambda: [
        "person", "dining table", "chair", "bench", "refrigerator", "tv", "couch", "bed",
    ])
    """물건으로 보지 않는 종류 (가구 등)."""

    @property
    def enabled(self) -> bool:
        return bool(self.areas)


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
    write_risk: bool = True
    risk_name: str = "risk_events.jsonl"
    payments_name: str = "payments.jsonl"
    shelf_name: str = "shelf_visits.jsonl"
    actions_name: str = "actions.jsonl"
    alerts_dir: str = "alerts"
    """HIGH_RISK 순간의 장면을 이미지로 저장할 폴더 (결과 폴더 아래)."""
    show_window: bool = False
    draw_zones: bool = True
    draw_trail: bool = True
    trail_length: int = 30


@dataclass
class CameraConfig:
    """카메라 2대 실행에서 카메라 하나 (docs/multiview-design.md M-1).

    role:
      overview — 전체 카메라 (angle1). 매장 신원·동선·결제·출구. zones 는 바닥 지도 파일(floor.yaml).
      detail   — 선반 카메라 (angle0). 손·선반 물건. zones 는 화면 구역 파일(선반 칸 = SHELF 구역).
    """

    name: str = "cam0"
    source: str = ""
    role: str = "overview"
    zones: str | None = None
    offset_ms: float = 0.0
    """기준(전체) 카메라 시간축으로 밀 양. 녹화본은 labels/<촬영>/sync.csv 에서 채운다."""
    stride: int = 1
    """N 장면에 1장 처리. 60FPS 전체 카메라는 4 (= 15FPS)."""
    detector: str = "yolo"
    """yolo / yolo-pose. 선반 카메라는 손목이 필요해 yolo-pose."""
    identity: dict = field(default_factory=dict)
    """이 카메라의 신원 계층만 바꿀 값 (IdentityConfig 칸 이름 → 값). 예: 화각 때문에 늘 잘리는 변을 가장자리에서 빼기."""


@dataclass
class FusionConfig:
    """⑦ 합치기 — 교차 화면 연결과 선반 사건 묶기 (설계 M-4·M-5·M-7·M-8). 기준값은 정상(N) 영상으로 정한다."""

    window_seconds: float = 0.5
    """짝을 다시 정하는 조각 길이 (Complementary-View MHT 의 짧은 조각 단위 연결)."""
    position_weight: float = 0.7
    """점수 = w × 위치 + (1 − w) × 외형. 위치가 1순위 (위치를 빼면 화면 사이 맞히기 84.0 → 15.5)."""
    min_score: float = 0.5
    """이보다 낮으면 짝 없음 — 한쪽 화면에만 있는 사람."""
    keep_bonus: float = 0.1
    """지금 짝에 더해 주는 점수. 한 조각의 흔들림으로 짝을 바꾸지 않게."""
    max_band_gap: float = 0.25
    """선반 앞 띠 위치 차이가 이만큼이면 위치 점수 0."""
    position_mode: str = "band"
    """두 화면의 위치를 무엇으로 견주나. "band" = 선반 앞 띠 한 줄 좌표 (2차 촬영 — 선반 카메라에 발이 안 보임, B-14).
    "floor" = 두 카메라의 발을 같은 바닥 좌표(cm)로 옮겨 **바닥 위 거리**로 (3차 촬영 — 선반 카메라에도 바닥 지도가 있을 때).
    발목이 안 보이는 사람(화면 아래로 잘림, 앞사람·가구에 가림)은 점이 아니라 '상자 아래에서 화면 아래쪽으로 뻗는 반직선'으로 보고
    그 반직선까지의 거리를 잰다 — 발이 어디인지는 몰라도 그 줄 위 어딘가인 것은 맞다. 바닥 좌표가 아예 없는 표본만 띠 위치로 대신한다."""
    max_floor_gap_cm: float = 60.0
    """position_mode "floor": 두 카메라가 본 발의 바닥 위 거리가 이만큼이면 위치 점수 0. 외형이 없으면 절반(30cm) 안이어야 짝이 맺힌다.
    정상(N) 한 명 영상에서 같은 사람의 두 카메라 발 거리 분포로 정한다 (scripts/link_diagnose.py --shoot 3 fit)."""
    ray_slack_cm: float = 25.0
    """position_mode "floor": 한쪽이라도 발이 가려 반직선으로 견준 표본은 이 거리 안이면 '같은 자리'(거리 0)로 본다.
    반직선까지의 거리는 그만큼의 차이를 가를 정밀도가 없다 — 정상 한 명 영상에서 같은 사람인데도 중앙 15cm·90% 24cm 가 나온다
    (발목이 두 화면 모두 보이면 중앙 4cm·90% 6cm). 0 이면 두 반직선이 우연히 겹친 다른 사람(거리 0)이 7~18cm 로 나온 진짜 짝을
    이긴다 (3차 39쌍, 2026-11-13). 이 안에서는 외형과 직전 짝(keep_bonus)이 가른다. 25 = 정상 영상 같은 사람 거리의 90%."""
    solo_appearance_min: float = 0.85
    """선반 앞에 비교할 전체 카메라 사람이 한 명뿐이고 외형이 이 이상이면, 위치가 어긋나도 잇는다 (0 = 끔).
    헷갈릴 상대가 없을 때만 외형을 믿는다 — 선반 카메라 띠 위치가 계산대·팔 뻗기에서 어긋나는 B-14 대책 (2026-11-07).
    0.85 = 정상(N) 영상에서 다른 사람끼리 나온 외형의 최댓값(0.844, 표본 9) 바로 위. 0.8 이면 그 다른 사람도 잇는다."""
    link_customers_only: bool = False
    """전체 카메라 신원 중 **매장 안 바닥을 밟은 근거가 있는 신원**만 두 화면 짝의 후보로 삼는다:
    ① 손님으로 센 신원(매장 안 바닥에 risk.inside_seconds 이상 — 손님 장부의 seen_inside), 또는
    ② 그 조각(0.5초)에 발목이 보이고 그 발이 매장 안 바닥 구역(risk.inside_zones)에 있던 신원 (막 들어온 손님).
    창가 겉옷·가방에 생긴 가짜 사람 상자는 매장 안 바닥을 밟지 않아 손님이 아닌데, 그 반직선이 손님 자리를 지나가면
    위치 점수 만점으로 손님의 짝을 가로챘다 (3차 N09 2조각 · N12 1조각, 후속 39). 새 기준값은 없다 — 손님 규칙과 구역을 그대로 쓴다.
    후보에서만 빠질 뿐 표본·손님 장부·판정은 그대로다. 기본 꺼짐, 3차 평가(`dual_eval.py --shoot 3`)에서 켠다 (후속 41).
    ② 는 risk.inside_excludes_exit(문 구역 안은 매장 안 아님, 후속 29)를 따르지 않는다 — 통로가 문 구역을 덮은 곳에 선 사람도 후보다.
    문 앞 손님을 후보에서 빼면 2차에서 짝 6조각을 잃었다."""
    keep_bonus_recent_only: bool = False
    """keep_bonus(직전 짝 가산점)를 **바로 앞 조각에서 맺힌 짝**에게만 준다. 꺼져 있으면 선반 카메라에서 몇 초 사라졌다 돌아온 번호도
    사라지기 전의 짝에게 가산점을 준다 — 그 사이 번호가 다른 사람에게 붙었으면 틀린 짝을 붙든다
    (3차 C15 18.7초: 4초 전 화면 끝에서 두 장면 잘못 붙은 번호의 짝이 가산점으로 진짜 짝 0.97 을 1.01 로 이겼다, 후속 42).
    '바로 앞'은 짝 구간을 잇는 기준과 같다 (조각 길이의 1.5배 안). 새 기준값 없음."""
    ankle_hidden_behind: bool = False
    """position_mode "floor": 상자 아래 가운데가 **더 앞에 선 다른 사람의 상자 안**이면 그 사람의 발목 관절을 믿지 않고
    상자 아래에서 뻗는 반직선으로 본다. 앞사람에게 다리가 가려 상자가 허리께에서 잘려도 자세 모델은 발목을 상자 안 어딘가에
    짐작해 찍는다 — 그 점을 믿으면 70~80cm 어긋나 짝이 끊긴다 (3차 C15 9.1초 82cm · N09 20.8초 65cm, 후속 39·42).
    '더 앞' = 그 상자의 아래 변이 내 상자 아래 변보다 내 상자 높이의 5% 넘게 아래 (발목을 상자 안으로 치는 여유와 같은 값)."""
    pair_tolerance_seconds: float = 0.15
    """두 화면 표본을 '같은 때'로 보는 시간 차 — 시차 추정 오차(설계 §2-3)를 견디게."""
    event_tolerance_seconds: float = 1.0
    """같은 칸의 선반 변화·센서·손 접촉을 한 사건으로 묶는 시간 차 (M-8)."""
    shelf_zone: str = "shelf"
    """두 카메라의 집기 동작·선반 방문을 손님 장부에서 맞출 공통 구역 이름."""
    release_delay_seconds: float = 1.5
    """선반 카메라의 손 접촉·선반 방문을 끝나고 이만큼 기다렸다 주인을 정한다 — 짝 기록·칸 그림 결과가 쌓이게.
    (처음엔 손 접촉 1.0초였는데 2차 C1 에서 손 접촉과 방문의 주인이 갈려 1.5초로 맞췄다)"""
    unchanged_clears_all: bool = True
    """물건 탐지·칸 그림이 모두 '그대로'인 방문이면 그때 선반 앞 띠의 다른 손님 손 뻗기도 '만지기만'으로 (2차 C12 를 보고 추가)."""
    sensor_path: str | None = None
    """선반 센서 기록 (CSV: time_sec, slot, delta_g, confidence). 없으면 센서 없이."""
    sensor_unit_g: dict = field(default_factory=dict)
    """칸별 물건 하나 무게(g). 무게 변화를 개수로 바꾼다."""
    overview_reach_seconds: float = 0.0
    """전체 카메라 팔 뻗기: 손목이 전체 카메라 화면의 선반 영역(floor.yaml `shelf_image`)에 이 시간 이상 들어가면 한 번.
    0 이면 끔. 전체 카메라를 yolo-pose 로 돌려야 손목이 있다 (2026-11-04 사용자 제안 — 위에서 보면 팔 뻗기가 잘 보인다).
    쓰임: 선반 변화의 주인 정하기(그 시각 팔을 뻗은 사람이 한 명이면 그 사람)와 손 뻗기의 두 번째 출처.
    """
    overview_reach_conf: float = 0.3
    """전체 카메라 손목 신뢰도 하한."""
    overview_reach_takes: bool = False
    """전체 카메라 팔 뻗기를 그 손님의 손 뻗기(집기 행동)로도 넘길지. 기본 끔 — 주인 정하기에만 쓴다.
    켜면 선반 변화가 없는 팔 뻗기(만지기·같이 온 사람이 가리키기)가 '선반으로 확인 못 한 집기 동작'으로 1개씩
    세어져 복합 정상 장면에 거짓 경보 4개(C1·C4·C12·C21), 절도 장면 이득 0 (2026-11-04 3차 비교).
    """


@dataclass
class AppConfig:
    video: VideoConfig = field(default_factory=VideoConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    zones: ZonesConfig = field(default_factory=ZonesConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    shelves: ShelfConfig = field(default_factory=ShelfConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    cameras: list[CameraConfig] = field(default_factory=list)
    """비어 있으면 카메라 한 대(video.source) — 지금까지와 같다."""
    fusion: FusionConfig = field(default_factory=FusionConfig)
    detail_interaction: InteractionConfig | None = None
    """선반 카메라의 손 뻗기 설정 (손목이 칸 안). 없으면 interaction 을 그대로 쓴다."""

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
            risk=RiskConfig(**(raw.get("risk") or {})),
            alerts=AlertConfig(**(raw.get("alerts") or {})),
            shelves=ShelfConfig(**(raw.get("shelves") or {})),
            output=OutputConfig(**(raw.get("output") or {})),
            cameras=[CameraConfig(**c) for c in raw.get("cameras") or []],
            fusion=FusionConfig(**(raw.get("fusion") or {})),
            detail_interaction=(InteractionConfig(**raw["detail_interaction"])
                                if raw.get("detail_interaction") else None),
        )


def _build_tracker_config(raw: dict[str, Any]) -> TrackerConfig:
    """tracker 설정 안에 중첩된 reid 블록을 따로 조립한다."""
    raw = dict(raw)
    reid = ReIDConfig(**(raw.pop("reid", None) or {}))
    return TrackerConfig(reid=reid, **raw)


def apply_judge_overrides(config: AppConfig, path: str | Path | None) -> dict[str, Any]:
    """분석 도구의 '수치 조정'에서 적용한 판정 값을 얹는다 → 얹은 값들. 파일이 없으면 아무 일도 없다.

    파일은 `절.키: 값` 한 줄씩 (예: `risk.dwell_pay_seconds: 2.0`). 판정 단계에서 쓰는 절(risk·fusion·zones)만 받는다 —
    탐지·추적·신원 값은 영상을 다시 돌려야 반영되므로 여기에 두지 않는다.
    """
    if path is None or not Path(path).exists():
        return {}
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    applied = {}
    for key, value in raw.items():
        section, _, name = str(key).partition(".")
        if section not in ("risk", "fusion", "zones") or not hasattr(getattr(config, section), name):
            raise ValueError(f"{path}: 모르는 판정 값 '{key}'")
        setattr(getattr(config, section), name, value)
        applied[str(key)] = value
    return applied
