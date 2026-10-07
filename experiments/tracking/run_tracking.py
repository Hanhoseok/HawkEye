"""1차 구현 실행 진입점.

    python run_tracking.py --source data/videos/sample.mp4
    python run_tracking.py --source data/videos/sample.mp4 --max-frames 100 --show

설정값은 config.yaml 에서 읽고, 명령행 인자는 그 위에 덮어쓴다.
근거: 05 아키텍처 §4 (임계값·모델 경로 하드코딩 금지)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.core.config import AppConfig
from src.detectors.yolo_detector import build_detector, build_item_detector
from src.pipeline import TrackingPipeline
from src.identity.embedder import build_embedder
from src.interaction.detector import InteractionDetector
from src.risk.engine import RiskEngine
from src.risk.payments import PaymentFeed
from src.shelf.watcher import ShelfWatcher, parse_area
from src.sinks.alert_sink import AlertSink
from src.viz.live_panel import LivePanel, parse_truth
from src.identity.registry import IdentityRegistry
from src.trackers import build_tracker
from src.zones.zone_map import ZoneMap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="사람 탐지 + 추적 파이프라인 (1차 구현)")
    parser.add_argument("--config", default="config.yaml", help="설정 파일 경로")
    parser.add_argument("--source", help="입력 영상 경로 또는 카메라 인덱스")
    parser.add_argument("--model", help="YOLO 가중치 경로 (예: yolov8n.pt)")
    parser.add_argument(
        "--detector", choices=["yolo", "yolo-pose"], help="yolo-pose 는 손 위치까지 함께 준다"
    )
    parser.add_argument(
        "--signal",
        choices=["dwell", "hand", "raise", "stage", "reach"],
        help="TAKE 후보의 판정 근거. stage = 체류로 거르고 손 올림으로 시점 찍기",
    )
    parser.add_argument("--tracker", choices=["bytetrack", "botsort"], help="사용할 tracker")
    parser.add_argument("--device", help="cpu / cuda:0")
    parser.add_argument("--conf", type=float, help="detection confidence 임계값")
    parser.add_argument("--imgsz", type=int, help="YOLO 입력 크기. 작은 객체에는 키우면 유리")
    parser.add_argument(
        "--classes", help="탐지할 클래스를 쉼표로 (예: person 또는 bowl,cup). 비우면 전체"
    )
    parser.add_argument("--start-frame", type=int, help="이 프레임부터 처리 (구간 실험용)")
    parser.add_argument(
        "--blackout", help="START:END 원본 프레임 구간을 검은 화면으로 덮는다 (긴 공백 실험용)"
    )
    parser.add_argument("--max-frames", type=int, help="처리할 최대 프레임 수")
    parser.add_argument("--stride", type=int, help="N 프레임마다 1장 처리")
    parser.add_argument(
        "--scale", type=float, help="축소 배율(0~1). 카메라가 멀어진 상황을 흉내낸다"
    )
    parser.add_argument(
        "--scale-mode",
        choices=["shrink", "resize"],
        help="shrink=화면 크기 유지하고 내용만 축소(실제 원거리에 가까움) / resize=프레임 자체 축소",
    )
    parser.add_argument(
        "--tracker-fps",
        type=int,
        help="tracker 가 가정할 실효 fps. lost_track_buffer 가 프레임 단위이므로 "
        "원본 fps/stride 에 맞춰 주는 것이 맞다",
    )
    parser.add_argument("--lost-buffer", type=int, help="사라진 track 을 몇 프레임 기억할지")
    parser.add_argument("--match-thresh", type=float, help="IoU 매칭 임계값 (낮출수록 관대)")
    parser.add_argument("--min-consec", type=int, help="track 확정에 필요한 연속 매칭 프레임 수")
    parser.add_argument(
        "--reid-proximity",
        type=float,
        help="외형 비교를 허용할 IoU 거리 상한. 1.0 이면 게이트 없음 (botsort 전용)",
    )
    parser.add_argument("--reid-appearance", type=float, help="외형 거리 상한 (botsort 전용)")
    parser.add_argument(
        "--no-reid",
        action="store_true",
        help="botsort 에서 외형 모델을 끈다. 외형의 기여를 분리 측정하는 대조 실험용",
    )
    parser.add_argument("--out-dir", help="결과 저장 폴더")
    parser.add_argument("--show", action="store_true", help="실시간 창으로 보기 (상황판 포함)")
    parser.add_argument("--panel", action="store_true", help="영상 옆에 손님별 상태·최근 사건 상황판을 붙인다 (결과 영상에도)")
    parser.add_argument("--truth", help="정답 구간(초) — 상황판에 표시. 예: '24-73;76-80'")
    parser.add_argument("--no-video", action="store_true", help="결과 영상 저장 안 함")
    parser.add_argument("--zones", help="구역 정의 파일 경로 (기본 zones.yaml)")
    parser.add_argument("--no-takes", action="store_true", help="TAKE 후보 판정(계층 3)을 끈다")
    parser.add_argument("--payments", help="결제 기록 CSV (src/risk/payments.py 형식)")
    parser.add_argument("--no-risk", action="store_true", help="손님 상태·결제·출구 판정을 끈다")
    parser.add_argument("--alert-server", help="경보 서버 주소 (예: http://127.0.0.1:8000). API 키는 환경변수 HAWKEYE_API_KEY")
    parser.add_argument("--camera-id", help="경보 서버에 보낼 카메라 이름 (기본 cam1)")
    parser.add_argument(
        "--shelf", action="append",
        help="지켜볼 선반 영역 [이름=]x1,y1,x2,y2[@SHELF구역] (화면 비율 0~1, 여러 번 가능). "
             "주면 선반에서 물건이 몇 개 줄었는지 세어 손님 기록에 붙인다",
    )
    parser.add_argument("--dwell", type=float, help="선반 앞 체류 시간 기준(초)")
    parser.add_argument("--max-speed", type=float, help="'멈춤' 기준 속도 (체구 높이/초)")
    parser.add_argument("--no-zones", action="store_true", help="구역을 쓰지 않는다")
    parser.add_argument(
        "--no-identity", action="store_true", help="매장 단위 신원 레지스트리(계층 2)를 끈다"
    )
    parser.add_argument(
        "--identity-threshold", type=float, help="같은 사람으로 볼 외형 유사도 하한(0~1)"
    )
    parser.add_argument(
        "--retire-after",
        type=float,
        help="이 초 동안 '연속으로' 안 보이면 매장을 나간 것으로 본다 (체류 시간 상한이 아님)",
    )
    return parser.parse_args()


def apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    if args.source:
        config.video.source = args.source
    if args.model:
        config.detector.model_path = args.model
    if args.tracker:
        config.tracker.name = args.tracker
    if args.detector:
        config.detector.name = args.detector
        if not args.model:
            config.detector.model_path = (
                "yolov8n-pose.pt" if args.detector == "yolo-pose" else "yolov8n.pt"
            )
    if args.signal:
        config.interaction.signal = args.signal
    if args.device:
        config.detector.device = args.device
    if args.conf is not None:
        config.detector.conf_threshold = args.conf
    if args.scale_mode:
        config.video.scale_mode = args.scale_mode
    if args.scale is not None:
        config.video.scale = args.scale
    if args.blackout:
        config.video.blackout = args.blackout
    if args.start_frame is not None:
        config.video.start_frame = args.start_frame
    if args.imgsz is not None:
        config.detector.imgsz = args.imgsz
    if args.classes is not None:
        config.detector.classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if args.max_frames is not None:
        config.video.max_frames = args.max_frames
    if args.stride is not None:
        config.video.stride = args.stride
    if args.tracker_fps is not None:
        config.tracker.frame_rate = args.tracker_fps
    if args.lost_buffer is not None:
        config.tracker.lost_track_buffer = args.lost_buffer
    if args.match_thresh is not None:
        config.tracker.minimum_matching_threshold = args.match_thresh
    if args.min_consec is not None:
        config.tracker.minimum_consecutive_frames = args.min_consec
    if args.no_reid:
        config.tracker.reid.weights = ""
    if args.reid_proximity is not None:
        config.tracker.reid.proximity_threshold = args.reid_proximity
    if args.reid_appearance is not None:
        config.tracker.reid.appearance_threshold = args.reid_appearance
    if args.out_dir:
        config.output.dir = args.out_dir
    if args.show:
        config.output.show_window = True
    if args.no_video:
        config.output.write_video = False
    if args.zones:
        config.zones.path = args.zones
    if args.payments:
        config.risk.payments = args.payments
    if args.shelf:
        config.shelves.areas = [parse_area(text, i) for i, text in enumerate(args.shelf)]
    if args.no_risk:
        config.risk.enabled = False
    if args.no_takes:
        config.interaction.enabled = False
    if args.dwell is not None:
        config.interaction.dwell_seconds = args.dwell
    if args.max_speed is not None:
        config.interaction.max_speed = args.max_speed
    if args.no_zones:
        config.zones.enabled = False
    if args.no_identity:
        config.identity.enabled = False
    if args.identity_threshold is not None:
        config.identity.match_threshold = args.identity_threshold
    if args.retire_after is not None:
        config.identity.retire_after_seconds = args.retire_after
    return config


def main() -> int:
    args = parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        config = AppConfig.load(config_path)
    else:
        print(f"[warn] 설정 파일이 없어 기본값으로 실행합니다: {config_path}")
        config = AppConfig()
    config = apply_overrides(config, args)

    print(f"detector : {config.detector.name} ({config.detector.model_path}, {config.detector.device})")
    print(
        f"tracker  : {config.tracker.name} "
        f"(fps={config.tracker.frame_rate}, lost_buffer={config.tracker.lost_track_buffer})"
    )

    detector = build_detector(config.detector)
    tracker = build_tracker(config.tracker)

    registry = None
    if config.identity.enabled:
        embedder = build_embedder(config.tracker.reid)
        registry = IdentityRegistry(
            config.identity, embedder, imgsz=config.detector.imgsz
        )
        print(
            f"identity : 매장 단위 신원 레지스트리 켜짐 "
            f"(유사도 {config.identity.match_threshold}, "
            f"퇴장 {config.identity.retire_after_seconds:g}초 미관측, "
            f"최소 텐서 {config.identity.min_tensor_height}px)"
        )

    zone_map = ZoneMap.empty()
    if config.zones.enabled:
        zone_path = Path(config.zones.path)
        if zone_path.exists():
            zone_map = ZoneMap.load(zone_path)
        else:
            print(f"[warn] 구역 파일이 없어 구역 없이 실행합니다: {zone_path}")

    interactions = None
    if config.interaction.enabled and zone_map and registry is not None:
        interactions = InteractionDetector(config.interaction, zone_map)
        print(
            f"take     : 선반 앞 체류 {config.interaction.dwell_seconds:g}초 이상, "
            f"속도 {config.interaction.max_speed:g} 체구/초 이하, "
            f"겹침 {config.interaction.min_overlap:g} 이상"
        )

    risk = None
    if config.risk.enabled and interactions is not None:
        payments = PaymentFeed.load(config.risk.payments) if config.risk.payments else PaymentFeed.empty()
        risk = RiskEngine(
            config.risk, zone_map, payments,
            consistency=registry.recent_consistency if registry is not None else None,
        )
        types = {z.type.value for z in zone_map.zones}
        print(
            f"risk     : 미결제 상태로 출구에 다가서면 WARNING, 출구에서 사라져 "
            f"{config.risk.exit_confirm_seconds:g}초 안 돌아오면 HIGH_RISK "
            f"(결제 기록 {len(payments)}건)"
        )
        if "EXIT" not in types:
            print("[warn] 구역 파일에 EXIT 가 없어 출구 판정이 일어나지 않습니다.")
        if len(payments) and "CHECKOUT" not in types:
            print("[warn] 구역 파일에 CHECKOUT 이 없어, person_id 가 적히지 않은 결제는 손님에게 연결되지 않습니다.")

    alert_sink = None
    if risk is not None:
        alert_sink = AlertSink.create(args.alert_server or config.alerts.server,
                                      args.camera_id or config.alerts.camera_id, config.alerts.client_path)
        if alert_sink is not None:
            print(f"alerts   : {alert_sink.server} 로 경보 전송 (카메라 {alert_sink.camera_id}, 실행 {alert_sink.run_id})")

    shelves = None
    if config.shelves.enabled:
        if registry is None:
            print("[warn] 선반 지도는 매장 단위 신원(identity)이 켜져 있어야 손님에게 붙일 수 있어 끕니다.")
        else:
            scfg = config.shelves
            shelves = ShelfWatcher(scfg, build_item_detector(
                scfg.model_path, scfg.imgsz, scfg.conf, config.detector.device, scfg.ignore))
            print(
                f"shelf    : 선반 {len(scfg.areas)}곳에서 물건 수 확인 "
                f"({scfg.model_path}, 해상도 {scfg.imgsz}, 관찰 {scfg.stable_seconds:g}초)"
            )

    panel = None
    if args.panel or args.show:
        panel = LivePanel(Path(str(config.video.source)).name, parse_truth(args.truth))

    stats = TrackingPipeline(
        config, detector, tracker, registry, zone_map, interactions, risk, alert_sink, shelves, panel
    ).run()
    print(stats.summary())

    if stats.frames == 0:
        print("[error] 처리된 프레임이 없습니다. 입력 영상을 확인하세요.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
