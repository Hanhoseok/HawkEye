"""선반 지도 1단계 시연 — 영상에서 '어느 자리의 물건이 사라졌고, 그때 누가 있었나'를 찾는다.

사람: YOLO 'person' -> ByteTrack 으로 번호. 물건: 같은 YOLO 의 나머지 종류(일반 YOLO, 2단계에서 우리 상품 모델로 교체).
선반 영역은 화면 비율(0~1)로 준다.

    python scripts/shelf_demo.py --video data/videos/pexels/pexels_10566665.mp4 \\
        --shelf 0.35,0.62,0.80,0.95 --out outputs/shelf_demo/10566665

출력: 콘솔에 변화 기록, out 폴더에 변화 전/후 장면 이미지(사라진 자리 빨간 상자).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import AppConfig  # noqa: E402
from src.core.types import Detection, Frame  # noqa: E402
from src.shelf.monitor import ShelfItem, ShelfMonitor  # noqa: E402
from src.identity.embedder import build_embedder  # noqa: E402
from src.identity.registry import IdentityRegistry  # noqa: E402
from src.trackers.factory import build_tracker  # noqa: E402

IGNORE = {"person", "dining table", "chair", "bench", "refrigerator", "tv", "couch", "bed", "toilet_seat"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--shelf", required=True, help="x1,y1,x2,y2 (화면 비율 0~1)")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=1280, help="물건 탐지 해상도 (작은 물건용으로 크게)")
    parser.add_argument("--person-imgsz", type=int, default=640,
                        help="사람 탐지 해상도. 크게 하면 가까운 사람이 여러 조각 상자로 잡혔다(10566665 실측)")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--stable", type=float, default=0.6, help="지도를 확정하는 데 필요한 연속 관찰 시간(초)")
    parser.add_argument("--no-identity", action="store_true",
                        help="신원 유지(계층 2)를 끄고 추적 번호를 그대로 쓴다 (비교용)")
    parser.add_argument("--out", default="outputs/shelf_demo")
    args = parser.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    item_classes = [i for i, n in model.names.items() if n not in IGNORE]

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sx1, sy1, sx2, sy2 = (float(v) for v in args.shelf.split(","))
    shelf_box = (sx1 * w, sy1 * h, sx2 * w, sy2 * h)

    config = AppConfig.load("config.yaml")
    config.tracker.name = "bytetrack"
    config.tracker.frame_rate = int(round(fps / args.stride))
    tracker = build_tracker(config.tracker)
    registry = None if args.no_identity else IdentityRegistry(
        config.identity, build_embedder(config.tracker.reid), imgsz=config.detector.imgsz)
    monitor = ShelfMonitor("shelf", shelf_box, stable_seconds=args.stable)
    per_window = max(1, int(round(args.stable * fps / args.stride)))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    snapshots: dict[float, object] = {}
    index = -1
    print(f"영상 {w}x{h} {fps:.0f}fps, 선반 영역 {tuple(int(v) for v in shelf_box)}\n")
    while True:
        ok, image = cap.read()
        if not ok:
            break
        index += 1
        if index % args.stride:
            continue
        now = index * 1000.0 / fps
        # 물건과 사람을 다른 해상도로 찾는다 — 작은 물건은 크게, 사람은 기본 크기로.
        # 찾을 종류를 매번 명시한다 — ultralytics 는 앞 호출의 classes 설정을 다음 호출에 그대로 남긴다.
        result = model.predict(image, imgsz=args.imgsz, conf=args.conf, classes=item_classes, verbose=False)[0]
        items = []
        for box, cls, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(),
                                   result.boxes.conf.cpu().numpy()):
            name = model.names[int(cls)]
            if name not in IGNORE:
                items.append(ShelfItem(name, tuple(float(v) for v in box), float(score)))
        presult = model.predict(image, imgsz=args.person_imgsz, conf=0.25, classes=[0], verbose=False)[0]
        people_det = [Detection(bbox=tuple(float(v) for v in b), score=float(c), class_id=0, class_name="person")
                      for b, c in zip(presult.boxes.xyxy.cpu().numpy(), presult.boxes.conf.cpu().numpy())]
        frame = Frame(index=index, timestamp="", pts_ms=now, image=image)
        tracks = tracker.update(frame, people_det)
        if registry is None:
            people = [(t.track_id, t.bbox) for t in tracks]
        else:
            # 실제 파이프라인처럼 추적 번호 대신 매장 단위 신원을 쓴다 (쪼개진 번호를 한 사람으로 잇는다).
            # 판정 보류(-1)인 사람도 선반을 가리기는 하므로 가림에는 넣되 번호는 -1 로 둔다.
            people = [(registry.resolve(o.person_id) if o.person_id > 0 else -1, o.bbox)
                      for o in registry.assign(frame, tracks)]

        snapshots[now] = image
        snapshots = {t: im for t, im in snapshots.items() if now - t < 20000}   # 최근 20초만
        for change in monitor.update(now, items, people, per_window):
            print(f"  {change.start_ms / 1000:5.1f}s ~ {change.end_ms / 1000:5.1f}s  {change.describe()}")
            before_t = max((t for t in snapshots if t <= change.start_ms), default=min(snapshots))
            vis_b, vis_a = snapshots[before_t].copy(), image.copy()
            for s in change.removed:
                x1, y1, x2, y2 = (int(v) for v in s.item.bbox)
                cv2.rectangle(vis_b, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(vis_b, f"row{s.row} #{s.index} {s.item.label}", (x1, max(15, y1 - 6)), 0, 0.6, (0, 0, 255), 2)
            for s in change.added:
                x1, y1, x2, y2 = (int(v) for v in s.item.bbox)
                cv2.rectangle(vis_a, (x1, y1), (x2, y2), (0, 200, 0), 3)
            for vis, tag in ((vis_b, "before"), (vis_a, "after")):
                x1, y1, x2, y2 = (int(v) for v in shelf_box)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 200, 0), 1)
                cv2.putText(vis, tag, (10, 30), 0, 1.0, (255, 255, 0), 2)
            cv2.imwrite(str(out / f"change_{change.end_ms / 1000:05.1f}s.jpg"),
                        cv2.hconcat([vis_b, vis_a]))
    cap.release()

    print()
    if monitor.state is not None:
        print("영상 끝 선반 지도: " + ", ".join(s.name for s in monitor.state))
    print(f"변화 {len(monitor.changes)}건, 이미지: {out}")


if __name__ == "__main__":
    main()
