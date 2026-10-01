"""저장된 신원 기록(identities.jsonl) 위에서 TAKE 판정 → 손님 상태 → 결제 → 출구 판정만 다시 돌린다.

왜 필요한가: 결제 기록이나 구역, 판정 기준을 바꿀 때마다 YOLO 부터 다시 돌리면 영상 하나에 몇 분씩 걸린다.
뒤 계층은 IdentityObservation 만 받으므로 저장된 기록으로 몇 초 만에 다시 볼 수 있다.

    python scripts/replay_risk.py --identities outputs/store/identities.jsonl \\
        --video data/videos/store.mp4 --zones zones.yaml --payments payments.csv

--video 는 fps 와 해상도를 알아내는 데만 쓴다(프레임을 읽지 않는다).
영상을 읽지 않으므로 판정 직전 신원 확인(외형 비교)은 하지 않는다. 그 확인까지 보려면 run_tracking.py 로 돌린다.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import AppConfig, InteractionConfig  # noqa: E402
from src.core.types import Frame, RiskLevel  # noqa: E402
from src.interaction.detector import InteractionDetector  # noqa: E402
from src.risk.engine import RiskEngine  # noqa: E402
from src.risk.payments import PaymentFeed  # noqa: E402
from src.sinks.observation_log import ObservationLog, read_identities  # noqa: E402
from src.zones.zone_map import ZoneMap  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identities", required=True)
    parser.add_argument("--video", required=True, help="fps·해상도 확인용")
    parser.add_argument("--zones", default="zones.yaml")
    parser.add_argument("--payments", help="결제 기록 CSV")
    parser.add_argument("--merges", help="identity_merges.jsonl (파이프라인 출력일 때)")
    parser.add_argument("--interaction", help="TAKE 판정 설정 JSON (기본: config.yaml)")
    parser.add_argument("--out", help="risk_events.jsonl / payments.jsonl 을 쓸 폴더")
    args = parser.parse_args()

    config = AppConfig.load("config.yaml")
    icfg = config.interaction
    if args.interaction:
        icfg = InteractionConfig(**json.loads(Path(args.interaction).read_text(encoding="utf-8")))

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    zone_map = ZoneMap.load(args.zones)
    zone_map.resolve(width, height)
    payments = PaymentFeed.load(args.payments) if args.payments else PaymentFeed.empty()
    engine = RiskEngine(config.risk, zone_map, payments)
    detector = InteractionDetector(icfg, zone_map)

    merges_by_frame = defaultdict(list)
    if args.merges and Path(args.merges).exists():
        for line in Path(args.merges).read_text(encoding="utf-8").splitlines():
            if line.strip():
                m = json.loads(line)
                merges_by_frame[m["frame"]].append(m)

    by_frame = defaultdict(list)
    for obs in read_identities(args.identities):
        by_frame[obs.frame].append(obs)

    print(f"영상 {width}x{height} {fps:.1f}fps, 관측 프레임 {len(by_frame)}개, 결제 기록 {len(payments)}건")
    print(f"구역: {zone_map.summary()}\n")

    merges: list[dict] = []
    events_all, records_all = [], []
    takes_by_person = defaultdict(list)
    last_ms = 0.0
    for index in sorted(by_frame):
        frame = Frame(index=index, timestamp="", pts_ms=index * 1000.0 / fps, image=None)
        merges.extend(merges_by_frame.get(index, []))
        done = detector.update(frame, by_frame[index])
        for c in done:
            takes_by_person[c.person_id].append(c)
            print(f"  [take] {c.start_frame / fps:7.2f}~{c.end_frame / fps:6.2f}s  손님 {c.person_id}  ({c.zone})")
        events, records = engine.update(frame, by_frame[index], done, merges)
        for r in records:
            who = f"손님 {r.person_id}" if r.person_id is not None else "연결 실패"
            print(f"  [pay ] {r.time_sec:7.2f}s  {r.items}개 -> {who} ({r.method}, 후보 {r.candidates}명)")
        for e in events:
            mark = {RiskLevel.HIGH_RISK: "!!", RiskLevel.REVIEW: "??", RiskLevel.WARNING: "!?"}.get(e.level, "ok")
            print(f"  [{mark}  ] {e.time_sec:7.2f}s  손님 {e.person_id}  {e.level.value}  {e.reason}")
        events_all += events
        records_all += records
        last_ms = frame.pts_ms
    engine.add_late_takes(detector.flush(), last_ms)

    print()
    print(engine.summary())

    if args.out:
        out = Path(args.out)
        with ObservationLog(out / "risk_events.jsonl") as log:
            log.write_many(events_all)
        with ObservationLog(out / "payments.jsonl") as log:
            log.write_many(records_all)
        print(f"기록: {out}")


if __name__ == "__main__":
    main()
