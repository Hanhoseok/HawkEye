"""저장된 추적 결과(observations.jsonl) 위에서 신원 레지스트리만 다시 돌린다.

왜 필요한가: 신원 레지스트리 설정을 바꿀 때마다 파이프라인 전체(YOLO 포함)를
다시 돌리면 영상 하나에 5분, MERL 35개면 3시간이 걸린다.
레지스트리는 추적 결과와 사람 crop 만 있으면 되므로, 저장된 추적 결과를 읽고
영상은 임베딩이 필요한 프레임의 이미지를 얻는 데만 쓴다.

계층을 나눠 둔 덕분에 가능한 일이다 — 계층 2는 TrackObservation 만 받는다.

    # 한 영상
    python scripts/replay_identity.py --runs outputs/merl --names 40_1 --out outputs/merl_stitch

    # stitching 을 끄고 돌려 원래 결과가 재현되는지 확인 (도구 검증용)
    python scripts/replay_identity.py --runs outputs/merl --names 40_1 --out outputs/merl_replay --no-stitch

나중에 합치기(late merge)가 일어나면, 영상이 끝난 뒤 합친 기록을 반영해 identities.jsonl 을
다시 쓴다. 뒤 계층(상태 관리)이 합친 기록을 받아 두 신원의 이력을 하나로 합친 것과 같은 상태이며,
평가(merl_eval)는 이 상태를 기준으로 한다. 합친 기록 자체는 identity_merges.jsonl 에 남긴다.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import AppConfig  # noqa: E402
from src.core.types import Frame  # noqa: E402
from src.identity.embedder import build_embedder  # noqa: E402
from src.identity.registry import IdentityRegistry  # noqa: E402
from src.sinks.observation_log import ObservationLog, read_observations  # noqa: E402

MERL_VIDEOS = Path("data/merl/Videos_MERL_Shopping_Dataset")


def replay(video: Path, observations: Path, out: Path, registry: IdentityRegistry) -> dict:
    by_frame: dict[int, list] = defaultdict(list)
    for obs in read_observations(observations):
        by_frame[obs.frame].append(obs)
    if not by_frame:
        return {"frames": 0, "people": 0, "stitched": 0, "merges": 0}

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    last = max(by_frame)
    registry.reset()
    out.parent.mkdir(parents=True, exist_ok=True)

    written = []
    stitched = 0
    index = -1
    while index < last:
        ok, image = cap.read()
        if not ok:
            break
        index += 1
        current = by_frame.get(index)
        if not current:
            continue  # 관측이 없는 프레임은 레지스트리 상태를 바꾸지 않는다
        frame = Frame(index=index, timestamp=current[0].timestamp,
                      pts_ms=index * 1000.0 / fps, image=image)
        identities = registry.assign(frame, current)
        written.extend(identities)
        stitched += sum(1 for o in identities if o.stitched)
    cap.release()

    resolved = [
        dataclasses.replace(o, person_id=registry.resolve(o.person_id)) if o.person_id > 0 else o
        for o in written
    ]
    with ObservationLog(out) as log:
        log.write_many(resolved)
    out.with_name("identity_merges.jsonl").write_text(
        "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in registry.merges), encoding="utf-8"
    )
    people = {o.person_id for o in resolved if o.person_id > 0}
    return {"frames": len(by_frame), "people": len(people), "stitched": stitched,
            "merges": len(registry.merges)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="outputs/merl", help="observations.jsonl 이 있는 폴더들의 상위")
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--out", required=True, help="identities.jsonl 을 쓸 상위 폴더")
    parser.add_argument("--video-dir", default=str(MERL_VIDEOS))
    parser.add_argument("--video-suffix", default="_crop.mp4")
    parser.add_argument("--video", help="영상 경로를 직접 지정 (--names 가 하나일 때, MERL 이 아닌 영상용)")
    parser.add_argument("--no-stitch", action="store_true")
    parser.add_argument("--stitch-seconds", type=float)
    parser.add_argument("--stitch-distance", type=float)
    parser.add_argument("--stitch-threshold", type=float)
    parser.add_argument("--match-threshold", type=float)
    parser.add_argument("--no-late-merge", action="store_true")
    parser.add_argument("--refresh-seconds", type=float)
    parser.add_argument("--edge-margin", type=float)
    parser.add_argument("--late-merge-threshold", type=float)
    args = parser.parse_args()

    config = AppConfig.load("config.yaml")
    ident = config.identity
    if args.no_stitch:
        ident.stitch_enabled = False
    if args.stitch_seconds is not None:
        ident.stitch_max_seconds = args.stitch_seconds
    if args.stitch_distance is not None:
        ident.stitch_max_distance = args.stitch_distance
    if args.stitch_threshold is not None:
        ident.stitch_threshold = args.stitch_threshold
    if args.match_threshold is not None:
        ident.match_threshold = args.match_threshold
    if args.refresh_seconds is not None:
        ident.template_refresh_seconds = args.refresh_seconds
    if args.edge_margin is not None:
        ident.edge_margin = args.edge_margin
    if args.no_late_merge:
        ident.late_merge_enabled = False
    if args.late_merge_threshold is not None:
        ident.late_merge_threshold = args.late_merge_threshold

    embedder = build_embedder(config.tracker.reid)
    registry = IdentityRegistry(ident, embedder, imgsz=config.detector.imgsz)
    print(
        f"stitching={'켬' if ident.stitch_enabled else '끔'}  "
        f"(생김새 기준 {ident.match_threshold} / 완화 {ident.stitch_threshold}, "
        f"{ident.stitch_max_seconds}초, 체구 {ident.stitch_max_distance}배)  "
        f"나중에 합치기={'켬' if ident.late_merge_enabled else '끔'} "
        f"(기준 {ident.late_merge_threshold}, 확인 시점 {list(ident.late_merge_samples)}장)\n"
    )

    for name in args.names:
        started = time.perf_counter()
        video = Path(args.video) if args.video else Path(args.video_dir) / f"{name}{args.video_suffix}"
        stats = replay(
            video,
            Path(args.runs) / name / "observations.jsonl",
            Path(args.out) / name / "identities.jsonl",
            registry,
        )
        print(
            f"  {name}: 신원 {stats['people']}명, 위치·시간으로 이어붙인 관측 {stats['stitched']}건, "
            f"나중에 합침 {stats['merges']}건 "
            f"({time.perf_counter() - started:.0f}초)"
        )


if __name__ == "__main__":
    main()
