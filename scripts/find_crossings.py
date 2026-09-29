"""같은 프레임에서 두 track 의 bbox 가 겹치는 구간(교차)을 찾는다.

Phase 3 실험 #2(사람 교차)에서 "어디를 봐야 하는지"를 먼저 좁히기 위한 도구.
교차 구간의 frame 번호를 얻은 뒤 결과 영상의 해당 지점만 확인하면 된다.

    python scripts/find_crossings.py outputs/people-bytetrack/observations.jsonl --min-iou 0.1
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.types import TrackObservation
from src.sinks.observation_log import read_observations


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--min-iou", type=float, default=0.1, help="이 값 이상 겹치면 교차로 본다")
    args = parser.parse_args()

    by_frame: dict[int, list[TrackObservation]] = defaultdict(list)
    for obs in read_observations(args.path):
        by_frame[obs.frame].append(obs)

    # 겹친 프레임을 (track 쌍) 별로 묶어 연속 구간으로 만든다
    events: dict[tuple[int, int], list[tuple[int, float]]] = defaultdict(list)
    for frame in sorted(by_frame):
        for a, b in combinations(by_frame[frame], 2):
            overlap = iou(a.bbox, b.bbox)
            if overlap >= args.min_iou:
                pair = tuple(sorted((a.track_id, b.track_id)))
                events[pair].append((frame, overlap))

    if not events:
        print(f"IoU {args.min_iou} 이상 겹치는 구간이 없습니다. (교차 사건 없음)")
        return

    print(f"{'track 쌍':>12} {'시작':>6} {'종료':>6} {'프레임':>6} {'최대IoU':>8}")
    print("-" * 46)
    for pair, hits in sorted(events.items()):
        frames = [f for f, _ in hits]
        print(f"{str(pair):>12} {min(frames):>6} {max(frames):>6} {len(frames):>6} {max(o for _, o in hits):>8.2f}")
    print("-" * 46)
    print(f"교차 사건 {len(events)}건")
    print("\n이 frame 구간의 결과 영상을 확인해 ID 가 서로 뒤바뀌었는지 눈으로 검증한다.")


if __name__ == "__main__":
    main()
