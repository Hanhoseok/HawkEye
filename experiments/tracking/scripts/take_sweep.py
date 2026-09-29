"""저장된 identities.jsonl 을 다시 돌려 TAKE 후보 임계값을 스윕한다.

추론을 다시 하지 않으므로 초 단위로 끝난다.
계층을 나눠 둔 덕분에 가능한 일이다 — 계층 3은 IdentityObservation 만 있으면 동작한다.

    python scripts/take_sweep.py outputs/take-base/identities.jsonl --fps 59.9
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import InteractionConfig  # noqa: E402
from src.core.types import Frame  # noqa: E402
from src.interaction.detector import InteractionDetector  # noqa: E402
from src.sinks.observation_log import read_identities  # noqa: E402
from src.zones.zone_map import ZoneMap  # noqa: E402


def replay(observations, zone_map, fps: float, **cfg) -> list:
    det = InteractionDetector(InteractionConfig(**cfg), zone_map)
    by_frame: dict[int, list] = {}
    for o in observations:
        by_frame.setdefault(o.frame, []).append(o)
    dummy = np.zeros((2, 2, 3), dtype=np.uint8)
    for idx in sorted(by_frame):
        det.update(Frame(index=idx, timestamp="", pts_ms=idx * 1000.0 / fps, image=dummy), by_frame[idx])
    det.flush()
    return det.completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("identities")
    parser.add_argument("--zones", default="zones.yaml")
    parser.add_argument("--fps", type=float, required=True, help="원본 영상 fps")
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=404)
    args = parser.parse_args()

    zone_map = ZoneMap.load(args.zones)
    zone_map.resolve(args.width, args.height)
    observations = [o for o in read_identities(args.identities) if o.person_id > 0]
    people = {o.person_id for o in observations}
    span = (max(o.frame for o in observations) - min(o.frame for o in observations)) / args.fps
    person_time = len(observations) / args.fps * 2  # stride 2 기준 근사

    print(f"관측 {len(observations):,}건 / {len(people)}명 / 영상 {span:.0f}초")
    print(f"사람이 화면에 있던 총 시간 약 {person_time:.0f}초\n")
    print(f"{'속도상한':>8} {'체류(초)':>8} {'후보':>6} {'인원':>5} {'총체류(초)':>10} {'체류/person-time':>16}")
    print("-" * 62)

    for speed in (0.35, 0.20, 0.10, 0.05, 0.03):
        for dwell in (1.0, 2.0, 3.0):
            got = replay(observations, zone_map, args.fps, max_speed=speed, dwell_seconds=dwell)
            total = sum(c.duration_sec for c in got)
            ratio = total / person_time * 100 if person_time else 0
            print(
                f"{speed:>8.2f} {dwell:>8.1f} {len(got):>6} "
                f"{len({c.person_id for c in got}):>5} {total:>10.1f} {ratio:>15.0f}%"
            )

    print()
    print("체류/person-time 이 100% 를 넘을 수 있는 이유: 한 사람이 두 선반 구역에")
    print("동시에 걸쳐 있으면 각각 후보가 생기기 때문이다.")


if __name__ == "__main__":
    main()
