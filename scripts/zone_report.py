"""신원별로 어느 구역에 얼마나 머물렀는지 집계한다.

구역 좌표가 제대로 잡혔는지 검증하는 도구이자, TAKE 판정의 전 단계다.
(TAKE 는 "선반 구역에 일정 시간 이상 머물렀는가"에서 출발한다)

    python scripts/zone_report.py outputs/full-botsort/identities.jsonl --min-overlap 0.15
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.types import ZoneType  # noqa: E402
from src.sinks.observation_log import read_identities  # noqa: E402
from src.zones.zone_map import ZoneMap  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("identities", help="identities.jsonl 경로")
    parser.add_argument("--zones", default="zones.yaml")
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=404)
    parser.add_argument(
        "--min-overlap", type=float, default=0.15, help="SHELF 는 이 비율 이상 겹쳐야 '접촉'으로 센다"
    )
    args = parser.parse_args()

    zone_map = ZoneMap.load(args.zones)
    zone_map.resolve(args.width, args.height)
    print(f"구역: {zone_map.summary()}\n")

    frames: dict[int, set[int]] = defaultdict(set)
    touch: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for obs in read_identities(args.identities):
        if obs.person_id <= 0:
            continue
        frames[obs.person_id].add(obs.frame)
        for hit in zone_map.evaluate(obs.bbox):
            if hit.zone_type in (ZoneType.EXIT, ZoneType.CHECKOUT, ZoneType.ENTRANCE):
                if hit.contains_foot:
                    touch[obs.person_id][hit.zone] += 1
            elif hit.overlap >= args.min_overlap:
                touch[obs.person_id][hit.zone] += 1

    names = [z.name for z in zone_map.zones]
    header = f"{'person':>7} {'관측':>6} " + " ".join(f"{n[:13]:>14}" for n in names)
    print(header)
    print("-" * len(header))
    for pid in sorted(frames):
        total = len(frames[pid])
        cells = []
        for n in names:
            c = touch[pid].get(n, 0)
            cells.append(f"{c:>5} ({c/total*100:>3.0f}%)" if c else f"{'-':>14}")
        print(f"{pid:>7} {total:>6} " + " ".join(cells))
    print()
    print("SHELF 는 겹침 비율 기준, EXIT/CHECKOUT 은 발 위치 기준으로 셌다.")
    print("괄호 안은 그 사람의 전체 관측 중 비율이다.")


if __name__ == "__main__":
    main()
