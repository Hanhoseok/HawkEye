"""observations.jsonl 을 track 단위로 요약한다.

Phase 2·3 의 확인 항목("같은 사람이 이동해도 동일 ID 유지", "ID switch 기록")을
영상을 일일이 돌려보지 않고도 숫자로 먼저 좁히기 위한 도구.

    python scripts/analyze_observations.py outputs/store-aisle/observations.jsonl
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sinks.observation_log import read_observations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="observations.jsonl 경로")
    parser.add_argument("--min-frames", type=int, default=1, help="이 길이 미만 track 은 '깜빡임'으로 따로 센다")
    args = parser.parse_args()

    frames_by_track: dict[int, list[int]] = defaultdict(list)
    for obs in read_observations(args.path):
        frames_by_track[obs.track_id].append(obs.frame)

    if not frames_by_track:
        print("observation 이 없습니다.")
        return

    all_frames = sorted({f for fs in frames_by_track.values() for f in fs})
    print(f"프레임 범위 : {all_frames[0]} ~ {all_frames[-1]} ({len(all_frames)} frames)")
    print(f"고유 track  : {len(frames_by_track)}")
    print()
    print(f"{'track_id':>8} {'등장':>6} {'소멸':>6} {'길이':>6} {'끊김':>6}  비고")
    print("-" * 60)

    flicker = 0
    for track_id in sorted(frames_by_track):
        fs = sorted(frames_by_track[track_id])
        # 연속이어야 할 프레임 사이에 빈 구간이 몇 번 있었는지 = 재매칭(가림 복귀) 횟수
        step = min((b - a) for a, b in zip(fs, fs[1:])) if len(fs) > 1 else 1
        gaps = sum(1 for a, b in zip(fs, fs[1:]) if b - a > step)
        note = ""
        if len(fs) < args.min_frames:
            note = "깜빡임 의심"
            flicker += 1
        if gaps:
            note = (note + " / " if note else "") + f"가림 후 복귀 {gaps}회"
        print(f"{track_id:>8} {fs[0]:>6} {fs[-1]:>6} {len(fs):>6} {gaps:>6}  {note}")

    print("-" * 60)
    print(f"깜빡임({args.min_frames} frame 미만) track: {flicker}개")
    print()
    print("해석 요령:")
    print("  · 실제 등장 인원보다 track 수가 많으면 ID switch 를 의심한다.")
    print("  · '길이'가 매우 짧은 track 이 많으면 minimum_consecutive_frames 를 올려본다.")
    print("  · '끊김'이 잦으면 lost_track_buffer 를 늘려본다.")


if __name__ == "__main__":
    main()
