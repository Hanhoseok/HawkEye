"""ID 가 다른 사람에게 옮겨간 지점(identity handoff)을 찾는다.

track 수만 세면 ID switch 를 놓친다. ID 가 A 에서 B 로 옮겨가고 A 가 새 ID 를 받으면
총 개수는 그대로이기 때문이다. 실제 신호는 '속도'가 아니라 다음 조합이다.

    어떤 track 이 원래 자리에서 멀리 튀어나가고,
    그 빈 자리에 새 track 이 태어난다.

    python scripts/detect_id_jumps.py outputs/*/observations.jsonl
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sinks.observation_log import read_observations


def center_x(bbox) -> float:
    return (bbox[0] + bbox[2]) / 2


def analyze(path: str, jump_ratio: float, near_ratio: float, slack: int) -> int:
    tracks = defaultdict(list)
    for o in read_observations(path):
        tracks[o.track_id].append(o)
    for obs in tracks.values():
        obs.sort(key=lambda o: o.frame)

    births = {tid: obs[0] for tid, obs in tracks.items()}
    findings = []

    for tid, obs in tracks.items():
        for a, b in zip(obs, obs[1:]):
            width = max(1.0, a.bbox[2] - a.bbox[0])
            moved = abs(center_x(b.bbox) - center_x(a.bbox))
            if moved < jump_ratio * width:
                continue  # 자연스러운 이동
            # 튀어나간 자리에 새로 태어난 track 이 있는가
            for other, birth in births.items():
                if other == tid:
                    continue
                if not (a.frame - slack <= birth.frame <= b.frame + slack):
                    continue
                if abs(center_x(birth.bbox) - center_x(a.bbox)) <= near_ratio * width:
                    findings.append((tid, a.frame, b.frame, moved, other, birth.frame))
                    break

    name = Path(path).parent.name
    if not findings:
        print(f"  {name:26s} ID 이동 정황 없음 ✅")
        return 0
    print(f"  {name:26s} ID 이동 의심 {len(findings)}건 ❌")
    for tid, f1, f2, moved, other, bf in findings:
        print(
            f"      track {tid} 이(가) frame {f1}->{f2} 사이에 {moved:.0f}px 튀어나가고,"
            f" 그 자리에 track {other} 이(가) frame {bf} 에 새로 생김"
        )
    return len(findings)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--jump-ratio", type=float, default=0.7, help="bbox 폭 대비 이 배수 이상 이동하면 점프로 본다")
    parser.add_argument("--near-ratio", type=float, default=1.0, help="원래 자리로부터 이 배수 안에서 태어나면 '그 자리'로 본다")
    parser.add_argument("--slack", type=int, default=10, help="탄생 시점 허용 오차(프레임)")
    args = parser.parse_args()
    print(f"판정: 폭의 {args.jump_ratio}배 이상 튀어나간 track + 그 자리에 {args.slack}프레임 내 새 track 탄생\n")
    for p in args.paths:
        analyze(p, args.jump_ratio, args.near_ratio, args.slack)


if __name__ == "__main__":
    main()
