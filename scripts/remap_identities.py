"""두 실행 사이의 person_id 대응표를 만든다.

왜 필요한가: 라벨은 **특정 실행의 person_id** 로 찍힌다.
detector 를 바꾸거나 설정을 고치면 번호가 달라져 라벨이 그대로는 쓸 수 없다.
실제로 yolo -> yolo-pose 로 바꾸자 P4·P5 가 P5·P6 으로 밀렸다.

대응은 **같은 프레임에서 bbox 가 겹치는 정도**로 찾는다.
번호가 바뀌어도 같은 사람은 같은 시간·같은 자리에 있기 때문이다.

    python scripts/remap_identities.py outputs/take-base/identities.jsonl \
        outputs/pose-base/identities.jsonl
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sinks.observation_log import read_identities  # noqa: E402
from src.zones.geometry import overlap_ratio  # noqa: E402


def load(path: str) -> dict[int, dict[int, tuple]]:
    """person_id -> {frame: bbox}"""
    out: dict[int, dict[int, tuple]] = defaultdict(dict)
    for o in read_identities(path):
        if o.person_id > 0:
            out[o.person_id][o.frame] = o.bbox
    return out


def build_mapping(reference: str, target: str, tolerance: int = 2) -> dict[int, int]:
    """target 의 person_id -> reference 의 person_id"""
    ref = load(reference)
    tgt = load(target)

    scores: dict[tuple[int, int], float] = defaultdict(float)
    for t_pid, t_frames in tgt.items():
        for frame, t_box in t_frames.items():
            for r_pid, r_frames in ref.items():
                r_box = None
                for offset in range(tolerance + 1):
                    r_box = r_frames.get(frame - offset) or r_frames.get(frame + offset)
                    if r_box:
                        break
                if not r_box:
                    continue
                poly = (
                    (r_box[0], r_box[1]), (r_box[2], r_box[1]),
                    (r_box[2], r_box[3]), (r_box[0], r_box[3]),
                )
                scores[(t_pid, r_pid)] += overlap_ratio(poly, t_box)

    mapping: dict[int, int] = {}
    used: set[int] = set()
    for (t_pid, r_pid), score in sorted(scores.items(), key=lambda kv: -kv[1]):
        if t_pid in mapping or r_pid in used:
            continue
        if score < 1.0:  # 사실상 안 겹침
            continue
        mapping[t_pid] = r_pid
        used.add(r_pid)
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", help="라벨이 찍힌 기준 identities.jsonl")
    parser.add_argument("target", help="새로 돌린 identities.jsonl")
    args = parser.parse_args()

    mapping = build_mapping(args.reference, args.target)
    tgt = load(args.target)
    ref = load(args.reference)

    print(f"{'새 실행':>10} -> {'기준(라벨)':>10}   {'새 구간':>16} {'기준 구간':>16}")
    print("-" * 60)
    for t_pid in sorted(tgt):
        r_pid = mapping.get(t_pid)
        t_span = f"{min(tgt[t_pid])}-{max(tgt[t_pid])}"
        if r_pid is None:
            print(f"{'P'+str(t_pid):>10} -> {'(대응 없음)':>10}   {t_span:>16}")
        else:
            r_span = f"{min(ref[r_pid])}-{max(ref[r_pid])}"
            print(f"{'P'+str(t_pid):>10} -> {'P'+str(r_pid):>10}   {t_span:>16} {r_span:>16}")
    missing = set(ref) - set(mapping.values())
    if missing:
        print(f"\n기준에만 있고 새 실행에 없는 person: {sorted('P'+str(m) for m in missing)}")


if __name__ == "__main__":
    main()
