"""TAKE 후보를 정답 라벨과 대조해 정밀도/재현율을 계산한다.

이것이 라벨링의 목적이다. 지금까지는 "후보가 줄었다"까지만 말할 수 있었고
"정확해졌다"는 말할 수 없었다. 이 스크립트가 그 벽을 넘게 한다.

    # 저장된 후보를 평가
    python scripts/eval_takes.py labels/store-aisle.csv --takes outputs/take-base/take_candidates.jsonl

    # 임계값을 스윕하면서 평가 (정답 기준으로 최적점 찾기)
    python scripts/eval_takes.py labels/store-aisle.csv \
        --identities outputs/take-base/identities.jsonl --fps 59.9 --sweep

판정 규칙
    TP  후보가 같은 person 의 TAKE 라벨과 프레임 구간이 겹친다 (라벨 하나당 후보 하나만 인정)
    FP  겹치는 TAKE 라벨이 없다
        그중 BROWSE 라벨과 겹치는 것은 '구경을 집기로 오인' 으로 따로 센다
    FN  후보가 붙지 않은 TAKE 라벨
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sinks.observation_log import read_identities  # noqa: E402
from src.zones.zone_map import ZoneMap  # noqa: E402


@dataclass
class Label:
    person_id: int
    action: str
    start_frame: int
    end_frame: int


def load_labels(path: str) -> list[Label]:
    out = []
    with Path(path).open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out.append(
                Label(
                    int(row["person_id"]),
                    row["action"].strip().upper(),
                    int(row["start_frame"]),
                    int(row["end_frame"]),
                )
            )
    return out


def overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start <= b_end and b_start <= a_end


def evaluate(candidates, labels: list[Label]) -> dict:
    takes = [l for l in labels if l.action == "TAKE"]
    browses = [l for l in labels if l.action == "BROWSE"]

    matched_labels: set[int] = set()
    tp, fp_browse, fp_other = 0, 0, 0

    for cand in sorted(candidates, key=lambda c: c.start_frame):
        hit = None
        for i, label in enumerate(takes):
            if i in matched_labels:
                continue
            if label.person_id == cand.person_id and overlaps(
                cand.start_frame, cand.end_frame, label.start_frame, label.end_frame
            ):
                hit = i
                break
        if hit is not None:
            matched_labels.add(hit)
            tp += 1
        elif any(
            b.person_id == cand.person_id
            and overlaps(cand.start_frame, cand.end_frame, b.start_frame, b.end_frame)
            for b in browses
        ):
            fp_browse += 1
        else:
            fp_other += 1

    fp = fp_browse + fp_other
    fn = len(takes) - len(matched_labels)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "fp_browse": fp_browse, "fp_other": fp_other,
        "precision": precision, "recall": recall, "f1": f1,
        "labels_take": len(takes), "labels_browse": len(browses),
    }


def print_result(r: dict, prefix: str = "") -> None:
    print(f"{prefix}TP {r['tp']}  FP {r['fp']} (구경오인 {r['fp_browse']}, 기타 {r['fp_other']})  FN {r['fn']}")
    print(f"{prefix}정밀도 {r['precision']:.2f}  재현율 {r['recall']:.2f}  F1 {r['f1']:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels")
    parser.add_argument("--takes", help="평가할 take_candidates.jsonl")
    parser.add_argument("--identities", help="스윕할 때 재생할 identities.jsonl")
    parser.add_argument("--zones", default="zones.yaml")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=404)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    labels = load_labels(args.labels)
    takes = [l for l in labels if l.action == "TAKE"]
    browses = [l for l in labels if l.action == "BROWSE"]
    print(f"정답 라벨: TAKE {len(takes)}건, BROWSE {len(browses)}건, "
          f"전체 {len(labels)}건 / {len({l.person_id for l in labels})}명\n")
    if not takes:
        raise SystemExit("TAKE 라벨이 없습니다. 최소 몇 건은 있어야 평가할 수 있습니다.")

    if args.takes:
        from src.core.types import TakeCandidate
        import json

        candidates = []
        with Path(args.takes).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    candidates.append(TakeCandidate(**json.loads(line)))
        print(f"후보 {len(candidates)}건 평가")
        print_result(evaluate(candidates, labels), "  ")
        return

    if not args.identities:
        raise SystemExit("--takes 또는 --identities 중 하나가 필요합니다")

    from take_sweep import replay  # 같은 폴더

    zone_map = ZoneMap.load(args.zones)
    zone_map.resolve(args.width, args.height)
    observations = [o for o in read_identities(args.identities) if o.person_id > 0]

    print(f"{'속도':>6} {'체류':>5} {'후보':>5} {'TP':>4} {'FP':>4} {'FN':>4} "
          f"{'정밀도':>7} {'재현율':>7} {'F1':>6}")
    print("-" * 58)
    best = None
    for speed in (0.30, 0.20, 0.10, 0.05, 0.03):
        for dwell in (1.0, 2.0, 3.0):
            cands = replay(observations, zone_map, args.fps, max_speed=speed, dwell_seconds=dwell)
            r = evaluate(cands, labels)
            print(f"{speed:>6.2f} {dwell:>5.1f} {len(cands):>5} {r['tp']:>4} {r['fp']:>4} "
                  f"{r['fn']:>4} {r['precision']:>7.2f} {r['recall']:>7.2f} {r['f1']:>6.2f}")
            if best is None or r["f1"] > best[0]["f1"]:
                best = (r, speed, dwell)
    if best:
        r, speed, dwell = best
        print(f"\n최고 F1: max_speed={speed}, dwell_seconds={dwell}")
        print_result(r, "  ")
        print("\n주의: 같은 데이터로 고른 값이다. 다른 영상에서도 되는지는 별도 확인이 필요하다.")


if __name__ == "__main__":
    main()
