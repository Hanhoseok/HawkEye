"""TAKE 후보를 정답 라벨과 대조해 정밀도/재현율을 계산한다.

이것이 라벨링의 목적이다. 지금까지는 "후보가 줄었다"까지만 말할 수 있었고
"정확해졌다"는 말할 수 없었다. 이 스크립트가 그 벽을 넘게 한다.

    # 저장된 후보를 평가
    python scripts/eval_takes.py labels/store-aisle.csv --takes outputs/take-base/take_candidates.jsonl

(머묾 기준값을 훑는 --sweep 은 지웠다 — 2026-11-21, 후속 60.)

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
    parser.add_argument("--takes", required=True, help="평가할 take_candidates.jsonl")
    args = parser.parse_args()

    labels = load_labels(args.labels)
    takes = [l for l in labels if l.action == "TAKE"]
    browses = [l for l in labels if l.action == "BROWSE"]
    print(f"정답 라벨: TAKE {len(takes)}건, BROWSE {len(browses)}건, "
          f"전체 {len(labels)}건 / {len({l.person_id for l in labels})}명\n")
    if not takes:
        raise SystemExit("TAKE 라벨이 없습니다. 최소 몇 건은 있어야 평가할 수 있습니다.")

    import json

    from src.core.types import TakeCandidate

    candidates = []
    with Path(args.takes).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                candidates.append(TakeCandidate.from_dict(json.loads(line)))
    print(f"후보 {len(candidates)}건 평가")
    print_result(evaluate(candidates, labels), "  ")


if __name__ == "__main__":
    main()
