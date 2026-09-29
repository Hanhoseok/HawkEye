"""MERL 로 TAKE 후보 판정을 평가한다. train 에서 고르고 test 에서만 잰다.

지금까지의 모든 수치는 **임계값을 고른 것과 같은 데이터로 평가**한 값이었다.
MERL 은 subject 단위로 train(1-20) / val(21-26) / test(27-41) 가 나뉘어 있어
그 문제 없이 잴 수 있다.

    python scripts/merl_eval.py --train 1_1 2_1 3_1 --test 27_1 28_1 29_1

    # 신원 레지스트리만 바꿔 다시 돌린 결과(scripts/replay_identity.py)로 test 를 평가
    python scripts/merl_eval.py --train 1_1 ... --test 27_1 ... --test-runs outputs/merl_late
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.merl_to_labels import convert  # noqa: E402
from scripts.eval_takes import Label, evaluate, overlaps  # noqa: E402
from src.core.config import InteractionConfig  # noqa: E402
from src.core.types import Frame  # noqa: E402
from src.interaction.detector import InteractionDetector  # noqa: E402
from src.sinks.observation_log import read_identities  # noqa: E402
from src.zones.zone_map import ZoneMap  # noqa: E402

LABEL_DIR = Path("data/merl/Labels_MERL_Shopping_Dataset")
FPS = 30.0


def load_clip(name: str, take_span: str, runs: str = "outputs/merl"):
    """한 영상의 (관측, 라벨) 을 읽는다. person_id 는 관측이 가장 많은 사람으로 맞춘다."""
    identities = Path(runs) / name / "identities.jsonl"
    observations = [o for o in read_identities(identities) if o.person_id > 0]
    if not observations:
        return [], []
    # MERL 영상의 주인공은 쇼핑객 한 명이다. 관측이 가장 많은 사람을 그 사람으로 본다.
    main = Counter(o.person_id for o in observations).most_common(1)[0][0]
    observations = [o for o in observations if o.person_id == main]
    rows = convert(LABEL_DIR / f"{name}_label.mat", main, take_span)
    labels = [Label(r["person_id"], r["action"], r["start_frame"], r["end_frame"]) for r in rows]
    return observations, labels


def run_clip(observations, zone_map, **cfg):
    detector = InteractionDetector(InteractionConfig(**cfg), zone_map)
    by_frame: dict[int, list] = {}
    for o in observations:
        by_frame.setdefault(o.frame, []).append(o)
    dummy = np.zeros((2, 2, 3), dtype=np.uint8)
    for idx in sorted(by_frame):
        detector.update(
            Frame(index=idx, timestamp="", pts_ms=idx * 1000.0 / FPS, image=dummy),
            by_frame[idx],
        )
    detector.flush()
    return detector.completed


def score(clips, zone_map, **cfg) -> dict:
    """여러 영상을 합쳐 채점한다."""
    total = {"tp": 0, "fp": 0, "fn": 0, "fp_browse": 0}
    durations, locs = [], []
    for observations, labels in clips:
        candidates = run_clip(observations, zone_map, **cfg)
        r = evaluate(candidates, labels)
        for k in total:
            total[k] += r[k]
        takes = [l for l in labels if l.action == "TAKE"]
        for c in candidates:
            durations.append(c.duration_sec)
            hit = sum(
                min(c.end_frame, l.end_frame) - max(c.start_frame, l.start_frame) + 1
                for l in takes
                if l.person_id == c.person_id
                and overlaps(c.start_frame, c.end_frame, l.start_frame, l.end_frame)
            )
            locs.append(hit / max(1, c.end_frame - c.start_frame + 1))
    tp, fp, fn = total["tp"], total["fp"], total["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        **total,
        "n": len(durations),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0,
        "duration": st.mean(durations) if durations else 0.0,
        "localization": st.mean(locs) * 100 if locs else 0.0,
    }


def show(tag: str, r: dict) -> None:
    print(
        f"{tag:<34} {r['n']:>4} {r['tp']:>4} {r['fp']:>4} {r['fn']:>4} "
        f"{r['precision']:>7.2f} {r['recall']:>7.2f} {r['f1']:>6.2f} "
        f"{r['duration']:>7.1f}초 {r['localization']:>7.0f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--test", nargs="+", required=True)
    parser.add_argument("--zones", default="zones_merl.yaml")
    parser.add_argument("--take-span", choices=["reach", "interaction"], default="reach")
    parser.add_argument("--runs", default="outputs/merl", help="train 의 identities.jsonl 상위 폴더")
    parser.add_argument("--test-runs", default=None, help="test 쪽 폴더 (기본: --runs 와 같음)")
    args = parser.parse_args()

    zone_map = ZoneMap.load(args.zones)
    zone_map.resolve(920, 680)

    train = [load_clip(n, args.take_span, args.runs) for n in args.train]
    test = [load_clip(n, args.take_span, args.test_runs or args.runs) for n in args.test]
    for tag, clips, names in (("train", train, args.train), ("test", test, args.test)):
        takes = sum(1 for _, ls in clips for l in ls if l.action == "TAKE")
        browses = sum(1 for _, ls in clips for l in ls if l.action == "BROWSE")
        print(f"{tag}: {len(names)}개 영상, TAKE {takes}건 / BROWSE {browses}건")
    print()

    header = (
        f"{'설정':<34} {'후보':>4} {'TP':>4} {'FP':>4} {'FN':>4} "
        f"{'정밀도':>7} {'재현율':>7} {'F1':>6} {'평균길이':>8} {'국소화':>8}"
    )

    print("=== 1단계 · train 에서 임계값 고르기 ===")
    print(header)
    print("-" * 104)
    best = None
    for wz in (0.10, 0.15, 0.20, 0.25, 0.35):
        for dwell in (0.2, 0.4, 0.6):
            cfg = dict(
                signal="reach", wrist_zone_max=wz, dwell_seconds=dwell,
                gap_tolerance_seconds=0.2, min_overlap=0.0, max_speed=99.0,
            )
            r = score(train, zone_map, **cfg)
            show(f"reach 거리<={wz} 지속>={dwell}초", r)
            if best is None or r["f1"] > best[0]["f1"]:
                best = (r, cfg)
    r, cfg = best
    print(f"\ntrain 최고 F1 {r['f1']:.2f} -> 거리<={cfg['wrist_zone_max']}, 지속>={cfg['dwell_seconds']}초")

    print("\n=== 2단계 · test 에서 그 설정으로만 평가 ===")
    print(header)
    print("-" * 104)
    show("(참고) dwell 기준선", score(test, zone_map, signal="dwell", min_overlap=0.05,
                                   max_speed=0.30, dwell_seconds=3.0))
    show("reach (train 에서 고른 설정)", score(test, zone_map, **cfg))
    print("\n두 번째 줄이 처음으로 '튜닝과 평가가 분리된' 수치다.")


if __name__ == "__main__":
    main()
