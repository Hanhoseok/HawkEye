"""MERL 영상을 묶어서 파이프라인에 통과시킨다.

지금까지의 MERL 결론은 영상 6개(train 3 / test 3)에 얹혀 있었다.
표본을 늘려 같은 결론이 유지되는지 확인하기 위한 배치 처리기다.

이미 처리한 영상은 건너뛰므로 중간에 멈춰도 이어서 돌릴 수 있다.

    python scripts/merl_batch.py --split train --limit 20
    python scripts/merl_batch.py --split test --limit 15
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT / "data/merl/Videos_MERL_Shopping_Dataset"
OUT_DIR = ROOT / "outputs/merl"

# MERL 공식 분할 (ReadMe.md 기준)
SPLITS = {
    "train": range(1, 21),    # subject 1~20
    "val": range(21, 27),     # subject 21~26
    "test": range(27, 42),    # subject 27~41
}


def pick(split: str, limit: int, sessions: int) -> list[str]:
    """subject 당 앞쪽 세션부터 고른다. 사람 다양성을 우선한다."""
    wanted = SPLITS[split]
    chosen: list[str] = []
    for subject in wanted:
        found = 0
        for session in (1, 2, 3):
            name = f"{subject}_{session}"
            if (VIDEO_DIR / f"{name}_crop.mp4").exists():
                chosen.append(name)
                found += 1
                if found >= sessions:
                    break
        if len(chosen) >= limit:
            break
    return chosen[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--sessions", type=int, default=1, help="subject 당 최대 세션 수")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--force", action="store_true", help="이미 처리한 것도 다시")
    args = parser.parse_args()

    names = pick(args.split, args.limit, args.sessions)
    print(f"{args.split}: 영상 {len(names)}개 — {', '.join(names)}\n")

    started = time.perf_counter()
    done = skipped = failed = 0
    for i, name in enumerate(names, 1):
        out = OUT_DIR / name
        if not args.force and (out / "identities.jsonl").exists():
            skipped += 1
            print(f"[{i}/{len(names)}] {name} 건너뜀 (이미 처리됨)")
            continue
        cmd = [
            sys.executable, str(ROOT / "run_tracking.py"),
            "--source", str(VIDEO_DIR / f"{name}_crop.mp4"),
            "--detector", "yolo-pose",
            "--tracker", "bytetrack",   # MERL 은 쇼핑객 1명이라 ReID 불필요. 약 30% 빠르다
            "--zones", str(ROOT / "zones_merl.yaml"),
            "--tracker-fps", "30",
            "--stride", str(args.stride),
            "--no-video", "--no-takes",
            "--out-dir", str(out),
        ]
        t = time.perf_counter()
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if result.returncode != 0:
            failed += 1
            print(f"[{i}/{len(names)}] {name} 실패\n{result.stderr[-400:]}")
            continue
        done += 1
        elapsed = time.perf_counter() - t
        remaining = (len(names) - i) * elapsed
        print(f"[{i}/{len(names)}] {name} 완료 ({elapsed:.0f}초, 남은 예상 {remaining/60:.0f}분)")

    total_min = (time.perf_counter() - started) / 60
    print(f"\n완료 {done} / 건너뜀 {skipped} / 실패 {failed}  (총 {total_min:.1f}분)")


if __name__ == "__main__":
    main()
