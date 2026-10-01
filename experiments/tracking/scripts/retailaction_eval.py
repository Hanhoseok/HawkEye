"""RetailAction 으로 '선반 구역 없이 자세만으로' 집기를 찾을 수 있는지 잰다.

왜: 지금 집기 판정은 "손이 선반 구역에 얼마나 가까운가"로 판단한다(MERL, 천장 시점).
매장마다 선반 구역을 손으로 그려야 하고, 구역이 없으면 판정할 수 없다.
RetailAction(실제 편의점 10곳, 천장 카메라)은 샘플마다 손님 주변만 잘라 둔 영상이라 선반 위치를 모른다.
그래서 이 데이터로는 '구역 없이 자세만으로' 되는지를 본다.

신호: 손(손목·손 중 먼 것)과 목 사이 거리 / 어깨너비.
  위에서 보면 물건을 집으려고 팔을 뻗을 때 손이 몸통에서 멀어진다.
  어깨너비로 나누는 이유: 사람 크기·카메라 거리를 맞추기 위해. 위에서 볼 때 가장 덜 변한다.

자세 좌표는 데이터에 들어 있는 것(정답 대상 손님만)을 쓴다. 좌표 순서는 (행, 열)이다.
기준값은 샘플 번호 짝수에서 정하고, 홀수에서만 채점한다.

    python scripts/retailaction_eval.py
    python scripts/retailaction_eval.py --norm torso      # 몸통 길이로 나누기 비교

데이터: Standard AI RetailAction (Standard.AI Dataset License). 원본은 저장소에 올리지 않는다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DATA = Path("data/retailaction/test")
HANDS = ("left_wrist", "right_wrist", "left_hand", "right_hand")


def point(pose: dict, name: str):
    """(행, 열) -> (x, y)."""
    v = pose.get(name)
    return None if v is None else np.array([v[1], v[0]], dtype=float)


def scale_of(frame: dict | None, norm: str) -> float:
    """한 장면의 몸 크기(정규화 기준). 계산할 수 없으면 nan."""
    if not frame or not frame.get("pose"):
        return np.nan
    pose = frame["pose"]
    neck = point(pose, "neck")
    if norm == "shoulder":
        ls, rs = point(pose, "left_shoulder"), point(pose, "right_shoulder")
        return np.nan if ls is None or rs is None else float(np.linalg.norm(ls - rs))
    waist = point(pose, "middle_of_waist")
    return np.nan if neck is None or waist is None else float(np.linalg.norm(neck - waist))


def reach(frame: dict | None) -> float:
    """손(손목·손 중 먼 것)과 목 사이 거리 (정규화 전). 계산할 수 없으면 nan."""
    if not frame or not frame.get("pose"):
        return np.nan
    pose = frame["pose"]
    neck = point(pose, "neck")
    if neck is None:
        return np.nan
    dists = [np.linalg.norm(p - neck) for p in (point(pose, h) for h in HANDS) if p is not None]
    return max(dists) if dists else np.nan


def extension(frame: dict | None, norm: str) -> float:
    """한 장면의 '팔 뻗음' 값 (그 장면의 몸 크기로 나눔). 계산할 수 없으면 nan."""
    if not frame or not frame.get("pose"):
        return np.nan
    pose = frame["pose"]
    neck = point(pose, "neck")
    if neck is None:
        return np.nan
    if norm == "shoulder":
        ls, rs = point(pose, "left_shoulder"), point(pose, "right_shoulder")
        scale = None if ls is None or rs is None else np.linalg.norm(ls - rs)
    else:
        waist = point(pose, "middle_of_waist")
        scale = None if waist is None else np.linalg.norm(neck - waist)
    if not scale or scale < 1e-3:
        return np.nan
    dists = [np.linalg.norm(p - neck) for p in (point(pose, h) for h in HANDS) if p is not None]
    return max(dists) / scale if dists else np.nan


def clip_signal(meta: dict, norm: str, bins: int = 32, clip_scale: bool = False) -> np.ndarray:
    """두 카메라의 신호를 같은 시간 비율 칸(bins)에 모아, 칸마다 더 큰 값을 쓴다.

    clip_scale: 몸 크기를 장면마다 재지 않고 그 샘플(카메라별) 전체의 중앙값으로 쓴다.
    위에서 볼 때 몸을 옆으로 돌리면 어깨너비가 거의 0 이 되어 값이 10 배 넘게 튀었다.
    """
    out = np.full(bins, np.nan)
    for cam in ("rank0", "rank1"):
        poses = (meta["action_cam"].get(cam) or {}).get("poses")
        if not poses:
            continue
        n = len(poses)
        scales = np.array([scale_of(fr, norm) for fr in poses])
        median = float(np.nanmedian(scales)) if np.any(~np.isnan(scales)) else float("nan")
        for k, fr in enumerate(poses):
            if clip_scale:
                r = reach(fr)
                v = r / median if not np.isnan(median) and median > 1e-3 else np.nan
            else:
                v = extension(fr, norm)
            if np.isnan(v):
                continue
            b = min(bins - 1, int(round(k / max(1, n - 1) * (bins - 1))))
            out[b] = v if np.isnan(out[b]) else max(out[b], v)
    return out


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """양성 점수가 음성보다 클 확률 (0.5 = 못 가름, 1.0 = 완벽)."""
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm", choices=["shoulder", "torso"], default="shoulder")
    parser.add_argument("--tolerance", type=float, default=0.05, help="정답 구간을 앞뒤로 넓힐 비율")
    parser.add_argument("--clip-scale", action="store_true", help="몸 크기를 샘플 전체 중앙값으로")
    parser.add_argument("--sustain", type=int, default=1, help="최고값 대신 연속 N 장면의 최솟값 중 최고")
    parser.add_argument("--report-test", action="store_true",
                        help="채점 쪽(홀수) 결과를 출력한다. 방법을 다 고른 뒤 마지막에 한 번만 쓴다")
    args = parser.parse_args()

    clips = []
    for f in sorted(DATA.glob("*/metadata.json")):
        meta = json.load(open(f, encoding="utf-8"))["content"]
        acts = meta["labels"]["action"]
        sig = clip_signal(meta, args.norm, clip_scale=args.clip_scale)
        if args.sustain > 1:   # 연속 N 칸 중 가장 작은 값 -> 한 장면만 튀는 값을 누른다
            w = args.sustain
            sig = np.array([np.nanmin(sig[i:i + w]) if np.any(~np.isnan(sig[i:i + w])) else np.nan
                            for i in range(len(sig) - w + 1)] + [np.nan] * (w - 1))
        clips.append({
            "id": int(f.parent.name),
            "spans": [(a["start"], a["end"]) for a in acts],
            "labels": [a["label"] for a in acts],
            "sig": sig,
        })
    tune = [c for c in clips if c["id"] % 2 == 0]
    test = [c for c in clips if c["id"] % 2 == 1]
    print(f"샘플 {len(clips)}개 (기준 정하기 {len(tune)} / 채점 {len(test)}), 정규화: {args.norm}, "
          f"샘플 중앙값={args.clip_scale}, 연속={args.sustain}")
    if not args.report_test:
        print("(방법을 고르는 중: 아래 '채점 쪽' 수치도 기준 쪽 데이터로 계산했다)")
        test = tune
    print()

    # ---------------------------------------------------------------- A. 언제 집었나
    bins = 32
    grid = np.linspace(0, 1, bins)

    def inside(f, spans, tol):
        return any(s - tol <= f <= e + tol for s, e in spans)

    hits = base = n = 0
    for c in test:
        if not c["spans"] or np.all(np.isnan(c["sig"])):
            continue
        peak = grid[int(np.nanargmax(c["sig"]))]
        hits += inside(peak, c["spans"], args.tolerance)
        valid = ~np.isnan(c["sig"])
        base += np.mean([inside(g, c["spans"], args.tolerance) for g in grid[valid]])  # 아무 순간이나 고를 때
        n += 1
    print("A. 언제 집었나 — 신호가 가장 큰 순간이 정답 구간(앞뒤 ±{:.0%}) 안에 드는 비율".format(args.tolerance))
    print(f"   신호 최고점      : {hits / n:6.1%}  ({hits}/{n})")
    print(f"   아무 순간이나    : {base / n:6.1%}  (비교 기준)\n")

    # ---------------------------------------------------------------- B. 헛잡기
    def score(c):
        return np.nanmax(c["sig"]) if not np.all(np.isnan(c["sig"])) else np.nan

    def split(cs):
        pos = np.array([score(c) for c in cs if c["spans"]])
        neg = np.array([score(c) for c in cs if not c["spans"]])
        return pos, neg

    tp_pos, tp_neg = split(tune)
    te_pos, te_neg = split(test)
    print("B. 헛잡기 — 행동 있는 샘플과 아무것도 안 한 샘플을 '최대 팔 뻗음'으로 가르기")
    print(f"   AUC (0.5=못 가름, 1.0=완벽): 기준 쪽 {auc(tp_pos, tp_neg):.3f} / 채점 쪽 {auc(te_pos, te_neg):.3f}")
    print(f"   행동 있음 {len(te_pos)}개 / 없음 {len(te_neg)}개 (채점 쪽)")
    for recall in (0.80, 0.90, 0.95):
        th = float(np.nanpercentile(tp_pos, (1 - recall) * 100))   # 기준 쪽에서 정한 문턱
        r = np.nanmean(te_pos >= th)
        fp = np.nanmean(te_neg >= th)
        print(f"   행동의 {recall:.0%} 를 잡는 문턱 {th:.2f} -> 채점 쪽: 행동 {r:.1%} 잡음 / 아무것도 안 한 샘플 {fp:.1%} 를 헛잡음")


if __name__ == "__main__":
    sys.exit(main())
