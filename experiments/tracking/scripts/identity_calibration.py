"""신원 판정 방식별로 '같은 사람을 이어주는 비율 / 다른 사람을 잘못 합치는 비율'을 잰다.

왜 따로 만들었나: 처음 보정 실험은 영상 전체에서 띄엄띄엄 뽑은 사진 8장을 평균했는데,
실제 레지스트리는 새 track 이 생긴 직후 **연속 8프레임(약 0.5초)** 을 평균한다.
연속 프레임은 자세가 거의 같아 잡음이 상쇄되지 않으므로, 앞의 실험은 지나치게 낙관적이었다
(config 에 반영했더니 오히려 한 사람이 더 많이 쪼개졌다).

이 스크립트는 레지스트리가 실제로 만나는 상황을 그대로 재현한다.

    과거 기록 (새 track 이 생기기 전까지 그 사람에게서 모인 사진)
        vs
    새 track 에서 모은 사진 (방식별로 다름)

같은 사람 사례 = MERL 영상에서 손님이 새 track 으로 다시 나타난 순간 (track 이 새로 생긴 시점).
다른 사람 사례 = 같은 새 track 을 **다른 영상 손님의 과거 기록**과 비교.
MERL 은 영상마다 손님이 한 명이므로, 관측이 많은 신원은 모두 그 손님이다.

    python scripts/identity_calibration.py              # train 1~20
    python scripts/identity_calibration.py --subjects 27 28 29 ...
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import AppConfig  # noqa: E402
from src.identity.embedder import build_embedder  # noqa: E402
from src.sinks.observation_log import read_identities  # noqa: E402

MERL_VIDEOS = Path("data/merl/Videos_MERL_Shopping_Dataset")
CACHE = Path("outputs/identity_calib")
SUBJECT_MIN_OBS = 100  # 이보다 관측이 적은 신원은 잡음(오탐)으로 보고 뺀다


def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) or 1.0)


def load_subject(name: str, embedder) -> dict | None:
    """손님 관측 전체의 (시각, track, 임베딩). 한 번 뽑으면 캐시한다."""
    cache = CACHE / f"{name}.npz"
    if cache.exists():
        d = np.load(cache)
        return {"t": d["t"], "track": d["track"], "emb": d["emb"]}

    ids = [o for o in read_identities(f"outputs/merl/{name}/identities.jsonl") if o.person_id > 0]
    counts = Counter(o.person_id for o in ids)
    subject = {pid for pid, n in counts.items() if n >= SUBJECT_MIN_OBS}
    obs = sorted((o for o in ids if o.person_id in subject), key=lambda o: o.frame)
    if not obs:
        return None
    by_frame = defaultdict(list)
    for o in obs:
        by_frame[o.frame].append(o)

    cap = cv2.VideoCapture(str(MERL_VIDEOS / f"{name}_crop.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    t, track, emb = [], [], []
    index, last = -1, obs[-1].frame
    while index < last:
        ok, image = cap.read()
        if not ok:
            break
        index += 1
        for o in by_frame.get(index, []):
            v = embedder.embed(image, np.array([o.bbox], dtype=np.float32))
            if v is None:
                continue
            t.append(index / fps)
            track.append(o.track_id)
            emb.append(v[0])
    cap.release()
    out = {"t": np.array(t), "track": np.array(track), "emb": np.array(emb, dtype=np.float32)}
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


# ---------------------------------------------------------------- 새 track 쪽 사진 고르기
def spaced(times: np.ndarray, spacing: float) -> list[int]:
    """앞에서부터 spacing 초 간격으로 고른 관측 인덱스 전부."""
    chosen, next_t = [], times[0] if len(times) else 0.0
    for i, t in enumerate(times):
        if t >= next_t:
            chosen.append(i)
            next_t = t + spacing
    return chosen


def pick(times: np.ndarray, count: int, spacing: float) -> list[int] | None:
    """새 track 의 관측 중에서 spacing 초 간격으로 count 장. 모자라면 None."""
    chosen = spaced(times, spacing)[:count]
    return chosen if len(chosen) == count else None


QUERIES = {
    # 이름: (장수, 간격 초)
    "사진 1장 (즉시)": (1, 0.0),
    "연속 8장 (≈0.5초)": (8, 0.0),
    "0.25초 간격 8장 (≈2초)": (8, 0.25),
    "0.5초 간격 6장 (≈3초)": (6, 0.5),
    "0.5초 간격 10장 (≈5초)": (10, 0.5),
    # 레지스트리의 템플릿 갱신 주기(1초)에 맞춘 경우 — '나중에 합치기'가 실제로 쓰는 증거
    "1초 간격 3장 (≈3초)": (3, 1.0),
    "1초 간격 5장 (≈5초)": (5, 1.0),
    "1초 간격 8장 (≈8초)": (8, 1.0),
}


# ---------------------------------------------------------------- 과거 기록 쪽 대표값
def templates(times: np.ndarray, embs: np.ndarray, every: float = 1.0, keep: int = 5) -> np.ndarray:
    """레지스트리처럼 1초마다 한 장씩 보관하되 최근 keep 장만 남긴 것에 가깝게."""
    return embs[spaced(times, every)][-keep:]


def score(query: np.ndarray, hist_t: np.ndarray, hist_e: np.ndarray, how: str) -> float:
    if how == "평균 vs 평균":
        return float(unit(query.mean(0)) @ unit(hist_e.mean(0)))
    if how == "평균 vs 템플릿 최댓값":
        return float(max(unit(query.mean(0)) @ t for t in templates(hist_t, hist_e)))
    if how == "장별 템플릿 최댓값의 평균":
        T = templates(hist_t, hist_e)
        return float(np.mean([max(q @ t for t in T) for q in query]))
    raise ValueError(how)


HOWS = ("평균 vs 평균", "평균 vs 템플릿 최댓값", "장별 템플릿 최댓값의 평균")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 21)))
    parser.add_argument("--min-history", type=float, default=3.0, help="과거 기록이 이보다 짧으면 사례에서 뺀다(초)")
    parser.add_argument("--false-merge", type=float, default=0.05, help="허용할 오병합 비율")
    args = parser.parse_args()

    config = AppConfig.load("config.yaml")
    embedder = build_embedder(config.tracker.reid)
    data = {}
    for s in args.subjects:
        name = f"{s}_1"
        d = load_subject(name, embedder)
        if d is not None and len(d["t"]) > 50:
            data[name] = d
    names = list(data)

    # 같은 사람 사례: 첫 track 이 아닌 track 의 탄생 순간
    events = []  # (영상, 과거 끝 인덱스, 새 track 관측 인덱스들)
    for name, d in data.items():
        seen = set()
        for i, tr in enumerate(d["track"]):
            if tr in seen:
                continue
            seen.add(tr)
            if i == 0 or d["t"][i] - d["t"][0] < args.min_history:
                continue
            members = np.where(d["track"] == tr)[0]
            events.append((name, i, members))
    print(f"손님 {len(names)}명, 다시 나타난 순간(같은 사람 사례) {len(events)}건\n")

    header = f"{'새 track 쪽':<24}{'비교 방식':<26}{'사례':>5}{'기준':>7}{'같은 사람 이어줌':>16}"
    print(f"(다른 사람 오병합을 {args.false_merge:.0%} 로 묶었을 때)\n")
    print(header)
    print("-" * len(header.encode("euc-kr", "ignore")))
    for qname, (count, spacing) in QUERIES.items():
        for how in HOWS:
            if count == 1 and how != "평균 vs 템플릿 최댓값":
                continue  # 한 장이면 세 방식이 사실상 같다
            same, diff = [], []
            for name, end, members in events:
                d = data[name]
                sel = pick(d["t"][members], count, spacing)
                if sel is None:
                    continue  # track 이 너무 짧아 이 방식으로는 판정 못 함
                q = d["emb"][members][sel]
                same.append(score(q, d["t"][:end], d["emb"][:end], how))
                for other in names:
                    if other == name:
                        continue
                    o = data[other]
                    diff.append(score(q, o["t"], o["emb"], how))
            if not same:
                continue
            same, diff = np.array(same), np.array(diff)
            th = float(np.percentile(diff, 100 * (1 - args.false_merge)))
            print(f"{qname:<24}{how:<26}{len(same):>5}{th:>7.3f}{np.mean(same >= th) * 100:>15.1f}%")
        print()


if __name__ == "__main__":
    main()
