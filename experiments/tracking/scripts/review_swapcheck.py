"""두 손님이 겹친 순간 전후로 신원이 뒤바뀌었는지 — 전체 카메라 겉모습(몸통 색 분포)으로 점검.

오판 장면 보기 데이터(`outputs/review/data/<설정>/<쌍>.js`, review_export.py 가 만듦)와 원본 영상을 읽는다.
정답 2명 이상인 쌍에서, 매장 안 손님 신원 두 개의 상자가 IoU 0.15 이상이거나 작은 쪽의 50% 이상 덮이면 '겹침'.
겹침 시작 1.5~0.3초 전, 끝 0.3~1.5초 뒤 각 신원 상자 몸통(위 15~65%, 좌우 20% 안쪽)의 HSV 색 분포 평균을 내어
'그대로' 유사도(전A·후A + 전B·후B)와 '바꿈' 유사도(전A·후B + 전B·후A)를 비교한다. 바꿈이 0.05 넘게 크면 의심.

색 분포는 가림·창가 역광에도 흔들리므로 '의심'은 영상으로 직접 확인할 후보일 뿐이다 (2026-11-05: 의심 7곳 모두 실제로는 그대로).
HEVC 원본은 OpenCV 구간 이동이 어긋나므로 처음부터 차례로 읽는다.

    python scripts/review_swapcheck.py [--run edgegap]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
DATA = Path(__file__).resolve().parents[1] / "outputs" / "review" / "data"


def load(p: Path) -> dict:
    s = p.read_text(encoding="utf-8")
    return json.loads(s[s.index(",") + 1: s.rindex(")")])


def overlap(a, b) -> tuple[float, float]:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0, 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter), inter / min(aa, bb)


def torso_hist(img, box):
    x1, y1, x2, y2 = (int(v) for v in box)
    h, w = y2 - y1, x2 - x1
    crop = img[max(0, y1 + int(.15 * h)): y1 + int(.65 * h), max(0, x1 + int(.2 * w)): x2 - int(.2 * w)]
    if crop.size < 300:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 6, 6], [0, 180, 0, 256, 0, 256]).flatten()
    return hist / (hist.sum() + 1e-9)


def sim(a, b) -> float:
    return float(np.minimum(a, b).sum())


def check_pair(d: dict) -> list[tuple]:
    cust = {int(k) for k, v in d["customers"].items() if v["inside"]}
    cam = d["tracks"]["angle1"]
    fps, stride = cam["fps"], cam["stride"]
    frames = {int(k): v for k, v in cam["frames"].items()}
    keys = sorted(frames)
    per: dict[int, dict[int, tuple]] = {}
    for f in keys:
        for r in frames[f]:
            if r[1] in cust:
                per.setdefault(r[1], {})[f] = (r[3:7], r[2])
    ids = sorted(per)
    events = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            run: list[int] = []
            for f in sorted(set(per[a]) & set(per[b])):
                iou, cover = overlap(per[a][f][0], per[b][f][0])
                if iou >= .15 or cover >= .5:
                    if run and f - run[-1] > stride * 3:
                        events.append((a, b, run[0], run[-1]))
                        run = []
                    run.append(f)
            if run:
                events.append((a, b, run[0], run[-1]))
    if not events:
        return []
    need: dict[int, list] = {}
    for ev in events:
        lo0, lo1 = ev[2] - int(1.5 * fps), ev[2] - int(.3 * fps)
        hi0, hi1 = ev[3] + int(.3 * fps), ev[3] + int(1.5 * fps)
        for f in keys:
            if lo0 <= f <= lo1 or hi0 <= f <= hi1:
                need.setdefault(f, []).append(ev)
    hists: dict[tuple, list] = {}
    cap = cv2.VideoCapture(str(REPO / d["video"]["angle1"].lstrip("/")))
    f, last = 0, max(need) if need else -1
    while f <= last and cap.grab():
        if f in need:
            _, img = cap.retrieve()
            for ev in need[f]:
                phase = "pre" if f < ev[2] else "post"
                for pid in ev[:2]:
                    if f in per[pid]:
                        h = torso_hist(img, per[pid][f][0])
                        if h is not None:
                            hists.setdefault((ev, pid, phase), []).append(h)
        f += 1
    cap.release()
    out = []
    for ev in events:
        a, b = ev[0], ev[1]
        mean = {k: np.mean(hists[(ev, *k)], 0) for k in ((a, "pre"), (b, "pre"), (a, "post"), (b, "post")) if hists.get((ev, *k))}
        if len(mean) < 4:
            out.append((a, b, ev[2] / fps, ev[3] / fps, None, None))
            continue
        keep = sim(mean[(a, "pre")], mean[(a, "post")]) + sim(mean[(b, "pre")], mean[(b, "post")])
        swap = sim(mean[(a, "pre")], mean[(b, "post")]) + sim(mean[(b, "pre")], mean[(a, "post")])
        out.append((a, b, ev[2] / fps, ev[3] / fps, keep, swap))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="edgegap")
    args = ap.parse_args()
    total = suspects = 0
    for p in sorted((DATA / args.run).glob("*.js")):
        d = load(p)
        if d["truth"]["people"] < 2:
            continue
        rows = check_pair(d)
        print(f"{p.stem:5} 정답 {d['truth']['people']}명 — 겹침 {len(rows)}곳")
        for a, b, t0, t1, keep, swap in rows:
            if keep is None:
                print(f"    손님{a}·{b} {t0:5.1f}~{t1:5.1f}s  전후 모습 부족")
                continue
            total += 1
            flag = swap > keep + 0.05
            suspects += flag
            print(f"    손님{a}·{b} {t0:5.1f}~{t1:5.1f}s  그대로 {keep:.2f} 바꿈 {swap:.2f}{'  ← 뒤바뀜 의심 (영상 확인)' if flag else ''}")
    print(f"\n전후 비교한 겹침 {total}곳, 뒤바뀜 의심 {suspects}곳")


if __name__ == "__main__":
    main()
