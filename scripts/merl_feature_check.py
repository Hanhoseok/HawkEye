"""MERL 라벨 기준으로 pose 특징의 판별력을 측정한다.

우리 결론의 상당 부분이 **영상 하나(65초, 6명)** 에 얹혀 있었다.
MERL 은 41명 / 106영상 / 5,377건이므로, 같은 차이가 재현되는지 확인할 수 있다.

특히 MERL 은 **거의 수직 천장 시점**이다. 우리 영상(완만한 각도)에서 찾은
"손 높이" 신호가 여기서도 유효한지는 전혀 다른 질문이다 —
위에서 내려다보면 손을 들어도 이미지상 y 좌표가 크게 변하지 않기 때문이다.

    python scripts/merl_feature_check.py --videos 5 --stride 5
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.merl_to_labels import load_tlabs  # noqa: E402
from src.core.types import (  # noqa: E402
    KP_LEFT_HIP,
    KP_LEFT_SHOULDER,
    KP_LEFT_WRIST,
    KP_RIGHT_HIP,
    KP_RIGHT_SHOULDER,
    KP_RIGHT_WRIST,
)

VIDEO_DIR = Path("data/merl/Videos_MERL_Shopping_Dataset")
LABEL_DIR = Path("data/merl/Labels_MERL_Shopping_Dataset")

REACH, RETRACT, HAND_IN, INSPECT_PRODUCT, INSPECT_SHELF = 0, 1, 2, 3, 4
CLASS_KO = {
    REACH: "선반으로 손 뻗기",
    RETRACT: "선반에서 손 빼기",
    HAND_IN: "손이 선반 안에",
    INSPECT_PRODUCT: "물건 들고 봄",
    INSPECT_SHELF: "선반 보기만",
}


def frame_labels(tlabs, n_frames: int) -> dict[int, set[int]]:
    """프레임 번호 -> 그 프레임에 걸린 행동 클래스들."""
    out: dict[int, set[int]] = defaultdict(set)
    for cls, spans in enumerate(tlabs):
        for s, e in spans:
            for f in range(max(1, s), min(n_frames, e) + 1):
                out[f].add(cls)
    return out


ZONE_MAP = None


def features(kp, bbox, img_h: float = 680.0, img_w: float = 920.0) -> dict[str, float]:
    """우리가 store-aisle 에서 쓰던 것과 같은 정의로 계산한다."""
    height = bbox[3] - bbox[1]
    out: dict[str, float] = {}
    if height <= 0:
        return out

    heights, extensions, laterals = [], [], []
    cx = (bbox[0] + bbox[2]) / 2
    for wrist, shoulder, hip in (
        (KP_LEFT_WRIST, KP_LEFT_SHOULDER, KP_LEFT_HIP),
        (KP_RIGHT_WRIST, KP_RIGHT_SHOULDER, KP_RIGHT_HIP),
    ):
        wx, wy, wc = kp[wrist]
        sx, sy, sc = kp[shoulder]
        hx, hy, hc = kp[hip]
        if wc < 0.5 or (wx == 0 and wy == 0):
            continue
        laterals.append(abs(wx - cx) / height)
        if hc >= 0.5 and not (hx == 0 and hy == 0):
            heights.append((hy - wy) / height)
        if sc >= 0.5 and not (sx == 0 and sy == 0):
            extensions.append(((wx - sx) ** 2 + (wy - sy) ** 2) ** 0.5 / height)
    if heights:
        out["손높이"] = max(heights)
    if extensions:
        out["팔뻗음"] = max(extensions)
    if laterals:
        out["좌우벗어남"] = max(laterals)

    # --- 수직 시점용 후보 ---
    # 천장에서 내려다보면 이미지가 거의 평면도가 된다.
    # 선반은 화면 위쪽에 있으므로, 선반으로 손을 뻗으면 손목의 이미지 y 가 작아진다.
    wys = [kp[i][1] for i in (KP_LEFT_WRIST, KP_RIGHT_WRIST)
           if kp[i][2] >= 0.5 and not (kp[i][0] == 0 and kp[i][1] == 0)]
    if wys:
        out["손목 화면y"] = min(wys) / img_h              # 작을수록 선반(위)에 가까움
        out["손목-몸 위쪽거리"] = (min(wys) - bbox[1]) / height  # 몸 상단 기준
    # 손목 -> 선반 구역 최단 거리 (체구 정규화). 구역만 정의되면 카메라가 달라도 같은 의미.
    if ZONE_MAP is not None:
        from src.interaction.pose_features import wrist_zone_distance

        kp_t = tuple(tuple(float(v) for v in row) for row in kp)
        d = wrist_zone_distance(ZONE_MAP, kp_t, 0.5, height)
        if d is not None:
            out["손목-선반거리"] = d
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=int, default=5, help="처리할 영상 수 (train set 앞에서부터)")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    from ultralytics import YOLO

    from src.zones.zone_map import ZoneMap

    global ZONE_MAP
    ZONE_MAP = ZoneMap.load("zones_merl.yaml")
    ZONE_MAP.resolve(920, 680)

    model = YOLO("yolov8n-pose.pt")

    # train set = subject 1~20 (평가용 test set 은 건드리지 않는다)
    videos = sorted(VIDEO_DIR.glob("*_crop.mp4"))
    videos = [v for v in videos if int(v.stem.split("_")[0]) <= 20][: args.videos]

    buckets: dict[int, list[dict]] = defaultdict(list)
    no_label = []
    processed = 0

    for video in videos:
        stem = video.stem.replace("_crop", "")
        label_file = LABEL_DIR / f"{stem}_label.mat"
        if not label_file.exists():
            continue
        tlabs = load_tlabs(label_file)
        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        per_frame = frame_labels(tlabs, total)

        idx = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            idx += 1
            if idx % args.stride:
                continue
            result = model.predict(
                source=img, conf=0.25, imgsz=args.imgsz, device="cpu", verbose=False
            )[0]
            boxes = result.boxes
            if boxes is None or len(boxes) == 0 or result.keypoints is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            kps = result.keypoints.data.cpu().numpy()
            # 쇼핑객 = 가장 큰 사람. MERL 영상에는 보통 한 명이지만 간혹 다른 사람이 걸린다.
            best = max(range(len(xyxy)), key=lambda i: (xyxy[i][2] - xyxy[i][0]) * (xyxy[i][3] - xyxy[i][1]))
            feats = features(kps[best], xyxy[best], img.shape[0], img.shape[1])
            if not feats:
                continue
            processed += 1
            classes = per_frame.get(idx)
            if not classes:
                no_label.append(feats)
            for cls in classes or ():
                buckets[cls].append(feats)
        cap.release()
        print(f"  처리: {video.name}")

    print(f"\n영상 {len(videos)}개, 표본 {processed:,}개 (stride {args.stride})\n")
    for name in ("손목-선반거리", "손높이", "손목 화면y"):
        print(f"[{name}]")
        rows = []
        for cls in (REACH, HAND_IN, RETRACT, INSPECT_PRODUCT, INSPECT_SHELF):
            values = sorted(f[name] for f in buckets[cls] if name in f)
            if not values:
                continue
            rows.append((CLASS_KO[cls], len(values), st.median(values),
                         values[len(values) // 4], values[3 * len(values) // 4]))
        others = sorted(f[name] for f in no_label if name in f)
        if others:
            rows.append(("(라벨 없음)", len(others), st.median(others),
                         others[len(others) // 4], others[3 * len(others) // 4]))
        for label, n, med, q1, q3 in rows:
            print(f"   {label:<16} n={n:>6,}  중앙값 {med:>6.3f}   25%={q1:>6.3f}  75%={q3:>6.3f}")
        reach = [f[name] for f in buckets[REACH] if name in f]
        shelf = [f[name] for f in buckets[INSPECT_SHELF] if name in f]
        if reach and shelf:
            ratio = st.median(reach) / st.median(shelf) if st.median(shelf) else float("inf")
            print(f"   -> 손뻗기 / 선반보기 중앙값 비율 = {ratio:.2f}배")
        print()


if __name__ == "__main__":
    main()
