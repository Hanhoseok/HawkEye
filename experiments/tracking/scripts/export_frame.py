"""구역 좌표를 손으로 정하기 위해, 격자를 입힌 프레임을 저장한다.

    python scripts/export_frame.py data/videos/store-aisle-detection.mp4 --frame 2000

0~1 상대 좌표도 함께 표시하므로 zones.yaml 에 normalized 로 적기 쉽다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--step", type=int, default=10, help="격자 간격(%)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {args.source}")
    if args.frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"frame {args.frame} 을 읽을 수 없습니다")

    h, w = img.shape[:2]
    canvas = img.copy()
    for pct in range(0, 101, args.step):
        x = int(w * pct / 100)
        y = int(h * pct / 100)
        strong = pct % 50 == 0
        color = (0, 255, 255) if strong else (90, 90, 90)
        cv2.line(canvas, (x, 0), (x, h), color, 2 if strong else 1)
        cv2.line(canvas, (0, y), (w, y), color, 2 if strong else 1)
        if pct % 20 == 0:
            cv2.putText(canvas, f"{pct/100:.1f}|{x}", (x + 3, 14), FONT, 0.35, (0, 255, 255), 1)
            cv2.putText(canvas, f"{pct/100:.1f}|{y}", (3, y - 4), FONT, 0.35, (0, 255, 255), 1)

    out = Path(args.out or f"outputs/frames/{Path(args.source).stem}_f{args.frame}_grid.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    print(f"저장: {out}  (프레임 {w}x{h})")
    print("격자 라벨은 '상대좌표|픽셀' 형식이다. zones.yaml 에는 상대좌표 사용을 권장한다.")


if __name__ == "__main__":
    main()
