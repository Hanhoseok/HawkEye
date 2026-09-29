"""Phase 0(영상 I/O)만 검증하기 위한 합성 영상 생성기.

주의: 여기서 만든 영상에는 실제 사람이 없으므로 YOLO 는 아무것도 탐지하지 않는다.
Phase 1 이후에는 반드시 실제 매장/사람이 찍힌 mp4 를 data/videos/ 에 넣어야 한다.

    python scripts/make_sample_video.py --out data/videos/synthetic.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/videos/synthetic.mp4")
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height)
    )

    for i in range(args.frames):
        canvas = np.full((args.height, args.width, 3), 30, dtype=np.uint8)
        # 서로 교차하는 두 사각형: 프레임 번호/시간이 제대로 흐르는지 눈으로 확인용
        x1 = int((i / args.frames) * (args.width - 60))
        x2 = int((1 - i / args.frames) * (args.width - 60))
        cv2.rectangle(canvas, (x1, 120), (x1 + 60, 260), (80, 180, 255), -1)
        cv2.rectangle(canvas, (x2, 140), (x2 + 60, 280), (180, 255, 120), -1)
        cv2.putText(
            canvas, f"frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        writer.write(canvas)

    writer.release()
    print(f"생성 완료: {out} ({args.frames} frames, {args.fps}fps)")


if __name__ == "__main__":
    main()
