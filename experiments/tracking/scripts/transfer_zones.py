"""구역 옮기기 — 같은 매장을 다시 찍었는데 카메라가 조금 움직였을 때, 구역 파일을 새 영상에 맞춰 옮긴다.

두 영상의 첫 장면(사람이 없는 배경)에서 특징점을 맞춰 카메라가 얼마나 이동·회전했는지 재고,
구역 꼭짓점과 선반 영역을 그만큼 옮긴다. 정답(행동)은 전혀 보지 않는다 — 배경만 쓴다.

    python scripts/transfer_zones.py --ref ../../source/N01.mp4 --target ../../source/T01.mp4 \\
        --zones zones_store01_N.yaml --shelf 0.02,0.70,0.45,0.88 --out zones_store01_T01.yaml

직접 찍은 영상에서 실제로 필요했다: 정상 영상(N)과 절도 영상(T) 사이에 카메라가 91px 옮겨지고 0.8° 돌아가 있었다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml


def first_frame(path: str) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    ok, image = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"영상을 읽을 수 없습니다: {path}")
    return image


def camera_motion(ref: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, int]:
    """ref 픽셀 좌표를 target 픽셀 좌표로 옮기는 2x3 행렬 (이동·회전·배율)과 맞은 점 수."""
    orb = cv2.ORB_create(4000)
    g1, g2 = (cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in (ref, target))
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    matches = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2), key=lambda m: m.distance)[:600]
    p1 = np.float32([k1[m.queryIdx].pt for m in matches])
    p2 = np.float32([k2[m.trainIdx].pt for m in matches])
    matrix, inliers = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=3)
    return matrix, int(inliers.sum())


def move(points_norm, matrix, size) -> list[list[float]]:
    """꼭짓점을 카메라 이동만큼 옮긴다. 단, 화면 테두리(0 또는 1)에 붙은 좌표는 그대로 둔다.

    화면 테두리는 카메라가 움직여도 화면 테두리다. 그대로 회전시키면 '화면 아래 끝까지' 그린 띠가
    끝에서 떨어진다 — 0.8° 회전에서 오른쪽 아래가 y 1.0 -> 0.976 으로 올라가, 발이 화면 끝(0.99)에 걸린
    사람이 출구 띠 밖으로 나갔다(T02 에서 발견).
    """
    w, h = size
    pts = np.array([[x * w, y * h, 1.0] for x, y in points_norm])
    out = pts @ matrix.T
    moved = []
    for (ox, oy), (x, y) in zip(points_norm, out):
        nx = ox if ox in (0.0, 1.0) else float(np.clip(x / w, 0, 1))
        ny = oy if oy in (0.0, 1.0) else float(np.clip(y / h, 0, 1))
        moved.append([round(nx, 4), round(ny, 4)])
    return moved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, help="구역을 그린 영상")
    parser.add_argument("--target", required=True, help="옮길 대상 영상")
    parser.add_argument("--zones", required=True, help="ref 기준 구역 파일 (normalized)")
    parser.add_argument("--shelf", help="ref 기준 선반 영역 x1,y1,x2,y2")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ref, target = first_frame(args.ref), first_frame(args.target)
    matrix, inliers = camera_motion(ref, target)
    angle = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    print(f"카메라 이동: x {matrix[0, 2]:+.0f}px, y {matrix[1, 2]:+.0f}px, 회전 {angle:+.2f}°, 맞은 점 {inliers}")
    if inliers < 50:
        print("[warn] 맞은 점이 적다 — 장면이 많이 달라졌으면 구역을 새로 그려야 한다")

    h, w = target.shape[:2]
    raw = yaml.safe_load(Path(args.zones).read_text(encoding="utf-8"))
    for zone in raw["zones"]:
        zone["polygon"] = move(zone["polygon"], matrix, (w, h))
    header = (f"# {Path(args.zones).name} 를 {Path(args.target).name} 에 맞춰 옮김 (scripts/transfer_zones.py)\n"
              f"# 카메라 이동 x {matrix[0, 2]:+.0f}px y {matrix[1, 2]:+.0f}px 회전 {angle:+.2f}° — 배경 특징점으로 잼, 정답은 보지 않음\n")
    if args.shelf:
        x1, y1, x2, y2 = (float(v) for v in args.shelf.split(","))
        corners = move([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], matrix, (w, h))
        xs, ys = [c[0] for c in corners], [c[1] for c in corners]
        shelf = f"{min(xs):.3f},{min(ys):.3f},{max(xs):.3f},{max(ys):.3f}"
        header += f"# 선반 영역(--shelf): {shelf}\n"
        print(f"선반 영역: {shelf}")
    Path(args.out).write_text(header + yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
