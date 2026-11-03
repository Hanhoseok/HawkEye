"""바닥 보정 — 높은 카메라 화면의 바닥 점과 실제 위치(cm)를 짝지어 바닥 지도 행렬을 만든다 (design 1-0, M-3).

바닥 타일 모서리처럼 화면에서 또렷하고 실제 위치를 잴 수 있는 점 4개 이상을 고른다.
점이 많고 넓게 퍼질수록 정확하다 (4점이면 오차가 늘 0 으로 나와 확인이 안 되므로 5점 이상 권장).

    # 화면에서 직접 찍기 (클릭 순서 = --floor 순서, 백스페이스 되돌리기, 엔터 끝)
    python scripts/calibrate_floor.py --video "../../source/1(angle1)/N01.mp4" --frame 30 \
        --floor "0,0;60,0;120,0;0,60;60,60;120,60" --out zones/store01_dual/floor.yaml

    # 화면 위치를 이미 알 때 (픽셀, 같은 순서)
    python scripts/calibrate_floor.py --image frame.jpg --pixels "812,640;905,641;..." --floor "..." --out ...

결과: floor.yaml (보정점, 기존 바닥 구역은 그대로 둠) + 확인 그림(<out>.check.jpg) — 바닥 격자를 화면에 다시 그린다.
격자가 실제 타일 줄과 겹치면 보정이 맞은 것이다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.zones.floor import FloorMap  # noqa: E402


def parse_points(spec: str) -> list[tuple[float, float]]:
    pts = []
    for part in spec.split(";"):
        part = part.strip()
        if part:
            x, y = part.split(",")
            pts.append((float(x), float(y)))
    return pts


def read_frame(args) -> "cv2.Mat":
    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            raise SystemExit(f"그림을 열 수 없습니다: {args.image}")
        return image
    cap = cv2.VideoCapture(args.video)
    image = None
    for _ in range(args.frame + 1):   # H.265 는 위치 이동(seek)이 불안정해 차례로 읽는다
        ok, image = cap.read()
        if not ok:
            raise SystemExit(f"{args.frame} 번째 장면을 읽지 못했습니다: {args.video}")
    cap.release()
    return image


def pick_points(image, count: int) -> list[tuple[float, float]]:
    """화면에서 count 개를 차례로 찍는다."""
    picked: list[tuple[float, float]] = []
    scale = min(1.0, 1400 / image.shape[1])
    shown = cv2.resize(image, None, fx=scale, fy=scale)

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(picked) < count:
            picked.append((x / scale, y / scale))

    cv2.namedWindow("floor")
    cv2.setMouseCallback("floor", on_mouse)
    while True:
        canvas = shown.copy()
        for i, (x, y) in enumerate(picked):
            cv2.circle(canvas, (int(x * scale), int(y * scale)), 5, (0, 0, 255), -1)
            cv2.putText(canvas, str(i + 1), (int(x * scale) + 6, int(y * scale) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(canvas, f"{len(picked)}/{count}  click in --floor order, Backspace=undo, Enter=done",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("floor", canvas)
        key = cv2.waitKey(30) & 0xFF
        if key == 8 and picked:
            picked.pop()
        elif key in (13, 10) and len(picked) == count:
            break
        elif key == 27:
            raise SystemExit("취소")
    cv2.destroyWindow("floor")
    return picked


def draw_check(image, fm: FloorMap, tile_cm: float):
    """바닥 격자(tile_cm 간격)와 보정점을 화면에 그린다."""
    h, w = image.shape[:2]
    out = image.copy()
    xs = [p[0] for p in fm.floor_points]
    ys = [p[1] for p in fm.floor_points]
    x0, x1 = min(xs) - tile_cm, max(xs) + tile_cm
    y0, y1 = min(ys) - tile_cm, max(ys) + tile_cm

    def to_px(fx, fy):
        u, v = fm.to_image(fx, fy)
        return int(u * w), int(v * h)

    x = x0
    while x <= x1 + 1e-6:
        cv2.line(out, to_px(x, y0), to_px(x, y1), (0, 255, 255), 1, cv2.LINE_AA)
        x += tile_cm
    y = y0
    while y <= y1 + 1e-6:
        cv2.line(out, to_px(x0, y), to_px(x1, y), (0, 255, 255), 1, cv2.LINE_AA)
        y += tile_cm
    for (ix, iy), (fx, fy) in zip(fm.image_points, fm.floor_points):
        cv2.circle(out, (int(ix * w), int(iy * h)), 6, (0, 0, 255), 2)
        cv2.circle(out, to_px(fx, fy), 3, (0, 255, 0), -1)
        cv2.putText(out, f"{fx:.0f},{fy:.0f}", (int(ix * w) + 8, int(iy * h) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--image")
    ap.add_argument("--frame", type=int, default=0, help="--video 일 때 쓸 장면 번호")
    ap.add_argument("--floor", required=True, help="바닥 위치 cm 'x,y;x,y;...' (4개 이상)")
    ap.add_argument("--pixels", help="화면 위치 픽셀 'x,y;...' (없으면 화면에서 찍음)")
    ap.add_argument("--camera", default="angle1")
    ap.add_argument("--tile", type=float, default=60.0, help="확인 격자 간격 cm (바닥 타일 크기)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    floor = parse_points(args.floor)
    image = read_frame(args)
    h, w = image.shape[:2]
    pixels = parse_points(args.pixels) if args.pixels else pick_points(image, len(floor))
    if len(pixels) != len(floor):
        raise SystemExit(f"화면 점 {len(pixels)}개와 바닥 점 {len(floor)}개의 수가 다릅니다")

    out = Path(args.out)
    zones = FloorMap.load(out).zones if out.exists() else []
    fm = FloorMap(args.camera, [(x / w, y / h) for x, y in pixels], floor, zones)
    fm.save(out)
    check = out.with_suffix(".check.jpg")
    cv2.imwrite(str(check), draw_check(image, fm, args.tile))
    print(f"저장: {out}  (보정점 {len(floor)}개, 바닥 구역 {len(zones)}개 유지)")
    print(f"보정점 평균 오차: {fm.reprojection_error_cm():.1f} cm" + ("  (4점이면 늘 0 — 5점 이상으로 확인 권장)" if len(floor) == 4 else ""))
    print(f"확인 그림: {check}  — 노란 격자가 실제 타일 줄과 겹치는지 보세요")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
