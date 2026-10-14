"""검증 영상 고르기 — 영상마다 '사람이 얼마나 크게 보이나, 선반 물건이 얼마나 잘 보이나'를 자동으로 잰다.

검증 대상 세 가지(사람 추적 / 집기·되돌려놓기 / 선반 물건으로 개수 세기)에 쓸 수 있는 영상을 추리기 위한 1차 거름망이다.
사람이 최종으로 고르기 전에, 볼 가치가 있는 영상을 숫자로 좁힌다.

영상마다 고르게 뽑은 장면 N장에서:
  - 사람: YOLO 640 으로 찾고, 모델 입력 속 사람 키(px)를 잰다 (docs/scale-limits.md: 90 이상 정상, 65 미만 위험)
  - 물건: YOLO 1280 으로 사람·가구 말고 다른 것을 찾고, 장면 절반 이상에 같은 자리에 있는 것만 '자리 잡은 물건'으로 센다
    (선반 지도와 같은 방식 — 한 장면만 튀는 오탐을 뺀다). 물건 크기는 모델 입력(1280) 속 크기로 잰다.
  - 상호작용: 사람 상자가 자리 잡은 물건들이 있는 영역과 겹친 장면 비율
  크기는 모델 입력 속 크기지만 **원본 해상도보다 크게 치지 않는다** — 작은 영상(320x240)을 키워 넣어도 정보는 늘지 않는다
  (README §6: imgsz 는 프레임 긴 변까지만 의미가 있다). 처음엔 키운 크기로 재서 UCF 물건이 4배 크게 나왔다.

    python scripts/screen_videos.py data/videos/pexels/*.mp4 --out outputs/screening/pexels.csv
    python scripts/screen_videos.py data/merl/Videos_MERL_Shopping_Dataset --limit 10 --out outputs/screening/merl.csv

출력: CSV 한 줄 = 영상 하나, 그리고 영상마다 확인용 사진(물건 초록, 사람 파랑)이 out 옆 폴더에 저장된다.
등급 기준은 아래 grade() — 숫자를 근거로 1차 분류할 뿐, 최종 선정은 사진을 보고 사람이 한다.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import ShelfConfig  # noqa: E402
from src.shelf.monitor import ShelfItem, _Consensus  # noqa: E402

VIDEO_EXT = {".mp4", ".avi", ".mpg", ".mov", ".mkv"}


def collect(paths: list[str], limit: int | None) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files = sorted(f for f in p.rglob("*") if f.suffix.lower() in VIDEO_EXT)
            out += files[:limit] if limit else files
        elif p.suffix.lower() in VIDEO_EXT:
            out.append(p)
    seen: set[Path] = set()
    return [f for f in out if not (f.resolve() in seen or seen.add(f.resolve()))]   # 폴더가 겹쳐도 한 번만


def overlaps(a, b) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def grade(row: dict) -> dict:
    """1차 분류. 사람 크기 기준은 scale-limits.md, 물건 크기 기준(32px)은 작은 물체 탐지의 통상적인 하한."""
    person = row["person_tensor_px"]
    track = ("좋음" if person >= 90 else "주의" if person >= 65 else "부족" if person > 0 else "사람 없음")
    items, size = row["stable_items"], row["item_tensor_px"]
    shelf = ("좋음" if items >= 3 and size >= 32 else "부분" if items >= 1 else "불가")
    # 개수를 세려면 물건이 여러 개 또렷해야 한다. 큰 물체 하나(냉장고 문 등)만 잡힌 '부분'은 빼고 사람이 직접 본다.
    both = shelf == "좋음" and track in ("좋음", "주의") and row["interaction_ratio"] >= 0.1
    return {"추적": track, "선반물건": shelf, "개수세기 후보": "예" if both else "아니오"}


def screen(path: Path, model, item_classes: list[int], frames: int, sheet_dir: Path | None) -> dict:
    cap = cv2.VideoCapture(str(path))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < fps * 2:  # 메타데이터를 못 믿는 녹화본(MPEG2: 17 프레임이라고 하지만 실제 1,521) — 직접 센다
        total = 0
        while cap.grab():
            total += 1
        cap.release()
        cap = cv2.VideoCapture(str(path))
    long_side = float(max(w, h)) or 1.0
    picks = sorted({int(i * (total - 1) / max(1, frames - 1)) for i in range(frames)}) if total else []

    cons = _Consensus(presence=0.5, gate=0.5)
    per_frame_items, people_per_frame, person_heights = [], [], []
    shots = []   # (이미지, 물건 상자, 사람 상자)
    for idx in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, image = cap.read()
        if not ok:
            continue
        # 사람과 물건은 해상도를 나눠 찾고, 찾을 종류를 매번 명시한다 (shelf-map.md: ultralytics 가 classes 를 남긴다)
        pr = model.predict(image, imgsz=640, conf=0.25, classes=[0], verbose=False)[0]
        people = [tuple(map(float, b)) for b in pr.boxes.xyxy.cpu().numpy()]
        ir = model.predict(image, imgsz=1280, conf=0.15, classes=item_classes, verbose=False)[0]
        items = [ShelfItem(model.names[int(c)], tuple(map(float, b)), float(s))
                 for b, c, s in zip(ir.boxes.xyxy.cpu().numpy(), ir.boxes.cls.cpu().numpy(), ir.boxes.conf.cpu().numpy())]
        cons.add(items, keep=len(picks))
        per_frame_items.append(len(items))
        people_per_frame.append(people)
        person_heights += [b[3] - b[1] for b in people]
        shots.append((image, items, people))
    cap.release()

    stable = cons.result()
    region = None
    if stable:
        xs1, ys1, xs2, ys2 = zip(*(s.bbox for s in stable))
        region = (min(xs1), min(ys1), max(xs2), max(ys2))
    touching = sum(1 for ppl in people_per_frame if region and any(overlaps(p, region) for p in ppl))
    n = max(1, len(shots))
    row = {
        "video": str(path).replace("\\", "/"),
        "size": f"{w}x{h}",
        "seconds": round(total / fps, 1),
        "frames_checked": len(shots),
        "person_ratio": round(sum(1 for p in people_per_frame if p) / n, 2),
        "max_people": max((len(p) for p in people_per_frame), default=0),
        "person_tensor_px": round(statistics.median(person_heights) * min(640, long_side) / long_side) if person_heights else 0,
        "stable_items": len(stable),
        "item_tensor_px": round(statistics.median(s.size for s in stable) * min(1280, long_side) / long_side) if stable else 0,
        "items_per_frame": round(statistics.median(per_frame_items), 1) if per_frame_items else 0,
        "interaction_ratio": round(touching / n, 2),
        "item_labels": ",".join(sorted({s.label for s in stable}))[:80],
    }
    row.update(grade(row))

    if sheet_dir is not None and shots:
        # 확인용 사진 3장: 처음, 사람이 물건 영역에 닿은 장면(없으면 가운데), 마지막
        touch = [i for i, (_, _, ppl) in enumerate(shots) if region and any(overlaps(p, region) for p in ppl)]
        mid = touch[len(touch) // 2] if touch else len(shots) // 2
        tiles = []
        for i in (0, mid, len(shots) - 1):
            image, items, people = shots[i]
            vis = image.copy()
            for it in stable:
                x1, y1, x2, y2 = map(int, it.bbox)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
            for x1, y1, x2, y2 in people:
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 120, 0), 2)
            scale = 480.0 / vis.shape[1]
            tiles.append(cv2.resize(vis, (480, int(vis.shape[0] * scale))))
        sheet = cv2.hconcat(tiles)
        label = (f"{path.name}  person {row['person_tensor_px']}px  items {row['stable_items']} "
                 f"@{row['item_tensor_px']}px  touch {row['interaction_ratio']}")
        cv2.putText(sheet, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(sheet, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        sheet_dir.mkdir(parents=True, exist_ok=True)
        name = "_".join(path.parts[-2:]).replace(".", "_") + ".jpg"
        cv2.imwrite(str(sheet_dir / name), sheet)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="영상 파일 또는 폴더")
    parser.add_argument("--frames", type=int, default=24, help="영상마다 고르게 뽑을 장면 수")
    parser.add_argument("--limit", type=int, help="폴더마다 앞에서 몇 개만")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--out", default="outputs/screening/screening.csv")
    parser.add_argument("--no-sheets", action="store_true", help="확인용 사진을 만들지 않는다")
    args = parser.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    skip = {n.lower() for n in ShelfConfig().ignore}
    item_classes = [i for i, n in model.names.items() if n.lower() not in skip]

    videos = collect(args.paths, args.limit)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet_dir = None if args.no_sheets else out.with_suffix("")
    rows = []
    for i, path in enumerate(videos, 1):
        row = screen(path, model, item_classes, args.frames, sheet_dir)
        rows.append(row)
        print(f"[{i}/{len(videos)}] {path.name:32s} 사람 {row['person_tensor_px']:4d}px({row['추적']})  "
              f"물건 {row['stable_items']:2d}개 @{row['item_tensor_px']:3d}px({row['선반물건']})  "
              f"닿음 {row['interaction_ratio']:.2f}  개수세기 후보: {row['개수세기 후보']}", flush=True)
    if rows:
        with out.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n{len(rows)}개 영상 -> {out}" + ("" if sheet_dir is None else f", 확인용 사진 -> {sheet_dir}"))


if __name__ == "__main__":
    main()
