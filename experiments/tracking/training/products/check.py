"""학습 데이터가 다 갖춰졌는지 본다 — 그림이 다 있는지, 가져온 그림이 맞는 그림인지, 라벨 · 목록이 서로 맞는지.

    python check.py            # 빠른 검사 (그림 수 · 파일 크기)
    python check.py --deep     # 그림 내용까지 (sha1, 1~2분)

이 폴더의 다른 파일만 쓴다 (저장소의 다른 코드에 기대지 않는다).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "dataset"
SIDES = ("train", "val", "test")
CAMERAS = ("angle0", "angle1")


def read_table(data: Path = DATA) -> list[dict]:
    """dataset/frames.csv — 그림마다 한 줄 (image · source · video · camera · split · boxes · bytes · sha1)."""
    with (data / "frames.csv").open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_list(path: Path) -> list[str]:
    """목록 파일(./images/이름.jpg 한 줄씩) → 그림 이름들."""
    if not path.exists():
        return []
    return [line.strip().rsplit("/", 1)[-1] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check(data: Path = DATA, deep: bool = False) -> dict:
    """문제를 까닭별로 모은다. ok = 학습을 시작해도 된다."""
    rows = read_table(data)
    out = {"images": len(rows), "present": 0, "boxes": 0, "by_split": {s: 0 for s in SIDES},
           "missing_images": [], "wrong_images": [], "missing_labels": [], "label_mismatch": [], "list_mismatch": [],
           "extra_images": [], "video_overlap": []}
    videos: dict[str, set] = {}
    for r in rows:
        image, label = data / "images" / r["image"], data / "labels" / (Path(r["image"]).stem + ".txt")
        videos.setdefault(r["video"], set()).add(r["split"])
        out["by_split"][r["split"]] += 1
        if not label.exists():
            out["missing_labels"].append(r["image"])
        else:
            lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            out["boxes"] += len(lines)
            if len(lines) != int(r["boxes"]):
                out["label_mismatch"].append(r["image"])
        if not image.exists():
            out["missing_images"].append(r["image"])
            continue
        out["present"] += 1
        if r.get("bytes") and image.stat().st_size != int(r["bytes"]):
            out["wrong_images"].append(r["image"])
        elif deep and r.get("sha1") and hashlib.sha1(image.read_bytes()).hexdigest() != r["sha1"]:
            out["wrong_images"].append(r["image"])
    known = {r["image"] for r in rows}
    if (data / "images").exists():
        out["extra_images"] = sorted(p.name for p in (data / "images").glob("*.jpg") if p.name not in known)
    for side in SIDES:
        want = sorted(r["image"] for r in rows if r["split"] == side)
        if sorted(read_list(data / f"{side}.txt")) != want:
            out["list_mismatch"].append(f"{side}.txt")
        for camera in CAMERAS:
            want = sorted(r["image"] for r in rows if r["split"] == side and r["camera"] == camera)
            if sorted(read_list(data / f"{side}_{camera}.txt")) != want:
                out["list_mismatch"].append(f"{side}_{camera}.txt")
    # 같은 영상이 두 쪽에 있으면 점수가 부푼다 (옆 장면은 거의 같은 그림이다)
    out["video_overlap"] = sorted(v for v, sides in videos.items() if len(sides) > 1)
    blocking = ("missing_images", "wrong_images", "missing_labels", "label_mismatch", "list_mismatch", "video_overlap")
    out["ok"] = bool(rows) and not any(out[k] for k in blocking)
    return out


def report(c: dict) -> str:
    lines = [f"그림 {c['present']} / {c['images']}장 · 상자 {c['boxes']}개 · "
             + " · ".join(f"{s} {c['by_split'][s]}" for s in SIDES)]
    words = {"missing_images": "없는 그림 (prepare.py 로 가져온다)", "wrong_images": "다른 그림 (크기나 내용이 표와 다르다)",
             "missing_labels": "없는 라벨", "label_mismatch": "상자 수가 표와 다른 라벨", "list_mismatch": "표와 다른 목록",
             "video_overlap": "두 쪽에 걸친 영상", "extra_images": "표에 없는 그림 (학습에는 안 쓰인다)"}
    for k, word in words.items():
        if c[k]:
            lines.append(f"  ! {word}: {len(c[k])}개 — 예: {', '.join(c[k][:3])}")
    lines.append("결과: 학습을 시작해도 된다" if c["ok"] else "결과: 아직 안 된다 (위 ! 줄)")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--deep", action="store_true", help="그림 내용(sha1)까지 견준다")
    args = ap.parse_args()
    c = check(deep=args.deep)
    print(report(c))
    return 0 if c["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
