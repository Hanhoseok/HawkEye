"""그림을 dataset/images/ 에 가져다 놓는다. 그림에는 얼굴이 있어 저장소에 없다 — 따로 받아 와야 한다.

    python prepare.py <그림이 있는 폴더>

<폴더>는 둘 중 하나면 된다 (알아서 가린다):
  - 원래 컴퓨터의 outputs/products_rf/images/ 를 옮겨 온 것   (이름이 C01_angle1_rf0005.jpg 같은 꼴)
  - Roboflow 에서 내보낸 폴더 (train · valid · test 가 들어 있는 것, 이름이 C01_mp4-0005_jpg.rf.….jpg 같은 꼴)
    — dataset/frames.csv 의 source 칸으로 짝을 찾아 이름을 바꿔 놓는다. **같은 내보내기**(파일 이름이 같은 것)여야 한다.

끝나면 python check.py 로 확인한다.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from check import DATA, check, read_table, report


def find(folder: Path, row: dict) -> Path | None:
    """표의 한 줄에 해당하는 그림을 folder 에서 찾는다."""
    source = Path(row["source"])                       # 'train/C01_mp4-0005_jpg.rf.….jpg'
    for cand in (folder / row["image"], folder / "images" / row["image"], folder / source, folder / source.name):
        if cand.is_file():
            return cand
    return None


def prepare(folder: Path, data: Path = DATA) -> dict:
    images = data / "images"
    images.mkdir(parents=True, exist_ok=True)
    stats = {"placed": 0, "already": 0, "missing": []}
    for r in read_table(data):
        dst = images / r["image"]
        if dst.exists() and (not r.get("bytes") or dst.stat().st_size == int(r["bytes"])):
            stats["already"] += 1
            continue
        src = find(folder, r)
        if src is None:
            stats["missing"].append(r["image"])
            continue
        shutil.copyfile(src, dst)
        stats["placed"] += 1
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", type=Path, help="그림이 있는 폴더")
    args = ap.parse_args()
    if not args.folder.is_dir():
        print(f"{args.folder} 는 폴더가 아니다")
        return 1
    s = prepare(args.folder)
    print(f"가져다 놓음 {s['placed']}장 · 이미 있음 {s['already']}장 · 못 찾음 {len(s['missing'])}장")
    if s["missing"]:
        print(f"  ! 못 찾은 그림 예: {', '.join(s['missing'][:3])}")
    c = check()
    print(report(c))
    return 0 if c["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
