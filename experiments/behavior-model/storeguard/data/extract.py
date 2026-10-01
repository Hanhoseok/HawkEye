"""AI-Hub 배포 zip → data/raw/{train,val}/{videos,labels} 압축 해제.

zip 안의 엔트리명이 '/파일명.mp4' 처럼 루트 슬래시로 시작해서
ZipFile.extractall 이 경로 검사에 걸린다. 엔트리별로 basename 만 꺼낸다.

PowerShell 예:
    D:\\computervision\\.venv\\Scripts\\python.exe -m storeguard.data.extract --what labels
    D:\\computervision\\.venv\\Scripts\\python.exe -m storeguard.data.extract --what videos --split val
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(r"D:\computervision")

# 두 데이터셋을 같은 data/raw 로 푼다. 파일명만으로 구분되므로 섞여도 문제없다.
#   238-2 이상행동  : 전도/파손/절도      (3 fps)
#   238-1 구매행동  : 매장이동(3fps) + 선택/시험/구매/반품/비교(10fps)
_CANDIDATES = {
    "238-2": [
        ROOT / "238-2.실내(편의점, 매장) 사람 이상행동 데이터" / "01-1.정식개방데이터",
    ],
    "238-1": [
        # 구매행동 원본은 용량이 커서 외장 드라이브에 둘 수 있다. 있는 쪽을 자동으로 고른다.
        Path(r"F:\무인매장 구매행동 raw 데이터")
        / "238-1.실내(편의점, 매장) 사람 구매행동 데이터" / "01-1.정식개방데이터",
        ROOT / "238-1.실내(편의점, 매장) 사람 구매행동 데이터" / "01-1.정식개방데이터",
    ],
}


def _pick(paths: list[Path]) -> Path:
    """실제로 존재하고 zip 이 들어 있는 첫 경로. 없으면 첫 후보를 돌려준다."""
    for p in paths:
        if p.is_dir() and any(p.rglob("*.zip")):
            return p
    return paths[0]


DATASETS = {k: _pick(v) for k, v in _CANDIDATES.items()}
DEFAULT_SRC = DATASETS["238-2"]
DEFAULT_DST = ROOT / "data" / "raw"

SPLIT_DIRS = {"train": "Training", "val": "Validation"}
KIND_DIRS = {"videos": "01.원천데이터", "labels": "02.라벨링데이터"}


def extract_zip(zip_path: Path, dst: Path, overwrite: bool = False) -> tuple[int, int]:
    """엔트리를 dst 에 평평하게 푼다. (새로 푼 개수, 건너뛴 개수)"""
    dst.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    with zipfile.ZipFile(zip_path) as zf:
        entries = [e for e in zf.infolist() if not e.is_dir() and Path(e.filename).name]
        for e in entries:
            name = Path(e.filename).name
            out = dst / name
            if out.exists() and not overwrite and out.stat().st_size == e.file_size:
                skipped += 1
                continue
            tmp = out.with_suffix(out.suffix + ".part")
            with zf.open(e) as src, open(tmp, "wb") as fh:
                shutil.copyfileobj(src, fh, length=1 << 20)
            tmp.replace(out)
            written += 1
            if written % 50 == 0:
                print(f"    {written}/{len(entries)}", flush=True)
    return written, skipped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="배포 zip 압축 해제")
    ap.add_argument("--dataset", choices=["238-2", "238-1", "all"], default="238-2",
                    help="238-2=이상행동, 238-1=구매행동. --src 를 주면 무시된다")
    ap.add_argument("--src", type=Path, default=None)
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST)
    ap.add_argument("--split", choices=["train", "val", "all"], default="all")
    ap.add_argument("--what", choices=["videos", "labels", "all"], default="all")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    splits = ["train", "val"] if a.split == "all" else [a.split]
    kinds = ["labels", "videos"] if a.what == "all" else [a.what]
    if a.src is not None:
        sources = {"custom": a.src}
    elif a.dataset == "all":
        sources = dict(DATASETS)
    else:
        sources = {a.dataset: DATASETS[a.dataset]}

    total_w = total_s = 0
    empty: list[str] = []
    for ds_name, src in sources.items():
        if not src.is_dir():
            print(f"[경고] 데이터셋 폴더 없음: {src}", file=sys.stderr)
            continue
        for split in splits:
            for kind in kinds:
                src_dir = src / SPLIT_DIRS[split] / KIND_DIRS[kind]
                if not src_dir.is_dir():
                    print(f"[경고] 없음: {src_dir}", file=sys.stderr)
                    continue
                dst_dir = a.dst / split / kind
                zips = sorted(src_dir.glob("*.zip"))
                if not zips:
                    print(f"[경고] zip 없음: {src_dir}", file=sys.stderr)
                for z in zips:
                    if z.stat().st_size == 0:
                        # 다운로드가 끝나지 않으면 0바이트 zip + .irx623 등 임시 파일만 남는다.
                        empty.append(str(z))
                        print(f"[건너뜀] {z.name} 이 0바이트다. 다운로드가 끝나지 않았다.",
                              file=sys.stderr)
                        continue
                    print(f"[{ds_name} {split}/{kind}] {z.name} -> {dst_dir}", flush=True)
                    w, s = extract_zip(z, dst_dir, a.overwrite)
                    total_w += w
                    total_s += s
                    print(f"    완료: 신규 {w}, 기존 유지 {s}", flush=True)

    print(f"총 신규 {total_w}, 기존 유지 {total_s}")
    if empty:
        print(f"\n[중요] 0바이트 zip {len(empty)}개를 건너뛰었다. 다운로드를 완료해야 한다:",
              file=sys.stderr)
        for p in empty[:10]:
            print(f"  {p}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
