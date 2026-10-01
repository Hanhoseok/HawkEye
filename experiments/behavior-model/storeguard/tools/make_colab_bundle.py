"""Colab 으로 옮길 업로드 묶음 만들기.

만들어지는 것 (기본 위치: D:\\computervision\\colab_upload)

  storeguard_code.zip   코드 + configs + 인덱스/분할 (수 MB)
  cache_jpz.tar         JPEG 로 포장한 프레임 캐시 (약 5 GB)

왜 tar 로 묶나
  구글 드라이브는 **작은 파일 2,169개보다 큰 파일 1개**를 훨씬 빨리 주고받는다.
  Colab 에서 Drive 를 마운트해 파일을 하나씩 읽는 것도 느리다.
  큰 tar 하나를 로컬 디스크로 복사한 뒤 풀어 쓰는 것이 가장 빠르다.
  JPEG 는 이미 압축되어 있으므로 tar 는 압축하지 않는다(압축해도 거의 안 줄고 느리기만 하다).

PowerShell 예:
    # 1) 캐시를 JPEG 로 포장 (약 3분)
    D:\\computervision\\.venv\\Scripts\\python.exe -m storeguard.data.pack_cache --workers 4
    # 2) 업로드 묶음 생성
    D:\\computervision\\.venv\\Scripts\\python.exe -m storeguard.tools.make_colab_bundle
"""
from __future__ import annotations

import argparse
import json
import tarfile
import time
import zipfile
from pathlib import Path

from storeguard.config import Config

CODE_INCLUDE = [
    "storeguard/**/*.py",
    "configs/*.yaml",
    "requirements.txt",
    "AGENTS.md",
]
INDEX_FILES = ["clips.jsonl", "splits.json", "cache_meta.json", "takes.json"]


def build_code_zip(root: Path, out: Path, index_dir: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for pattern in CODE_INCLUDE:
            for p in sorted(root.glob(pattern)):
                if p.is_file() and "__pycache__" not in p.parts:
                    z.write(p, p.relative_to(root).as_posix())
                    n += 1
        for name in INDEX_FILES:
            p = index_dir / name
            if p.exists():
                z.write(p, f"data/index/{name}")
                n += 1
    return n


def build_cache_tar(cache_dir: Path, out: Path) -> tuple[int, int]:
    out.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(cache_dir.iterdir())
    files = [f for f in files if f.is_file()]
    total = 0
    tmp = out.with_suffix(".tar.part")
    with tarfile.open(tmp, "w") as t:          # 압축 없음 (JPEG 는 이미 압축됨)
        for i, f in enumerate(files, 1):
            t.add(f, arcname=f"data/cache/{f.name}")
            total += f.stat().st_size
            if i % 250 == 0:
                print(f"    {i}/{len(files)}", flush=True)
    tmp.replace(out)
    return len(files), total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--root", type=Path, default=Path(r"D:\computervision"))
    ap.add_argument("--cache", type=Path, default=None,
                    help="기본: paths.cache. 이름이 _jpz 로 끝나지 않으면 "
                         "pack_cache 산출물(<paths.cache>_jpz)을 찾는다")
    ap.add_argument("--out", type=Path, default=Path(r"D:\computervision\colab_upload"))
    a = ap.parse_args(argv)

    cfg = Config.load(a.config)
    cache = a.cache
    if cache is None:
        p = cfg.path("paths.cache")
        # 이미 jpz 캐시를 직접 쓰고 있으면 그대로, 아니면 pack_cache 산출물을 찾는다
        cache = p if p.name.endswith("_jpz") else p.parent / (p.name + "_jpz")
    if not cache.is_dir() or not any(cache.iterdir()):
        print(f"[오류] JPEG 캐시가 없다: {cache}")
        print("       먼저 실행: python -m storeguard.data.pack_cache --workers 4")
        return 2

    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("1/2 코드 + 인덱스 압축")
    code_zip = a.out / "storeguard_code.zip"
    n_code = build_code_zip(a.root, code_zip, cfg.path("paths.index"))
    print(f"    {code_zip.name}  {code_zip.stat().st_size / 1e6:.1f} MB  ({n_code} files)")

    print("2/2 프레임 캐시 tar 묶기")
    cache_tar = a.out / "cache_jpz.tar"
    n_cache, raw = build_cache_tar(cache, cache_tar)
    print(f"    {cache_tar.name}  {cache_tar.stat().st_size / 1024**3:.2f} GB  ({n_cache} files)")

    info = {
        "code_zip": str(code_zip),
        "code_zip_MB": round(code_zip.stat().st_size / 1e6, 1),
        "cache_tar": str(cache_tar),
        "cache_tar_GB": round(cache_tar.stat().st_size / 1024**3, 2),
        "cache_files": n_cache,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (a.out / "bundle_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n다음 단계")
    print(f"  1) 구글 드라이브의 MyDrive/storeguard/ 폴더에 아래 두 파일을 올린다.")
    print(f"       {code_zip}")
    print(f"       {cache_tar}")
    print(f"  2) notebooks/colab_train.ipynb 를 Colab 에서 연다.")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
