"""프레임 캐시(.npy) → JPEG 포장(.jpz.npz) 변환.

외부 GPU(Colab 등)로 학습을 옮길 때 업로드할 용량을 줄이기 위한 도구다.

(실측, 표본 20개 영상)

| 품질 | 전체 용량 | 원본 대비 | 클립16장 디코딩 | PSNR |
|---|---|---|---|---|
| 95 | 5.01 GB | 21.0 % | 4.0 ms | 31.3 dB |
| 90 | 3.62 GB | 15.2 % | 3.5 ms | 30.2 dB |
| 85 | 2.93 GB | 12.3 % | 2.9 ms | 29.1 dB |

참고로 .npy 를 zlib 으로 압축하면 88 % 까지밖에 줄지 않는다(무의미).

기본값은 품질 95 다. 5 GB 면 구글 드라이브 무료 용량(15 GB)에 들어가고 화질 손실이 가장 작다.

PowerShell 예:
    D:\\computervision\\.venv\\Scripts\\python.exe -m storeguard.data.pack_cache --workers 4
    D:\\computervision\\.venv\\Scripts\\python.exe -m storeguard.data.pack_cache --quality 90 --out D:\\upload\\cache
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from storeguard.config import Config
from storeguard.data.framestore import JPZ
from storeguard.utils import write_json


def pack_one(args: tuple) -> dict:
    src, dst, quality = args
    import cv2

    out = Path(dst)
    if out.exists():
        try:
            z = np.load(out)
            return {"stem": out.name[: -len(JPZ)], "status": "cached",
                    "frames": int(len(z["off"]) - 1), "bytes": out.stat().st_size}
        except Exception:
            out.unlink(missing_ok=True)

    arr = np.load(src, mmap_mode="r")
    blobs = []
    for fr in arr:
        ok, buf = cv2.imencode(".jpg", np.asarray(fr)[:, :, ::-1],
                               [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if not ok:
            return {"stem": out.name, "status": "encode_failed"}
        blobs.append(buf.tobytes())

    lens = np.array([len(b) for b in blobs], dtype=np.int64)
    off = np.zeros(len(blobs) + 1, dtype=np.int64)
    np.cumsum(lens, out=off[1:])
    blob = np.frombuffer(b"".join(blobs), dtype=np.uint8)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    # JPEG 는 이미 압축되어 있으므로 savez_compressed 를 쓰지 않는다(느리기만 하다).
    with open(tmp, "wb") as fh:
        np.savez(fh, jpeg=blob, off=off)
    tmp.replace(out)
    return {"stem": out.name[: -len(JPZ)], "status": "ok",
            "frames": int(len(blobs)), "bytes": out.stat().st_size,
            "src_bytes": int(arr.nbytes)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--src", type=Path, default=None, help="기본: paths.cache")
    ap.add_argument("--out", type=Path, default=None,
                    help="기본: <paths.cache>_jpz. 업로드용으로 따로 모으려면 지정")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)

    cfg = Config.load(a.config)
    src = a.src or cfg.path("paths.cache")
    out = a.out or src.parent / (src.name + "_jpz")
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("*.npy"))
    if a.limit:
        files = files[: a.limit]
    if not files:
        print(f"변환할 .npy 가 없다: {src}")
        return 2

    jobs = [(str(f), str(out / (f.stem + JPZ)), a.quality) for f in files]
    print(f"대상 {len(jobs)}개, 품질 {a.quality}, workers={a.workers}")
    print(f"  {src}  ->  {out}")

    t0 = time.time()
    results = []
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(pack_one, j) for j in jobs]
        for f in as_completed(futs):
            results.append(f.result())
            done += 1
            if done % 100 == 0 or done == len(jobs):
                el = time.time() - t0
                rate = done / max(el, 1e-6)
                print(f"  {done}/{len(jobs)}  {rate:.0f}/s  "
                      f"남은시간 {(len(jobs)-done)/max(rate,1e-6)/60:.1f}분", flush=True)

    src_bytes = sum(int(f.stat().st_size) for f in files)
    dst_bytes = sum(r.get("bytes", 0) for r in results)
    bad = [r for r in results if r["status"] not in ("ok", "cached")]

    # 클립 생성기가 쓰는 프레임 수 메타를 같이 갱신한다.
    meta_path = cfg.path("paths.index") / "cache_meta.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    for r in results:
        if r.get("frames"):
            meta[r["stem"]] = int(r["frames"])
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "src": str(src), "out": str(out), "quality": a.quality,
        "count": len(results), "failed": bad[:20],
        "src_GB": round(src_bytes / 1024**3, 3),
        "out_GB": round(dst_bytes / 1024**3, 3),
        "비율": f"{dst_bytes / max(src_bytes,1) * 100:.1f}%",
        "elapsed_sec": round(time.time() - t0, 1),
    }
    write_json(cfg.path("paths.reports") / "pack_cache_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
