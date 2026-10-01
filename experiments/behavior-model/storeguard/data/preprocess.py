"""영상 → 프레임 캐시(.npy) 전처리.

왜 캐시하나
  원천 영상은 1920x1080 MP4 이고 학습 중 클립마다 디코딩하면 i7-7700 에서 데이터로더가 병목이 된다.
  영상당 한 번만 디코딩해 (T, H, W, 3) uint8 배열로 저장하면 학습이 디스크 순차 읽기만 하면 된다.

크기 근거
  torchvision 의 Kinetics 비디오 프리셋과 동일하게 resize(H=128, W=171) 후 crop 112 를 쓴다.
  사전학습 가중치(r2plus1d_18, Kinetics-400)가 이 전처리로 학습되었기 때문이다.
  영상 1개 = 180 x 128 x 171 x 3 = 약 11.8 MB. 2,169개 전체 = 약 25 GB.

PowerShell 예:
    D:\\computervision\\.venv\\Scripts\\python.exe -m storeguard.data.preprocess --workers 4
    D:\\computervision\\.venv\\Scripts\\python.exe -m storeguard.data.preprocess --only-split test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from storeguard.config import Config


def decode_one(args: tuple) -> dict:
    video_path, out_path, height, width, expect_frames, target_fps = args
    import cv2

    out = Path(out_path)
    if out.exists():
        try:
            arr = np.load(out, mmap_mode="r")
            return {"stem": out.stem, "status": "cached", "frames": int(arr.shape[0])}
        except Exception:
            out.unlink(missing_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"stem": out.stem, "status": "open_failed"}
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    # 캐시는 항상 sample_fps 로 맞춘다. 원천 이상행동 영상은 이미 3fps 라 전부 보존되지만,
    # 직접 촬영한 정상 영상(보통 30fps)은 여기서 3fps 로 줄어야 시간 기준이 일치한다.
    step = 1.0
    if target_fps and src_fps and src_fps > target_fps * 1.05:
        step = src_fps / target_fps
    frames = []
    try:
        i = 0
        next_keep = 0.0
        while True:
            if i + 1e-9 >= next_keep:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                frames.append(frame[:, :, ::-1])   # BGR -> RGB
                next_keep += step
            else:
                # 버릴 프레임은 grab() 만 한다. 디코딩은 하되 색변환/복사를 건너뛰어 더 빠르다.
                # 구매행동 영상은 10fps → 3fps 로 줄이므로 3프레임 중 2프레임이 여기로 온다.
                if not cap.grab():
                    break
            i += 1
    finally:
        cap.release()

    if not frames:
        return {"stem": out.stem, "status": "no_frames"}

    arr = np.ascontiguousarray(np.stack(frames, axis=0), dtype=np.uint8)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".npy.part")
    # 주의: np.save 에 **경로 문자열**을 주면 파일명이 .npy 로 끝나지 않을 때 .npy 를 덧붙인다.
    # ('X.npy.part' → 'X.npy.part.npy'). 파일 객체를 주면 그런 일이 없다.
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
    tmp.replace(out)
    status = "ok"
    if step == 1.0 and expect_frames and abs(arr.shape[0] - expect_frames) > 1:
        status = f"frame_mismatch(got={arr.shape[0]},expect={expect_frames})"
    return {"stem": out.stem, "status": status, "frames": int(arr.shape[0]),
            "src_fps": round(src_fps, 3), "resampled": step != 1.0}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--only-split", choices=["train", "val", "test"], default=None,
                    help="splits.json 의 특정 분할만 처리")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)

    cfg = Config.load(a.config, a.overrides)
    index_dir = cfg.path("paths.index")
    cache_dir = cfg.path("paths.cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    H = int(cfg["data.cache_height"])
    W = int(cfg["data.cache_width"])
    target_fps = float(cfg["data.sample_fps"])

    rows = [json.loads(l) for l in (index_dir / "clips.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    wanted: set[str] | None = None
    if a.only_split:
        splits = json.loads((index_dir / "splits.json").read_text(encoding="utf-8"))
        wanted = set(splits[a.only_split])

    jobs = []
    for r in rows:
        if not r.get("has_video"):
            continue
        if wanted is not None and r["stem"] not in wanted:
            continue
        jobs.append((r["video_path"], str(cache_dir / f"{r['stem']}.npy"), H, W,
                     r.get("label_frames") or 0, target_fps))
    if a.limit:
        jobs = jobs[: a.limit]

    print(f"대상 {len(jobs)}개, workers={a.workers}, 크기 {H}x{W}")
    t0 = time.time()
    results = []
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(decode_one, j) for j in jobs]
        for f in as_completed(futs):
            res = f.result()
            results.append(res)
            done += 1
            if done % 50 == 0 or done == len(jobs):
                el = time.time() - t0
                rate = done / max(el, 1e-6)
                print(f"  {done}/{len(jobs)}  {rate:.1f}/s  남은시간 {(len(jobs)-done)/max(rate,1e-6)/60:.1f}분",
                      flush=True)

    from collections import Counter
    stat = Counter(r["status"] if r["status"] in ("ok", "cached") else r["status"] for r in results)
    bad = [r for r in results if r["status"] not in ("ok", "cached")]
    manifest = {
        "cache_dir": str(cache_dir), "height": H, "width": W, "sample_fps": target_fps,
        "count": len(results), "status": dict(stat),
        "resampled_videos": sum(1 for r in results if r.get("resampled")),
        "elapsed_sec": round(time.time() - t0, 1),
        "failed": bad[:50],
    }
    # 캐시된 프레임 수를 기록한다. 클립 생성기는 라벨 프레임 수가 아니라 이 값을 써야
    # 시간 리샘플링된 영상에서도 anchor 범위가 맞는다.
    meta_path = a.out_index if getattr(a, "out_index", None) else (
        Path(str(cfg.path("paths.index"))) / "cache_meta.json")
    meta: dict = {}
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
    manifest["cache_meta"] = str(meta_path)
    (cfg.path("paths.reports")).mkdir(parents=True, exist_ok=True)
    (cfg.path("paths.reports") / "preprocess_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2)[:2000])
    return 0 if not bad else 0


if __name__ == "__main__":
    sys.exit(main())
