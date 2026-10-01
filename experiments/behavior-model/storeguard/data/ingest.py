"""zip → 프레임 캐시 **한 번에** (중간 mp4 를 디스크에 쌓지 않는다).

왜 만들었나
  기존 흐름은 두 단계였다.
      1) extract_paired : zip → mp4 를 디스크에 전부 푼다 (구매행동 전량이면 927 GB)
      2) preprocess     : mp4 를 읽어 프레임 캐시를 만든다
  같은 데이터를 디스크에 한 번 쓰고 다시 읽는 낭비가 있고, 927 GB 를 담을 공간도 필요하다.

  ingest 는 영상 하나씩 임시 파일로 꺼내 → 디코딩·리샘플링·리사이즈 → 캐시 저장 → 임시 파일 삭제
  를 한 프로세스 안에서 한다. 동시에 디스크에 존재하는 mp4 는 워커 수만큼(수백 MB)뿐이다.

실측 근거 (구매 영상 16개, i7-7700)
  | 설정 | videos/s |
  |---|---|
  | 워커 4, cv2 스레드 기본 (기존) | 0.25 |
  | 워커 6, cv2 스레드 1          | 0.31 |
  | 워커 8, cv2 스레드 1          | 0.33 |
  워커마다 OpenCV 가 8스레드를 쓰려 해서 서로 경합한다. 워커당 1스레드로 고정해야 한다.

  구간별 비중(단일 스레드): zip 해제 35.8 %, 디코딩 29.6 %, 리사이즈 28.9 %,
  JPEG 인코딩 0.6 %, 저장 0.2 %.

GPU 를 쓰지 않는 이유 (실측)
  | 방식 | 처리량 |
  |---|---|
  | CPU 1스레드 | 147 fps |
  | NVDEC (GTX 1050 Ti, PyNvVideoCodec) | 325 fps |
  | **CPU 6워커 합산** | **880 fps** |
  1050 Ti 에는 NVDEC 엔진이 **하나뿐**이라 병렬화가 안 된다. CPU 6코어가 2.7배 빠르다.
  NVDEC 엔진이 여러 개인 GPU(A100 등)에서는 반대가 되지만, 그러려면 원본 zip 670 GB 를
  먼저 올려야 해서 업로드가 더 오래 걸린다.

캐시 형식은 기존과 **완전히 동일**하다(학습 코드 수정 불필요).
  .npy      : (T, 128, 171, 3) uint8 RGB
  .jpz.npz  : jpeg(uint8 1-D), off(int64 T+1)

PowerShell 예:
    D:\\computervision\\.venv\\Scripts\\python.exe -m storeguard.data.ingest --dry-run
    D:\\computervision\\.venv\\Scripts\\python.exe -m storeguard.data.ingest ^
        --cap moving=250 --max-per-class 0 --workers 6
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import tempfile
import time
import traceback
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from storeguard import naming
from storeguard.config import Config
from storeguard.data.extract import DATASETS, KIND_DIRS, SPLIT_DIRS
from storeguard.data.extract_paired import choose, label_stems, scan_video_zips
from storeguard.data.framestore import JPZ, store_path
from storeguard.utils import write_json


# ----------------------------------------------------------------- 워커
def _cache_ok(path: Path, fmt: str) -> int:
    """캐시가 정상인지 확인하고 프레임 수를 돌려준다. 깨졌으면 0.

    파일이 있다는 이유만으로 완료로 보지 않는다(중간에 죽어 반쯤 쓰인 파일 방지).
    """
    try:
        if fmt == "npy":
            a = np.load(path, mmap_mode="r")
            return int(a.shape[0]) if a.ndim == 4 and a.shape[0] > 0 else 0
        z = np.load(path)
        n = int(len(z["off"]) - 1)
        return n if n > 0 and len(z["jpeg"]) == int(z["off"][-1]) else 0
    except Exception:
        return 0


def ingest_one(job: tuple) -> dict:
    (stem, zip_path, inner, out_path, fmt, H, W, target_fps, quality, work_dir) = job
    import cv2

    cv2.setNumThreads(1)          # 워커끼리 스레드 경합 방지 (실측 1.3배 차이)
    out = Path(out_path)
    res = {"stem": stem, "status": "FAILED", "frames": 0, "bytes": 0,
           "t_extract": 0.0, "t_decode": 0.0, "t_encode": 0.0, "t_write": 0.0,
           "src_frames": 0, "src_bytes": 0}
    t_all = time.perf_counter()
    tmp_mp4 = Path(work_dir) / f"{os.getpid()}_{stem}.mp4"
    try:
        # 1) zip 엔트리 → 임시 mp4 (처리 후 바로 지운다)
        t = time.perf_counter()
        with zipfile.ZipFile(zip_path) as zf, zf.open(inner) as src, \
                open(tmp_mp4, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        res["t_extract"] = time.perf_counter() - t
        res["src_bytes"] = tmp_mp4.stat().st_size

        # 2) 디코딩 + 시간 리샘플링 + 리사이즈
        t = time.perf_counter()
        cap = cv2.VideoCapture(str(tmp_mp4))
        if not cap.isOpened():
            raise RuntimeError("VideoCapture 열기 실패")
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        step = (src_fps / target_fps) if (target_fps and src_fps > target_fps * 1.05) else 1.0
        frames = []
        i, nxt = 0, 0.0
        try:
            while True:
                if i + 1e-9 >= nxt:
                    ok, fr = cap.read()
                    if not ok:
                        break
                    fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
                    frames.append(fr[:, :, ::-1])       # BGR → RGB
                    nxt += step
                else:
                    # 버릴 프레임은 grab() 만 (색변환·복사를 건너뛴다)
                    if not cap.grab():
                        break
                i += 1
        finally:
            cap.release()
        res["t_decode"] = time.perf_counter() - t
        res["src_frames"] = i
        if not frames:
            raise RuntimeError("프레임 0개")

        # 3) 인코딩
        t = time.perf_counter()
        if fmt == "npy":
            payload = np.ascontiguousarray(np.stack(frames, 0), dtype=np.uint8)
        else:
            blobs = []
            for f in frames:
                ok, buf = cv2.imencode(".jpg", f[:, :, ::-1],
                                       [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
                if not ok:
                    raise RuntimeError("JPEG 인코딩 실패")
                blobs.append(buf.tobytes())
            lens = np.array([len(b) for b in blobs], dtype=np.int64)
            off = np.zeros(len(blobs) + 1, dtype=np.int64)
            np.cumsum(lens, out=off[1:])
            payload = (np.frombuffer(b"".join(blobs), dtype=np.uint8), off)
        res["t_encode"] = time.perf_counter() - t

        # 4) 저장 — 임시 파일에 쓴 뒤 rename (중간에 죽어도 깨진 캐시가 남지 않는다)
        t = time.perf_counter()
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp_out = out.with_name(out.name + ".tmp")
        with open(tmp_out, "wb") as fh:
            if fmt == "npy":
                np.save(fh, payload)
            else:
                np.savez(fh, jpeg=payload[0], off=payload[1])
        tmp_out.replace(out)
        res["t_write"] = time.perf_counter() - t

        n = _cache_ok(out, fmt)
        if n != len(frames):
            raise RuntimeError(f"저장 검증 실패 (기대 {len(frames)}, 실제 {n})")
        res.update(status="SUCCESS", frames=len(frames), bytes=out.stat().st_size)
    except Exception as exc:
        res["error"] = f"{type(exc).__name__}: {exc}"
        res["trace"] = traceback.format_exc(limit=3)
        try:
            out.with_name(out.name + ".tmp").unlink(missing_ok=True)
        except Exception:
            pass
    finally:
        try:
            tmp_mp4.unlink(missing_ok=True)
        except Exception:
            pass
    res["t_total"] = time.perf_counter() - t_all
    return res


# ----------------------------------------------------------------- 메인
def parse_caps(pairs: list[str], default: int) -> dict[str, int]:
    caps = {}
    for p in pairs or []:
        if "=" not in p:
            raise ValueError(f"--cap 형식은 클래스=개수: {p}")
        k, v = p.split("=", 1)
        caps[k.strip()] = int(v)
    caps.setdefault("_default", default)
    return caps


def select(paired: list[str], caps: dict[str, int], seed: int,
           prefer: set[str]) -> tuple[list[str], dict]:
    """클래스별 상한을 따로 줄 수 있게 choose() 를 클래스 단위로 호출한다."""
    by_cls = collections.defaultdict(list)
    for s in paired:
        try:
            by_cls[naming.parse_stem(s).class_name].append(s)
        except naming.NameParseError:
            pass
    picked, report = [], {}
    for cls, stems in sorted(by_cls.items()):
        cap = caps.get(cls, caps["_default"])
        got, rep = choose(stems, cap, seed, prefer=prefer)
        picked += got
        report[cls] = {"짝맞음": len(stems), "상한": cap or "무제한",
                       "선택": len(got)}
    return sorted(picked), report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--dataset", default="238-1", choices=list(DATASETS))
    ap.add_argument("--raw", type=Path, default=Path(r"D:\computervision\data\raw"),
                    help="라벨 XML 위치")
    ap.add_argument("--out", type=Path, default=None,
                    help="캐시 출력 폴더 (기본: paths.cache)")
    ap.add_argument("--format", choices=["jpz", "npy"], default="jpz",
                    help="jpz 는 npy 의 약 1/6 용량. 학습 코드는 둘 다 읽는다")
    ap.add_argument("--quality", type=int, default=95, help="jpz JPEG 품질")
    ap.add_argument("--max-per-class", type=int, default=0,
                    help="클래스별 상한(0=무제한). --cap 으로 개별 지정 가능")
    ap.add_argument("--cap", action="append", default=[],
                    help="클래스별 상한. 예: --cap moving=250")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--work-dir", type=Path, default=None,
                    help="임시 mp4 위치 (기본: 시스템 임시 폴더)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--limit", type=int, default=0, help="처리 개수 제한(시험용)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="이미 있는 캐시도 다시 만든다")
    a = ap.parse_args(argv)

    cfg = Config.load(a.config)
    out_dir = a.out or cfg.path("paths.cache")
    H = int(cfg["data.cache_height"])
    W = int(cfg["data.cache_width"])
    target_fps = float(cfg["data.sample_fps"])
    src = DATASETS[a.dataset]

    labels = label_stems(a.raw)
    videos = scan_video_zips(src)
    paired = sorted(set(labels) & set(videos))
    print(f"라벨 {len(labels)} | zip 안 영상 {len(videos)} | 짝 맞음 {len(paired)}")
    if not paired:
        print("[오류] 짝이 맞는 영상이 없다. 라벨을 먼저 풀 것.")
        return 2

    caps = parse_caps(a.cap, a.max_per_class)
    # 이미 캐시가 있는 영상의 take 를 우선 고른다(재처리 낭비 방지)
    cached = {p.name[: -len(JPZ)] if p.name.endswith(JPZ) else p.stem
              for p in out_dir.glob("*") if p.is_file()} if out_dir.is_dir() else set()
    picked, report = select(paired, caps, a.seed, prefer=cached)

    print(f"\n{'클래스':10s} {'짝맞음':>7s} {'상한':>8s} {'선택':>6s}")
    for cls, r in report.items():
        print(f"{cls:10s} {r['짝맞음']:7d} {str(r['상한']):>8s} {r['선택']:6d}")
    print(f"총 선택 {len(picked)}개")

    # 이미 정상 캐시가 있는 것은 건너뛴다 (resume)
    todo, skipped = [], 0
    ext = JPZ if a.format == "jpz" else ".npy"
    for s in picked:
        out = out_dir / f"{s}{ext}"
        if not a.force:
            found = store_path(out_dir, s)
            if found and _cache_ok(found[0], "jpz" if found[1] == "jpz" else "npy"):
                skipped += 1
                continue
        zpath, inner, split, size = videos[s]
        todo.append((s, str(zpath), inner, str(out), a.format, H, W,
                     target_fps, a.quality, ""))
    if a.limit:
        todo = todo[: a.limit]
    print(f"이미 캐시 있음 {skipped}개 | 처리할 것 {len(todo)}개")

    if a.dry_run:
        print("\n--dry-run 이므로 실제 처리는 하지 않았다.")
        return 0
    if not todo:
        print("할 일이 없다.")
        return 0

    work = Path(a.work_dir) if a.work_dir else Path(tempfile.mkdtemp(prefix="ingest_"))
    work.mkdir(parents=True, exist_ok=True)
    todo = [j[:-1] + (str(work),) for j in todo]
    print(f"임시 폴더: {work}  (동시에 존재하는 mp4 는 워커 수만큼뿐)")

    manifest_path = cfg.path("paths.reports") / f"ingest_{a.dataset}.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fail_log = cfg.path("paths.reports") / f"ingest_{a.dataset}_failed.log"

    t0 = time.perf_counter()
    agg = collections.Counter()
    times = collections.defaultdict(float)
    done = ok = 0
    with open(manifest_path, "a", encoding="utf-8") as mf, \
            ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(ingest_one, j) for j in todo]
        for f in as_completed(futs):
            r = f.result()
            done += 1
            agg[r["status"]] += 1
            for k in ("t_extract", "t_decode", "t_encode", "t_write", "t_total"):
                times[k] += r.get(k, 0.0)
            times["frames"] += r.get("frames", 0)
            times["src_frames"] += r.get("src_frames", 0)
            times["src_bytes"] += r.get("src_bytes", 0)
            if r["status"] == "SUCCESS":
                ok += 1
            else:
                with open(fail_log, "a", encoding="utf-8") as lf:
                    lf.write(f"{r['stem']}\t{r.get('error')}\n{r.get('trace','')}\n")
            mf.write(json.dumps({k: v for k, v in r.items() if k != "trace"},
                                ensure_ascii=False) + "\n")
            if done % 25 == 0 or done == len(todo):
                el = time.perf_counter() - t0
                rate = done / el
                print(f"  {done}/{len(todo)}  {rate:.2f} videos/s  "
                      f"{times['src_frames']/el:.0f} frames/s  "
                      f"{times['src_bytes']/el/1e6:.0f} MB/s  "
                      f"남은시간 {(len(todo)-done)/max(rate,1e-9)/60:.0f}분  "
                      f"실패 {done-ok}", flush=True)

    el = time.perf_counter() - t0
    wall = {k: round(times[k] / max(done, 1) * 1000, 1)
            for k in ("t_extract", "t_decode", "t_encode", "t_write", "t_total")}
    summary = {
        "dataset": a.dataset, "format": a.format, "quality": a.quality,
        "workers": a.workers, "선택": len(picked), "이미있음": skipped,
        "처리": done, "성공": ok, "실패": done - ok,
        "elapsed_sec": round(el, 1),
        "videos_per_sec": round(done / el, 3),
        "frames_per_sec": round(times["src_frames"] / el, 1),
        "MB_per_sec": round(times["src_bytes"] / el / 1e6, 1),
        "영상당_구간_ms(워커 내부 합계)": wall,
        "manifest": str(manifest_path),
    }
    if done - ok:
        summary["실패_로그"] = str(fail_log)
    write_json(cfg.path("paths.reports") / f"ingest_{a.dataset}_summary.json", summary)
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))

    if not a.work_dir:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if ok == done else 1


if __name__ == "__main__":
    raise SystemExit(main())
