"""외부 데이터셋(폴더 단위 라벨) 으로 오탐률을 측정한다.

용도
  학습에 쓴 AI-Hub 데이터와 **완전히 다른 매장·카메라**의 영상으로
  모델이 실제로 일반화되는지, 정상 영상에서 얼마나 잘못 울리는지 본다.
  폴더 하나 = 라벨 하나다(XML 라벨이 없어도 된다).

    --normal <폴더>     이 안의 영상에서 나온 이상행동 점수는 전부 오탐이다
    --positive <폴더>   이 안의 영상은 해당 행동이 실제로 있다 (--positive-class)

왜 전용 도구가 필요한가 — 짧은 영상 문제
  학습 클립은 16프레임 @3fps = **5.33초**다. 그런데 외부 시험 영상이 그보다 짧으면
  (예: Zenodo shoplifting 데이터는 3.1~4.0초) 3fps 로는 16프레임을 못 채운다.
  그래서 샘플링 방식을 두 가지로 두고 **둘 다 재서 민감도를 같이 보고한다.**

    uniform : 영상 전체에 16프레임을 균등 배치한다(TSN 방식).
              실제 움직임만 쓰지만 시간 축이 학습(5.33초) 대비 압축된다.
    fps_pad : 학습과 같은 3fps 로 뽑고 모자라면 마지막 프레임을 반복해 채운다.
              시간 축은 맞지만 뒤쪽 몇 프레임이 정지 화면이 된다.

  어느 쪽도 학습 조건과 완전히 같지는 않다. 두 결과가 많이 다르면
  그 차이 자체가 "이 모델은 이 길이의 영상에 쓰기 어렵다" 는 신호다.

측정 항목
  - 영상별 클래스 최대 점수
  - 임계값에서의 오탐률(정상 영상 중 알림이 뜬 비율)
  - 임계값 스윕 (0.1 ~ 0.9)
  - AUC : 정상 vs 양성 폴더를 점수만으로 얼마나 가르는지 (임계값과 무관한 지표)

PowerShell 예:
    D:\\computervision\\.venv\\Scripts\\python.exe -m storeguard.tools.eval_external ^
        --run unified_r2p1d ^
        --normal "F:\\...\\zenodo test 데이터\\normal" ^
        --positive "F:\\...\\zenodo test 데이터\\shoplifting" --positive-class theft
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from storeguard.config import Config
from storeguard.evaluate import load_run
from storeguard.utils import get_logger, write_json


def read_clip(path: Path, clip_len: int, H: int, W: int,
              sample_fps: float, mode: str) -> np.ndarray | None:
    """영상 1개 → (clip_len, H, W, 3) uint8 RGB. 실패하면 None."""
    import cv2

    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0 or src_fps <= 0:
            return None

        if mode == "uniform":
            # 영상 전체에 균등 배치
            want = sorted({int(round(i * (total - 1) / max(clip_len - 1, 1)))
                           for i in range(clip_len)})
            while len(want) < clip_len:
                want.append(want[-1])
            want = want[:clip_len]
        else:
            # 학습과 같은 3fps 간격. 모자라면 뒤에서 반복해 채운다.
            step = max(src_fps / sample_fps, 1.0)
            want = [int(round(i * step)) for i in range(clip_len)]
            want = [min(w, total - 1) for w in want]

        frames, idx_set = {}, set(want)
        i = 0
        while i <= max(want):
            ok = cap.grab()
            if not ok:
                break
            if i in idx_set:
                ok, fr = cap.retrieve()
                if not ok:
                    break
                fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
                frames[i] = fr[:, :, ::-1]
            i += 1
        if not frames:
            return None
        last = None
        out = []
        for w in want:
            f = frames.get(w, last)
            if f is None:
                f = next(iter(frames.values()))
            out.append(f)
            last = f
        return np.ascontiguousarray(np.stack(out, 0), dtype=np.uint8)
    finally:
        cap.release()


@torch.no_grad()
def score_folder(folder: Path, cfg: Config, model, device, mode: str,
                 log, limit: int = 0) -> tuple[list[str], np.ndarray]:
    from storeguard.data.dataset import preprocess_clip

    files = sorted(folder.glob("*.mp4"))
    if limit:
        files = files[:limit]
    L = int(cfg["data.clip_len"])
    H, W = int(cfg["data.cache_height"]), int(cfg["data.cache_width"])
    fps = float(cfg["data.sample_fps"])
    amp = bool(cfg.get("train.amp", True)) and device.type == "cuda"
    bs = int(cfg["eval.batch_size"])

    names, probs, batch, bnames, failed = [], [], [], [], []
    t0 = time.perf_counter()
    for k, p in enumerate(files, 1):
        clip = read_clip(p, L, H, W, fps, mode)
        if clip is None:
            failed.append(p.name)
            continue
        batch.append(preprocess_clip(clip, cfg))
        bnames.append(p.name)
        if len(batch) >= bs or k == len(files):
            x = torch.stack(batch).to(device)
            with torch.autocast("cuda", enabled=amp):
                out = model(x)
            probs.append(torch.softmax(out.float(), 1).cpu().numpy())
            names += bnames
            batch, bnames = [], []
        if k % 100 == 0:
            log.info("  %s %d/%d (%.1f videos/s)", folder.name, k, len(files),
                     k / (time.perf_counter() - t0))
    if failed:
        log.warning("%s: 열지 못한 영상 %d개 %s", folder.name, len(failed), failed[:3])
    return names, (np.concatenate(probs) if probs else np.zeros((0, len(cfg.class_names))))


def auc(pos: np.ndarray, neg: np.ndarray) -> float | None:
    """AUC = 무작위로 뽑은 양성 점수가 음성 점수보다 클 확률 (순위 기반)."""
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    # 동점 처리
    u, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    mean_rank = np.zeros(len(u))
    for i in range(len(u)):
        pass
    s = np.argsort(allv, kind="mergesort")
    rr = np.empty(len(allv))
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[s[j + 1]] == allv[s[i]]:
            j += 1
        rr[s[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    r_pos = rr[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--runs-dir", type=Path, default=Path(r"D:\computervision\runs"))
    ap.add_argument("--normal", type=Path, required=True, help="정상 영상 폴더")
    ap.add_argument("--positive", type=Path, default=None, help="양성 영상 폴더(선택)")
    ap.add_argument("--positive-class", default="theft")
    ap.add_argument("--sampling", choices=["uniform", "fps_pad", "both"], default="both")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--save-scores", action="store_true",
                    help="영상별 클래스 점수를 결과 JSON 에 함께 저장")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    run_dir = a.runs_dir / a.run
    dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev)
    cfg, model, state, ckpt = load_run(run_dir, device)
    if a.batch_size:
        cfg.set("eval.batch_size", a.batch_size)
    log = get_logger("eval_external", run_dir / "eval_external.log")
    classes = state.get("classes") or cfg.class_names
    alerting = [c for c in cfg.action_classes if c in classes]
    log.info("체크포인트 %s | 클래스 %s", ckpt.name, classes)
    log.info("알림 대상 %s", alerting)

    modes = ["uniform", "fps_pad"] if a.sampling == "both" else [a.sampling]
    result = {
        "run": a.run, "checkpoint": str(ckpt), "model_classes": classes,
        "alerting": alerting, "normal_dir": str(a.normal),
        "positive_dir": str(a.positive) if a.positive else None,
        "positive_class": a.positive_class if a.positive else None,
        "주의": "학습에 쓴 AI-Hub 데이터와 다른 출처의 영상이다. "
                "영상 길이가 학습 클립(5.33초)보다 짧아 샘플링 방식을 두 가지로 재서 함께 보고한다.",
        "modes": {},
    }

    for mode in modes:
        log.info("=== 샘플링 방식: %s", mode)
        n_names, n_probs = score_folder(a.normal, cfg, model, device, mode, log, a.limit)
        entry = {"normal": {"n": len(n_names)}}

        idx = {c: classes.index(c) for c in alerting}
        thresholds = {c: float(cfg.alert_params(c).get("threshold", 0.6)) for c in alerting}
        per_class = {}
        for c in alerting:
            s = n_probs[:, idx[c]] if len(n_probs) else np.zeros(0)
            th = thresholds[c]
            per_class[c] = {
                "임계값": th,
                "오탐_영상수": int((s >= th).sum()),
                "오탐률": round(float((s >= th).mean()), 4) if len(s) else None,
                "점수_평균": round(float(s.mean()), 4) if len(s) else None,
                "점수_p95": round(float(np.percentile(s, 95)), 4) if len(s) else None,
                "점수_최대": round(float(s.max()), 4) if len(s) else None,
            }
        # 하나라도 울린 영상
        if len(n_probs):
            any_fire = np.zeros(len(n_probs), dtype=bool)
            for c in alerting:
                any_fire |= n_probs[:, idx[c]] >= thresholds[c]
            entry["normal"]["어느_하나라도_알림"] = int(any_fire.sum())
            entry["normal"]["어느_하나라도_알림_비율"] = round(float(any_fire.mean()), 4)
            pred = n_probs.argmax(1)
            entry["normal"]["최빈_예측"] = {
                classes[i]: int((pred == i).sum()) for i in sorted(set(pred.tolist()))}
        entry["normal"]["클래스별"] = per_class

        # 임계값 스윕
        sweep = {}
        for c in alerting:
            s = n_probs[:, idx[c]] if len(n_probs) else np.zeros(0)
            sweep[c] = {f"{t:.1f}": round(float((s >= t).mean()), 4) for t in
                        np.arange(0.1, 1.0, 0.1)} if len(s) else {}
        entry["정상_오탐률_임계값_스윕"] = sweep

        if a.positive and a.positive.is_dir():
            p_names, p_probs = score_folder(a.positive, cfg, model, device, mode, log, a.limit)
            pc = a.positive_class
            entry["positive"] = {"n": len(p_names), "class": pc}
            if pc in classes and len(p_probs):
                pi = classes.index(pc)
                th = thresholds.get(pc, 0.6)
                entry["positive"]["임계값에서_탐지율"] = round(
                    float((p_probs[:, pi] >= th).mean()), 4)
                entry["positive"]["점수_평균"] = round(float(p_probs[:, pi].mean()), 4)
                entry["positive"]["AUC_정상대비"] = (
                    round(auc(p_probs[:, pi], n_probs[:, pi]), 4) if len(n_probs) else None)
                pred = p_probs.argmax(1)
                entry["positive"]["최빈_예측"] = {
                    classes[i]: int((pred == i).sum()) for i in sorted(set(pred.tolist()))}
            else:
                entry["positive"]["note"] = f"'{pc}' 는 이 모델의 클래스가 아니다"
        # 운영 곡선: 임계값을 바꿔 가며 (정상 오탐률, 양성 탐지율) 을 같이 본다.
        # 학습 데이터에서 정한 임계값이 다른 도메인에서도 맞는지 확인하는 용도다.
        if a.positive and "positive" in entry and a.positive_class in classes:
            pi = classes.index(a.positive_class)
            ns = n_probs[:, pi] if len(n_probs) else np.zeros(0)
            ps = p_probs[:, pi] if len(p_probs) else np.zeros(0)
            curve = []
            for t in np.arange(0.05, 0.96, 0.05):
                fa = float((ns >= t).mean()) if len(ns) else 0.0
                det = float((ps >= t).mean()) if len(ps) else 0.0
                curve.append({"임계값": round(float(t), 2),
                              "정상_오탐률": round(fa, 4),
                              "양성_탐지율": round(det, 4),
                              "탐지율-오탐률": round(det - fa, 4)})
            entry["운영곡선"] = curve
            best = max(curve, key=lambda r: r["탐지율-오탐률"])
            entry["최적_임계값(탐지율-오탐률 최대)"] = best

        if a.save_scores:
            entry["_scores"] = {
                "classes": classes,
                "normal": {n: [round(float(v), 5) for v in row]
                           for n, row in zip(n_names, n_probs)},
            }
            if a.positive and len(p_probs):
                entry["_scores"]["positive"] = {
                    n: [round(float(v), 5) for v in row]
                    for n, row in zip(p_names, p_probs)}

        result["modes"][mode] = entry

    out = a.out or (run_dir / "eval_external.json")
    write_json(out, result)

    # 사람이 읽는 요약
    print("\n" + "=" * 72)
    print(f"모델 {a.run} | 클래스 {classes}")
    for mode, e in result["modes"].items():
        print(f"\n--- 샘플링 {mode}")
        print(f"  정상 영상 {e['normal']['n']}개")
        for c, v in e["normal"]["클래스별"].items():
            print(f"    {c:8s} 임계값 {v['임계값']:.2f}  오탐 {v['오탐_영상수']:4d}개 "
                  f"({v['오탐률']:.1%})  점수 평균 {v['점수_평균']:.3f} 최대 {v['점수_최대']:.3f}")
        print(f"    → 하나라도 알림: {e['normal'].get('어느_하나라도_알림')}개 "
              f"({e['normal'].get('어느_하나라도_알림_비율', 0):.1%})")
        print(f"    최빈 예측: {e['normal'].get('최빈_예측')}")
        if "positive" in e and "임계값에서_탐지율" in e["positive"]:
            p = e["positive"]
            print(f"  양성 영상({p['class']}) {p['n']}개: 탐지율 {p['임계값에서_탐지율']:.1%}, "
                  f"점수 평균 {p['점수_평균']:.3f}, AUC {p['AUC_정상대비']}")
            print(f"    최빈 예측: {p.get('최빈_예측')}")
        if "운영곡선" in e:
            print(f"  운영 곡선 ({e['positive']['class']} 점수 기준)")
            print(f"    {'임계값':>6s} {'정상 오탐률':>10s} {'양성 탐지율':>10s}")
            for r in e["운영곡선"]:
                if round(r["임계값"] * 100) % 10 == 0 or r["임계값"] in (0.05, 0.15):
                    print(f"    {r['임계값']:6.2f} {r['정상_오탐률']:9.1%} {r['양성_탐지율']:9.1%}")
            b = e["최적_임계값(탐지율-오탐률 최대)"]
            print(f"    최적: 임계값 {b['임계값']:.2f} → 탐지율 {b['양성_탐지율']:.1%}, "
                  f"오탐률 {b['정상_오탐률']:.1%}")
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
