"""평가 스크립트.

두 단계로 평가한다.

A. 클립 단위 (모델 자체 성능)
   클래스별 Precision / Recall / F1, 혼동행렬.

B. 사건 단위 (시스템 성능)
   각 시험 영상에 대해 실시간과 동일한 인과적 슬라이딩 윈도(anchor=창의 끝)로 추론하고,
   그 점수열을 실시간과 **같은 AlertEngine** 에 넣어 알림을 만든 뒤 정답 사건과 매칭한다.

   매칭 규칙 (명시)
     - 알림(클래스 c, 시각 t)은 같은 클래스의 정답 사건 [start-tol, end+tol] 안에 들어가면 매칭.
     - 정답 사건 하나에는 **가장 빠른 알림 1개만** TP 로 센다. 이후 매칭되는 알림은 '중복 알림'으로
       따로 집계하며 기본값에서는 FP 로도 TP 로도 세지 않는다(eval.count_duplicate_as_fp 로 변경 가능).
     - 어떤 정답 사건에도 매칭되지 않는 알림은 오탐(FP).
     - 탐지 지연 = (알림 시각) - (정답 사건 시작 시각). 음수면 0 으로 자르지 않고 그대로 기록한다.

   측정 불가 항목은 "미측정" 으로 남긴다. 특히 **정상 연속 영상의 시간당 오경보 수**는
   이 데이터셋에 정상 영상이 없으므로 --normal-dir 로 정상 영상을 주지 않으면 측정하지 않는다.

PowerShell 예:
    D:\\computervision\\.venv\\Scripts\\python.exe -m storeguard.evaluate --run baseline_r2p1d --split test
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from storeguard.config import Config, env_report
from storeguard.data.clips import clips_for_video, event_seconds
from storeguard.data.dataset import ClipDataset, load_index, load_splits
from storeguard.data.framestore import open_store, store_path
from storeguard.infer.alerting import AlertEngine
from storeguard.models.factory import build_model
from storeguard.utils import confusion_matrix, get_logger, macro_f1, prf_per_class, write_json


def load_run(run_dir: Path, device: torch.device):
    cfg = Config.load(run_dir / "config.yaml")
    ckpt = run_dir / "best.pt"
    if not ckpt.exists():
        ckpt = run_dir / "last.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"체크포인트 없음: {run_dir}")
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model = build_model(str(cfg["model.arch"]), len(cfg.class_names),
                        pretrained=False, dropout=float(cfg["model.dropout"]))
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return cfg, model, state, ckpt


@torch.no_grad()
def clip_level(cfg: Config, model, device, stems: list[str], rows: list[dict], amp: bool) -> dict:
    ds = ClipDataset(cfg, stems, train=False, rows=rows)
    loader = DataLoader(ds, batch_size=int(cfg["eval.batch_size"]), shuffle=False,
                        num_workers=int(cfg["train.num_workers"]), pin_memory=True)
    ys, ps = [], []
    t0 = time.perf_counter()
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            out = model(x)
        ps.append(torch.softmax(out.float(), 1).argmax(1).cpu().numpy())
        ys.append(y.numpy())
    dt = time.perf_counter() - t0
    if not ys:
        return {"n": 0, "note": "클립 0개"}
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    cm = confusion_matrix(y, p, len(cfg.class_names))
    per = prf_per_class(cm)
    return {
        "n_clips": int(len(y)), "n_videos": len({s.stem for s in ds.specs}),
        "accuracy": round(float((y == p).mean()), 4),
        "macro_f1": macro_f1(cm),
        "per_class": {c: per[i] for i, c in enumerate(cfg.class_names)},
        "confusion_matrix": {"labels": cfg.class_names, "matrix": cm.tolist(),
                             "설명": "행=정답, 열=예측"},
        "clip_counts": ds.class_counts(),
        "elapsed_sec": round(dt, 2),
        "clips_per_sec": round(len(y) / dt, 2) if dt else None,
    }


@torch.no_grad()
def score_video(cfg: Config, model, device, stem: str, row: dict, amp: bool,
                every: int) -> tuple[list[int], np.ndarray, int]:
    """영상 전체를 인과적 슬라이딩 윈도로 추론. (anchor 목록, (N, C) 확률, 캐시 프레임 수)"""
    store = open_store(cfg.path("paths.cache"), stem)
    T = store.n_frames
    L = int(cfg["data.clip_len"])
    s = int(cfg["data.frame_stride"])
    span = (L - 1) * s
    crop = int(cfg["data.eval_crop"])
    mean = np.array(cfg["data.mean"], dtype=np.float32)
    std = np.array(cfg["data.std"], dtype=np.float32)
    probe = store.read([0])
    H, W = probe.shape[1], probe.shape[2]
    y0 = max(0, (H - crop) // 2)
    x0 = max(0, (W - crop) // 2)

    anchors = list(range(span, T, max(1, every)))
    if not anchors:
        return [], np.zeros((0, len(cfg.class_names)), dtype=np.float32), T

    probs = []
    bs = int(cfg["eval.batch_size"])
    for i in range(0, len(anchors), bs):
        batch = []
        for a in anchors[i:i + bs]:
            idxs = [a - span + k * s for k in range(L)]
            clip = store.read(idxs)[:, y0:y0 + crop, x0:x0 + crop, :]
            clip = (clip.astype(np.float32) / 255.0 - mean) / std
            batch.append(clip.transpose(3, 0, 1, 2))
        x = torch.from_numpy(np.ascontiguousarray(np.stack(batch))).to(device)
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            out = model(x)
        probs.append(torch.softmax(out.float(), 1).cpu().numpy())
    return anchors, np.concatenate(probs), T


def event_level(cfg: Config, model, device, stems: list[str], rows: list[dict],
                amp: bool, log) -> dict:
    by_stem = {r["stem"]: r for r in rows}
    fps = float(cfg["data.sample_fps"])
    every = int(cfg["runtime.infer_every_frames"])
    tol = float(cfg["eval.event_match_tolerance_sec"])
    actions = cfg.action_classes
    params = {a: cfg.alert_params(a) for a in actions}
    dup_as_fp = bool(cfg.get("eval.count_duplicate_as_fp", False))

    gt_total = defaultdict(int)
    tp = defaultdict(int)
    fp = defaultdict(int)
    dup = defaultdict(int)
    latencies = defaultdict(list)
    total_seconds = 0.0
    per_video = []
    missing_cache = []

    for n, stem in enumerate(stems):
        row = by_stem.get(stem)
        if row is None:
            continue
        if store_path(cfg.path("paths.cache"), stem) is None:
            missing_cache.append(stem)
            continue
        anchors, probs, T = score_video(cfg, model, device, stem, row, amp, every)
        if not anchors:
            continue
        # 캐시는 sample_fps 로 리샘플되어 있으므로 캐시 길이가 곧 시간 기준이다.
        total_seconds += T / fps

        # 실시간과 동일한 AlertEngine. 시계는 영상 내 시각으로 대체한다.
        media_now = {"t": 0.0}
        eng = AlertEngine(stem, actions, params, clock=lambda: media_now["t"])
        alerts = []
        for a, p in zip(anchors, probs):
            media_now["t"] = a / fps
            scores = {c: float(p[cfg.class_to_idx[c]]) for c in actions}
            started, _ended = eng.step(scores, media_now["t"], now=media_now["t"])
            for ev in started:
                alerts.append({"action": ev.action, "t": ev.started_media_ts,
                               "score": ev.score})

        gts = []
        for i, e in enumerate(row.get("events", [])):
            # 라벨 프레임은 원본 fps 좌표다. 캐시(sample_fps) 좌표로 바꾼 뒤 초로 환산한다.
            s, en = event_seconds(row, e, T, fps)
            gts.append({"action": e["action"], "start": s, "end": en, "matched": False})
            gt_total[e["action"]] += 1

        v_tp = v_fp = v_dup = 0
        for al in sorted(alerts, key=lambda x: x["t"]):
            cand = [g for g in gts if g["action"] == al["action"]
                    and (g["start"] - tol) <= al["t"] <= (g["end"] + tol)]
            if not cand:
                fp[al["action"]] += 1
                v_fp += 1
                continue
            unmatched = [g for g in cand if not g["matched"]]
            if unmatched:
                g = min(unmatched, key=lambda g: abs(al["t"] - g["start"]))
                g["matched"] = True
                tp[al["action"]] += 1
                v_tp += 1
                latencies[al["action"]].append(round(al["t"] - g["start"], 3))
            else:
                dup[al["action"]] += 1
                v_dup += 1
                if dup_as_fp:
                    fp[al["action"]] += 1

        per_video.append({"stem": stem, "class": row["class_name"], "store": row.get("store"),
                          "camera": row.get("camera"), "n_gt": len(gts), "seconds": T / fps,
                          "n_alerts": len(alerts), "tp": v_tp, "fp": v_fp, "dup": v_dup})
        if (n + 1) % 25 == 0:
            log.info("사건 평가 %d/%d", n + 1, len(stems))

    # 매장별 집계 — "다른 매장에서도 같은 성능"을 가정하지 않기 위해 편차를 드러낸다.
    by_store: dict[str, dict] = defaultdict(
        lambda: {"영상수": 0, "정답_사건수": 0, "탐지": 0, "오탐": 0, "초": 0.0})
    for v in per_video:
        s = by_store[v.get("store") or "?"]
        s["영상수"] += 1
        s["정답_사건수"] += v["n_gt"]
        s["탐지"] += v["tp"]
        s["오탐"] += v["fp"]
        s["초"] += v.get("seconds", 0.0)
    store_summary = {}
    for k, v in sorted(by_store.items()):
        h = v["초"] / 3600.0
        store_summary[k] = {
            "영상수": v["영상수"], "정답_사건수": v["정답_사건수"], "탐지": v["탐지"],
            "사건단위_탐지율": round(v["탐지"] / v["정답_사건수"], 4) if v["정답_사건수"] else None,
            "오탐": v["오탐"], "시간": round(h, 3),
            "오경보_시간당": round(v["오탐"] / h, 2) if h else None,
        }

    hours = total_seconds / 3600.0
    summary = {}
    for a in actions:
        g = gt_total[a]
        summary[a] = {
            "정답_사건수": g,
            "탐지(TP)": tp[a],
            "사건단위_탐지율": round(tp[a] / g, 4) if g else None,
            "오탐(FP)": fp[a],
            "중복알림": dup[a],
            "지연_초": {
                "n": len(latencies[a]),
                "median": round(float(np.median(latencies[a])), 2) if latencies[a] else None,
                "mean": round(float(np.mean(latencies[a])), 2) if latencies[a] else None,
                "p90": round(float(np.percentile(latencies[a], 90)), 2) if latencies[a] else None,
                "max": round(float(np.max(latencies[a])), 2) if latencies[a] else None,
            },
            "오경보_시간당": round(fp[a] / hours, 2) if hours else None,
        }
    return {
        "매칭규칙": f"같은 클래스, 알림 시각이 [start-{tol}s, end+{tol}s] 안. 사건당 최초 1건만 TP.",
        "추론간격_프레임": every,
        "추론간격_초": round(every / fps, 3),
        "평가_영상수": len(per_video),
        "총_영상시간_초": round(total_seconds, 1),
        "총_영상시간_시": round(hours, 3),
        "클래스별": summary,
        "오경보_시간당_전체": round(sum(fp.values()) / hours, 2) if hours else None,
        "주의": "여기서 센 오경보는 '연출된 이상행동 영상' 위의 오경보다. 이 영상들은 "
                "대부분의 시간에 행동 직전의 접근·물색 동작이 이어지므로 실제 정상 매장보다 어렵다. "
                "진짜 정상 매장 영상의 시간당 오경보는 storeguard.data.add_normal 로 정상 영상을 "
                "등록한 뒤 'evaluate --split normal' 로 측정해야 한다.",
        "매장별": store_summary,
        "매장별_주의": "이 매장들은 모두 학습에도 등장한다. 완전히 새로운 매장 성능이 아니다. "
                       "매장 분리 평가는 split.mode=store 로 다시 학습해야 한다.",
        "캐시_없어서_건너뛴_영상": missing_cache[:20],
        "영상별": per_video,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="runs/ 아래 실행 이름")
    ap.add_argument("--runs-dir", type=Path, default=Path(r"D:\computervision\runs"))
    ap.add_argument("--split", choices=["val", "test", "normal"], default="test",
                    help="normal 은 add_normal.py 로 등록한 정상 영상. "
                         "정답 사건이 0 이므로 모든 알림이 오탐이 되어 시간당 오경보를 실제로 측정한다")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-event", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="학습이 GPU 를 쓰는 중이면 cpu 로 돌려 VRAM 충돌을 피할 수 있다")
    # 전처리·모델 설정은 체크포인트 옆 config.yaml 을 그대로 따른다(재현성).
    # 실행 자원에만 관계된 값은 덮어쓸 수 있게 둔다.
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=-1)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    run_dir = a.runs_dir / a.run
    dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev)
    cfg, model, state, ckpt = load_run(run_dir, device)
    if a.batch_size:
        cfg.set("eval.batch_size", a.batch_size)
    if a.num_workers >= 0:
        cfg.set("train.num_workers", a.num_workers)
    log = get_logger("eval", run_dir / "eval.log")
    log.info("체크포인트=%s (epoch %s, best=%s)", ckpt.name, state.get("epoch"), state.get("best_metric"))

    rows = load_index(cfg)
    splits = load_splits(cfg)
    if a.split not in splits:
        print(f"'{a.split}' 분할이 splits.json 에 없다. "
              f"정상 영상을 먼저 등록할 것: storeguard.data.add_normal --dir <폴더>")
        return 2
    stems = splits[a.split]
    if a.limit:
        stems = stems[: a.limit]
    amp = bool(cfg["train.amp"])

    result = {
        "run": a.run, "split": a.split, "checkpoint": str(ckpt),
        "checkpoint_epoch": state.get("epoch"), "best_metric": state.get("best_metric"),
        "n_videos_요청": len(stems),
        "env": env_report(),
        "클립단위": clip_level(cfg, model, device, stems, rows, amp),
    }
    if not a.skip_event:
        result["사건단위"] = event_level(cfg, model, device, stems, rows, amp, log)
    else:
        result["사건단위"] = "미측정 (--skip-event)"

    if a.split == "normal":
        ev = result.get("사건단위") or {}
        result["정상영상_오경보"] = {
            "설명": "사람이 정상이라고 확인한 영상에서 발생한 알림. 전부 오경보다.",
            "총_영상시간_시": ev.get("총_영상시간_시"),
            "시간당_오경보_전체": ev.get("오경보_시간당_전체"),
            "클래스별_시간당_오경보": {
                k: v.get("오경보_시간당") for k, v in (ev.get("클래스별") or {}).items()},
        }
        result["미측정_항목"] = ["새 매장 추가 학습 전후 비교 (직접 촬영 데이터가 있으면 측정 가능)"]
    else:
        result["미측정_항목"] = [
            "정상 매장 연속 영상에서의 시간당 오경보 "
            "(정상 영상 없음. storeguard.data.add_normal 로 등록 후 --split normal 로 측정)",
            "새 매장 추가 학습 전후 비교 (직접 촬영 데이터 없음)",
        ]

    out = a.out or (run_dir / f"eval_{a.split}.json")
    write_json(out, result)
    print(json.dumps({k: v for k, v in result.items() if k != "env"},
                     ensure_ascii=False, indent=2)[:6000])
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
