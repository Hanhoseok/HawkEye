"""정상 행동 영상에서의 오경보 측정.

왜 필요한가
  이상행동 데이터(238-2)에는 정상 영상이 없다. 그래서 "정상 매장 영상에서 시간당 몇 번
  잘못 울리는가" 를 오랫동안 측정하지 못했다.
  구매행동 데이터(238-1)의 매장이동·선택·시험·구매·반품·비교는 **정상 매장 영상**이다.
  여기서 나오는 전도/파손/절도 알림은 전부 오경보이므로, 이 영상들로 그 수치를 잰다.

무엇을 재나
  실시간과 **같은 AlertEngine** 으로 알림을 만든 뒤, 알림 수 / 영상 시간 을 계산한다.
  정상 영상에는 이상행동 정답 사건이 없으므로 모든 알림이 오탐이다.

실측 기록 (이상행동만 학습한 baseline_r2p1d, test 분할 구매행동 133개 / 3.42시간)
    전도 0.0  파손 40.9  절도 12.0  → 합계 시간당 52.9건

PowerShell 예:
    D:\\computervision\\.venv\\Scripts\\python.exe -m storeguard.tools.false_alarm --run baseline_r2p1d
    D:\\computervision\\.venv\\Scripts\\python.exe -m storeguard.tools.false_alarm --run unified_r2p1d --compare baseline_r2p1d
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from storeguard.config import Config
from storeguard.data.dataset import load_index, load_splits
from storeguard.evaluate import event_level, load_run
from storeguard.utils import get_logger, write_json

# 정상으로 간주할 클래스 (구매행동 6종). 이 영상들에는 이상행동이 없다.
NORMAL_CLASSES = {"moving", "select", "test", "buying", "return", "compare"}


def measure(run_dir: Path, split: str, device: torch.device, log,
            limit: int = 0) -> dict | None:
    cfg, model, state, ckpt = load_run(run_dir, device)
    # 인덱스/캐시는 현재 것을 본다(체크포인트 설정의 경로는 학습 당시 것일 수 있다)
    cur = Config.load()
    for k in ("paths.index", "paths.cache", "paths.raw"):
        cfg.set(k, cur[k])

    rows = load_index(cfg)
    sp = load_splits(cfg)
    by_stem = {r["stem"]: r for r in rows}
    stems = [s for s in sp.get(split, [])
             if by_stem.get(s, {}).get("class_name") in NORMAL_CLASSES]
    if limit:
        stems = stems[:limit]
    if not stems:
        log.warning("%s 분할에 정상(구매행동) 영상이 없다.", split)
        return None

    log.info("%s: 정상 영상 %d개로 오경보 측정", run_dir.name, len(stems))
    res = event_level(cfg, model, device, stems, rows, bool(cfg["train.amp"]), log)
    return {
        "run": run_dir.name,
        "checkpoint": str(ckpt),
        "model_classes": state.get("classes"),
        "split": split,
        "n_videos": res["평가_영상수"],
        "hours": res["총_영상시간_시"],
        "false_alarms_per_hour": res["오경보_시간당_전체"],
        "per_class": {k: {"오탐": v["오탐(FP)"], "시간당": v["오경보_시간당"]}
                      for k, v in res["클래스별"].items()},
        "설명": "구매행동(정상) 영상에서 이상행동 모델이 낸 알림. 전부 오경보다.",
        "_detail": res,
    }


def show(d: dict) -> None:
    print(f"\n=== {d['run']}  ({d['split']} 분할)")
    print(f"    모델 클래스: {d['model_classes']}")
    print(f"    정상 영상 {d['n_videos']}개 / {d['hours']}시간")
    for k, v in d["per_class"].items():
        print(f"      {k:8s} 오탐 {v['오탐']:4d}건   시간당 {v['시간당']}")
    print(f"    합계 시간당 오경보: {d['false_alarms_per_hour']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="측정할 실행 이름")
    ap.add_argument("--compare", default=None, help="함께 비교할 이전 실행 이름")
    ap.add_argument("--runs-dir", type=Path, default=Path(r"D:\computervision\runs"))
    ap.add_argument("--split", choices=["test", "val", "train"], default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    a = ap.parse_args(argv)

    dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev)
    log = get_logger("false_alarm")

    results = []
    for name in ([a.run] + ([a.compare] if a.compare else [])):
        d = measure(a.runs_dir / name, a.split, device, log, a.limit)
        if d:
            results.append(d)
            out = a.runs_dir / name / f"false_alarm_normal_{a.split}.json"
            write_json(out, d)
            show(d)
            print(f"    저장: {out}")

    if len(results) == 2:
        new, old = results[0], results[1]
        print("\n=== 비교")
        print(f"    {old['run']:20s} 시간당 {old['false_alarms_per_hour']}")
        print(f"    {new['run']:20s} 시간당 {new['false_alarms_per_hour']}")
        o, n = old["false_alarms_per_hour"] or 0, new["false_alarms_per_hour"] or 0
        if o:
            print(f"    변화: {(n - o) / o * 100:+.1f}%")
    elif len(results) == 1 and not a.compare:
        print("\n참고: --compare <이전실행> 을 주면 개선폭을 함께 볼 수 있다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
