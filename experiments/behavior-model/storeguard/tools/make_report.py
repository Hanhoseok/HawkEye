"""학습·평가·실시간 측정 결과를 하나의 마크다운 리포트로 모은다.

측정하지 않은 항목은 '미측정'으로 남긴다. 값을 만들어 채우지 않는다.

PowerShell 예:
    D:\\computervision\\.venv\\Scripts\\python.exe -m storeguard.tools.make_report --run baseline_r2p1d
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

NA = "미측정"


def _load(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _fmt(v, nd=4):
    if v is None:
        return NA
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def clip_table(ev: dict, names: list[str]) -> str:
    if not ev or "per_class" not in ev:
        return f"({NA})\n"
    rows = ["| 클래스 | Precision | Recall | F1 | 지원(클립) |", "|---|---|---|---|---|"]
    for c in names:
        d = ev["per_class"].get(c)
        if not d:
            continue
        rows.append(f"| {c} | {_fmt(d['precision'])} | {_fmt(d['recall'])} "
                    f"| {_fmt(d['f1'])} | {d['support']} |")
    rows.append(f"\n정확도 **{_fmt(ev.get('accuracy'))}**, macro-F1 **{_fmt(ev.get('macro_f1'))}** "
                f"(클립 {ev.get('n_clips')}개 / 영상 {ev.get('n_videos')}개)")
    return "\n".join(rows) + "\n"


def cm_table(ev: dict) -> str:
    cm = (ev or {}).get("confusion_matrix")
    if not cm:
        return f"({NA})\n"
    labels = cm["labels"]
    head = "| 정답＼예측 | " + " | ".join(labels) + " |"
    sep = "|---" * (len(labels) + 1) + "|"
    rows = [head, sep]
    for i, lab in enumerate(labels):
        rows.append(f"| **{lab}** | " + " | ".join(str(x) for x in cm["matrix"][i]) + " |")
    return "\n".join(rows) + "\n"


def event_table(ev: dict) -> str:
    if not isinstance(ev, dict) or "클래스별" not in ev:
        return f"({NA})\n"
    rows = ["| 클래스 | 정답 사건 | 탐지(TP) | 사건 탐지율 | 오탐(FP) | 중복 알림 | 지연 중앙값(초) | 지연 p90(초) |",
            "|---|---|---|---|---|---|---|---|"]
    for c, d in ev["클래스별"].items():
        lat = d.get("지연_초", {})
        rows.append(
            f"| {c} | {d['정답_사건수']} | {d['탐지(TP)']} | {_fmt(d['사건단위_탐지율'])} "
            f"| {d['오탐(FP)']} | {d['중복알림']} | {_fmt(lat.get('median'), 2)} "
            f"| {_fmt(lat.get('p90'), 2)} |")
    return "\n".join(rows) + "\n"


def build(run_dir: Path, reports: Path, realtime: Path | None) -> str:
    hist = _load(run_dir / "history.json") or []
    best = _load(run_dir / "best_val.json")
    stats = _load(run_dir / "dataset_stats.json") or {}
    env = _load(run_dir / "env.json") or {}
    ev_test = _load(run_dir / "eval_test.json")
    ev_norm = _load(run_dir / "eval_normal.json")
    probes = sorted(reports.glob("vram_probe_*.json"))
    probe = _load(probes[0]) if probes else None
    rt = _load(realtime) if realtime else None

    names = (ev_test or {}).get("클립단위", {}).get("per_class", {})
    names = list(names) or ["background_unlabeled", "fall", "broken", "theft"]

    out = [f"# 실행 결과 리포트 — `{run_dir.name}`", ""]
    out.append("> `storeguard.tools.make_report` 자동 생성. 모든 수치는 실제 실행 산출물이다.")
    out.append("> 값이 `미측정` 이면 실행하지 않았다는 뜻이며, 추정치를 넣지 않는다.")
    out.append("")

    out.append("## 1. 실행 환경")
    out.append("")
    if env:
        out.append("| 항목 | 값 |")
        out.append("|---|---|")
        for k in ("python", "platform", "torch", "cuda_version", "gpu", "gpu_total_mem_gb", "opencv"):
            if k in env:
                out.append(f"| {k} | {env[k]} |")
    else:
        out.append(f"({NA})")
    out.append("")

    out.append("## 2. 데이터")
    out.append("")
    if stats:
        out.append("| 분할 | 클립 | 영상 | 클래스별 클립 |")
        out.append("|---|---|---|---|")
        for k, v in stats.items():
            out.append(f"| {k} | {v['clips']} | {v['videos']} | {v['class_counts']} |")
    else:
        out.append(f"({NA})")
    out.append("")

    out.append("## 3. 학습 곡선")
    out.append("")
    if hist:
        out.append("| epoch | train loss | train acc | val macro-F1 | val acc | 소요(초) |")
        out.append("|---|---|---|---|---|---|")
        for h in hist:
            out.append(f"| {h['epoch']} | {_fmt(h['train_loss'])} | {_fmt(h['train_acc'])} "
                       f"| {_fmt(h['val_macro_f1'])} | {_fmt(h['val_acc'])} | {h['sec']} |")
        if best:
            out.append("")
            out.append(f"best 체크포인트: epoch **{best.get('epoch')}**, "
                       f"val macro-F1 **{_fmt(best.get('macro_f1'))}**")
    else:
        out.append(f"({NA})")
    out.append("")

    out.append("## 4. 시험(test) 분할 — 클립 단위")
    out.append("")
    out.append("test 분할은 배포 Validation 세트다. 모델 선택·임계값 조정에 쓰지 않았다.")
    out.append("")
    out.append(clip_table((ev_test or {}).get("클립단위", {}), names))
    out.append("### 혼동행렬")
    out.append("")
    out.append(cm_table((ev_test or {}).get("클립단위", {})))

    out.append("## 5. 시험 분할 — 사건 단위")
    out.append("")
    ev = (ev_test or {}).get("사건단위")
    if isinstance(ev, dict):
        out.append(f"매칭 규칙: {ev.get('매칭규칙')}")
        out.append("")
        out.append(f"추론 간격 {ev.get('추론간격_초')}초, 평가 영상 {ev.get('평가_영상수')}개, "
                   f"총 {ev.get('총_영상시간_시')}시간")
        out.append("")
    out.append(event_table(ev))
    if isinstance(ev, dict):
        out.append(f"이상행동 영상 위에서의 시간당 오경보(전체): **{_fmt(ev.get('오경보_시간당_전체'), 2)}**")
        out.append("")
        out.append(f"> {ev.get('주의', '')}")
        out.append("")
        stores = ev.get("매장별")
        if stores:
            out.append("### 매장별 (편차 확인용)")
            out.append("")
            out.append("| 매장 | 영상 | 정답 사건 | 탐지 | 사건 탐지율 | 오탐 | 시간 | 시간당 오경보 |")
            out.append("|---|---|---|---|---|---|---|---|")
            for k, v in stores.items():
                out.append(f"| {k} | {v['영상수']} | {v['정답_사건수']} | {v['탐지']} "
                           f"| {_fmt(v['사건단위_탐지율'])} | {v['오탐']} | {v['시간']} "
                           f"| {_fmt(v['오경보_시간당'], 2)} |")
            out.append("")
            out.append(f"> {ev.get('매장별_주의', '')}")
            out.append("")

    out.append("## 6. 정상 영상 — 시간당 오경보")
    out.append("")
    if ev_norm and ev_norm.get("정상영상_오경보"):
        d = ev_norm["정상영상_오경보"]
        out.append(f"- 총 영상 시간: {d.get('총_영상시간_시')} 시간")
        out.append(f"- **시간당 오경보(전체): {_fmt(d.get('시간당_오경보_전체'), 2)}**")
        out.append(f"- 클래스별: {d.get('클래스별_시간당_오경보')}")
    else:
        out.append(f"**{NA}** — 이 데이터셋에는 정상 영상이 없다.")
        out.append("")
        out.append("측정하려면 정상 영상을 모아 다음을 실행한다.")
        out.append("")
        out.append("```powershell")
        out.append("python -m storeguard.data.add_normal --dir <정상영상폴더>")
        out.append("python -m storeguard.data.preprocess --workers 4")
        out.append(f"python -m storeguard.evaluate --run {run_dir.name} --split normal")
        out.append("```")
    out.append("")

    out.append("## 7. 하드웨어 실측 (VRAM / 처리량)")
    out.append("")
    if probe:
        out.append(f"모델 `{probe.get('arch')}` · 입력 `{probe.get('input')}` · "
                   f"AMP `{probe.get('amp')}` · 파라미터 {probe.get('params_M')}M")
        out.append("")
        out.append("| 배치 | 학습 peak VRAM(GB) | 학습 clips/s | 추론 peak VRAM(GB) | 추론 clips/s |")
        out.append("|---|---|---|---|---|")
        tr = {r["batch"]: r for r in probe.get("train_steps", []) if r.get("ok")}
        inf = {r["batch"]: r for r in probe.get("infer_steps", []) if r.get("ok")}
        for b in sorted(set(tr) | set(inf)):
            t = tr.get(b, {})
            i = inf.get(b, {})
            out.append(f"| {b} | {t.get('peak_gb', NA)} | {t.get('clips_per_sec', NA)} "
                       f"| {i.get('peak_gb', NA)} | {i.get('clips_per_sec', NA)} |")
        out.append("")
        out.append(f"실시간(배치 1) 추론 지연: **{probe.get('실시간_추론_1클립_지연_ms')} ms**")
    else:
        out.append(f"({NA})")
    out.append("")

    out.append("## 8. 실시간 파이프라인 실측")
    out.append("")
    if rt:
        src = rt.get("source", {})
        lat = rt.get("infer_latency_ms", {})
        out.append("| 항목 | 값 |")
        out.append("|---|---|")
        out.append(f"| 입력 | {src.get('kind')} — {src.get('url')} |")
        out.append(f"| 실제 수신 FPS | {src.get('measured_fps')} |")
        out.append(f"| 수신 프레임 / 폐기 | {src.get('frames_received')} / {src.get('frames_dropped')} |")
        out.append(f"| 재접속 횟수 | {src.get('reconnects')} |")
        out.append(f"| 추론 횟수 | {rt.get('inferences')} (초당 {rt.get('inferences_per_sec')}) |")
        out.append(f"| 추론 지연 평균 / p90 | {lat.get('mean')} ms / {lat.get('p90')} ms |")
        out.append(f"| 분석 상태 | {rt.get('analysis_state')} |")
    else:
        out.append(f"({NA}) — `storeguard.infer.runner --print` 의 최종 상태를 JSON 으로 저장해 넘길 것")
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--runs-dir", type=Path, default=Path(r"D:\computervision\runs"))
    ap.add_argument("--reports", type=Path, default=Path(r"D:\computervision\reports"))
    ap.add_argument("--realtime", type=Path, default=None,
                    help="실시간 실행 최종 상태 JSON 경로")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    run_dir = a.runs_dir / a.run
    text = build(run_dir, a.reports, a.realtime)
    out = a.out or (a.reports / f"results_{a.run}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
