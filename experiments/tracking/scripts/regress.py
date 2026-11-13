"""회귀 검사 — 모든 촬영·모든 판정 방식을 한 번에 돌려 기준(baselines/)과 달라진 짝을 뽑는다.

    python scripts/regress.py check                       # 2·3차 전부, 기준과 비교
    python scripts/regress.py check --set risk.exit_hold_seconds=0        # 판정 단계 설정을 바꿔 보기
    python scripts/regress.py check --replay --set identity.late_merge_threshold=0.875   # 신원·손 뻗기 설정은 저장분으로 다시 계산
    python scripts/regress.py update                      # 지금 결과를 기준으로

- 기준: baselines/shoot<N>_<방식>.json — 짝별 판정·맞음·손님 수 (작은 파일, 커밋함).
  방식: 3차 v(영상만)·s(센서 L1,L2)·p(결제 라벨)·ps(센서+결제), 2차 v·p.
- 실행 기록: 기본 실행 폴더(3차 outputs/dual_eval_shoot3, 2차 outputs/dual_eval_pose3). --replay 는 그 폴더의 저장분(cache)으로
  outputs/regress_tmp_shoot<N> 에 관찰 층을 다시 계산한 뒤 거기서 판정한다 (docs/replay-cache.md).
- 종료 코드: 0 = 기준과 같거나 좋아짐, 1 = 나빠진 짝 있음(맞던 짝이 틀림·손님 수가 틀려짐), 2 = 검사가 불완전(빠진 짝·실패한 짝).
- update 는 결과가 불완전하면(실패·건너뜀·빈 결과·기준에 있던 짝이 빠짐) 기준을 쓰지 않는다. 일부러 짝을 뺄 때만 --allow-remove.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

BASELINES = ROOT / "baselines"
MODES = {
    3: {"v": {}, "s": {"sensor": True, "lanes": ("L1", "L2")}, "p": {"payments": True},
        "ps": {"sensor": True, "lanes": ("L1", "L2"), "payments": True}},
    2: {"v": {}, "p": {"payments": True}},
}
MODE_NAMES = {"v": "영상만", "s": "센서 L1·L2", "p": "결제 라벨", "ps": "센서+결제"}
JUDGE_SECTIONS = {"risk", "fusion"}
"""판정 단계만 다시 돌려도 반영되는 설정 절."""
REPLAY_SECTIONS = {"identity", "interaction", "detail_interaction", "zones"}
"""관찰 층 설정 — 저장분으로 다시 계산(--replay)해야 반영된다."""
KEEP = ("pair", "verdicts", "correct", "people", "people_ok", "no_verdict")


def slim(row: dict) -> dict:
    return {k: row.get(k) for k in KEEP}


def check_sets(sets: list[str], replay: bool, identity_check: bool) -> str | None:
    """--set 이 이 실행 방식으로 반영되는지. 문제가 있으면 안내 문장, 없으면 None."""
    for item in sets:
        key = item.split("=", 1)[0]
        section = key.split(".", 1)[0]
        if section in REPLAY_SECTIONS and not replay:
            return f"{key}: 관찰 층 설정은 판정만 다시 해서는 반영되지 않습니다 — --replay 를 붙이세요"
        if section not in JUDGE_SECTIONS | REPLAY_SECTIONS:
            return (f"{key}: 탐지기·추적기·선반 설정은 저장분으로 시험할 수 없습니다 — "
                    "dual_eval.py run 으로 직접 돌린 뒤 그 폴더를 --out 으로 판정하세요")
        if key.startswith("risk.swap_") and not identity_check:
            return f"{key}: 신원 뒤바뀜 확인은 --identity-check 를 켜야 판정에 들어갑니다"
    return None


def compare(base: list[dict], rows: list[dict]) -> dict:
    """기준과 이번 결과의 차이. 짝 집합이 다르면 missing/extra 에."""
    b = {r["pair"]: r for r in base}
    n = {r["pair"]: r for r in rows}
    both = [p for p in b if p in n]
    return {
        "score": (sum(bool(r["correct"]) for r in b.values()), sum(bool(r["correct"]) for r in n.values())),
        "people": (sum(r["people_ok"] is True for r in b.values()), sum(r["people_ok"] is True for r in n.values())),
        "better": [p for p in both if n[p]["correct"] and not b[p]["correct"]],
        "worse": [p for p in both if b[p]["correct"] and not n[p]["correct"]],
        "people_worse": [p for p in both if b[p]["people_ok"] is True and n[p]["people_ok"] is False],
        "people_better": [p for p in both if b[p]["people_ok"] is False and n[p]["people_ok"] is True],
        "changed": [p for p in both if b[p]["verdicts"] != n[p]["verdicts"] and b[p]["correct"] == n[p]["correct"]],
        "missing": [p for p in b if p not in n],
        "extra": [p for p in n if p not in b],
    }


def exit_code(diffs: list[dict], incomplete: bool) -> int:
    if incomplete or any(d["missing"] or d["extra"] for d in diffs):
        return 2
    if any(d["worse"] or d["people_worse"] for d in diffs):
        return 1
    return 0


def baseline_path(shoot: int, mode: str, root: Path = BASELINES) -> Path:
    return root / f"shoot{shoot}_{mode}.json"


def load_baseline(shoot: int, mode: str, root: Path = BASELINES) -> list[dict] | None:
    path = baseline_path(shoot, mode, root)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def write_baseline(shoot: int, mode: str, rows: list[dict], *, failed=(), skipped=(), allow_remove: bool = False,
                   root: Path = BASELINES) -> str | None:
    """기준을 쓴다. 불완전하면 쓰지 않고 이유를 돌려준다 (쓰면 None)."""
    if failed or skipped:
        return f"실패 {[f['pair'] if isinstance(f, dict) else f for f in failed]} · 건너뜀 {list(skipped)} — 기준을 쓰지 않음"
    if not rows:
        return "결과가 비었음 — 기준을 쓰지 않음"
    old = load_baseline(shoot, mode, root)
    if old is not None and not allow_remove:
        gone = sorted({r["pair"] for r in old} - {r["pair"] for r in rows})
        if gone:
            return f"기준에 있던 짝이 이번 결과에 없음 {gone} — 기준을 쓰지 않음 (일부러 뺀 것이면 --allow-remove)"
    path = baseline_path(shoot, mode, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([slim(r) for r in rows], ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return None


def run_mode(shoot: int, mode: str, out: Path | None, sets: list[str], identity_check: bool) -> dict:
    """그 촬영·방식으로 판정. {rows, failed, skipped}."""
    import dual_eval

    spec = MODES[shoot][mode]
    dual_eval.set_profile(shoot)
    if out is not None:
        dual_eval.OUT = out
    dual_eval.OVERRIDES = list(sets)
    dual_eval.SENSOR_LANES = spec.get("lanes")
    dual_eval.IDENTITY_CHECK = identity_check
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            dual_eval.judge(None, True, spec.get("sensor", False), f"regress_{mode}", False, spec.get("payments", False))
        last = dual_eval.LAST_JUDGE
        return {"rows": [slim(r) for r in last["rows"]], "failed": list(last["failed"]), "skipped": list(last["skipped"])}
    finally:
        dual_eval.OVERRIDES, dual_eval.SENSOR_LANES, dual_eval.IDENTITY_CHECK = [], None, False


def replay(shoot: int, sets: list[str]) -> tuple[Path, list[dict]]:
    """기본 실행 폴더의 저장분으로 관찰 층을 다시 계산. (쓴 폴더, 실패 목록)."""
    import dual_eval

    dual_eval.set_profile(shoot)
    source = dual_eval.OUT
    out = ROOT / "outputs" / f"regress_tmp_shoot{shoot}"
    dual_eval.OUT = out
    dual_eval.OVERRIDES = list(sets)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            dual_eval.run(None, replay_from=source)
        return out, list(dual_eval.LAST_RUN["failed"])
    finally:
        dual_eval.OVERRIDES = []


def describe(shoot: int, mode: str, d: dict) -> str:
    line = (f"{shoot}차 {MODE_NAMES[mode]:8s} 점수 {d['score'][0]} → {d['score'][1]}   "
            f"손님 수 {d['people'][0]} → {d['people'][1]}")
    for key, label in (("better", "좋아짐"), ("worse", "나빠짐"), ("people_better", "손님 수 고쳐짐"),
                       ("people_worse", "손님 수 틀려짐"), ("changed", "판정만 바뀜"), ("missing", "빠진 짝"), ("extra", "새 짝")):
        if d[key]:
            line += f"\n    {label}: {', '.join(d[key])}"
    return line


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["check", "update"])
    ap.add_argument("--shoot", type=int, nargs="+", choices=sorted(MODES), default=sorted(MODES))
    ap.add_argument("--set", action="append", default=[], metavar="절.키=값")
    ap.add_argument("--replay", action="store_true", help="저장분으로 관찰 층을 다시 계산한 뒤 판정 (신원·손 뻗기 설정 시험)")
    ap.add_argument("--identity-check", action="store_true", help="판정에 신원 뒤바뀜 확인을 넣는다 (기본 끔)")
    ap.add_argument("--allow-remove", action="store_true", help="update: 기준에 있던 짝이 빠져도 쓴다")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    problem = check_sets(args.set, args.replay, args.identity_check)
    if problem:
        print(problem)
        return 2
    if args.action == "update" and (args.set or args.replay or args.identity_check):
        print("update 는 기본 설정·기본 실행 폴더의 결과만 기준으로 씁니다 (--set·--replay·--identity-check 없이)")
        return 2
    diffs, incomplete = [], False
    for shoot in args.shoot:
        out = None
        if args.replay:
            out, failed = replay(shoot, args.set)
            if failed:
                incomplete = True
                print(f"{shoot}차 다시 계산 실패: " + "; ".join(f"{f['pair']} ({f['error']})" for f in failed[:5]))
        for mode in MODES[shoot]:
            result = run_mode(shoot, mode, out, args.set, args.identity_check)
            if result["failed"] or result["skipped"]:
                incomplete = True
                print(f"{shoot}차 {MODE_NAMES[mode]}: 실패 {[f['pair'] for f in result['failed']]} · 기록 없음 {result['skipped']}")
            if args.action == "update":
                why = write_baseline(shoot, mode, result["rows"], failed=result["failed"], skipped=result["skipped"],
                                     allow_remove=args.allow_remove)
                print(f"{shoot}차 {MODE_NAMES[mode]}: " + (why or f"기준 씀 ({sum(r['correct'] for r in result['rows'])}/{len(result['rows'])})"))
                incomplete = incomplete or bool(why)
                continue
            base = load_baseline(shoot, mode)
            if base is None:
                incomplete = True
                print(f"{shoot}차 {MODE_NAMES[mode]}: 기준 파일 없음 — 먼저 update")
                continue
            d = compare(base, result["rows"])
            diffs.append(d)
            print(describe(shoot, mode, d))
    code = exit_code(diffs, incomplete)
    if args.action == "check":
        print({0: "결과: 기준과 같거나 좋아짐", 1: "결과: 나빠진 짝 있음", 2: "결과: 검사가 불완전 (빠진 짝·실패)"}[code])
    return code


if __name__ == "__main__":
    raise SystemExit(main())
