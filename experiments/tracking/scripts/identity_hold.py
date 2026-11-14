"""신원 보류 두 가지(후속 45)와 관절로 가르기(후속 46)를 저장분으로 켜고 끄며 견준다.

    python scripts/identity_hold.py --shoot 3 grid                      # 기준 후보들을 한 번에 (정상 N 짝 / 전체 따로 집계)
    python scripts/identity_hold.py --shoot 3 try nested=0.7 score=0.4   # 한 조합을 짝별로 자세히 (전체 카메라에만)
    python scripts/identity_hold.py --shoot 3 try nested=0.7 both=1      # 두 카메라 모두에
    python scripts/identity_hold.py --shoot 3 who C03                    # 그 짝 전체 카메라의 신원 흐름 (끔 → 켬)
    python scripts/identity_hold.py --shoot 3 links nested=0.7           # 두 화면 짝이 달라지나 (짝 있음 · 끊김 · 짝 바뀜 · 손님 아닌 번호와 맺힘)
    python scripts/identity_hold.py --shoot 3 held nested=0.7            # 포개져 보류된 상자마다: 얼마나 보류됐고 그 뒤 무엇이 됐나
    python scripts/identity_hold.py --shoot 3 gaps                       # 포개진 두 상자의 관절이 얼마나 떨어져 있나 (관절 기준의 근거)
    python scripts/identity_hold.py --shoot 3 posegrid                   # 포개짐 × 관절 기준 후보를 한 번에 (후속 46)
    python scripts/identity_hold.py --shoot 3 try nested=0.7 pose=0.5    # 포개짐 보류 + 관절이 0.5 넘게 떨어지면 다른 사람

- nested = identity.nested_hold_overlap (이미 신원이 있는 상자와 포개진 상자에는 새 신원을 만들지 않고 보류)
- score  = identity.min_identity_score (탐지 점수가 낮은 상자로는 신원을 만들지도 되살리지도 않고 보류)
- pose   = identity.nested_pose_gap (포개져 있어도 두 상자의 공통 관절이 이만큼 넘게 떨어져 있으면 다른 사람 — 보류하지 않음)
- 저장분(`dual_eval.py run --record-cache`)이 기본 실행 폴더에 있어야 한다. 다시 계산한 기록은 outputs/hold_<이름>_shoot<N>.
- 견주는 것: 판정(영상만·센서·결제 방식 모두) · 손님 수 · 전체 카메라 신원 수 · 보류한 줄 수(이유별) · 신원이 붙은 줄 수.
- 기준은 정상(N) 짝만 보고 고른다 (한계 A-15). grid 는 N 짝과 전체를 따로 적는다.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import dual_eval  # noqa: E402
import regress  # noqa: E402
from src.identity.registry import IdentityRegistry  # noqa: E402

KEYS = {"nested": "nested_hold_overlap", "score": "min_identity_score", "pose": "nested_pose_gap"}
VALUES = [{"nested": 0.5}, {"nested": 0.7}, {"nested": 0.9}, {"score": 0.3}, {"score": 0.35}, {"score": 0.4}, {"score": 0.5}]
GRID = [{}] + VALUES + [{**v, "both": 1.0} for v in VALUES]
"""both 가 없으면 전체 카메라에만, 있으면 두 카메라 모두에 건다."""
POSE_GRID = [{}] + [{"nested": n} for n in (0.5, 0.7, 0.9)] + [{"nested": n, "pose": g} for n in (0.5, 0.7, 0.9) for g in (0.3, 0.5, 0.8)]
"""포개짐만 / 포개짐 + 관절 기준 (후속 46)."""


def name_of(combo: dict) -> str:
    parts = [f"{k}{str(v).replace('.', '')}" for k, v in sorted(combo.items()) if k != "both"]
    return "_".join(parts + (["both"] if combo.get("both") else [])) or "off"


def sets_of(combo: dict) -> list[str]:
    """--set 목록. 두 값을 늘 적는다 — dual_eval 의 기본이 무엇이든 '끔'은 끔이고, 조합에 없는 값은 0 이다.
    전체 카메라에만 걸 때는 여기서는 둘 다 0 으로 두고(선반 카메라 몫) 값은 dual_eval.OVERVIEW_IDENTITY 로 건다."""
    both = bool(combo.get("both"))
    return [f"identity.{key}={combo.get(k, 0.0) if both else 0.0}" for k, key in KEYS.items()]


def overview_of(combo: dict) -> dict:
    """전체 카메라의 신원 설정에 덮어쓸 값 (카메라별 값이 --set 보다 나중에 먹는다)."""
    return {key: combo.get(k, 0.0) for k, key in KEYS.items()}


def replay(shoot: int, combo: dict, only: str | None = None) -> Path:
    """그 조합으로 관찰 층을 다시 계산한 폴더. only: 그 짝만."""
    dual_eval.set_profile(shoot)
    source = dual_eval.OUT
    out = ROOT / "outputs" / f"hold_{name_of(combo)}_shoot{shoot}"
    dual_eval.OUT, dual_eval.OVERRIDES, dual_eval.OVERVIEW_IDENTITY = out, sets_of(combo), overview_of(combo)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            dual_eval.run(only, replay_from=source)
        failed = list(dual_eval.LAST_RUN["failed"])
        if failed:
            raise SystemExit("다시 계산 실패: " + "; ".join(f"{f['pair']} ({f['error']})" for f in failed[:5]))
        return out
    finally:
        dual_eval.OVERRIDES, dual_eval.OVERVIEW_IDENTITY = [], {}


def identity_stats(out: Path, pair: str, combo: dict) -> dict:
    """전체 카메라 기록에서: 신원 수, 신원이 붙은 줄, 보류 줄(이유별)."""
    by_frame = defaultdict(list)
    path = out / pair / "angle1" / "identities.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        by_frame[r["frame"]].append(r)
    people, rows, held = set(), 0, {"score": 0, "nested": 0, "other": 0}
    score, nested = combo.get("score", 0.0), combo.get("nested", 0.0)
    for frame_rows in by_frame.values():
        named = [r["bbox"] for r in frame_rows if r["person_id"] > 0]
        for r in frame_rows:
            if r["person_id"] > 0:
                people.add(r["person_id"])
                rows += 1
            elif score and r["score"] < score:
                held["score"] += 1
            elif nested and any(IdentityRegistry._nested(r["bbox"], b) >= nested for b in named):   # 관절 기준은 안 봄 (어림)
                held["nested"] += 1
            else:
                held["other"] += 1
    return {"identities": len(people), "rows": rows, "held": held}


def measure(shoot: int, combo: dict) -> dict:
    """{방식: {짝: 판정 줄}} 과 {짝: 신원 통계}."""
    out = replay(shoot, combo)
    modes = {}
    for mode in regress.MODES[shoot]:
        result = regress.run_mode(shoot, mode, out, sets_of(combo), False)
        if result["failed"] or result["skipped"]:
            raise SystemExit(f"{mode}: 실패 {result['failed']} · 기록 없음 {result['skipped']}")
        modes[mode] = {r["pair"]: r for r in result["rows"]}
    pairs = list(next(iter(modes.values())))
    return {"modes": modes, "ids": {p: identity_stats(out, p, combo) for p in pairs}, "pairs": pairs, "out": out}


def total(m: dict, pairs: list[str]) -> dict:
    ids = [m["ids"][p] for p in pairs]
    return {
        "score": {mode: sum(bool(rows[p]["correct"]) for p in pairs) for mode, rows in m["modes"].items()},
        "people": sum(m["modes"]["v"][p]["people_ok"] is True for p in pairs),
        "identities": sum(i["identities"] for i in ids),
        "rows": sum(i["rows"] for i in ids),
        "held": {k: sum(i["held"][k] for i in ids) for k in ("score", "nested", "other")},
    }


def line(label: str, t: dict, n: int) -> str:
    scores = " ".join(f"{regress.MODE_NAMES[k]} {v}" for k, v in t["score"].items())
    h = t["held"]
    return (f"  {label:22s} 판정 [{scores}] / {n}   손님 수 {t['people']}/{n}   신원 {t['identities']}   "
            f"신원 붙은 줄 {t['rows']}   보류 줄: 점수 {h['score']} · 포개짐 {h['nested']} · 그 밖 {h['other']}")


def diff_pairs(base: dict, now: dict) -> list[str]:
    out = []
    for p in base["pairs"]:
        notes = []
        for mode, rows in now["modes"].items():
            b, n = base["modes"][mode][p], rows[p]
            if b["correct"] != n["correct"] or b["verdicts"] != n["verdicts"]:
                mark = "맞음→틀림" if b["correct"] and not n["correct"] else "틀림→맞음" if n["correct"] and not b["correct"] else "판정만"
                notes.append(f"{regress.MODE_NAMES[mode]} {mark} {b['verdicts']}→{n['verdicts']}")
        b, n = base["modes"]["v"][p], now["modes"]["v"][p]
        if b["people"] != n["people"]:
            ok = "맞음" if n["people_ok"] else "틀림" if n["people_ok"] is False else "-"
            notes.append(f"손님 수 {b['people']}→{n['people']}({ok})")
        bi, ni = base["ids"][p], now["ids"][p]
        if bi["identities"] != ni["identities"]:
            notes.append(f"신원 {bi['identities']}→{ni['identities']}")
        if notes:
            out.append(f"    {p:6s} " + " · ".join(notes))
    return out


def grid(shoot: int, combos: list[dict] = GRID) -> None:
    results = [(combo, measure(shoot, combo)) for combo in combos]
    base = results[0][1]
    normal = [p for p in base["pairs"] if p.startswith("N")]
    for title, pairs in (("정상(N) 짝 — 기준은 여기서 고른다", normal), ("전체", base["pairs"])):
        print(f"{shoot}차 {title} ({len(pairs)}짝)")
        for combo, m in results:
            print(line(name_of(combo), total(m, pairs), len(pairs)))
    for combo, m in results[1:]:
        changed = diff_pairs(base, m)
        print(f"\n{name_of(combo)} 에서 달라진 짝 ({len(changed)})")
        print("\n".join(changed) if changed else "    없음")


def try_one(shoot: int, combo: dict) -> None:
    base, now = measure(shoot, {}), measure(shoot, combo)
    normal = [p for p in base["pairs"] if p.startswith("N")]
    for title, pairs in (("정상(N)", normal), ("전체", base["pairs"])):
        print(f"{shoot}차 {title} ({len(pairs)}짝)")
        print(line("끔", total(base, pairs), len(pairs)))
        print(line(name_of(combo), total(now, pairs), len(pairs)))
    changed = diff_pairs(base, now)
    print(f"\n달라진 짝 ({len(changed)})")
    print("\n".join(changed) if changed else "    없음")
    print(f"\n기록: {now['out']}")


def who(shoot: int, pair: str, combo: dict) -> None:
    """그 짝 전체 카메라에서 신원이 정해지거나 바뀐 순간 (끔 / 켬)."""
    for label, c in (("끔", {}), (name_of(combo), combo)):
        out = replay(shoot, c, only=pair)
        meta = json.loads((out / pair / "meta.json").read_text(encoding="utf-8"))
        fps = meta["angle1"]["fps"]
        print(f"[{label}] {pair} 전체 카메라 — 추적 번호의 신원이 바뀐 순간")
        last: dict[int, int] = {}
        for text in (out / pair / "angle1" / "identities.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(text)
            t, p = r["track_id"], r["person_id"]
            if last.get(t) == p:
                continue
            sim = r.get("matched_similarity")
            how = "새 신원" if r.get("is_new") else "보류" if p < 0 else f"대조 {sim:.2f}" + (" 위치로 이음" if r.get("stitched") else "") if sim is not None else "이어짐"
            box = ",".join(str(int(v)) for v in r["bbox"])
            print(f"  {r['frame'] / fps:6.2f}초  추적 {t:<3d} 신원 {last.get(t, '-')}→{p:<3d} 탐지 {r['score']:.2f}  {how}  [{box}]")
            last[t] = p


def links(shoot: int, combo: dict) -> None:
    """두 화면 짝(선반 카메라 사람 × 0.5초 조각)이 그 조합에서 달라지나. link_diagnose.py gate 와 같은 셈."""
    import link_diagnose as ld
    from src.core.config import AppConfig
    from src.fusion import judge as jd

    orig_finish = jd.DualJudge.finish
    print(f"{shoot}차 — 조각 = 선반 카메라 사람 × 0.5초")
    print(f"  {'':24s} {'조각':>5s} {'짝 있음':>7s} {'끊김':>4s} {'짝 바뀜':>6s} {'손님 아닌 번호와 맺힘':>12s}")
    kept: dict[str, dict] = {}
    # 다시 계산을 먼저 다 해 둔다 — 기록기를 단 채로 다시 계산하면 그 판정의 조각까지 앞 기록기에 섞여 들어간다
    runs = [(label, replay(shoot, c)) for label, c in (("끔", {}), (name_of(combo), combo))]
    orig_update = ld.cv.CrossViewLinker.update
    for label, out in runs:
        dual_eval.set_profile(shoot)
        dual_eval.OUT = out
        rec, seen = ld.Recorder(), {}
        rec.install()

        def finish(judge, frame, resolve=None, rec=rec, seen=seen):
            done = orig_finish(judge, frame, resolve)
            seen[rec.pair] = {pid for pid, cust in judge.risk.book.customers.items() if cust.seen_inside}
            return done

        jd.DualJudge.finish = finish
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ld._judge_all(rec, AppConfig.load(ROOT / "config.yaml"))
        finally:
            jd.DualJudge.finish = orig_finish
            ld.cv.CrossViewLinker.update = orig_update
            dual_eval.set_profile(shoot)
        for name, keep in (("정상(N)", lambda p: p.startswith("N")), ("전체", lambda p: True)):
            part = ld.Recorder()
            part.rows = [r for r in rec.rows if keep(r["pair"])]
            ghost = sum(1 for r in part.rows if r["o"] is not None and r["o"] not in seen[r["pair"]])
            n_breaks, n_switch, _ = part.breaks()
            print(f"  {label:16s} {name:7s} {len(part.rows):5d} {part.linked_pct():6.1f}% {n_breaks:4d} {n_switch:6d} {ghost:12d}")
        kept[label] = {"rec": rec, "seen": seen}
    (_, a), (_, b) = kept.items()
    for pair in dict.fromkeys(r["pair"] for r in a["rec"].rows):
        def summary(k):
            part = ld.Recorder()
            part.rows = [r for r in k["rec"].rows if r["pair"] == pair]
            ghost = sum(1 for r in part.rows if r["o"] is not None and r["o"] not in k["seen"][pair])
            n_breaks, n_switch, _ = part.breaks()
            return (len(part.rows), sum(r["reason"] == "linked" for r in part.rows), n_breaks, n_switch, ghost)
        x, y = summary(a), summary(b)
        if x != y:
            print(f"    {pair:6s} 조각 {x[0]}→{y[0]} · 짝 있음 {x[1]}→{y[1]} · 끊김 {x[2]}→{y[2]} · 짝 바뀜 {x[3]}→{y[3]} · 손님 아닌 번호 {x[4]}→{y[4]}")


def held(shoot: int, combo: dict) -> None:
    """전체 카메라에서 포개져 보류된 상자(추적 번호)마다: 언제부터 몇 줄, 그 뒤 어떤 신원이 됐나.

    '포개진 그 신원으로 이어짐' = 같은 사람 위의 두 번째 상자였을 가능성이 크다(또는 두 사람이 한 신원으로 합쳐졌다 — 기록만으로는 못 가른다).
    '나중에 새 신원' = 다른 사람이었고 신원이 그만큼 늦게 붙었다. 보류가 긴 줄은 진짜 사람이 신원 없이 서 있었을 수 있다."""
    nested = combo.get("nested", 0.0)
    if not nested:
        raise SystemExit("held nested=0.7 처럼 포개짐 기준을 준다")
    out = replay(shoot, combo)
    kinds: dict[str, int] = defaultdict(int)
    waits: list[float] = []
    for pair in dual_eval.pairs(None):
        path = out / pair / "angle1" / "identities.jsonl"
        if not path.exists():
            continue
        fps = json.loads((out / pair / "meta.json").read_text(encoding="utf-8"))["angle1"]["fps"]
        by_frame = defaultdict(list)
        for text in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(text)
            by_frame[r["frame"]].append(r)
        runs: dict[int, list] = defaultdict(list)
        for f in sorted(by_frame):
            named = [(r["bbox"], r["person_id"]) for r in by_frame[f] if r["person_id"] > 0]
            for r in by_frame[f]:
                host = [pid for b, pid in named if IdentityRegistry._nested(r["bbox"], b) >= nested]
                if r["person_id"] < 0 and host:
                    runs[r["track_id"]].append((f, host[0], r["score"]))
        for t, rows in runs.items():
            first, host = rows[0][0], rows[0][1]
            after = next((r for f in sorted(by_frame) if f > first for r in by_frame[f]
                          if r["track_id"] == t and r["person_id"] > 0), None)
            kind = ("끝까지 신원 없음" if after is None else "포개진 그 신원으로 이어짐" if after["person_id"] == host
                    else "나중에 새 신원" if after.get("is_new") else "다른 기존 신원으로 이어짐")
            wait = (after["frame"] - first) / fps if after else None
            kinds[kind] += 1
            if wait is not None:
                waits.append(wait)
            tail = f" → {wait:.2f}초 뒤 신원 {after['person_id']}" if after else ""
            print(f"  {pair:6s} 추적 {t:<3d} {first / fps:6.2f}초 · 포개져 보류된 줄 {len(rows)} (포개진 신원 {host}, "
                  f"탐지 {min(s for *_, s in rows):.2f}~{max(s for *_, s in rows):.2f}) {kind}{tail}")
    waits.sort()
    print(f"{shoot}차 {name_of(combo)}: 보류된 추적 번호 {sum(kinds.values())}개 — " + " · ".join(f"{k} {v}" for k, v in kinds.items()))
    if waits:
        print(f"  첫 보류부터 신원이 붙기까지: 중앙 {waits[len(waits) // 2]:.2f}초 · 1초 넘게 {sum(w > 1.0 for w in waits)}개 · 최장 {waits[-1]:.2f}초"
              " (그 사이 추적이 끊기거나 상자가 작아 못 정한 시간 포함)")


def _row_gap(a: dict, b: dict) -> tuple[float | None, int]:
    """기록 두 줄의 관절 거리(장부와 같은 셈)와 공통 관절 수."""
    from src.core.types import TrackObservation
    from src.identity.registry import POSE_CONF

    def obs(r):
        kp = r.get("keypoints")
        return TrackObservation(r["track_id"], tuple(r["bbox"]), r["frame"], "", keypoints=tuple(map(tuple, kp)) if kp else None)

    shared = sum(1 for ka, kb in zip(a.get("keypoints") or [], b.get("keypoints") or []) if ka[2] >= POSE_CONF and kb[2] >= POSE_CONF)
    return IdentityRegistry._pose_gap(obs(a), obs(b)), shared


def gaps(shoot: int, nested: float = 0.5) -> None:
    """포개짐 보류(관절 안 봄)로 보류된 줄마다, 그 상자와 포개진 신원 상자의 관절 거리. 관절 기준(pose)의 근거.

    정상(N) 영상은 거의 한 명이라 거기서 보류된 줄은 '한 사람 위의 두 상자'(또는 가짜 상자)다 → 같은 사람일 때 거리가 얼마까지 나오나를 N 으로 본다.
    다른 사람일 때의 거리는 두 명 영상(T · C)에서만 나온다 — 그 줄들은 기준을 고르는 데 쓰지 않고 확인에만 쓴다."""
    out = replay(shoot, {"nested": nested})
    rows_out: list[dict] = []
    for pair in dual_eval.pairs(None):
        path = out / pair / "angle1" / "identities.jsonl"
        if not path.exists():
            continue
        fps = json.loads((out / pair / "meta.json").read_text(encoding="utf-8"))["angle1"]["fps"]
        by_frame = defaultdict(list)
        for text in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(text)
            by_frame[r["frame"]].append(r)
        for f in sorted(by_frame):
            named = [r for r in by_frame[f] if r["person_id"] > 0]
            for r in by_frame[f]:
                host = next((h for h in named if IdentityRegistry._nested(r["bbox"], h["bbox"]) >= nested), None)
                if r["person_id"] > 0 or host is None:
                    continue
                gap, shared = _row_gap(r, host)
                after = next((x for g in sorted(by_frame) if g > f for x in by_frame[g]
                              if x["track_id"] == r["track_id"] and x["person_id"] > 0), None)
                kind = ("신원 없음" if after is None else "그 신원으로" if after["person_id"] == host["person_id"]
                        else "새 신원" if after.get("is_new") else "다른 신원")
                rows_out.append({"pair": pair, "gap": gap, "shared": shared, "kind": kind})
                text = "관절 없음" if gap is None else f"거리 {gap:4.2f} (공통 관절 {shared})"
                print(f"  {pair:6s} {f / fps:6.2f}초 추적 {r['track_id']:<3d} ↔ 신원 {host['person_id']}  포개짐 "
                      f"{IdentityRegistry._nested(r['bbox'], host['bbox']):.2f}  {text}  → {kind}")
    for title, keep in (("정상(N)", lambda p: p.startswith("N")), ("그 밖(T · C)", lambda p: not p.startswith("N"))):
        part = [r for r in rows_out if keep(r["pair"])]
        vals = sorted(r["gap"] for r in part if r["gap"] is not None)
        if not part:
            continue
        cuts = " · ".join(f"{c} 이하 {sum(v <= c for v in vals)}" for c in (0.3, 0.5, 0.8))
        print(f"{shoot}차 {title}: 보류된 줄 {len(part)} · 공통 관절 없음 {len(part) - len(vals)} · 거리 {cuts} · 0.8 초과 {sum(v > 0.8 for v in vals)}"
              + (f" · 최대 {vals[-1]:.2f}" if vals else ""))


def parse_combo(items: list[str]) -> dict:
    combo = {}
    for item in items:
        k, v = item.split("=", 1)
        if k not in (*KEYS, "both"):
            raise SystemExit(f"{k}: nested · score · both")
        combo[k] = float(v)
    return combo


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["grid", "try", "who", "links", "held", "gaps", "posegrid"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--shoot", type=int, choices=[2, 3], default=3)
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if a.action == "grid":
        grid(a.shoot)
    elif a.action == "try":
        try_one(a.shoot, parse_combo(a.args))
    elif a.action == "links":
        links(a.shoot, parse_combo(a.args))
    elif a.action == "held":
        held(a.shoot, parse_combo(a.args))
    elif a.action == "gaps":
        gaps(a.shoot)
    elif a.action == "posegrid":
        grid(a.shoot, POSE_GRID)
    else:
        if not a.args:
            ap.error("who <짝> [nested=.. score=..]")
        who(a.shoot, a.args[0], parse_combo(a.args[1:]) or {"nested": 0.7, "score": 0.4})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
