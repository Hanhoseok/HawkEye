"""수치 조정 — 판정 값을 초안으로 바꿔 다시 판정하고, 전후를 비교한 뒤 적용한다.

여기서 바꿀 수 있는 것은 **저장된 기록으로 판정만 다시 하면 반영되는 값**이다 (결제·출구·손님 조건·두 화면 짝·선반 사건 묶기).
탐지·추적·신원·칸 그림 기준처럼 영상을 다시 돌려야 하는 값은 여기 없다.

순서와 잠금 (limitations A-15 — 채점용 영상을 보며 값을 맞추면 그 영상에만 맞는 값이 된다):
  1. 정상(N) 영상으로만 확인한다. 몇 번이든.
  2. 값을 정했으면 절도·복합 채점을 **연다**. 연 것은 기록에 남는다 (<그 촬영의 라벨 폴더>/tune_log.jsonl).
  3. 적용하면 그 촬영의 판정 값 파일(2차 configs/store01_dual_judge.yaml · 3차 configs/store01_shoot3_judge.yaml)에 적히고,
     평가(dual_eval)·실행(run_tracking --dual-config)·분석 도구가 그 값을 쓴다.

촬영 회차마다 따로다 (Tuner 는 그 회차의 Builder 를 받는다) — 카메라 자리가 다르면 맞는 값도 다르다.
초안으로 판정한 결과는 <그 촬영의 보기 데이터>/drafts/<값들의 지문>/ 에 두어 한 쌍 분석 화면에서 그대로 열어 볼 수 있다.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from pathlib import Path

import yaml

from workbench.data import ROOT, Builder, totals

KNOBS = [
    # key, 묶음, 이름, 설명, 종류, (최소, 최대, 간격) 또는 고를 값, 단위
    ("risk.dwell_pay_seconds", "결제", "계산대에 서 있어야 하는 시간",
     "계산대 구역에 이만큼 이어서 서 있으면 그때까지 가진 것을 모두 결제한 것으로 본다. 늘리면 스쳐 지나간 것이 결제로 잡히는 일이 줄고, "
     "빨리 계산한 정상 손님을 놓칠 수 있다. 정상 영상의 계산대 체류는 2.4~9초였다.", "number", (0.5, 6.0, 0.1), "초"),
    ("risk.zone_grace_seconds", "결제", "구역에서 잠깐 벗어나도 이어서 세기",
     "계산대·출구 구역에서 발이 이 시간 이하로 벗어나면 머문 시간을 처음부터 다시 세지 않는다. 0 이면 끔. "
     "켜면 정상 결제는 잘 잡히지만 계산대를 스친 도둑도 결제로 잡힐 수 있다 (11-08 에 재 봄).", "number", (0.0, 2.0, 0.1), "초"),
    ("risk.exit_min_seconds", "나감과 경보", "출구에 들어섰다고 보는 시간",
     "발이 출구 구역에 이만큼 이어서 있어야 '출구에 다가섬'으로 본다. 출구 옆을 스치는 것을 거른다.", "number", (0.1, 2.0, 0.1), "초"),
    ("risk.exit_confirm_seconds", "나감과 경보", "나갔다고 확정하기까지 기다리는 시간",
     "출구 쪽에서 사라진 뒤 이만큼 다시 안 보여야 나간 것으로 확정하고 최종 판정한다. 줄이면 판정이 빨리 나오지만, "
     "탐지가 잠깐 끊긴 것을 나갔다고 잘못 볼 수 있다.", "number", (0.5, 6.0, 0.5), "초"),
    ("risk.exit_rearm_seconds", "나감과 경보", "출구 근처로 치는 시간",
     "출구 구역을 벗어난 지 이 시간 안에 사라져야 '출구 쪽에서 사라짐'으로 본다. 같은 방문에 판정이 여러 번 나가는 것도 막는다.",
     "number", (0.5, 5.0, 0.5), "초"),
    ("risk.inside_seconds", "손님으로 세는 조건", "매장 안 바닥에 서 있어야 하는 시간",
     "발이 선반 앞·계산대·통로에 이만큼 이어서 있어야 손님으로 센다. 문 앞 조각·창밖 행인·외투를 거르는 값. "
     "줄이면 가짜 손님이 늘고, 늘리면 금방 나간 진짜 손님을 놓친다.", "number", (0.0, 3.0, 0.1), "초"),
    ("zones.foot_point", "손님으로 세는 조건", "발 위치를 무엇으로 볼까",
     "구역 판단에 쓰는 발 점. 발목은 관절 모델의 두 발목 가운데(안 잡히면 상자 아래 가운데), 상자는 사람 상자 아래 가운데. "
     "지금 구역은 발목에 맞춰 그렸다.", "choice", (("ankle", "발목"), ("bbox", "상자 아래 가운데")), ""),
    ("zones.ankle_conf", "손님으로 세는 조건", "발목을 믿는 최소 점수",
     "발목 점수가 이보다 낮으면 상자 아래 가운데를 대신 쓴다.", "number", (0.1, 0.9, 0.05), ""),
    ("fusion.min_score", "두 화면 같은 사람", "짝으로 인정하는 최소 점수",
     "위치와 외형을 합친 점수가 이보다 낮으면 짝을 맺지 않는다. 낮추면 짝이 늘지만 틀린 짝도 는다.", "number", (0.2, 0.9, 0.05), ""),
    ("fusion.position_weight", "두 화면 같은 사람", "위치를 얼마나 믿을까",
     "점수 = 이 값 × 위치 + (1 − 이 값) × 외형. 2차 촬영은 선반 카메라의 위치가 한 줄 좌표라 어긋날 때가 있다 (한계 B-14).",
     "number", (0.0, 1.0, 0.05), ""),
    ("fusion.max_band_gap", "두 화면 같은 사람", "위치가 이만큼 다르면 다른 사람",
     "선반 앞 띠 위에서 위치 차이가 이 값이면 위치 점수가 0 이 된다.", "number", (0.05, 0.6, 0.05), ""),
    ("fusion.max_floor_gap_cm", "두 화면 같은 사람", "바닥 위에서 이만큼 떨어지면 다른 사람",
     "두 카메라가 본 발의 바닥 위 거리가 이 값이면 위치 점수가 0 이 된다. 정상 영상에서 같은 사람의 거리는 발목이 보일 때 "
     "중앙 4cm · 90% 6cm 였다. 40~80 사이에서는 결과가 거의 같았다 (후속 27).", "number", (20.0, 150.0, 5.0), "cm"),
    ("fusion.ray_slack_cm", "두 화면 같은 사람", "가린 발은 이만큼까지 같은 자리로 보기",
     "발목이 안 보이면(화면 아래로 잘림·가림) 발이 있을 방향만 안다. 그렇게 잰 거리는 이 값까지 0 으로 친다 — "
     "몇 cm 차이로 사람을 가르지 않고 외형과 '지금 짝 유지'가 정하게. 정상 영상에서 가린 발의 거리는 중앙 15cm · 90% 24cm.",
     "number", (0.0, 60.0, 5.0), "cm"),
    ("fusion.keep_bonus", "두 화면 같은 사람", "지금 짝을 유지하는 가산점",
     "이미 맺은 짝에 더해 주는 점수. 키우면 짝이 덜 바뀐다.", "number", (0.0, 0.4, 0.05), ""),
    ("fusion.solo_appearance_min", "두 화면 같은 사람", "후보가 한 명일 때 외형만으로 잇는 기준",
     "선반 앞에 비교할 사람이 한 명뿐이면 위치가 어긋나도 외형이 이 이상일 때 잇는다. 0 이면 끔. "
     "0.85 는 정상 영상에서 다른 사람끼리 나온 외형 최댓값(0.844) 바로 위.", "number", (0.0, 0.99, 0.01), ""),
    ("fusion.release_delay_seconds", "선반 사건의 주인", "주인을 정하기 전에 기다리는 시간",
     "손 접촉·선반 방문이 끝나고 이만큼 기다렸다 누구 것인지 정한다 — 짝 기록과 칸 그림 결과가 쌓이게.", "number", (0.5, 4.0, 0.5), "초"),
    ("fusion.event_tolerance_seconds", "선반 사건의 주인", "한 사건으로 묶는 시간 차",
     "같은 칸의 선반 변화·손 접촉·센서 기록이 이 시간 안이면 한 사건으로 묶는다.", "number", (0.2, 3.0, 0.1), "초"),
    ("fusion.unchanged_clears_all", "선반 사건의 주인", "선반이 그대로면 옆 사람 손 뻗기도 지우기",
     "물건 탐지와 칸 그림이 모두 그대로인 방문이면, 그때 선반 앞에 있던 다른 손님의 손 뻗기도 '만지기만'으로 본다.", "bool", (), ""),
]
KNOB = {k[0]: k for k in KNOBS}
SHOOT_ONLY = {"fusion.max_band_gap": (2,), "fusion.max_floor_gap_cm": (3,), "fusion.ray_slack_cm": (3,)}
"""그 촬영에서만 뜻이 있는 값 — 2차는 선반 앞 띠 한 줄로, 3차는 바닥 위 거리로 두 화면 짝을 맺는다."""


def knobs_for(shoot: int) -> list[tuple]:
    return [k for k in KNOBS if shoot in SHOOT_ONLY.get(k[0], (shoot,))]
RERUN_ONLY = [
    "사람 탐지 점수·모델, 추적기 설정", "사람 아님 영역", "신원 기준(가장자리 변, 되살아난 번호 다시 판정)",
    "선반 물건 탐지, 칸 그림 기준(8%)", "손-칸 접촉 시간(0.3초)",
]
"""영상을 다시 돌려야 반영되는 값 — 이 화면에서는 못 바꾼다 (화면에 안내만)."""


def coerce(key: str, value):
    """화면에서 온 값을 그 칸의 종류로 바꾸고 범위 안으로 맞춘다. 모르는 칸·틀린 값이면 ValueError."""
    if key not in KNOB:
        raise ValueError(f"모르는 값: {key}")
    kind, spec = KNOB[key][4], KNOB[key][5]
    if kind == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"{key}: 참/거짓이어야 합니다")
        return value
    if kind == "choice":
        if value not in [v for v, _ in spec]:
            raise ValueError(f"{key}: {value!r} 는 고를 수 없는 값입니다")
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key}: 숫자가 아닙니다") from None
    if number != number:
        raise ValueError(f"{key}: 숫자가 아닙니다")
    lo, hi, _ = spec
    return round(min(hi, max(lo, number)), 4)


def changed_values(values: dict, current: dict) -> dict:
    """지금 값과 다른 것만 (칸 순서대로)."""
    out = {}
    for key, *_ in KNOBS:
        if key in values:
            value = coerce(key, values[key])
            if key not in current:
                raise ValueError(f"{key}: 이 촬영에서는 쓰지 않는 값입니다")
            if value != current[key]:
                out[key] = value
    return out


def draft_id(changed: dict, base: str = "") -> str:
    """값 묶음의 지문 — 같은 값이면 같은 초안 폴더를 다시 쓴다.
    base 는 비교 기준(적용한 값·구역·시차·정답표·기록)의 지문. 기준이 바뀌면 같은 값이라도 다른 초안이다."""
    return hashlib.sha1((base + json.dumps(changed, sort_keys=True, ensure_ascii=False)).encode()).hexdigest()[:12]


def to_set(changed: dict) -> list[str]:
    """dual_eval --set 과 같은 '절.키=값' 목록."""
    return [f"{k}={json.dumps(v)}" for k, v in changed.items()]


def describe(changed: dict, current: dict) -> list[str]:
    out = []
    for key, value in changed.items():
        _, _, name, _, kind, spec, unit = KNOB[key]

        def show(v):
            if kind == "bool":
                return "켬" if v else "끔"
            if kind == "choice":
                return dict(spec).get(v, v)
            return f"{v:g}{unit}"
        out.append(f"{name} {show(current[key])} → {show(value)}")
    return out


BRIEF = ("headline", "outcome", "correct", "no_verdict", "verdicts", "people", "people_ok", "taken", "link_pct", "reason")


def compare(before: list[dict], after: list[dict]) -> dict:
    """같은 짝들의 적용본 카드와 초안 카드 → 점수 전후와 달라진 짝."""
    prev = {c["pair"]: c for c in before}
    both = [c for c in after if c["pair"] in prev]
    rows = []
    for c in both:
        b = prev[c["pair"]]
        diff = [k for k in ("verdicts", "outcome", "people", "taken", "link_pct") if b[k] != c[k]]
        if not diff:
            continue
        effect = "same"
        if b["correct"] != c["correct"]:
            effect = "better" if c["correct"] else "worse"
        elif b["no_verdict"] != c["no_verdict"]:
            effect = "better" if not c["no_verdict"] else "worse"
        elif b["people_ok"] != c["people_ok"] and c["people_ok"] is not None:
            effect = "better" if c["people_ok"] else "worse"
        rows.append({"pair": c["pair"], "set": c["set"], "set_name": c["set_name"], "effect": effect, "diff": diff,
                     "before": {k: b[k] for k in BRIEF}, "after": {k: c[k] for k in BRIEF}})
    order = {"worse": 0, "better": 1, "same": 2}
    rows.sort(key=lambda r: order[r["effect"]])
    return {"before": totals([prev[c["pair"]] for c in both]), "after": totals(both), "rows": rows, "n": len(both)}


class Tuner:
    def __init__(self, builder: Builder) -> None:
        self.builder = builder
        self.knobs = knobs_for(builder.shoot)
        self.log_file = builder.profile.labels / "tune_log.jsonl"
        self.judge_file = builder.profile.judge_file
        self.drafts = builder.out / "drafts"
        self.job = {"running": False, "id": None, "scope": None, "error": None}
        self._current: dict | None = None

    # -- 지금 값
    def current(self, refresh: bool = False) -> dict:
        """지금 판정에 쓰이는 값 (코드 기본 + 적용한 값)."""
        if self._current is None or refresh:
            self._current = self._values(with_file=True)
        return self._current

    def factory(self) -> dict:
        """적용한 값을 빼고 본 코드 기본."""
        return self._values(with_file=False)

    def _values(self, with_file: bool) -> dict:
        c = self.builder.ctx()
        pairs = self.builder.pairs()
        if not pairs:
            return {}
        with self.builder.engine() as de:
            cfg = de.dual_config(c["AppConfig"].load(ROOT / "config.yaml"), pairs[0],
                                 judge_file=self.judge_file if with_file else None, overrides=[])
        out = {}
        for key, *_ in self.knobs:
            section, name = key.split(".")
            out[key] = getattr(getattr(cfg, section), name)
        return out

    def applied(self) -> dict:
        if not self.judge_file.exists():
            return {}
        return yaml.safe_load(self.judge_file.read_text(encoding="utf-8")) or {}

    # -- 화면에 주는 것
    def page(self) -> dict:
        current, factory = self.current(refresh=True), self.factory()
        knobs = []
        if not current:
            raise ValueError("이 촬영의 기록이 없습니다 — 먼저 scripts/dual_eval.py run 으로 영상을 처리해 주세요")
        for key, group, name, text, kind, spec, unit in self.knobs:
            item = {"key": key, "group": group, "name": name, "help": text, "kind": kind, "unit": unit,
                    "value": current[key], "factory": factory[key]}
            if kind == "number":
                item.update(min=spec[0], max=spec[1], step=spec[2])
            elif kind == "choice":
                item["choices"] = [{"value": v, "name": n} for v, n in spec]
            knobs.append(item)
        log = self.log()
        rel = self.judge_file.relative_to(ROOT).as_posix()
        return {"knobs": knobs, "rerun_only": RERUN_ONLY, "log": log[-12:][::-1], "judge_file": rel,
                "shoot": self.builder.shoot, "shoot_name": self.builder.profile.name,
                "opens": sum(1 for r in log if r["kind"] in ("open", "apply")),
                "pairs": {"normal": len(self.scope("normal")), "all": len(self.builder.pairs())},
                "job": dict(self.job), "build": dict(self.builder.state)}

    def scope(self, name: str) -> list[str]:
        pairs = self.builder.pairs()
        return [p for p in pairs if p.startswith("N")] if name == "normal" else pairs

    def log(self) -> list[dict]:
        if not self.log_file.exists():
            return []
        return [json.loads(line) for line in self.log_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _append_log(self, row: dict) -> None:
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def opened(self, did: str) -> bool:
        return any(r.get("draft") == did and r["kind"] in ("open", "apply") for r in self.log())

    # -- 초안 판정
    def start(self, values: dict, scope: str) -> dict:
        """scope 'normal' = 정상 영상만 (몇 번이든), 'all' = 절도·복합 채점까지 연다 (기록에 남는다)."""
        if scope not in ("normal", "all"):
            raise ValueError("scope 는 normal 또는 all")
        current = self.current(refresh=True)
        changed = changed_values(values, current)
        if not changed:
            raise ValueError("바뀐 값이 없습니다")
        if self.job["running"] or self.builder.state["running"]:
            raise ValueError("다른 판정이 돌고 있습니다. 끝난 뒤 다시 눌러 주세요")
        did = draft_id(changed, self.builder.settings_signature())
        self.job = {"running": True, "id": did, "scope": scope, "error": None}
        threading.Thread(target=self._run, args=(did, changed, current, scope), daemon=True).start()
        return {"draft": did}

    def _run(self, did: str, changed: dict, current: dict, scope: str) -> None:
        try:
            if self.builder.stale():
                self.builder.build()   # 비교 기준(적용본)이 낡았으면 먼저 맞춘다
            out = self.drafts / did
            self._prune(keep=did)
            out.mkdir(parents=True, exist_ok=True)
            (out / "meta.json").write_text(json.dumps(
                {"draft": did, "values": changed, "changes": describe(changed, current)}, ensure_ascii=False), encoding="utf-8")
            errors = self.builder.build_draft(out, self.scope(scope), to_set(changed))
            if errors:
                raise RuntimeError("; ".join(errors))
            if scope == "all" and not self.opened(did):
                result = self.result(changed, current, did, force_all=True)
                self._append_log(self._log_row("open", did, changed, current, result))
        except Exception as e:  # noqa: BLE001 — 화면에 그대로 보인다
            self.job["error"] = repr(e)
        finally:
            self.job["running"] = False

    def _prune(self, keep: str, limit: int = 8) -> None:
        """초안 폴더가 쌓이지 않게 오래된 것부터 지운다 (다시 누르면 몇 초면 만들어진다)."""
        if not self.drafts.exists():
            return
        old = sorted((d for d in self.drafts.iterdir() if d.is_dir() and d.name != keep), key=lambda d: d.stat().st_mtime)
        for folder in old[:max(0, len(old) - (limit - 1))]:
            shutil.rmtree(folder, ignore_errors=True)

    def _log_row(self, kind: str, did: str, changed: dict, current: dict, result: dict) -> dict:
        def score(part):
            return None if part is None else {"before": f"{part['before']['correct']}/{part['n']}",
                                              "after": f"{part['after']['correct']}/{part['n']}",
                                              "changed": [f"{r['pair']}:{r['effect']}" for r in part["rows"]]}
        return {"time": time.strftime("%Y-%m-%d %H:%M"), "kind": kind, "draft": did, "values": changed,
                "changes": describe(changed, current), "normal": score(result["normal"]), "all": score(result["all"])}

    def _cards(self, folder: Path, pairs: list[str]) -> list[dict]:
        out = []
        for pair in pairs:
            path = folder / "cards" / f"{pair}.json"
            if path.exists():
                out.append(json.loads(path.read_text(encoding="utf-8")))
        return out

    def result(self, values: dict, current: dict | None = None, did: str | None = None, force_all: bool = False) -> dict:
        """초안의 전후 비교. 절도·복합이 들어간 비교(all)는 채점을 연 초안만 준다."""
        current = current or self.current()
        changed = changed_values(values, current)
        did = did or draft_id(changed, self.builder.settings_signature())
        folder = self.drafts / did
        normal_pairs, all_pairs = self.scope("normal"), self.scope("all")
        home = self.builder.out
        normal_after = self._cards(folder, normal_pairs)
        out = {"draft": did, "values": changed, "changes": describe(changed, current), "normal": None, "all": None,
               "opened": self.opened(did) or force_all, "job": dict(self.job)}
        if len(normal_after) == len(normal_pairs) and normal_pairs:
            out["normal"] = compare(self._cards(home, normal_pairs), normal_after)
        if out["opened"]:
            all_after = self._cards(folder, all_pairs)
            if len(all_after) == len(all_pairs):
                out["all"] = compare(self._cards(home, all_pairs), all_after)
        return out

    # -- 적용
    def apply(self, values: dict) -> dict:
        current = self.current(refresh=True)
        changed = changed_values(values, current)
        if not changed:
            raise ValueError("바뀐 값이 없습니다")
        if self.job["running"] or self.builder.state["running"]:
            raise ValueError("다른 판정이 돌고 있습니다. 끝난 뒤 다시 눌러 주세요")
        did = draft_id(changed, self.builder.settings_signature())
        result = self.result(changed, current, did)
        if result["normal"] is None:
            raise ValueError("먼저 정상 영상으로 확인해 주세요")
        factory = self.factory()
        merged = {**self.applied(), **changed}
        merged = {k: v for k, v in merged.items() if v != factory.get(k)}   # 코드 기본과 같아진 값은 파일에서 뺀다
        judge_file = self.judge_file
        if judge_file.exists():
            backup = self.builder.out / "tune_backup" / f"{time.strftime('%Y%m%d-%H%M%S')}_{judge_file.name}"
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(judge_file, backup)
        if merged:
            head = (f"# 분석 도구의 '수치 조정'에서 적용한 판정 값 ({self.builder.profile.name}) — 손으로 고치지 말고 도구에서 바꾼다.\n"
                    "# scripts/dual_eval.py 가 코드 기본 위에 이 값을 얹는다 (촬영 회차마다 파일이 다르다).\n"
                    f"# 바꾼 기록: {self.log_file.relative_to(ROOT).as_posix()}\n")
            judge_file.parent.mkdir(parents=True, exist_ok=True)
            judge_file.write_text(head + yaml.safe_dump(merged, allow_unicode=True, sort_keys=True), encoding="utf-8")
        elif judge_file.exists():
            judge_file.unlink()
        self._append_log(self._log_row("apply", did, changed, current, result))
        self._current = None
        self.builder.build_async()   # 갤러리를 새 값으로 다시 만든다
        return {"applied": merged, "draft": did}
