"""2차 촬영(두 각도 38쌍) 평가 — 카메라 2대 파이프라인 (docs/multiview-design.md §6 1-2~1-6).

    python scripts/dual_eval.py run        # 짝마다 두 카메라를 돌려 기록 (시차 0 으로 — 시차는 judge 에서 반영)
    python scripts/dual_eval.py sync       # 두 화면의 손님 위치 변화로 짝마다 시차 추정 → labels/store01_dual/sync.csv
    python scripts/dual_eval.py calibrate  # 정상(N) 영상으로 선반 카메라 x → 선반 앞 띠 위치 변환을 맞춤 → angle0.yaml
    python scripts/dual_eval.py judge      # 저장된 기록으로 합치기·판정만 다시 (몇 초)
    python scripts/dual_eval.py --shoot 3 run|calibrate|judge   # 3차 촬영 (source/2(angle*), 영상별 구역·실제 장면 시각 — scripts/shoot3_eval.py)
        --only "N*,T1"   짝 이름 패턴     --detail-off  선반 카메라 없이 (전체 카메라만, 1-2 기준선)
        --sensor          가짜 센서 기록(labels/store01_dual/fake_sensor/<짝>.csv)을 함께 (1-6)
        --payments        계산대 체류 대신 결제 라벨(<라벨 폴더>/payments.csv)을 결제 기록으로 (E-13).
                          3차는 바코드 삑 소리 초안(scripts/beep_payments.py)을 label_payments.py --shoot 3 으로 확인한 것

기준값은 정상(N) 영상으로 정하고 고정한 뒤 절도(T)·복합(C)으로 채점한다 (1차 촬영과 같은 방식).
영상은 저장소 밖 ../../source/1(angle0)·1(angle1).

채점 (2026-11-10, limitations A-15): 정답은 정답표(labels/store01_dual/truth.csv)의 expected 를 읽는다 — 짝 이름 첫 글자가 아니다
(C2 는 복합 세트지만 미결제 이탈). 판정이 하나도 안 나온 짝은 '판정 없음'으로 따로 센다. 정상 짝은 HIGH_RISK 가 없으면 맞음이라
판정 없음도 맞음에 들어가는데, 그 수를 함께 찍어 CLEAR 까지 간 것과 구분한다. 손님이 돌아와 판정이 바뀐 경우(E-15)를 보려고
'손님별 마지막 판정' 기준 점수도 함께 낸다.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.core.config import AppConfig, CameraConfig, InteractionConfig, apply_judge_overrides  # noqa: E402
from src.core.types import Frame, RiskLevel, TakeCandidate  # noqa: E402
from src.fusion.judge import DualJudge  # noqa: E402
from src.fusion.shelf_events import TimeAxis, load_sensor_events  # noqa: E402
from src.inputs.sync import SyncRow, SyncTable, best_lag  # noqa: E402
from src.risk.customers import ShelfVisit  # noqa: E402
from src.risk.engine import RiskEngine  # noqa: E402
from src.risk.payments import PaymentFeed  # noqa: E402
from src.sinks.observation_log import read_identities  # noqa: E402
from src.zones.floor import FloorMap, FloorZoneMap, detail_floor_map, shelf_image_zones  # noqa: E402
import sync_pair  # noqa: E402

OUT = ROOT / "outputs" / "dual_eval_pose3"
"""기본 기록 — 전체 카메라 관절 모델 + 발목 발 점 + 사용자 구역(2026-11-09 채택). 그 전 기본은 dual_eval_edgegap (기존 탐지 모델·상자 발 점)."""
ZONES = ROOT / "zones" / "store01_dual"
LABELS = ROOT / "labels" / "store01_dual"
STRIDE = {"angle1": 4, "angle0": 2}
JUDGE_FILE: Path | None = ROOT / "configs" / "store01_dual_judge.yaml"
"""분석 도구의 '수치 조정'에서 적용한 판정 값 (절.키: 값). 없으면 코드의 기본 그대로. None 이면 읽지 않는다."""
OVERRIDES: list[str] = []
OVERVIEW_POSE = False
OVERVIEW_DETECTOR = "yolo-pose"   # 발목 발 점에 관절이 필요 (2026-11-09 채택). 그 전 기본은 "yolo"
SHOOT = 2
"""촬영 회차. 3 이면 set_profile(3) 이 경로를 바꾸고, 영상별 구역·실제 장면 시각·카메라 움직임을 되돌린 선반 카메라 위치를 쓴다."""
_PAIR_ZONES: dict[str, Path] = {}
PAIR_ERRORS = (KeyError, ValueError, FileNotFoundError)
SENSOR_LANES: tuple[str, ...] | None = None
"""3차 --sensor 에서 쓸 레인 (None = 전부). --sensor-lanes L1,L2"""
"""3차에서 한 짝만 실패로 기록하고 넘어갈 오류 (카메라 맞추기 없음·장면 수 불일치·파일 없음). 2차는 그대로 멈춘다."""
TENTATIVE = 0.0   # 전체 카메라 임시 신원 시간 (identity.tentative_seconds, C-11) — 0 이면 끔
"""전체 카메라 팔 뻗기를 판정에 넣음 (--overview-pose). 관절 모델 자체는 OVERVIEW_DETECTOR 가 정한다."""
EDGE_SIDES = {"angle1": ["left", "right", "bottom"], "angle0": ["left", "top", "right"]}
"""가장자리 규칙을 적용할 변 — 화각 때문에 늘 잘리는 변(전체 카메라 위쪽 머리, 선반 카메라 아래쪽 발)은 뺀다 (C-18)."""
"""60FPS 전체 카메라는 4장에 1장, 30FPS 선반 카메라는 2장에 1장 — 둘 다 초당 15장."""


def set_profile(shoot: int) -> None:
    """촬영 회차별 경로. 2 는 지금까지 그대로."""
    global SHOOT, OUT, ZONES, LABELS
    SHOOT = shoot
    _PAIR_ZONES.clear()
    if shoot == 3:
        OUT = ROOT / "outputs" / "dual_eval_shoot3"
        ZONES = ROOT / "zones" / "store01_shoot3"
        LABELS = ROOT / "labels" / "store01_shoot3"
    else:
        OUT = ROOT / "outputs" / "dual_eval_pose3"
        ZONES = ROOT / "zones" / "store01_dual"
        LABELS = ROOT / "labels" / "store01_dual"


def zones_dir(pair: str) -> Path:
    """그 짝이 쓸 구역 폴더. 3차는 기본 구역을 그 영상의 카메라 위치로 옮긴 파일을 (한 번 실행에 한 번) 새로 만든다."""
    if SHOOT != 3:
        return ZONES
    if pair not in _PAIR_ZONES:
        import shoot3_eval
        _PAIR_ZONES[pair] = shoot3_eval.make_pair_zones(pair, OUT / "zones", ZONES)
    return _PAIR_ZONES[pair]


def detail_point_for(pair: str):
    """3차: 선반 카메라 사람 위치를 area_setting 좌표로 (band_map 입력). 2차: None (상자 가운데 x 그대로)."""
    if SHOOT != 3:
        return None
    import shoot3_eval
    return shoot3_eval.detail_point(shoot3_eval.homographies("angle0")[pair])


def kind(pair: str) -> str:
    return {"N": "정상", "T": "절도", "C": "복합 정상"}[pair[0]]


# ---------------------------------------------------------------------- 채점

def load_truth(path: Path | None = None) -> dict[str, dict]:
    """정답표: 짝 → {"expected": "HIGH_RISK" | "CLEAR", "people": 손님 수, "set": normal | theft | complex}."""
    path = path or LABELS / "truth.csv"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8", newline="") as f:
        return {r["pair"]: {"expected": r["expected"].strip(), "people": int(r["people"]), "set": r["set"].strip()}
                for r in csv.DictReader(f)}


def score(pair: str, finals: list[tuple[int, str]], people: int, truth: dict | None) -> dict:
    """한 짝 채점. finals = 시간 순서의 (손님 번호, 최종 판정) — WARNING 은 뺀 것.

    - expected: 정답표의 expected. 정답표에 없는 짝만 이름 첫 글자(T = 절도)로 대신하고 expected_source="name" 으로 남긴다.
    - correct: HIGH_RISK 가 한 번이라도 났는지가 정답과 같은가 (경보는 울리면 되돌릴 수 없다 — E-2).
    - no_verdict: 최종 판정이 하나도 없다. 정상 짝이면 correct 는 True 지만 CLEAR 까지 간 것이 아니다.
    - last_verdicts / correct_last: 손님별 마지막 판정만 본 것. 나갔다고 HIGH_RISK 를 낸 뒤 돌아와 결제하고
      CLEAR 로 나간 손님(E-15)은 여기서 경보가 아니다.
    - outcome: hit(절도 잡음) / miss(절도 놓침) / false_alarm(거짓 경보) / pass(정상 통과) / no_verdict(정상인데 판정 없음).
    """
    row = (truth or {}).get(pair)
    if row is None:
        expected, source = ("HIGH_RISK" if pair[0] == "T" else "CLEAR"), "name"
    else:
        expected, source = row["expected"], "truth"
    theft = expected == "HIGH_RISK"
    levels = [level for _, level in finals]
    last: dict[int, str] = {}
    for pid, level in finals:
        last[pid] = level
    alarm, alarm_last = "HIGH_RISK" in levels, "HIGH_RISK" in last.values()
    no_verdict = not levels
    if theft:
        outcome = "hit" if alarm else "miss"
    else:
        outcome = "false_alarm" if alarm else ("no_verdict" if no_verdict else "pass")
    people_expected = row["people"] if row else None
    return {
        "expected": expected, "expected_source": source, "outcome": outcome,
        "correct": alarm == theft, "correct_last": alarm_last == theft, "no_verdict": no_verdict,
        "last_verdicts": {str(pid): level for pid, level in last.items()},
        "people_expected": people_expected,
        "people_ok": None if people_expected is None else people == people_expected,
    }


def summarize(rows: list[dict]) -> dict:
    """짝별 채점(rows)을 세트별·전체로 모은다. 찍는 것은 print_summary."""
    def part(sel: list[dict]) -> dict:
        right = [r for r in sel if r["correct"]]
        return {
            "n": len(sel), "correct": len(right),
            "correct_decided": sum(not r["no_verdict"] for r in right),
            "correct_no_verdict": sum(r["no_verdict"] for r in right),
            "no_verdict": sum(r["no_verdict"] for r in sel),
            "correct_last": sum(r["correct_last"] for r in sel),
            "wrong": [r["pair"] for r in sel if not r["correct"]],
        }
    out = {k: part([r for r in rows if r["kind"] == k]) for k in ("정상", "절도", "복합 정상")}
    out["전체"] = part(rows)
    counted = [r for r in rows if r.get("people_ok") is not None]
    out["people"] = {"n": len(counted), "ok": sum(r["people_ok"] for r in counted),
                     "wrong": [r["pair"] for r in counted if not r["people_ok"]]}
    out["by_name"] = [r["pair"] for r in rows if r["expected_source"] == "name"]
    return out


def print_summary(summary: dict, label: str = "") -> None:
    head = f"[{label.strip()}] " if label.strip() else ""
    for k in ("정상", "절도", "복합 정상", "전체"):
        s = summary[k]
        if not s["n"]:
            continue
        note = f" (그중 판정 없이 맞음으로 센 짝 {s['correct_no_verdict']})" if s["correct_no_verdict"] else ""
        print(f"{head}{k}: {s['correct']}/{s['n']} 맞음{note}")
    s = summary["전체"]
    if s["n"]:
        print(f"{head}  판정이 나와서 맞음 {s['correct_decided']} · 판정 없이 맞음 {s['correct_no_verdict']} · "
              f"틀림 {s['n'] - s['correct']} ({', '.join(s['wrong']) or '-'}) · 판정 없는 짝 {s['no_verdict']}")
        print(f"{head}  손님별 마지막 판정 기준(돌아와 판정이 바뀐 경우 반영): {s['correct_last']}/{s['n']}")
    p = summary["people"]
    if p["n"]:
        print(f"{head}  손님 수 맞은 짝: {p['ok']}/{p['n']}" + (f" (틀림 {', '.join(p['wrong'])})" if p["wrong"] else ""))
    if summary["by_name"]:
        print(f"{head}  정답표에 없어 이름 첫 글자로 채점한 짝: {', '.join(summary['by_name'])}")


def pairs(pattern: str | None) -> list[str]:
    names = sync_pair.all_pairs(SHOOT)
    if not pattern:
        return names
    pats = [p.strip() for p in pattern.split(",")]
    return [n for n in names if any(fnmatch.fnmatch(n, p) for p in pats)]


_UNSET = object()


def dual_config(base: AppConfig, pair: str, offset_ms: float = 0.0, judge_file=_UNSET, overrides=_UNSET) -> AppConfig:
    """judge_file·overrides 를 안 주면 모듈 값(JUDGE_FILE·OVERRIDES)을 쓴다. 주면 그 값으로만 — 모듈 값을 건드리지 않고
    '적용한 값 없이'·'--set 없이' 설정을 보고 싶을 때 (분석 도구의 수치 조정)."""
    judge_file = JUDGE_FILE if judge_file is _UNSET else judge_file
    overrides = OVERRIDES if overrides is _UNSET else overrides
    cfg = deepcopy(base)
    # 끊겼다 되살아난 추적 번호는 다시 판정 (C-22) — 가장자리 변과 함께 켜면 손님 수 36→38/38, 판정 변화 없음
    # (2026-11-04 3차 비교). --set identity.track_gap_recheck_seconds=0 으로 끌 수 있다.
    cfg.identity.track_gap_recheck_seconds = 1.0
    # 바닥 구역 판단의 발 점 = 발목 (B-15, 2026-11-09 채택) — 사용자가 실제 바닥 자리에 맞춰 그린 구역과 짝.
    # 관절이 없는 예전 기록은 상자 아래 가운데로 대신한다. --set zones.foot_point=bbox 로 예전 방식.
    cfg.zones.foot_point = "ankle"
    # 3차: 선반 카메라에도 바닥 지도가 있다 → 두 화면 짝을 바닥 위 거리로 (후속 27, 2026-11-13).
    # 39쌍에서 끊김 23 → 2번, 다른 사람과 맺힘은 그대로. --set fusion.position_mode=band 로 예전 방식(띠 한 줄).
    if SHOOT == 3:
        cfg.fusion.position_mode = "floor"
    _apply_set(cfg, overrides)   # 아래에서 이 값을 베껴 쓰는 설정(선반 카메라 손 뻗기 등)에도 반영되게 먼저 한 번
    video, ref = sync_pair.pair_files(pair, SHOOT)
    zdir = zones_dir(pair)
    cfg.cameras = [
        CameraConfig("angle1", str(ref), "overview", str(zdir / "floor.yaml"), 0.0, STRIDE["angle1"],
                     OVERVIEW_DETECTOR,
                     {"edge_sides": EDGE_SIDES["angle1"], "tentative_seconds": TENTATIVE}),
        CameraConfig("angle0", str(video), "detail", str(zdir / "angle0.yaml"), offset_ms, STRIDE["angle0"], "yolo-pose",
                     {"edge_sides": EDGE_SIDES["angle0"]}),
    ]
    # 선반 카메라 손 뻗기 = 손목이 칸(SHELF 구역) 안에 0.3초 이상 (설계 M-6). 기준값은 N 으로 다시 본다.
    cfg.detail_interaction = InteractionConfig(**{**cfg.interaction.__dict__, "signal": "hand",
                                                  "dwell_seconds": 0.3, "hand_conf": 0.3})
    cfg.risk.payment_mode = "dwell"   # 결제 기록 없음: 계산대 체류 = 결제 (사용자 확인 규칙)
    # 매장 안 바닥(선반 앞·계산대·통로)에 1초 이상 들어온 사람만 손님 — 문 앞·창밖 행인·입장 순간 겹친 상자 제외 (C-20)
    cfg.risk.inside_zones, cfg.risk.inside_seconds = ["shelf", "counter", "aisle"], 1.0
    # 영상 끝에 출구 구역 안이면 나가는 중 — 화면 끝 조건을 끔 (2·3차 모두, 후속 23·25).
    # 3차는 출구가 화면 가운데, 2차는 출구(오른쪽 위)에서 상자가 위쪽 끝에만 걸려 '왼쪽·오른쪽 끝' 조건을 못 넘었다.
    cfg.risk.exit_end_at_edge_only = False
    if OVERVIEW_POSE:
        cfg.fusion.overview_reach_seconds = 0.3   # 선반 카메라 손-칸 접촉과 같은 기준에서 시작 (N 으로 다시 본다)
    cfg.output.write_video = False
    cfg.shelves.log_items = True
    from fake_sensor import UNIT_G
    cfg.fusion.sensor_unit_g = dict(UNIT_G)
    # 여기까지가 코드의 기본. 그 위에 적용한 판정 값(수치 조정), 그 위에 이번 실행만의 --set 을 얹는다.
    # (2026-11-10 까지는 --set 을 위에서 먼저 얹어, risk.inside_seconds·payment_mode 는 바로 아래 기본값에 덮여 무시됐다.)
    apply_judge_overrides(cfg, judge_file)
    _apply_set(cfg, overrides)
    return cfg


def _apply_set(cfg: AppConfig, overrides: list[str]) -> None:
    for item in overrides:
        key, value = item.split("=", 1)
        section, name = key.split(".")
        setattr(getattr(cfg, section), name, yaml.safe_load(value))


# ---------------------------------------------------------------------- run

def run(pattern: str | None) -> None:
    import cv2
    from src.detectors.yolo_detector import build_detector, build_item_detector
    from src.identity.embedder import build_embedder
    from src.identity.registry import IdentityRegistry
    from src.interaction.detector import InteractionDetector
    from src.multicam import build_multicam
    from src.shelf.watcher import ShelfWatcher
    from src.trackers import build_tracker

    base = AppConfig.load(ROOT / "config.yaml")
    for pair in pairs(pattern):
        out = OUT / pair
        if (out / ".done").exists():
            print(f"{pair}: 이미 있음 (건너뜀)", flush=True)
            continue
        try:
            cfg = dual_config(base, pair)
        except PAIR_ERRORS as e:
            if SHOOT != 3:
                raise
            print(f"{pair}: 실패 {type(e).__name__}: {e} — 건너뜀", flush=True)
            continue
        pipe = build_multicam(cfg, out, build_detector=build_detector, build_tracker=build_tracker,
                              build_embedder=build_embedder, build_item_detector=build_item_detector,
                              IdentityRegistry=IdentityRegistry, InteractionDetector=InteractionDetector,
                              ShelfWatcher=ShelfWatcher, RiskEngine=RiskEngine, PaymentFeed=PaymentFeed)
        stats = pipe.run()
        meta = {}
        for cam in cfg.cameras:
            cap = cv2.VideoCapture(cam.source)
            meta[cam.name] = {"fps": cap.get(cv2.CAP_PROP_FPS), "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                              "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "stride": cam.stride}
            cap.release()
        (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        if pipe.detail is not None and pipe.detail.shelves is not None and pipe.detail.shelves.item_log:
            (out / "angle0" / "shelf_items.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in pipe.detail.shelves.item_log), encoding="utf-8")
        (out / ".done").write_text("ok", encoding="utf-8")
        levels = [e.level.value for e in stats.events if e.level != RiskLevel.WARNING]
        total = sum(stats.frames.values())
        print(f"{pair}: 완료 {total / max(stats.elapsed_sec, 1e-6):.1f} 장면/초, 판정 {levels}", flush=True)


# ---------------------------------------------------------------------- 기록 읽기

class PairLog:
    def __init__(self, pair: str) -> None:
        d = OUT / pair
        self.pair = pair
        self.meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        self.ids = {cam: self._by_frame(d / cam / "identities.jsonl") for cam in ("angle0", "angle1")}
        self.takes = {cam: self._takes(d / cam / "take_candidates.jsonl") for cam in ("angle0", "angle1")}
        self.merges = {cam: self._jsonl(d / cam / "identity_merges.jsonl") for cam in ("angle0", "angle1")}
        self.visits = self._jsonl(d / "angle0" / "shelf_visits.jsonl")
        self.slot_changes = self._jsonl(d / "angle0" / "slot_changes.jsonl")
        self.times: dict[str, list[float] | None] = {"angle0": None, "angle1": None}
        """3차: 카메라별 실제 장면 시각(ms). 2차: None (번호 ÷ fps)."""
        if SHOOT == 3:
            import shoot3_eval
            video, ref = sync_pair.pair_files(pair, SHOOT)
            for cam, path in (("angle0", video), ("angle1", ref)):
                self.times[cam] = shoot3_eval.cached_times(path, d / f"times_{cam}.json", self.meta[cam]["frames"])
        self.app: dict[tuple[str, int], list[tuple[float, np.ndarray]]] = {}
        for r in self._jsonl(d / "appearance.jsonl"):
            self.app.setdefault((r["camera"], r["person_id"]), []).append((r["ms"], np.asarray(r["vec"])))

    @staticmethod
    def _jsonl(path: Path) -> list[dict]:
        if not path.exists():
            return []
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    @staticmethod
    def _by_frame(path: Path) -> dict[int, list]:
        out: dict[int, list] = {}
        if path.exists():
            for o in read_identities(path):
                out.setdefault(o.frame, []).append(o)
        return out

    @classmethod
    def _takes(cls, path: Path) -> list[TakeCandidate]:
        return [TakeCandidate(**r) for r in cls._jsonl(path)]

    def fps(self, cam: str) -> float:
        return self.meta[cam]["fps"]

    def ms(self, cam: str, idx: int) -> float:
        """장면 번호 → 그 카메라 시각(ms, 시차 전). 3차는 실제 장면 시각."""
        times = self.times.get(cam)
        if times:
            return times[max(0, min(idx, len(times) - 1))]
        return idx * 1000.0 / self.meta[cam]["fps"]

    def appearance(self, cam: str, offset: float, now_ms: float):
        def lookup(pid: int):
            best = None
            for ms, vec in self.app.get((cam, pid), []):
                if ms + offset <= now_ms + 1e-6:
                    best = vec
            return best
        return lookup

    def resolver(self, cam: str, upto_frame: int):
        alias = {m["from"]: m["into"] for m in self.merges[cam] if m.get("frame", 0) <= upto_frame}

        def resolve(pid: int) -> int:
            while pid in alias:
                pid = alias[pid]
            return pid
        return resolve


# ---------------------------------------------------------------------- 판정 재실행

def judge_pair(pair: str, base: AppConfig, offset_ms: float, band_map, use_detail: bool = True,
               sensor: bool = False, payments: bool = False, truth: dict | None = None) -> dict:
    """truth = load_truth() 결과. 안 주면 정답표를 읽는다 (오판 장면 보기처럼 한 짝씩 부르는 쪽을 위해)."""
    if truth is None:
        truth = load_truth()
    log = PairLog(pair)
    cfg = dual_config(base, pair, offset_ms)
    zdir = zones_dir(pair)
    floor = FloorMap.load(zdir / "floor.yaml")
    floor.foot_mode, floor.ankle_conf = cfg.zones.foot_point, cfg.zones.ankle_conf
    fz = FloorZoneMap(floor)
    m1, m0 = log.meta["angle1"], log.meta["angle0"]
    fz.resolve(m1["width"], m1["height"])
    feed = PaymentFeed.empty()
    if payments:
        # 결제 라벨(scripts/label_payments.py)의 완료된 결제만 결제 기록으로 — 계산대 체류 규칙은 끈다
        from label_payments import load_feed
        cfg.risk.payment_mode = "feed"
        feed = load_feed(pair, LABELS / "payments.csv")
    risk = RiskEngine(cfg.risk, fz, feed)
    detail_setup = yaml.safe_load((zdir / "angle0.yaml").read_text(encoding="utf-8"))
    slot_shelf = {z["name"]: detail_setup["shelves"][0]["name"] for z in detail_setup["zones"]}
    sensors = []
    if sensor and SHOOT == 3:
        # 3차: 실제 센서 (angle0 시각 → +시차 = 공통 시각). SENSOR_LANES 로 레인 고름
        import shoot3_eval
        sensors = shoot3_eval.load_sensor(LABELS / "sensor" / f"{pair}.csv", offset_ms, SENSOR_LANES)
    elif sensor:
        path = LABELS / "fake_sensor" / f"{pair}.csv"
        if path.exists():
            sensors = load_sensor_events(path)
    floor_raw = yaml.safe_load((zdir / "floor.yaml").read_text(encoding="utf-8")) or {}
    t0, t1 = log.times["angle0"], log.times["angle1"]
    judge = DualJudge(cfg.fusion, risk, floor, band_map,
                      TimeAxis(m0["fps"], offset_ms, tuple(t0) if t0 else None), TimeAxis(m1["fps"], 0.0, tuple(t1) if t1 else None),
                      m0["width"], (m1["width"], m1["height"]), slot_shelf, sensors, shelf_image_zones(floor_raw),
                      detail_point=detail_point_for(pair), detail_floor=detail_floor_map(detail_setup),
                      detail_height=m0["height"])
    judge.reset()
    if use_detail:
        from src.shelf.slot_diff import SlotChange
        for r in log.slot_changes:
            judge.fuser.note_slot_change(SlotChange(**{**r, "start_ms": r["start_ms"] + offset_ms,
                                                       "end_ms": r["end_ms"] + offset_ms}))
    # 공통 시각 순서로 두 카메라 장면을 섞는다 (파이프라인과 같은 순서)
    timeline = [(log.ms("angle1", i), "angle1", i) for i in range(0, m1["frames"], m1["stride"])]
    if use_detail:
        timeline += [(log.ms("angle0", i) + offset_ms, "angle0", i) for i in range(0, m0["frames"], m0["stride"])]
    timeline.sort()
    take_at = {cam: {} for cam in ("angle0", "angle1")}
    for cam in take_at:
        for t in log.takes[cam]:
            take_at[cam].setdefault(t.end_frame, []).append(t)
    visit_at: dict[int, list] = {}
    for v in log.visits:
        visit_at.setdefault(v["end_frame"], []).append(v)
    merges_seen: list[dict] = []
    pending_merges = sorted(log.merges["angle1"], key=lambda m: m.get("frame", 0))
    last = None
    for ms, cam, idx in timeline:
        if cam == "angle0":
            visits = []
            for v in visit_at.get(idx, []):
                v = dict(v)
                start, end = v.pop("start_ms") + offset_ms, v.pop("end_ms") + offset_ms
                visits.append((ShelfVisit(**v), start, end))
            judge.detail_step(ms, log.ids["angle0"].get(idx, []), take_at["angle0"].get(idx, []), visits,
                              log.appearance("angle0", offset_ms, ms), log.resolver("angle0", idx))
        else:
            while pending_merges and pending_merges[0].get("frame", 0) <= idx:
                merges_seen.append(pending_merges.pop(0))
            frame = Frame(index=idx, timestamp="", pts_ms=ms, image=None, camera_id="angle1")
            judge.overview_step(frame, log.ids["angle1"].get(idx, []), take_at["angle1"].get(idx, []),
                                merges_seen, log.appearance("angle1", 0.0, ms), log.resolver("angle1", idx))
            last = frame
    resolve_end = log.resolver("angle1", m1["frames"])
    judge.finish(last, resolve_end)
    finals = [e for e in judge.events if e.level != RiskLevel.WARNING]
    levels = [e.level.value for e in finals]
    people = sum(1 for c in risk.book.customers.values() if c.seen_inside)
    # 손님별 마지막 판정은 나중에 합쳐진 신원을 한 사람으로 본다
    scored = score(pair, [(resolve_end(e.person_id), e.level.value) for e in finals], people, truth)
    return {
        "pair": pair, "kind": kind(pair), "verdicts": levels, **scored,
        "reasons": [f"P{e.person_id} {e.level.value}: {e.reason}" for e in finals],
        "people": people,
        "identities": len({o.person_id for f in log.ids["angle1"].values() for o in f if o.person_id > 0}),
        "links": len(judge.linker.links), "unowned_takes": judge.unowned_takes,
        "actions": [a.describe() for a in risk.all_actions() if a.kind != "LOOK"],
        "payments": sum(1 for p in risk.book.payments if p.method in ("dwell", "checkout", "given")),
        "unmatched_payments": sum(1 for p in risk.book.payments if p.method == "unmatched"),
        "cancels": sum(1 for p in risk.book.payments if p.method == "cancel"),
        "shelf_events": [r.to_dict() for r in judge.fuser.records],
    }


def load_band_map() -> tuple[float, float]:
    if SHOOT == 3:
        import shoot3_eval
        a, b = shoot3_eval.load_band_map(ZONES / "band_map.yaml")
        return float(a), float(b)
    raw = yaml.safe_load((ZONES / "angle0.yaml").read_text(encoding="utf-8"))
    a, b = raw.get("band_map") or (1.0, 0.0)
    return float(a), float(b)


def check_payment_labels(todo: list[str], path: Path | None = None) -> list[str]:
    """--payments 전에: 라벨 줄이 없는 짝은 '결제 없음'으로 읽히므로 멈춘다. 확인 안 한 초안 짝 목록을 돌려준다."""
    from label_payments import label_state, load_labels
    labels = load_labels(path or LABELS / "payments.csv")
    state = {p: label_state(labels, p) for p in todo}
    missing = [p for p, s in state.items() if s == "missing"]
    if missing:
        raise SystemExit(f"결제 라벨 없는 짝: {', '.join(missing)} — scripts/label_payments.py 로 먼저 찍는다")
    drafts = [p for p, s in state.items() if s == "draft"]
    if drafts:
        print(f"주의: 확인 안 한 결제 초안(beep-draft) {len(drafts)}짝 — {', '.join(drafts)}")
    return drafts


def judge(pattern: str | None, use_detail: bool, sensor: bool, label: str = "", before: bool = False,
          payments: bool = False) -> list[dict]:
    base = AppConfig.load(ROOT / "config.yaml")
    if before:
        base.fusion.release_delay_seconds, base.fusion.unchanged_clears_all = 1.0, False
    table = SyncTable.load(LABELS / "sync.csv")
    band = load_band_map()
    truth = load_truth()
    rows, failed = [], []
    if payments:
        check_payment_labels([p for p in pairs(pattern) if (OUT / p / ".done").exists()])
    for pair in pairs(pattern):
        if not (OUT / pair / ".done").exists():
            continue
        try:
            rows.append(judge_pair(pair, base, table.offset_ms(pair) or 0.0, band, use_detail, sensor, payments, truth))
        except PAIR_ERRORS as e:
            if SHOOT != 3:
                raise
            failed.append({"pair": pair, "error": f"{type(e).__name__}: {e}"})
    print_summary(summarize(rows), label)
    for f in failed:
        print(f"  실패  {f['pair']:6s} {f['error']} — 채점에서 뺌")
    for r in rows:
        # OK = 맞음, OK? = 정상인데 판정이 안 나와 맞음으로만 센 짝, XX = 틀림
        mark = "XX " if not r["correct"] else ("OK?" if r["no_verdict"] else "OK ")
        changed = "" if r["correct_last"] == r["correct"] else "  [마지막 판정 기준으로는 " + ("맞음" if r["correct_last"] else "틀림") + "]"
        people = f"{r['people']}" + ("" if r["people_ok"] in (True, None) else f"(정답 {r['people_expected']})")
        print(f"  {mark}{r['pair']:6s} 정답 {r['expected']:9s} {','.join(r['verdicts']) or '(판정 없음)'}{changed}  손님 {people}  짝 {r['links']}  "
              f"계산대 결제 {r['payments']}" + (f" 취소 {r['cancels']}" if r["cancels"] else "")
              + (f"  연결 못한 결제 {r['unmatched_payments']}" if r["unmatched_payments"] else "")
              + (f"  주인 없는 손 접촉 {r['unowned_takes']}" if r["unowned_takes"] else ""))
        if not r["correct"]:
            for reason in r["reasons"]:
                print(f"        {reason}")
            for a in r["actions"]:
                print(f"        행동: {a}")
    name = f"results{('_' + label.strip()) if label.strip() else ''}.json"
    (OUT / name).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    if failed:
        (OUT / name.replace("results", "failed")).write_text(json.dumps(failed, ensure_ascii=False, indent=1), encoding="utf-8")
    return rows


# ---------------------------------------------------------------------- 시차·띠 축 맞추기

def _tracks(log: PairLog):
    """한 명만 보이는 장면의 (ms, 위치): 선반 카메라 x(화면 비율), 전체 카메라 띠 위치."""
    floor = FloorMap.load(zones_dir(log.pair) / "floor.yaml")
    m1, m0 = log.meta["angle1"], log.meta["angle0"]
    point = detail_point_for(log.pair)
    a0 = []
    for idx, obs in sorted(log.ids["angle0"].items()):
        obs = [o for o in obs if o.person_id > 0]
        if len(obs) == 1:
            b = obs[0].bbox
            x = point(b, m0["width"]) if point else (b[0] + b[2]) / 2.0 / m0["width"]
            a0.append((log.ms("angle0", idx), x))
    a1 = []
    for idx, obs in sorted(log.ids["angle1"].items()):
        obs = [o for o in obs if o.person_id > 0]
        if len(obs) == 1:
            p = floor.foot_on_floor(obs[0].bbox, m1["width"], m1["height"])
            band = floor.band_position(p)
            if band is not None and -0.3 <= band <= 1.6:
                a1.append((log.ms("angle1", idx), band))
    return a0, a1


def _pairs_at(a0, a1, offset: float, tol: float = 80.0):
    xs, ys = [], []
    j = 0
    for ms, x in a0:
        t = ms + offset
        while j + 1 < len(a1) and a1[j + 1][0] <= t:
            j += 1
        best = min((a1[k] for k in (j, j + 1) if k < len(a1)), key=lambda s: abs(s[0] - t), default=None)
        if best is not None and abs(best[0] - t) <= tol:
            xs.append(x)
            ys.append(best[1])
    return np.asarray(xs), np.asarray(ys)


MIN_SYNC_SAMPLES = 60
MIN_SYNC_CORR = 0.6


def sync(pattern: str | None, max_lag_s: float = 3.0) -> None:
    """선반 카메라의 손님 x 와 전체 카메라의 띠 위치는 같은 움직임을 본다 → 상관이 가장 큰 시차.

    한 명만 보이는 장면이 적거나(표본 < 60) 상관이 낮으면(< 0.6) 그 값은 믿지 않고
    믿을 만한 짝들의 시차 중앙값을 쓴다 (method=trajectory-fallback).
    """
    table = SyncTable.load(LABELS / "sync.csv")
    found: dict[str, tuple[float, float, int]] = {}
    for pair in pairs(pattern):
        if not (OUT / pair / ".done").exists():
            continue
        row = table.rows.get(pair)
        if row and row.method == "manual":
            continue
        log = PairLog(pair)
        a0, a1 = _tracks(log)
        best = (0.0, 0.0, 0)
        for lag in np.arange(-max_lag_s * 1000, max_lag_s * 1000 + 1, 100.0):
            xs, ys = _pairs_at(a0, a1, lag)
            if len(xs) < 20 or xs.std() < 1e-3 or ys.std() < 1e-3:
                continue
            corr = abs(float(np.corrcoef(xs, ys)[0, 1]))
            if corr > best[1]:
                best = (float(lag), corr, len(xs))
        found[pair] = best
    good = [v[0] for v in found.values() if v[2] >= MIN_SYNC_SAMPLES and v[1] >= MIN_SYNC_CORR]
    fallback = float(np.median(good)) if good else 0.0
    for pair, (lag, corr, n) in found.items():
        video, ref = sync_pair.pair_files(pair, SHOOT)
        ok = n >= MIN_SYNC_SAMPLES and corr >= MIN_SYNC_CORR
        note = f"손님 위치 상관 {corr:.2f} (표본 {n})" + ("" if ok else f" — 믿기 어려워 중앙값 {fallback:+.0f}ms 사용 (추정 {lag:+.0f}ms)")
        table.upsert(SyncRow(pair, f"1(angle0)/{video.name}", f"1(angle1)/{ref.name}", lag if ok else fallback,
                             "trajectory" if ok else "trajectory-fallback", "", note + " — 확인 필요"))
        print(f"{pair}: {(lag if ok else fallback):+.0f}ms  {note}", flush=True)
    table.save(LABELS / "sync.csv")


def calibrate(pattern: str = "N*") -> None:
    """정상 영상(한 명)에서 선반 카메라 x → 띠 위치 1차식을 맞춘다 (시차 반영)."""
    table = SyncTable.load(LABELS / "sync.csv")
    xs_all, ys_all = [], []
    for pair in pairs(pattern):
        if not (OUT / pair / ".done").exists():
            continue
        row = table.rows.get(pair)
        if row is None or row.method not in ("trajectory", "manual"):
            continue   # 시차를 믿을 수 없는 짝은 빼고 맞춘다
        try:
            a0, a1 = _tracks(PairLog(pair))
        except PAIR_ERRORS as e:
            if SHOOT != 3:
                raise
            print(f"  {pair}: 실패 {type(e).__name__}: {e} — 빼고 맞춤")
            continue
        xs, ys = _pairs_at(a0, a1, row.offset_ms)
        xs_all += list(xs)
        ys_all += list(ys)
    xs, ys = np.asarray(xs_all), np.asarray(ys_all)
    a, b = np.polyfit(xs, ys, 1)
    resid = np.abs(ys - (a * xs + b))
    corr = np.corrcoef(xs, ys)[0, 1]
    print(f"띠 축 변환: band = {a:.3f}·x + {b:.3f}  (표본 {len(xs)}, 중앙 오차 {np.median(resid):.3f}, 상관 {corr:.2f})")
    if SHOOT == 3:
        import shoot3_eval
        shoot3_eval.save_band_map(a, b, f"dual_eval.py --shoot 3 calibrate ({pattern}): 표본 {len(xs)}, 중앙 오차 "
                                        f"{np.median(resid):.3f}, 상관 {corr:.2f}. 입력 x = area_setting 좌표로 옮긴 선반 카메라 사람 위치",
                                  ZONES / "band_map.yaml")
        _PAIR_ZONES.clear()   # 영상별 구역 파일에 새 band_map 이 들어가게 다음에 다시 만든다
        return
    path = ZONES / "angle0.yaml"
    text = path.read_text(encoding="utf-8")
    lines = [f"band_map: [{a:.4f}, {b:.4f}]" if l.startswith("band_map:") else l for l in text.splitlines()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["run", "sync", "calibrate", "judge"])
    ap.add_argument("--only")
    ap.add_argument("--detail-off", action="store_true")
    ap.add_argument("--sensor", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--before", action="store_true",
                    help="C1·C12 를 보고 고치기 전 규칙으로 (손 접촉 1.0초 뒤 주인 결정, 다른 손님 손 뻗기는 안 지움)")
    ap.add_argument("--out", help="기록 폴더 이름 (outputs/ 아래, 기본 dual_eval) — 설정을 바꿔 따로 돌려 비교할 때")
    ap.add_argument("--zones", help="구역 폴더 (floor.yaml·angle0.yaml, 기본 zones/store01_dual) — 구역 편집 페이지의 초안으로 판정할 때")
    ap.add_argument("--set", action="append", default=[], metavar="절.키=값",
                    help="config.yaml 값을 이번 실행만 바꿈, 예: identity.track_gap_recheck_seconds=1.0")
    ap.add_argument("--overview-pose", action="store_true",
                    help="'전체 카메라 팔 뻗기'를 판정에 넣음 (기본 끔, D-19). 전체 카메라 관절 모델은 이제 기본")
    ap.add_argument("--overview-detector", choices=["yolo-pose", "yolo"], default="yolo-pose",
                    help="전체 카메라 탐지 모델 (run 단계, 기본 yolo-pose — 발목 발 점용). yolo 는 11-08 까지의 기본")
    ap.add_argument("--all-edges", action="store_true",
                    help="가장자리 규칙을 네 변 모두에 (늘 잘리는 변을 빼기 전, C-18 비교용)")
    ap.add_argument("--tentative", type=float, default=0.0, metavar="초",
                    help="전체 카메라 새 신원을 임시로 두고, 발이 매장 안 바닥에 이만큼 연속으로 있어야 정식으로 (C-11, run 단계)")
    ap.add_argument("--sensor-lanes", help="3차 --sensor: 쓸 레인 쉼표 (기본 전부, 예: L1,L2)")
    ap.add_argument("--shoot", type=int, choices=[2, 3], default=2, help="촬영 회차 (기본 2). 3 = 2026-11-11 3차")
    ap.add_argument("--payments", action="store_true",
                    help="계산대 체류 대신 결제 라벨(<라벨 폴더>/payments.csv, scripts/label_payments.py)을 결제 기록으로")
    args = ap.parse_args()
    global OUT, OVERRIDES, OVERVIEW_POSE, OVERVIEW_DETECTOR, TENTATIVE, ZONES
    set_profile(args.shoot)
    global SENSOR_LANES
    SENSOR_LANES = tuple(x.strip() for x in args.sensor_lanes.split(",")) if args.sensor_lanes else None
    OVERVIEW_POSE = args.overview_pose
    OVERVIEW_DETECTOR = args.overview_detector
    TENTATIVE = args.tentative
    if args.all_edges:
        for cam in EDGE_SIDES:
            EDGE_SIDES[cam] = ["left", "top", "right", "bottom"]
    if args.out:
        OUT = ROOT / "outputs" / args.out
    if args.zones:
        ZONES = Path(args.zones).resolve()
    OVERRIDES = args.set
    if args.stage == "run":
        run(args.only)
    elif args.stage == "sync":
        sync(args.only)
    elif args.stage == "calibrate":
        calibrate(args.only or "N*")
    else:
        judge(args.only, not args.detail_off, args.sensor, args.label, args.before, args.payments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
