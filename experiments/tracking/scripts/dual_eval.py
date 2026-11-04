"""2차 촬영(두 각도 38쌍) 평가 — 카메라 2대 파이프라인 (docs/multiview-design.md §6 1-2~1-6).

    python scripts/dual_eval.py run        # 짝마다 두 카메라를 돌려 기록 (시차 0 으로 — 시차는 judge 에서 반영)
    python scripts/dual_eval.py sync       # 두 화면의 손님 위치 변화로 짝마다 시차 추정 → labels/store01_dual/sync.csv
    python scripts/dual_eval.py calibrate  # 정상(N) 영상으로 선반 카메라 x → 선반 앞 띠 위치 변환을 맞춤 → angle0.yaml
    python scripts/dual_eval.py judge      # 저장된 기록으로 합치기·판정만 다시 (몇 초)
        --only "N*,T1"   짝 이름 패턴     --detail-off  선반 카메라 없이 (전체 카메라만, 1-2 기준선)
        --sensor          가짜 센서 기록(labels/store01_dual/fake_sensor/<짝>.csv)을 함께 (1-6)

기준값은 정상(N) 영상으로 정하고 고정한 뒤 절도(T)·복합(C)으로 채점한다 (1차 촬영과 같은 방식).
영상은 저장소 밖 ../../source/1(angle0)·1(angle1).
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

from src.core.config import AppConfig, CameraConfig, InteractionConfig  # noqa: E402
from src.core.types import Frame, RiskLevel, TakeCandidate  # noqa: E402
from src.fusion.judge import DualJudge  # noqa: E402
from src.fusion.shelf_events import TimeAxis, load_sensor_events  # noqa: E402
from src.inputs.sync import SyncRow, SyncTable, best_lag  # noqa: E402
from src.risk.customers import ShelfVisit  # noqa: E402
from src.risk.engine import RiskEngine  # noqa: E402
from src.risk.payments import PaymentFeed  # noqa: E402
from src.sinks.observation_log import read_identities  # noqa: E402
from src.zones.floor import FloorMap, FloorZoneMap, shelf_image_zones  # noqa: E402
import sync_pair  # noqa: E402

OUT = ROOT / "outputs" / "dual_eval"
ZONES = ROOT / "zones" / "store01_dual"
LABELS = ROOT / "labels" / "store01_dual"
STRIDE = {"angle1": 4, "angle0": 2}
OVERRIDES: list[str] = []
OVERVIEW_POSE = False
"""전체 카메라도 yolo-pose 로 — 위에서 보이는 팔 뻗기 (--overview-pose)."""
EDGE_SIDES = {"angle1": ["left", "right", "bottom"], "angle0": ["left", "top", "right"]}
"""가장자리 규칙을 적용할 변 — 화각 때문에 늘 잘리는 변(전체 카메라 위쪽 머리, 선반 카메라 아래쪽 발)은 뺀다 (C-18)."""
"""60FPS 전체 카메라는 4장에 1장, 30FPS 선반 카메라는 2장에 1장 — 둘 다 초당 15장."""


def kind(pair: str) -> str:
    return {"N": "정상", "T": "절도", "C": "복합 정상"}[pair[0]]


def pairs(pattern: str | None) -> list[str]:
    names = sync_pair.all_pairs()
    if not pattern:
        return names
    pats = [p.strip() for p in pattern.split(",")]
    return [n for n in names if any(fnmatch.fnmatch(n, p) for p in pats)]


def dual_config(base: AppConfig, pair: str, offset_ms: float = 0.0) -> AppConfig:
    cfg = deepcopy(base)
    # 끊겼다 되살아난 추적 번호는 다시 판정 (C-22) — 가장자리 변과 함께 켜면 손님 수 36→38/38, 판정 변화 없음
    # (2026-11-04 3차 비교). --set identity.track_gap_recheck_seconds=0 으로 끌 수 있다.
    cfg.identity.track_gap_recheck_seconds = 1.0
    for item in OVERRIDES:
        key, value = item.split("=", 1)
        section, name = key.split(".")
        setattr(getattr(cfg, section), name, yaml.safe_load(value))
    video, ref = sync_pair.pair_files(pair)
    cfg.cameras = [
        CameraConfig("angle1", str(ref), "overview", str(ZONES / "floor.yaml"), 0.0, STRIDE["angle1"],
                     "yolo-pose" if OVERVIEW_POSE else "yolo",
                     {"edge_sides": EDGE_SIDES["angle1"]}),
        CameraConfig("angle0", str(video), "detail", str(ZONES / "angle0.yaml"), offset_ms, STRIDE["angle0"], "yolo-pose",
                     {"edge_sides": EDGE_SIDES["angle0"]}),
    ]
    # 선반 카메라 손 뻗기 = 손목이 칸(SHELF 구역) 안에 0.3초 이상 (설계 M-6). 기준값은 N 으로 다시 본다.
    cfg.detail_interaction = InteractionConfig(**{**cfg.interaction.__dict__, "signal": "hand",
                                                  "dwell_seconds": 0.3, "hand_conf": 0.3})
    cfg.risk.payment_mode = "dwell"   # 결제 기록 없음: 계산대 체류 = 결제 (사용자 확인 규칙)
    # 매장 안 바닥(선반 앞·계산대·통로)에 1초 이상 들어온 사람만 손님 — 문 앞·창밖 행인·입장 순간 겹친 상자 제외 (C-20)
    cfg.risk.inside_zones, cfg.risk.inside_seconds = ["shelf", "counter", "aisle"], 1.0
    if OVERVIEW_POSE:
        cfg.fusion.overview_reach_seconds = 0.3   # 선반 카메라 손-칸 접촉과 같은 기준에서 시작 (N 으로 다시 본다)
    cfg.output.write_video = False
    cfg.shelves.log_items = True
    from fake_sensor import UNIT_G
    cfg.fusion.sensor_unit_g = dict(UNIT_G)
    return cfg


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
        cfg = dual_config(base, pair)
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
               sensor: bool = False) -> dict:
    log = PairLog(pair)
    cfg = dual_config(base, pair, offset_ms)
    floor = FloorMap.load(ZONES / "floor.yaml")
    fz = FloorZoneMap(floor)
    m1, m0 = log.meta["angle1"], log.meta["angle0"]
    fz.resolve(m1["width"], m1["height"])
    risk = RiskEngine(cfg.risk, fz, PaymentFeed.empty())
    detail_setup = yaml.safe_load((ZONES / "angle0.yaml").read_text(encoding="utf-8"))
    slot_shelf = {z["name"]: detail_setup["shelves"][0]["name"] for z in detail_setup["zones"]}
    sensors = []
    if sensor:
        path = LABELS / "fake_sensor" / f"{pair}.csv"
        if path.exists():
            sensors = load_sensor_events(path)
    floor_raw = yaml.safe_load((ZONES / "floor.yaml").read_text(encoding="utf-8")) or {}
    judge = DualJudge(cfg.fusion, risk, floor, band_map, TimeAxis(m0["fps"], offset_ms), TimeAxis(m1["fps"]),
                      m0["width"], (m1["width"], m1["height"]), slot_shelf, sensors, shelf_image_zones(floor_raw))
    judge.reset()
    if use_detail:
        from src.shelf.slot_diff import SlotChange
        for r in log.slot_changes:
            judge.fuser.note_slot_change(SlotChange(**{**r, "start_ms": r["start_ms"] + offset_ms,
                                                       "end_ms": r["end_ms"] + offset_ms}))
    # 공통 시각 순서로 두 카메라 장면을 섞는다 (파이프라인과 같은 순서)
    timeline = [(i * 1000.0 / m1["fps"], "angle1", i) for i in range(0, m1["frames"], m1["stride"])]
    if use_detail:
        timeline += [(i * 1000.0 / m0["fps"] + offset_ms, "angle0", i) for i in range(0, m0["frames"], m0["stride"])]
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
    judge.finish(last, log.resolver("angle1", m1["frames"]))
    finals = [e for e in judge.events if e.level != RiskLevel.WARNING]
    levels = [e.level.value for e in finals]
    expected_theft = pair[0] == "T"
    hit = "HIGH_RISK" in levels
    correct = hit if expected_theft else not hit
    return {
        "pair": pair, "kind": kind(pair), "verdicts": levels, "correct": correct,
        "reasons": [f"P{e.person_id} {e.level.value}: {e.reason}" for e in finals],
        "people": sum(1 for c in risk.book.customers.values() if c.seen_inside),
        "identities": len({o.person_id for f in log.ids["angle1"].values() for o in f if o.person_id > 0}),
        "links": len(judge.linker.links), "unowned_takes": judge.unowned_takes,
        "actions": [a.describe() for a in risk.all_actions() if a.kind != "LOOK"],
        "payments": sum(1 for p in risk.book.payments if p.method == "dwell"),
        "cancels": sum(1 for p in risk.book.payments if p.method == "cancel"),
        "shelf_events": [r.to_dict() for r in judge.fuser.records],
    }


def load_band_map() -> tuple[float, float]:
    raw = yaml.safe_load((ZONES / "angle0.yaml").read_text(encoding="utf-8"))
    a, b = raw.get("band_map") or (1.0, 0.0)
    return float(a), float(b)


def judge(pattern: str | None, use_detail: bool, sensor: bool, label: str = "", before: bool = False) -> list[dict]:
    base = AppConfig.load(ROOT / "config.yaml")
    if before:
        base.fusion.release_delay_seconds, base.fusion.unchanged_clears_all = 1.0, False
    table = SyncTable.load(LABELS / "sync.csv")
    band = load_band_map()
    rows = []
    for pair in pairs(pattern):
        if not (OUT / pair / ".done").exists():
            continue
        rows.append(judge_pair(pair, base, table.offset_ms(pair) or 0.0, band, use_detail, sensor))
    for k in ("정상", "절도", "복합 정상"):
        part = [r for r in rows if r["kind"] == k]
        if part:
            ok = sum(r["correct"] for r in part)
            print(f"{label}{k}: {ok}/{len(part)} 정답")
    for r in rows:
        mark = "OK " if r["correct"] else "XX "
        print(f"  {mark}{r['pair']:6s} {','.join(r['verdicts']) or '(판정 없음)'}  손님 {r['people']}  짝 {r['links']}  "
              f"계산대 결제 {r['payments']}" + (f" 취소 {r['cancels']}" if r["cancels"] else "")
              + (f"  주인 없는 손 접촉 {r['unowned_takes']}" if r["unowned_takes"] else ""))
        if not r["correct"]:
            for reason in r["reasons"]:
                print(f"        {reason}")
            for a in r["actions"]:
                print(f"        행동: {a}")
    name = f"results{('_' + label.strip()) if label.strip() else ''}.json"
    (OUT / name).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return rows


# ---------------------------------------------------------------------- 시차·띠 축 맞추기

def _tracks(log: PairLog):
    """한 명만 보이는 장면의 (ms, 위치): 선반 카메라 x(화면 비율), 전체 카메라 띠 위치."""
    floor = FloorMap.load(ZONES / "floor.yaml")
    m1, m0 = log.meta["angle1"], log.meta["angle0"]
    a0 = []
    for idx, obs in sorted(log.ids["angle0"].items()):
        obs = [o for o in obs if o.person_id > 0]
        if len(obs) == 1:
            b = obs[0].bbox
            a0.append((idx * 1000.0 / m0["fps"], (b[0] + b[2]) / 2.0 / m0["width"]))
    a1 = []
    for idx, obs in sorted(log.ids["angle1"].items()):
        obs = [o for o in obs if o.person_id > 0]
        if len(obs) == 1:
            p = floor.foot_on_floor(obs[0].bbox, m1["width"], m1["height"])
            band = floor.band_position(p)
            if band is not None and -0.3 <= band <= 1.6:
                a1.append((idx * 1000.0 / m1["fps"], band))
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
        video, ref = sync_pair.pair_files(pair)
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
        a0, a1 = _tracks(PairLog(pair))
        xs, ys = _pairs_at(a0, a1, row.offset_ms)
        xs_all += list(xs)
        ys_all += list(ys)
    xs, ys = np.asarray(xs_all), np.asarray(ys_all)
    a, b = np.polyfit(xs, ys, 1)
    resid = np.abs(ys - (a * xs + b))
    print(f"띠 축 변환: band = {a:.3f}·x + {b:.3f}  (표본 {len(xs)}, 중앙 오차 {np.median(resid):.3f}, 상관 {np.corrcoef(xs, ys)[0, 1]:.2f})")
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
    ap.add_argument("--set", action="append", default=[], metavar="절.키=값",
                    help="config.yaml 값을 이번 실행만 바꿈, 예: identity.track_gap_recheck_seconds=1.0")
    ap.add_argument("--overview-pose", action="store_true",
                    help="전체 카메라도 yolo-pose 로 돌리고 '전체 카메라 팔 뻗기'를 판정에 넣음 (run·judge 둘 다 같이 줄 것)")
    ap.add_argument("--all-edges", action="store_true",
                    help="가장자리 규칙을 네 변 모두에 (늘 잘리는 변을 빼기 전, C-18 비교용)")
    args = ap.parse_args()
    global OUT, OVERRIDES, OVERVIEW_POSE
    OVERVIEW_POSE = args.overview_pose
    if args.all_edges:
        for cam in EDGE_SIDES:
            EDGE_SIDES[cam] = ["left", "top", "right", "bottom"]
    if args.out:
        OUT = ROOT / "outputs" / args.out
    OVERRIDES = args.set
    if args.stage == "run":
        run(args.only)
    elif args.stage == "sync":
        sync(args.only)
    elif args.stage == "calibrate":
        calibrate(args.only or "N*")
    else:
        judge(args.only, not args.detail_off, args.sensor, args.label, args.before)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
