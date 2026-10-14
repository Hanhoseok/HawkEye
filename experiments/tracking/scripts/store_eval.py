"""직접 찍은 매장 영상 일괄 평가 — 정상(N)·절도(T)·복합 정상(C).

    1) run    영상마다 구역을 배경으로 옮기고(N01 기준) 전체 프로그램을 결제 없이 돌려 기록을 남긴다 (느림, 한 번)
    2) pay    정상(N·C) 영상마다 계산대 체류 구간을 찾아 결제 기록 후보와 확인용 장면표를 만든다
              -> 사람이 장면표로 계산대에 실제로 있었는지 확인한 뒤 labels/store01/payments_<영상>.csv 로 쓴다
    3) judge  저장된 기록(신원·손 뻗기·선반 방문) + 결제로 판정만 다시 돌려 정답과 비교 (몇 초)

    python scripts/store_eval.py run --only "N*"        # 영상 묶음별로 나눠서 (CPU 로 영상당 1분 남짓)
    python scripts/store_eval.py pay
    python scripts/store_eval.py judge

judge 는 판정 직전 신원 확인(외형 비교)은 하지 않는다 — 영상을 다시 읽지 않기 때문이다(replay_risk.py 와 같음).
채점 단위: 정상·복합 정상 영상은 '아무에게도 HIGH_RISK 가 안 나야' 정답, 절도 영상은 'HIGH_RISK 가 하나 이상'이면 정답.
REVIEW(일부 결제 의심)는 따로 센다.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.core.config import AppConfig  # noqa: E402
from src.core.types import Frame, RiskLevel, TakeCandidate, ZoneType  # noqa: E402
from src.risk.customers import ShelfVisit  # noqa: E402
from src.risk.engine import RiskEngine  # noqa: E402
from src.risk.payments import PaymentFeed  # noqa: E402
from src.shelf.monitor import ShelfItem, ShelfMonitor  # noqa: E402
from src.sinks.observation_log import read_identities  # noqa: E402
from src.zones.zone_map import ZoneMap  # noqa: E402
from transfer_zones import camera_motion, first_frame, move  # noqa: E402

SOURCE = ROOT.parent.parent / "source"
OUT = ROOT / "outputs" / "store_eval"
LABELS = ROOT / "labels" / "store01"
REF = "N01"
REF_ZONES = ROOT / "zones_store01_N.yaml"
REF_SHELF = (0.02, 0.70, 0.45, 0.88)
SKIP = {"N01(1)"}   # N01 과 같은 파일
FPS = 30.0
STRIDE = 2
"""1080p 를 CPU 로 매 장면 돌리면 영상 20초에 200초(초당 3장)라 2장에 1장만 본다. 시간 규칙은 초 단위라 영향이 없다."""


def clips(pattern: str | None) -> list[str]:
    names = sorted(p.stem for p in SOURCE.glob("*.mp4") if p.stem not in SKIP)
    if pattern:
        names = [n for n in names if any(fnmatch.fnmatch(n, pat.strip()) for pat in pattern.split(","))]
    return names


def kind(name: str) -> str:
    return {"N": "정상", "T": "절도", "C": "복합 정상"}[name[0]]


def zones_for(name: str) -> tuple[Path, str]:
    """N01 기준 구역을 이 영상의 카메라 위치로 옮긴다 (배경 특징점, 정답은 안 봄)."""
    out = OUT / name / "zones.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    matrix, _ = camera_motion(first_frame(str(SOURCE / f"{REF}.mp4")), first_frame(str(SOURCE / f"{name}.mp4")))
    raw = yaml.safe_load(REF_ZONES.read_text(encoding="utf-8"))
    for zone in raw["zones"]:
        zone["polygon"] = move(zone["polygon"], matrix, (1920, 1080))
    out.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    x1, y1, x2, y2 = REF_SHELF
    corners = move([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], matrix, (1920, 1080))
    xs, ys = [c[0] for c in corners], [c[1] for c in corners]
    return out, f"{min(xs):.3f},{min(ys):.3f},{max(xs):.3f},{max(ys):.3f}"


# ---------------------------------------------------------------- 1) run

def run(pattern: str | None) -> None:
    for name in clips(pattern):
        out = OUT / name
        if (out / ".done").exists():
            print(f"{name}: 이미 있음 (건너뜀)")
            continue
        zones, shelf = zones_for(name)
        (out / "shelf.txt").write_text(shelf, encoding="utf-8")
        cmd = [sys.executable, "run_tracking.py", "--source", str(SOURCE / f"{name}.mp4"), "--zones", str(zones),
               "--shelf", f"shelf1={shelf}@shelf", "--out-dir", str(out), "--no-video", "--stride", str(STRIDE),
               "--shelf-log"]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        speed = next((l.split(":")[1].strip() for l in r.stdout.splitlines() if "처리 속도" in l), "?")
        if r.returncode == 0:
            (out / ".done").write_text("ok", encoding="utf-8")   # 중간에 끊긴 실행을 완료로 착각하지 않게
        print(f"{name}: 완료 ({speed})" if r.returncode == 0 else f"{name}: 실패\n{r.stderr[-800:]}", flush=True)


# ---------------------------------------------------------------- 2) pay

def counter_stays(name: str, min_seconds: float = 1.5) -> list[tuple[int, float, float]]:
    """(사람, 시작초, 끝초): 발이 계산대 구역에 min_seconds 이상 연속으로 있던 구간."""
    zm = ZoneMap.load(OUT / name / "zones.yaml")
    zm.resolve(1920, 1080)
    track: dict[int, list[tuple[float, bool]]] = {}
    for o in read_identities(OUT / name / "identities.jsonl"):
        if o.person_id <= 0:
            continue
        hit = any(h.zone_type == ZoneType.CHECKOUT and h.contains_foot for h in zm.evaluate(o.bbox))
        track.setdefault(o.person_id, []).append((o.frame / FPS, hit))
    stays = []
    for pid, seq in track.items():
        start = None
        for t, hit in seq + [(1e9, False)]:
            if hit and start is None:
                start = t
            elif not hit and start is not None:
                if prev - start >= min_seconds:
                    stays.append((pid, round(start, 1), round(prev, 1)))
                start = None
            prev = t
    return sorted(stays, key=lambda s: s[1])


def pay(pattern: str | None) -> None:
    LABELS.mkdir(parents=True, exist_ok=True)
    for name in clips(pattern):
        if name[0] == "T" or not (OUT / name / ".done").exists():
            continue
        stays = counter_stays(name)
        print(f"{name}: 계산대 체류 " + (", ".join(f"손님 {p} {a}~{b}s" for p, a, b in stays) or "없음"))
        # 확인용 장면표: 체류 가운데 장면
        cap, tiles, want = cv2.VideoCapture(str(SOURCE / f"{name}.mp4")), [], {int((a + b) / 2 * FPS): (p, a, b) for p, a, b in stays}
        i = 0
        while want and cap.grab():
            if i in want:
                _, im = cap.retrieve()
                p, a, b = want.pop(i)
                im = cv2.resize(im, (480, 270))
                cv2.putText(im, f"{name} P{p} {a}-{b}s", (6, 24), 0, 0.7, (0, 0, 255), 2)
                tiles.append(im)
            i += 1
        if tiles:
            cv2.imwrite(str(OUT / name / "counter_check.jpg"), cv2.hconcat(tiles))
        target = LABELS / f"payments_{name}.csv"
        if not target.exists():
            proposal = OUT / name / "payments_proposed.csv"
            with proposal.open("w", encoding="utf-8", newline="") as f:
                f.write("time_sec,items,person_id\n")
                for _, a, b in stays:
                    f.write(f"{(a + b) / 2:.1f},1,\n")


# ---------------------------------------------------------------- 3) judge

def load_takes(path: Path) -> list[TakeCandidate]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        out.append(TakeCandidate(**d))
    return out


def load_visits(path: Path) -> list[ShelfVisit]:
    if not path.exists():
        return []
    return [ShelfVisit(**json.loads(l)) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def replay_shelf(name: str, config: AppConfig, merges: list[dict], by_frame: dict, total: int) -> list[ShelfVisit]:
    """저장된 물건 탐지(shelf_items.jsonl)와 사람 위치로 선반 지도를 다시 돌린다 — 설정을 바꿔도 영상을 다시 안 읽는다."""
    out = OUT / name
    log = out / "shelf_items.jsonl"
    if not log.exists():
        return load_visits(out / "shelf_visits.jsonl")
    items_at = {}
    for line in log.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        items_at[r["frame"]] = [ShelfItem(l, (x1, y1, x2, y2), sc) for l, x1, y1, x2, y2, sc in r["items"]]
    x1, y1, x2, y2 = (float(v) for v in (out / "shelf.txt").read_text(encoding="utf-8").split(","))
    cfg = config.shelves
    monitor = ShelfMonitor("shelf1", (x1 * 1920, y1 * 1080, x2 * 1920, y2 * 1080), stable_seconds=cfg.stable_seconds,
                           map_score=cfg.map_score, weak_presence=cfg.weak_presence)
    per_window = max(1, int(round(cfg.stable_seconds * FPS / STRIDE)))
    alias: dict[int, int] = {}
    pending = sorted(merges, key=lambda m: m["frame"])

    def resolve(pid: int) -> int:
        while pid in alias:
            pid = alias[pid]
        return pid

    for idx in range(0, total, STRIDE):
        while pending and pending[0]["frame"] <= idx:
            m = pending.pop(0)
            alias[m["from"]] = m["into"]
        people = [(resolve(o.person_id) if o.person_id > 0 else -1, o.bbox) for o in by_frame.get(idx, [])]
        monitor.update(idx * 1000.0 / FPS, items_at.get(idx), people, per_window, idx)
    return [ShelfVisit(person_id=c.person_id if c.person_id is not None else -1, shelf="shelf1",
                       start_frame=c.start_frame, end_frame=c.end_frame, removed=len(c.removed), added=len(c.added),
                       zone="shelf", detail=c.describe()) for c in monitor.visits]


def judge_clip(name: str, config: AppConfig, use_shelf: bool = True) -> dict:
    out = OUT / name
    zm = ZoneMap.load(out / "zones.yaml")
    zm.resolve(1920, 1080)
    # 결제 기록이 없는 매장: 계산대에 서 있으면 결제, 다시 상품 쪽으로 가면 취소 (사용자 확인 규칙)
    config.risk.payment_mode = "dwell"
    feed = PaymentFeed.empty()
    engine = RiskEngine(config.risk, zm, feed)
    engine.reset()
    merges = [json.loads(l) for l in (out / "identity_merges.jsonl").read_text(encoding="utf-8").splitlines()] \
        if (out / "identity_merges.jsonl").exists() else []
    by_frame: dict[int, list] = {}
    for o in read_identities(out / "identities.jsonl"):
        by_frame.setdefault(o.frame, []).append(o)
    takes = load_takes(out / "take_candidates.jsonl")
    total = int(cv2.VideoCapture(str(SOURCE / f"{name}.mp4")).get(cv2.CAP_PROP_FRAME_COUNT))
    visits = replay_shelf(name, config, merges, by_frame, total) if use_shelf else []
    # 손 뻗기·선반 방문은 '끝난 프레임'에 들어온다 (파이프라인과 같은 시점)
    take_at: dict[int, list] = {}
    for t in takes:
        take_at.setdefault(t.end_frame, []).append(t)
    visit_at: dict[int, list] = {}
    for v in visits:
        visit_at.setdefault(v.end_frame, []).append(v)
    last = max(list(by_frame) + [0])
    frame = None
    for idx in range(0, last + 1):
        frame = Frame(index=idx, timestamp="", pts_ms=idx * 1000.0 / FPS, image=None)
        engine.update(frame, by_frame.get(idx, []), take_at.get(idx, []), merges, visit_at.get(idx, []))
    # 영상 끝까지 시간이 흐르게 (사람이 사라진 뒤 장면들)
    for idx in range(last + 1, total):
        frame = Frame(index=idx, timestamp="", pts_ms=idx * 1000.0 / FPS, image=None)
        engine.update(frame, [], take_at.get(idx, []), merges, visit_at.get(idx, []))   # 사람이 사라진 뒤 끝나는 기록도 넣는다
    engine.finish(frame, (1920, 1080))
    finals = [e for e in engine.events if e.level != RiskLevel.WARNING]
    levels = [e.level.value for e in finals]
    expected_theft = name[0] == "T"
    hit = "HIGH_RISK" in levels
    correct = hit if expected_theft else not hit
    return {"clip": name, "kind": kind(name), "verdicts": levels, "correct": correct,
            "review": levels.count("REVIEW"), "reasons": [f"P{e.person_id} {e.level.value}: {e.reason}" for e in finals],
            "payments": sum(1 for p in engine.book.payments if p.method == "dwell"),
            "cancels": sum(1 for p in engine.book.payments if p.method == "cancel")}


def judge(pattern: str | None, label: str = "", use_shelf: bool = True) -> list[dict]:
    config = AppConfig.load(ROOT / "config.yaml")
    rows = [judge_clip(n, config, use_shelf) for n in clips(pattern) if (OUT / n / ".done").exists()]
    for k in ("정상", "절도", "복합 정상"):
        part = [r for r in rows if r["kind"] == k]
        if part:
            ok = sum(r["correct"] for r in part)
            print(f"{label}{k}: {ok}/{len(part)} 정답" + (f", REVIEW {sum(r['review'] for r in part)}건" if k != "절도" else ""))
    for r in rows:
        mark = "OK " if r["correct"] else "XX "
        print(f"  {mark}{r['clip']:10s} {','.join(r['verdicts']) or '(판정 없음)'}  계산대 결제 {r['payments']}건"
              + (f", 취소 {r['cancels']}건" if r.get("cancels") else ""))
        if not r["correct"] or r["review"]:
            for reason in r["reasons"]:
                print(f"        {reason}")
    (OUT / f"results{label.strip() and '_' + label.strip()}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["run", "pay", "judge"])
    parser.add_argument("--only", help="영상 이름 패턴 (쉼표로 여러 개, 예: 'N*,T0*')")
    parser.add_argument("--no-shelf", action="store_true", help="judge: 선반 기록 없이 (손 뻗기만) — 원인 확인용")
    args = parser.parse_args()
    if args.stage == "judge":
        judge(args.only, "no_shelf " if args.no_shelf else "", use_shelf=not args.no_shelf)
    else:
        {"run": run, "pay": pay}[args.stage](args.only)


if __name__ == "__main__":
    main()
