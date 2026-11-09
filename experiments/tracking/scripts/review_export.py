"""카메라 2대 평가 기록 → '오판 장면 보기' 페이지 데이터 (outputs/review/).

각 설정(기준선·가장자리 변+다시 판정·전체 카메라 관절 …)의 기록을 dual_eval 과 같은 방식으로 다시 판정하고,
그 과정의 사람 상자·신원·관절·짝·손 접촉·선반 변화·판정을 장면 번호(공통 시각)별로 적어 둔다.
페이지는 원본 영상 위에 이것을 그려 '모델이 어디서 어떻게 잘못 봤는지'를 눈으로 보게 한다.

    python scripts/review_export.py                 # 전부
    python scripts/review_export.py --only C12,T2   # 일부 쌍만
    python scripts/review_serve.py                  # 페이지 열기

결과(outputs/review/)와 영상은 저장소에 올리지 않는다.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import dual_eval  # noqa: E402
import sync_pair  # noqa: E402
from src.core.config import AppConfig  # noqa: E402
from src.inputs.sync import SyncTable  # noqa: E402
from src.zones.floor import FloorMap  # noqa: E402

REVIEW = ROOT / "outputs" / "review"
REPO = ROOT.parents[1]

RUNS = [
    # key, 기록 폴더, 판정 때 옵션, 이름, 설명
    ("base", "dual_eval_base", {}, "기준선",
     "11-04 기준선. 가장자리 규칙이 네 변 모두, 되살아난 번호 다시 판정 없음."),
    ("edge", "dual_eval_edge", {}, "가장자리 변만",
     "카메라별 가장자리 변(전체 카메라 위쪽·선반 카메라 아래쪽 제외). 다시 판정 없음."),
    ("ankle", "dual_eval_pose3", {}, "관절 모델 + 발목 발 점 + 사용자 구역 (현재 기본)",
     "전체 카메라 yolov8n-pose, 바닥 구역 판단은 발목(B-15), 구역은 사용자가 편집 페이지에서 실제 바닥 자리에 맞춰 그린 것 + 외투 큰 상자용 사람 아님 영역(11-09). 팔 뻗기는 끔. 지금 카메라 2대 기본 설정."),
    ("edgegap", "dual_eval_edgegap", {"set": ["zones.foot_point=bbox"]}, "가장자리 변 + 다시 판정 (11-08 까지 기본)",
     "기존 탐지 모델(yolov8n), 상자 아래 가운데 발 점. 가장자리 변 + 1초 넘게 끊겼다 되살아난 추적 번호는 신원 다시 판정(3판) + 선반 앞 후보가 한 명이면 외형 0.85 이상으로 두 카메라 짝(11-07). 구역은 지금 적용본으로 판정하므로 그때 점수(32/38)와 다름."),
    ("posonly", "dual_eval_edgegap", {"set": ["zones.foot_point=bbox"], "fusion": {"solo_appearance_min": 0.0}}, "두 카메라 짝을 위치로만 (11-07 전)",
     "현재 기본과 같고, 두 카메라 짝을 위치 점수로만 정함(외형은 0.3 보조). 선반 카메라 띠 위치가 어긋나면 외형이 높아도 '짝 없음'."),
    ("tent", "dual_eval_tent", {}, "+ 임시 신원 1초",
     "가장자리 변 + 다시 판정 + 전체 카메라 새 신원은 임시로 두고 발이 매장 안 바닥에 1초 있어야 정식(대조·나중에 합치기 상대)."),
    ("gap1", "dual_eval_gap", {}, "다시 판정 1판 (가장자리 변 없음)",
     "되살아난 번호를 새 번호처럼 다시 판정. 가장자리 변 없이 시험 — C21 을 쪼갬."),
    ("gap3", "dual_eval_gap3", {}, "다시 판정 3판 (가장자리 변 없음)",
     "원래 주인 우대·가장자리면 미룸·작으면 보류. 가장자리 변 없이는 늘 '미룸'이라 효과 없음."),
    ("pose_reach", "dual_eval_pose", {"pose": True, "reach_takes": True, "set": ["zones.foot_point=bbox"]}, "관절 + 팔 뻗기를 집기로 (처음 구현)",
     "전체 카메라 yolov8n-pose. 팔 뻗기를 주인 정하기 + 손 뻗기(집기 행동)로 넘김 — C 거짓 경보 4개."),
    ("pose_owner", "dual_eval_pose", {"pose": True, "set": ["zones.foot_point=bbox"]}, "관절 + 팔 뻗기는 주인 정하기만",
     "전체 카메라 yolov8n-pose. 팔 뻗기는 주인 정하기에만 (지금 코드의 관절 켬 동작)."),
]

KP_KEEP = (5, 6, 7, 8, 9, 10, 15, 16)   # 어깨·팔꿈치·손목 (왼, 오) + 발목 (구역 편집 페이지의 발목 발 점, 2026-11-08)


# ---------------------------------------------------------------------- 판정 다시 돌리며 내부 붙잡기

class _Capture:
    judge = None


def _capturing(cls):
    class Captured(cls):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            _Capture.judge = self
    return Captured


dual_eval.DualJudge = _capturing(dual_eval.DualJudge)


def judge_with(run_dir: str, opts: dict, pair: str, base: AppConfig, table: SyncTable, band) -> tuple[dict, object]:
    dual_eval.OUT = ROOT / "outputs" / run_dir
    dual_eval.OVERVIEW_POSE = bool(opts.get("pose"))
    dual_eval.OVERRIDES = list(opts.get("set") or [])   # 예전 설정 기록은 그때의 발 점(상자)으로
    b = base
    if opts.get("reach_takes") or opts.get("fusion"):
        from copy import deepcopy
        b = deepcopy(base)
        if opts.get("reach_takes"):
            b.fusion.overview_reach_takes = True
        for k, v in (opts.get("fusion") or {}).items():
            setattr(b.fusion, k, v)
    result = dual_eval.judge_pair(pair, b, table.offset_ms(pair) or 0.0, band)
    return result, _Capture.judge


# ---------------------------------------------------------------------- 기록 → 보기 데이터

def _edge_sides(bbox, w, h, margin=0.02) -> str:
    x1, y1, x2, y2 = bbox
    s = ""
    if x1 <= margin * w:
        s += "L"
    if y1 <= margin * h:
        s += "T"
    if x2 >= (1 - margin) * w:
        s += "R"
    if y2 >= (1 - margin) * h:
        s += "B"
    return s


def _tracks(log, cam: str, resolve) -> dict:
    m = log.meta[cam]
    frames = {}
    for idx, obs in sorted(log.ids[cam].items()):
        rows = []
        for o in obs:
            x1, y1, x2, y2 = (round(v) for v in o.bbox)
            kp = []
            if o.keypoints:
                for j in KP_KEEP:
                    x, y, c = o.keypoints[j]
                    kp += [round(x), round(y), round(c * 100)]
            rows.append([
                o.person_id, resolve(o.person_id) if o.person_id > 0 else o.person_id, o.track_id,
                x1, y1, x2, y2,
                int(bool(o.is_new)) | (int(bool(o.stitched)) << 1),
                None if o.matched_similarity is None else round(o.matched_similarity, 3),
                kp,
            ])
        frames[idx] = rows
    return {"fps": m["fps"], "stride": m["stride"], "frames_total": m["frames"], "w": m["width"],
            "h": m["height"], "frames": frames}


def _births(log, cam: str) -> list[dict]:
    """새 신원이 생긴 순간 — 가장자리에 닿아 즉시 대조를 건너뛰었는지 (C-21)."""
    m = log.meta[cam]
    seen, out = set(), []
    for idx, obs in sorted(log.ids[cam].items()):
        for o in obs:
            if o.person_id > 0 and o.person_id not in seen:
                seen.add(o.person_id)
                out.append({"pid": o.person_id, "frame": idx, "ms": round(idx * 1000.0 / m["fps"], 1),
                            "edge": _edge_sides(o.bbox, m["width"], m["height"]), "track": o.track_id,
                            "sim": None if o.matched_similarity is None else round(o.matched_similarity, 3)})
    return out


def _revivals(log, cam: str, gap_ms: float = 1000.0) -> list[dict]:
    """추적 번호가 1초 넘게 끊겼다 되살아난 순간 (C-22)."""
    m = log.meta[cam]
    last, out = {}, []
    for idx, obs in sorted(log.ids[cam].items()):
        ms = idx * 1000.0 / m["fps"]
        for o in obs:
            prev = last.get(o.track_id)
            if prev is not None and ms - prev[0] > gap_ms:
                out.append({"track": o.track_id, "ms": round(ms, 1), "gap_ms": round(ms - prev[0]),
                            "before_pid": prev[1], "after_pid": o.person_id})
            last[o.track_id] = (ms, o.person_id)
    return out


def _shelf_items(run_dir: str, pair: str, w: int, h: int, fps: float, offset: float) -> list:
    p = ROOT / "outputs" / run_dir / pair / "angle0" / "shelf_items.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        items = [[n, round(x1), round(y1), round(x2), round(y2), round(c * 100)]
                 for n, x1, y1, x2, y2, c in r["items"] if c >= 0.25]
        out.append([r["frame"], items])
    return out


def _zones(m1: dict, m0: dict) -> dict:
    zdir = ROOT / "zones" / "store01_dual"
    floor_raw = yaml.safe_load((zdir / "floor.yaml").read_text(encoding="utf-8"))
    floor = FloorMap.load(zdir / "floor.yaml")
    w1, h1 = m1["width"], m1["height"]
    over = [{"name": z.name, "type": getattr(z.type, "value", str(z.type)),
             "poly": [[round(x), round(y)] for x, y in floor.image_polygon(z, w1, h1)]} for z in floor.zones]
    shelf_img = [[[round(x * w1), round(y * h1)] for x, y in poly] for poly in floor_raw.get("shelf_image") or []]
    ignore = [[round(v * (w1 if i % 2 == 0 else h1)) for i, v in enumerate(r)] for r in floor_raw.get("ignore_image") or []]
    d = yaml.safe_load((zdir / "angle0.yaml").read_text(encoding="utf-8"))
    w0, h0 = m0["width"], m0["height"]
    slots = [{"name": z["name"], "poly": [[round(x * w0), round(y * h0)] for x, y in z["polygon"]]} for z in d["zones"]]
    shelves = [{"name": s["name"], "box": [round(s["box"][0] * w0), round(s["box"][1] * h0),
                                            round(s["box"][2] * w0), round(s["box"][3] * h0)],
                **({"poly": [[round(x * w0), round(y * h0)] for x, y in s["polygon"]]} if s.get("polygon") else {})}
               for s in d["shelves"]]
    return {"overview": over, "shelf_image": shelf_img, "ignore": ignore, "slots": slots, "shelves": shelves}


def export_pair(key: str, run_dir: str, opts: dict, pair: str, base, table, band, truth) -> dict | None:
    d = ROOT / "outputs" / run_dir / pair
    if not (d / ".done").exists():
        return None
    result, judge = judge_with(run_dir, opts, pair, base, table, band)
    log = dual_eval.PairLog(pair)
    m1, m0 = log.meta["angle1"], log.meta["angle0"]
    f1, f0 = m1["fps"], m0["fps"]
    offset = table.offset_ms(pair) or 0.0
    r1 = log.resolver("angle1", m1["frames"])
    r0 = log.resolver("angle0", m0["frames"])
    risk = judge.risk
    video, ref = sync_pair.pair_files(pair)

    def ov_ms(frame):
        return round(frame * 1000.0 / f1, 1)

    events = [{"pid": e.person_id, "level": e.level.value, "zone": e.zone, "ms": round(e.time_sec * 1000.0, 1),
               "reason": e.reason, "bbox": [round(v) for v in e.bbox], "taken": e.taken, "paid": e.paid,
               "unpaid": e.unpaid} for e in judge.events]
    actions = [{"pid": a.person_id, "kind": a.kind, "name": a.name, "count": a.count, "evidence": a.evidence,
                "detail": a.detail, "start": ov_ms(a.start_frame), "end": ov_ms(a.end_frame),
                "text": a.describe()} for a in risk.all_actions()]
    customers = {pid: {"inside": c.seen_inside, "first": round(c.first_seen_ms), "last": round(c.last_seen_ms),
                       "paid": c.paid_items + c.dwell_paid_items}
                 for pid, c in risk.book.customers.items()}
    payments = [{"ms": round(p.time_sec * 1000.0), "items": p.items, "pid": p.person_id,
                 "method": getattr(p, "method", "")} for p in risk.book.payments]
    links = [{"start": round(l.start_ms), "end": round(l.end_ms), "detail": l.detail_id, "overview": l.overview_id,
              "score": round(l.score, 2)} for l in judge.linker.links]
    takes0 = [{"pid": t.person_id, "zone": t.zone, "start": round(t.start_frame * 1000.0 / f0 + offset),
               "end": round(t.end_frame * 1000.0 / f0 + offset)} for t in log.takes["angle0"]]
    takes1 = [{"pid": t.person_id, "zone": t.zone, "start": ov_ms(t.start_frame), "end": ov_ms(t.end_frame)}
              for t in log.takes["angle1"]]
    visits = [{"pid": v["person_id"], "start": round(v["start_ms"] + offset), "end": round(v["end_ms"] + offset),
               "removed": v["removed"], "added": v["added"], "detail": v.get("detail", ""),
               "slot_change": v.get("slot_change")} for v in log.visits]
    merges = {cam: [{"ms": round(mm.get("frame", 0) * 1000.0 / log.meta[cam]["fps"] + (offset if cam == "angle0" else 0)),
                     "from": mm["from"], "into": mm["into"], "sim": mm.get("similarity")}
                    for mm in log.merges[cam]] for cam in ("angle0", "angle1")}
    data = {
        "run": key, "pair": pair, "offset_ms": offset,
        "video": {"angle1": "/" + ref.relative_to(REPO).as_posix(), "angle0": "/" + video.relative_to(REPO).as_posix()},
        "truth": truth.get(pair, {}),
        "result": {k: result[k] for k in ("verdicts", "correct", "reasons", "people", "identities", "links",
                                          "payments", "cancels", "unowned_takes")},
        "zones": _zones(m1, m0),
        "tracks": {"angle1": _tracks(log, "angle1", r1), "angle0": _tracks(log, "angle0", r0)},
        "births": {"angle1": _births(log, "angle1"), "angle0": _births(log, "angle0")},
        "revivals": {"angle1": _revivals(log, "angle1"), "angle0": _revivals(log, "angle0")},
        "events": events, "actions": actions, "customers": customers, "payments": payments, "links": links,
        "takes0": takes0, "takes1": takes1, "visits": visits, "merges": merges,
        "shelf_events": result["shelf_events"],
        "shelf_items": _shelf_items(run_dir, pair, m0["width"], m0["height"], f0, offset),
    }
    out = REVIEW / "data" / key / f"{pair}.js"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"HE.loaded({json.dumps(key + '/' + pair)}, {json.dumps(data, ensure_ascii=False, separators=(',', ':'))});\n",
                   encoding="utf-8")
    return {"verdicts": result["verdicts"], "correct": result["correct"], "people": result["people"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="쌍 이름 (쉼표)")
    ap.add_argument("--runs", help="설정 key (쉼표)")
    args = ap.parse_args()
    base = AppConfig.load(ROOT / "config.yaml")
    table = SyncTable.load(dual_eval.LABELS / "sync.csv")
    band = dual_eval.load_band_map()
    truth = {r["pair"]: {"people": int(r["people"]), "expected": r["expected"], "notes": r["notes"], "set": r["set"]}
             for r in csv.DictReader(open(dual_eval.LABELS / "truth.csv", encoding="utf-8"))}
    only = set(args.only.split(",")) if args.only else None
    keys = set(args.runs.split(",")) if args.runs else None
    index_path = REVIEW / "data" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {"summary": {}}
    for key, run_dir, opts, name, desc in RUNS:
        if keys and key not in keys:
            continue
        for pair in sync_pair.all_pairs():
            if only and pair not in only:
                continue
            s = export_pair(key, run_dir, opts, pair, base, table, band, truth)
            if s is not None:
                index["summary"].setdefault(key, {})[pair] = s
                print(f"{key:10s} {pair:6s} {','.join(s['verdicts']) or '(판정 없음)'}  손님 {s['people']}", flush=True)
    index["runs"] = [{"key": k, "dir": d, "name": n, "desc": ds} for k, _d, _o, n, ds in RUNS for d in [_d]]
    index["pairs"] = sync_pair.all_pairs()
    index["truth"] = truth
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    (REVIEW / "data" / "index.js").write_text("HE.index(" + json.dumps(index, ensure_ascii=False) + ");\n",
                                              encoding="utf-8")
    shutil.copyfile(ROOT / "scripts" / "review_viewer.html", REVIEW / "index.html")
    shutil.copyfile(ROOT / "scripts" / "review_cases.js", REVIEW / "cases.js")
    print(f"보기 페이지: {REVIEW / 'index.html'}  →  python scripts/review_serve.py")


if __name__ == "__main__":
    main()
