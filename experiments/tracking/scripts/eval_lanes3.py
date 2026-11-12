"""3차 선반 카메라(angle0) 레인 그림 비교로 꺼냄/넣음을 찾고, 사람 정답(takes.csv)과 대조한다.

    python scripts/eval_lanes3.py pose            # 10Hz 사람 상자·손목 (yolov8n-pose) → outputs/lanes3/pose/<짝>.csv
    python scripts/eval_lanes3.py diff            # 레인 그림 비교 → outputs/lanes3/episodes.csv, status.csv
    python scripts/eval_lanes3.py eval            # 정답·센서와 대조 → 화면 요약, outputs/lanes3/eval.csv
    python scripts/eval_lanes3.py diff --set min_ratio=0.1   # SlotDiffConfig 값 바꿔 보기

레인 구역은 zones/store01_shoot3/angle0.yaml (area_setting 기준) 을 영상마다 옮겨 쓴다 (shoot3_zones.py).
그림 비교는 src/shelf/slot_diff.py SlotDiff 그대로 — 사람 상자가 레인을 가렸다가 다시 0.6초 안 가리면 전후 그림을 비교.

대조 규칙 (docs/multiview-progress.md 후속 20)
  후보   같은 레인, 바뀜(changed=1), 기준과 비교한(compared) 가림 구간 중 정답 시각 t ∈ [시작−1초, 끝+1초]
  배정   정답 하나는 최대 한 변화에 (t 를 품은 구간 > 가까운 구간, 방향 안 봄). 한 변화에 정답 여럿 허용.
  재현율 = 배정된 정답 / (정답 − 처리 실패 − 되돌림 쌍)          (정답 행 단위)
  정밀도 = 정답이 하나라도 붙은 변화 / 전체 변화                (변화 단위)
  방향   줄어듦↔꺼냄 · 늘어남↔넣음 = 맞음, 반대 = 틀림, 바뀜 = 모름 (배정된 정답 단위)
  되돌림 쌍  같은 레인 꺼냄·넣음이 같은 compared 구간(±1초) 안이고 그 구간이 안 바뀜 → 정상, 재현율에서 뺌
  센서 성공  sensor_check.csv 에서 그 정답 행이 맞음·경계·늦음 (센서는 방향까지 맞아야 짝지어짐)
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.shelf.slot_diff import SlotDiff, SlotDiffConfig  # noqa: E402

VIDEOS = ROOT.parent.parent / "source" / "2(angle0)"
LABELS = ROOT / "labels" / "store01_shoot3"
OUT = ROOT / "outputs" / "lanes3"
POSE = OUT / "pose"
POSE_MODEL = "yolov8n-pose.pt"
POSE_CONF = 0.3
HZ = 10.0
LANES = ("L1", "L2", "L3")
WINDOW_SEC = 1.0
POSE_FIELDS = ["ms", "frame", "person", "x1", "y1", "x2", "y2", "conf", "lwx", "lwy", "lwc", "rwx", "rwy", "rwc"]
EPISODE_FIELDS = ["pair", "lane", "start_ms", "end_ms", "state", "ratio", "baseline", "changed", "direction", "edge_ratio"]
SENSOR_OK = ("맞음", "경계", "늦음")


def pairs() -> list[str]:
    return sorted(p.stem for p in VIDEOS.glob("*.mp4") if p.stem != "area_setting")


# ---------------------------------------------------------------- 1. 사람 상자·손목

def pose_one(model, path: Path, hz: float = HZ) -> list[dict]:
    """실제 장면 시각 기준 hz 로 표본 (sync_floor extract 와 같은 방식). 사람이 없으면 person=-1 한 줄."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"{path} 열기 실패")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows, idx, nxt = [], -1, 0.0
    while cap.grab():
        idx += 1
        ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if idx == 0:
            ms = max(ms, 0.0) if ms < 1000.0 / fps else 0.0   # 첫 장면 시각은 엉뚱할 때가 있다 (frame_times.first_ms)
        if ms + 1e-6 < nxt:
            continue
        nxt += 1000.0 / hz
        ok, frame = cap.retrieve()
        if not ok:
            cap.release()
            raise ValueError(f"{path.name}: 장면 {idx} 읽기 실패 — 캐시를 쓰지 않음")
        res = model.predict(frame, imgsz=640, conf=POSE_CONF, classes=[0], verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            rows.append({"ms": f"{ms:.3f}", "frame": idx, "person": -1, **{k: "" for k in POSE_FIELDS[3:]}})
            continue
        kps = res.keypoints.data.cpu().numpy() if res.keypoints is not None else None
        for i, (box, conf) in enumerate(zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy())):
            kp = kps[i] if kps is not None else np.zeros((17, 3))
            rows.append({"ms": f"{ms:.3f}", "frame": idx, "person": i,
                         **{k: f"{v:.1f}" for k, v in zip(("x1", "y1", "x2", "y2"), box)}, "conf": f"{conf:.2f}",
                         "lwx": f"{kp[9][0]:.1f}", "lwy": f"{kp[9][1]:.1f}", "lwc": f"{kp[9][2]:.2f}",
                         "rwx": f"{kp[10][0]:.1f}", "rwy": f"{kp[10][1]:.1f}", "rwc": f"{kp[10][2]:.2f}"})
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if total and idx + 1 < total - 1:
        raise ValueError(f"{path.name}: 장면 {idx + 1}/{total} 에서 읽기가 끊김 — 캐시를 쓰지 않음")
    return rows


def covers(poses: dict, total: int, fps: float, hz: float = HZ) -> bool:
    """캐시가 영상 끝까지 표본을 가졌나 (마지막 표본이 끝에서 표본 간격 + 1장 안)."""
    return not total or max(poses) >= total - 1 - int(round(fps / hz)) - 1


def pose(force: bool, only: list[str] | None) -> None:
    from ultralytics import YOLO

    model = YOLO(str(ROOT / POSE_MODEL))
    POSE.mkdir(parents=True, exist_ok=True)
    todo = [p for p in pairs() if not only or p in only]
    for k, pair in enumerate(todo, 1):
        out = POSE / f"{pair}.csv"
        if out.exists() and not force:
            continue
        rows = pose_one(model, VIDEOS / f"{pair}.mp4")
        tmp = out.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=POSE_FIELDS)
            w.writeheader()
            w.writerows(rows)
        tmp.replace(out)
        print(f"  [{k}/{len(todo)}] {pair}: 표본 {len({r['frame'] for r in rows})}, 사람 {sum(r['person'] != -1 for r in rows)}",
              flush=True)


def load_pose(path: Path) -> dict[int, tuple[float, list[tuple[float, float, float, float]]]]:
    """장면 번호 → (시각 ms, 사람 상자들)."""
    out: dict[int, tuple[float, list]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            fr = int(r["frame"])
            ms, boxes = out.get(fr, (float(r["ms"]), []))
            if str(r["person"]).strip() not in ("", "-1"):
                boxes.append(tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2")))
            out[fr] = (ms, boxes)
    return out


# ---------------------------------------------------------------- 2. 레인 그림 비교

def parse_set(items: list[str] | None) -> SlotDiffConfig:
    """--set 키=값 → SlotDiffConfig (모르는 키는 오류)."""
    cfg = SlotDiffConfig()
    names = [f.name for f in dataclasses.fields(SlotDiffConfig)]
    for item in items or []:
        key, _, val = item.partition("=")
        key = key.strip()
        if key not in names:
            raise SystemExit(f"모르는 설정: {key} (가능: {', '.join(names)})")
        cur = getattr(cfg, key)
        setattr(cfg, key, int(float(val)) if isinstance(cur, int) else float(val))
    return cfg


def lane_boxes(pair: str, zones: dict, homs: dict) -> list[tuple[str, tuple[float, float, float, float]]]:
    import shoot3_zones as Z

    polys = Z.clip_polygons(pair, zones, homs)
    return [(k, (float(polys[k][:, 0].min()), float(polys[k][:, 1].min()),
                 float(polys[k][:, 0].max()), float(polys[k][:, 1].max()))) for k in LANES]


def dump_pair(sd: SlotDiff, before: dict, n_before: int, frame, out_dir: Path, pair: str) -> None:
    """이번 장면에 끝난 compared 가림마다 [기준 | 지금] 레인 그림을 저장 (눈 확인용)."""
    for e in sd.episodes[n_before:]:
        if e["state"] != "compared" or before.get(e["slot"]) is None:
            continue
        s = next(x for x in sd.slots if x.name == e["slot"])
        x1, y1, x2, y2 = s.box
        cur = frame[max(0, y1):y2, max(0, x1):x2]
        ref = before[e["slot"]]
        if ref.shape != cur.shape:
            continue
        img = cv2.resize(np.hstack([ref, np.full((ref.shape[0], 4, 3), 255, np.uint8), cur]), None, fx=4, fy=4,
                         interpolation=cv2.INTER_NEAREST)
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = "CH" if e["changed"] else "same"
        cv2.imwrite(str(out_dir / f"{pair}_{e['slot']}_{e['start_ms'] / 1000:05.1f}-{e['end_ms'] / 1000:05.1f}_{tag}"
                                  f"_r{e['ratio']:.2f}_b{e['baseline']:.2f}.png"), img)


def diff_one(path: Path, boxes, poses, cfg: SlotDiffConfig, dump: Path | None = None,
             pair: str = "") -> tuple[list[dict], int]:
    sd = SlotDiff(boxes, cfg)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"{path} 열기 실패")
    if not poses:
        cap.release()
        raise ValueError("사람 캐시가 비었다")
    total, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not covers(poses, total, fps):
        cap.release()
        raise ValueError(f"사람 캐시가 장면 {max(poses)} 에서 끝남 (영상 {total}장) — pose --force 로 다시")
    idx, last_ms, used = -1, 0.0, 0
    want = max(poses)
    while idx < want and cap.grab():
        idx += 1
        if idx not in poses:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            cap.release()
            raise ValueError(f"장면 {idx} 읽기 실패")
        ms, people = poses[idx]
        before = {s.name: s.reference for s in sd.slots} if dump else {}
        n_before = len(sd.episodes)
        sd.update(ms, frame, people)
        if dump:
            dump_pair(sd, before, n_before, frame, dump, pair)
        last_ms, used = ms, used + 1
    cap.release()
    if idx < want:
        raise ValueError(f"영상이 장면 {idx} 에서 끝남 (캐시는 {want} 까지)")
    sd.finish(last_ms)
    return sd.episodes, used


def fmt(v, spec: str) -> str:
    return "" if v is None else format(v, spec)


def diff(only: list[str] | None, cfg: SlotDiffConfig, dump: bool = False) -> None:
    import shoot3_zones as Z

    zones, homs = Z.load_zones(), Z.load_homographies()
    episodes, status = [], []
    for pair in pairs():
        if only and pair not in only:
            continue
        cache = POSE / f"{pair}.csv"
        reason = "카메라 맞추기 없음" if pair not in homs else ("사람 캐시 없음 (pose 먼저)" if not cache.exists() else "")
        if reason:
            status.append({"pair": pair, "ok": 0, "reason": reason, "frames": 0})
            print(f"  {pair}: {reason}")
            continue
        try:
            eps, used = diff_one(VIDEOS / f"{pair}.mp4", lane_boxes(pair, zones, homs), load_pose(cache), cfg,
                                 OUT / "dump" if dump else None, pair)
        except Exception as e:  # noqa: BLE001 — 한 영상 실패는 기록하고 넘어간다
            status.append({"pair": pair, "ok": 0, "reason": f"{type(e).__name__}: {e}", "frames": 0})
            print(f"  {pair}: 실패 {e}")
            continue
        for e in eps:
            episodes.append({"pair": pair, "lane": e["slot"], "start_ms": f"{e['start_ms']:.1f}",
                             "end_ms": f"{e['end_ms']:.1f}", "state": e["state"], "ratio": fmt(e["ratio"], ".4f"),
                             "baseline": fmt(e["baseline"], ".4f"), "changed": e["changed"],
                             "direction": e["direction"], "edge_ratio": fmt(e["edge_ratio"], ".3f")})
        status.append({"pair": pair, "ok": 1, "reason": "", "frames": used})
        ch = [e for e in eps if e["changed"]]
        print(f"  {pair}: 가림 {len(eps)}, 변화 {len(ch)} "
              + " ".join(f"{e['slot']}{e['direction']}@{e['start_ms'] / 1000:.1f}-{e['end_ms'] / 1000:.1f}" for e in ch),
              flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "episodes.csv", EPISODE_FIELDS, episodes)
    write_csv(OUT / "status.csv", ["pair", "ok", "reason", "frames"], status)
    print(f"설정 {dataclasses.asdict(cfg)}\n→ {OUT / 'episodes.csv'}")


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


# ---------------------------------------------------------------- 3. 대조

@dataclass
class Label:
    pair: str
    seq: int
    t_ms: float
    lane: str
    action: str


@dataclass
class Episode:
    pair: str
    lane: str
    start_ms: float
    end_ms: float
    state: str
    changed: bool
    direction: str


@dataclass
class Result:
    label: Label
    video: str = ""
    """찾음 / 놓침 / 되돌림 쌍 / 처리 실패"""
    cause: str = ""
    episode: int | None = None
    direction: str = ""
    """맞음 / 틀림 / 모름 (찾음일 때)"""
    sensor: str = ""
    sensor_ok: bool = False


def load_labels(path: Path = LABELS / "takes.csv") -> list[Label]:
    out = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("time_sec") or "").strip() and r["action"] in ("take", "return"):
                out.append(Label(r["pair"], int(r["seq"]), float(r["time_sec"]) * 1000.0, r["lane"], r["action"]))
    return out


def load_episodes(path: Path = OUT / "episodes.csv") -> list[Episode]:
    with path.open(encoding="utf-8", newline="") as f:
        return [Episode(r["pair"], r["lane"], float(r["start_ms"]), float(r["end_ms"]), r["state"],
                        str(r["changed"]) == "1", r["direction"]) for r in csv.DictReader(f)]


def load_status(path: Path = OUT / "status.csv") -> dict[str, bool]:
    with path.open(encoding="utf-8", newline="") as f:
        return {r["pair"]: str(r["ok"]) == "1" for r in csv.DictReader(f)}


def load_sensor_check(path: Path = LABELS / "sensor_check.csv") -> dict[tuple, str]:
    """(pair, lane, action, 정답 시각 ms 반올림) → 센서 판정 종류."""
    out: dict[tuple, str] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("label_sec") or "").strip():
                out[(r["pair"], r["lane"], r["direction"], round(float(r["label_sec"]) * 1000.0))] = r["kind"]
    return out


def near(e: Episode, t_ms: float, window_ms: float) -> bool:
    return e.start_ms - window_ms <= t_ms <= e.end_ms + window_ms


def gap(e: Episode, t_ms: float) -> float:
    return 0.0 if e.start_ms <= t_ms <= e.end_ms else min(abs(t_ms - e.start_ms), abs(t_ms - e.end_ms))


DIRECTION = {("줄어듦", "take"): "맞음", ("늘어남", "return"): "맞음",
             ("줄어듦", "return"): "틀림", ("늘어남", "take"): "틀림"}
CAUSE = {"no_reference": "기준 없음(첫 가림)", "unfinished": "끝까지 가림", "compared": "비교했지만 변화 없음"}


def evaluate(labels: list[Label], episodes: list[Episode], ok: dict[str, bool],
             sensor: dict[tuple, str] | None = None,
             window_ms: float = WINDOW_SEC * 1000.0) -> tuple[list[Result], list[dict]]:
    """정답 행마다 결과와, 변화(compared·changed)마다 {episode, lane, labels}. 규칙은 모듈 설명."""
    sensor = sensor or {}
    results = [Result(lb) for lb in labels]
    by: dict[tuple, list[int]] = {}
    for i, e in enumerate(episodes):
        by.setdefault((e.pair, e.lane), []).append(i)
    for r in results:
        lb = r.label
        r.sensor = sensor.get((lb.pair, lb.lane, lb.action, round(lb.t_ms)), "")
        r.sensor_ok = r.sensor in SENSOR_OK
        if not ok.get(lb.pair, False):
            r.video, r.cause = "처리 실패", "영상 처리 실패"
    # 되돌림 쌍: 같은 레인 꺼냄 뒤 넣음이 같은 compared·안 바뀐 구간 안
    for ra in results:
        a = ra.label
        if ra.video or a.action != "take":
            continue
        for rb in results:
            b = rb.label
            if rb.video or b.pair != a.pair or b.lane != a.lane or b.action != "return" or b.t_ms < a.t_ms:
                continue
            same = [k for k in by.get((a.pair, a.lane), []) if episodes[k].state == "compared"
                    and not episodes[k].changed and near(episodes[k], a.t_ms, window_ms)
                    and near(episodes[k], b.t_ms, window_ms)]
            if same:
                for x in (ra, rb):
                    x.video, x.cause, x.episode = "되돌림 쌍", "같은 가림 안 꺼냄·넣음, 그림 그대로", same[0]
                break
    # 배정: 정답 하나 → 최대 한 변화 (구간 안 > 가까운 구간 > 먼저 시작, 방향 안 봄)
    for r in results:
        if r.video:
            continue
        lb = r.label
        idxs = by.get((lb.pair, lb.lane), [])
        cands = [k for k in idxs if episodes[k].state == "compared" and episodes[k].changed
                 and near(episodes[k], lb.t_ms, window_ms)]
        if cands:
            k = min(cands, key=lambda k: (gap(episodes[k], lb.t_ms), episodes[k].start_ms))
            r.video, r.episode = "찾음", k
            r.direction = DIRECTION.get((episodes[k].direction, lb.action), "모름")
            continue
        r.video = "놓침"
        around = [k for k in idxs if near(episodes[k], lb.t_ms, window_ms)]
        if not around:
            r.cause = "가림 구간 없음"
        else:
            k = min(around, key=lambda k: gap(episodes[k], lb.t_ms))
            r.cause, r.episode = CAUSE.get(episodes[k].state, episodes[k].state), k
    changes = [{"episode": k, "lane": e.lane, "labels": [r for r in results if r.video == "찾음" and r.episode == k]}
               for k, e in enumerate(episodes) if e.state == "compared" and e.changed and ok.get(e.pair, False)]
    return results, changes


def lane_stats(results: list[Result], changes: list[dict], lane: str | None) -> dict:
    """lane=None 이면 전체. 재현율은 정답 행 단위, 정밀도는 변화 단위."""
    rs = [r for r in results if lane is None or r.label.lane == lane]
    cs = [c for c in changes if lane is None or c["lane"] == lane]
    excluded = sum(r.video in ("처리 실패", "되돌림 쌍") for r in rs)
    found = [r for r in rs if r.video == "찾음"]
    denom = len(rs) - excluded
    fake = sum(not c["labels"] for c in cs)
    return {"labels": len(rs), "excluded": excluded, "found": len(found),
            "recall": len(found) / denom if denom else 0.0, "changes": len(cs), "fake": fake,
            "precision": (len(cs) - fake) / len(cs) if cs else 0.0,
            "dir_ok": sum(r.direction == "맞음" for r in found), "dir_bad": sum(r.direction == "틀림" for r in found),
            "dir_unknown": sum(r.direction == "모름" for r in found),
            "multi": sum(len(c["labels"]) >= 2 for c in cs)}


def crosstab(results: list[Result], lane: str | None, video_ok) -> dict:
    rs = [r for r in results if (lane is None or r.label.lane == lane) and r.video not in ("처리 실패", "되돌림 쌍")]
    both = sum(video_ok(r) and r.sensor_ok for r in rs)
    v = sum(video_ok(r) and not r.sensor_ok for r in rs)
    s = sum(not video_ok(r) and r.sensor_ok for r in rs)
    return {"both": both, "video_only": v, "sensor_only": s, "neither": len(rs) - both - v - s, "n": len(rs)}


def summarize(results: list[Result], changes: list[dict]) -> list[str]:
    lines = [f"{'레인':<5}{'정답':>5}{'제외':>5}{'찾음':>6}{'재현율':>8}{'변화':>6}{'가짜':>6}{'정밀도':>8}"
             f"{'방향맞음':>9}{'틀림':>5}{'모름':>5}{'여러개':>7}"]
    for lane in LANES + (None,):
        s = lane_stats(results, changes, lane)
        lines.append(f"{lane or '전체':<5}{s['labels']:>5}{s['excluded']:>5}{s['found']:>6}{s['recall']:>8.0%}"
                     f"{s['changes']:>6}{s['fake']:>6}{s['precision']:>8.0%}{s['dir_ok']:>9}{s['dir_bad']:>5}"
                     f"{s['dir_unknown']:>5}{s['multi']:>7}")
    causes = _count(r.cause for r in results if r.video == "놓침")
    lines.append("놓친 원인: " + (", ".join(f"{k} {v}" for k, v in sorted(causes.items(), key=lambda kv: -kv[1])) or "-"))
    pairs_ = sum(r.video == "되돌림 쌍" for r in results) // 2
    lines.append(f"되돌림 쌍(정상, 재현율에서 뺌): {pairs_}쌍")
    for title, video_ok in (("(a) 영상 찾음(방향 무관) × 센서 성공(맞음·경계·늦음, 방향 포함)", lambda r: r.video == "찾음"),
                            ("(b) 영상 방향 맞음 × 센서 성공", lambda r: r.video == "찾음" and r.direction == "맞음")):
        lines.append(title)
        for lane in LANES + (None,):
            c = crosstab(results, lane, video_ok)
            lines.append(f"  {lane or '전체':<5} 둘 다 {c['both']:>3}  영상만 {c['video_only']:>3}  센서만 {c['sensor_only']:>3}"
                         f"  둘 다 놓침 {c['neither']:>3}  (정답 {c['n']})")
    return lines


def _count(items) -> dict:
    out: dict = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out


def run_eval() -> None:
    episodes = load_episodes()
    results, changes = evaluate(load_labels(), episodes, load_status(), load_sensor_check())
    for line in summarize(results, changes):
        print(line)
    fakes = [episodes[c["episode"]] for c in changes if not c["labels"]]
    if fakes:
        print("가짜 변화: " + ", ".join(f"{e.pair} {e.lane}{e.direction}@{e.start_ms / 1000:.1f}-{e.end_ms / 1000:.1f}"
                                    for e in fakes))
    missed = [r for r in results if r.video == "놓침"]
    if missed:
        print("놓침: " + ", ".join(f"{r.label.pair} {r.label.lane} {r.label.action}@{r.label.t_ms / 1000:.2f}({r.cause})"
                                 for r in missed))
    rows = [{"pair": r.label.pair, "seq": r.label.seq, "lane": r.label.lane, "action": r.label.action,
             "time_sec": f"{r.label.t_ms / 1000:.2f}", "video": r.video, "cause": r.cause, "direction": r.direction,
             "episode_start": "" if r.episode is None else f"{episodes[r.episode].start_ms / 1000:.2f}",
             "episode_end": "" if r.episode is None else f"{episodes[r.episode].end_ms / 1000:.2f}",
             "sensor": r.sensor, "sensor_ok": int(r.sensor_ok)} for r in results]
    write_csv(OUT / "eval.csv", list(rows[0]) if rows else ["pair"], rows)
    print(f"→ {OUT / 'eval.csv'}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["pose", "diff", "eval"])
    ap.add_argument("--only", help="짝 이름 쉼표 구분")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--set", action="append", metavar="키=값", help="diff: SlotDiffConfig 값 (여러 번)")
    ap.add_argument("--dump", action="store_true", help="diff: 비교한 가림마다 [기준|지금] 레인 그림 저장 (outputs/lanes3/dump)")
    args = ap.parse_args()
    only = [p.strip() for p in args.only.split(",")] if args.only else None
    if args.stage == "pose":
        pose(args.force, only)
    elif args.stage == "diff":
        diff(only, parse_set(args.set), args.dump)
    else:
        run_eval()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
