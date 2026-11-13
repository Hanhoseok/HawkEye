"""3차 촬영(카메라 1대 + 아두이노 레인 센서) 선반 행동 정답 라벨링 도구.

센서는 놓치고(T05), 늦고, 가짜로 찍기도 했다(T01 11초 ADD — 선반 앞에 아무도 없음).
그래서 **정답은 영상을 보고 사람이 찍는다**: 언제, 어느 레인에서, 집었는지/되돌렸는지, 몇 개.
화면에 그 영상의 센서 기록을 같이 띄워 찍으면서 대조하고, --compare 로 센서 정확도를 센다.

    python scripts/label_takes.py               # 검수 완료 안 된 영상부터 차례로
    python scripts/label_takes.py T01 T05       # 고른 영상만
    python scripts/label_takes.py --summary     # 찍은 결과 요약
    python scripts/label_takes.py --compare     # 검수 완료 영상만 센서와 대조
    python scripts/label_takes.py --review      # 대조에서 문제 난 곳(놓침·가짜·늦음)만 차례로 — 센서 보이게, 보기 전용

먼저 센서를 영상별로 나눠 둔다: python scripts/split_sensor_events.py "../../source/2(angle1)"

**시각 기준은 angle0**(선반 정면 카메라) 영상 시각이다. 센서 기록도 split_sensor_events 가 카메라 시차(sync.csv)로
angle0 시각에 옮겨 둔다 — sensor_index.csv 의 axis 열이 angle0 이 아니면 라벨·대조·보정을 하지 않는다.
옆 화면에 angle1 을 시차만큼 맞춰 같이 보여준다 (--single 로 끔).

시각은 영상 재생 시각(초). 손이 상품을 칸에서 들어 올린(또는 내려놓은) 순간에 찍는다.

조작키 (화면 HUD 에도 표시됨)
    SPACE   재생 / 일시정지
    ← →     1 프레임 뒤로 / 앞으로 (a d 도 됨)
    ↑ ↓     1초 뒤로 / 앞으로
    1 2 3   레인 L1 칸타타 / L2 동과루 / L3 애플사이다      4  센서 없는 다른 진열대(other)
    + -     수량 (1~9, 기본 1, 찍고 나면 1 로 돌아감)
    t       집음(take)      — 지금 시각, 고른 레인·수량
    r       되돌림(return)  — 지금 시각, 고른 레인·수량
    n       이 영상은 선반 행동 없음(none)
    u       마지막 줄 삭제
    g / b   다음 / 이전 센서 이벤트 시각으로 이동
    c       검수 완료 표시 (영상을 끝까지 보고 빠짐없이 찍었을 때). 다시 누르면 취소
    TAB     다음 영상 (저장)      BACKSPACE  이전 영상 (저장)
    s       저장          q/ESC   저장하고 종료

결과
    labels/store01_shoot3/takes.csv         pair,seq,time_sec,lane,action,qty,checked_by,note
    labels/store01_shoot3/takes_status.csv  pair,status,checked_by,updated     (status: draft / done)
    labels/store01_shoot3/sensor_check.csv  (--compare) 정답·센서 한 단위씩 대조 결과

라벨을 고치면(t r n u) 그 영상은 draft 로 돌아간다. --compare 는 done 영상만 센다 — 중간에 멈춘 영상의
아직 안 찍은 행동이 '센서 가짜'로 세지지 않게.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from label_payments import KEY_DOWN, KEY_UP, FrameCache, normalize_key  # noqa: E402
from src.inputs.sync import SyncTable  # noqa: E402

SOURCE = ROOT.parent.parent / "source" / "2(angle0)"
"""기준 화면·시각: 선반을 정면에서 보는 angle0."""
SIDE_SOURCE = ROOT.parent.parent / "source" / "2(angle1)"
"""옆 화면: angle1 (센서 시계를 연결한 영상). sync.csv 시차만큼 맞춰 보여준다."""
AXIS = "angle0"
LABELS = ROOT / "labels" / "store01_shoot3"
SYNC = LABELS / "sync.csv"
OUT = LABELS / "takes.csv"
STATUS = LABELS / "takes_status.csv"
CHECK = LABELS / "sensor_check.csv"
OFFSETS = LABELS / "sensor_offsets.csv"
STATUS_FIELDS = ("pair", "status", "checked_by", "updated", "mode")
OFFSET_FIELDS = ("pair", "shift_sec", "n", "spread_sec", "applied", "reason", "base_sec", "checked_by", "updated")
FIT_MIN_ROWS = 2        # 영상별 보정: 서로 다른 정답 행이 이만큼은 짝지어져야
FIT_MAX_SPREAD = 1.0    # 영상 안 dt 범위(최대-최소)가 이 안이어야 '일정한 어긋남'
FIT_MAX_SHIFT = 1.2     # 기록 시각 초 단위 절단(0~1초) + 측정 여유 — 넘으면 시각 오차가 아니라고 본다
FIELDS = ("pair", "seq", "time_sec", "lane", "action", "qty", "checked_by", "note")
LANES = ("L1", "L2", "L3", "other")
LANE_NAME = {"L1": "L1 cantata", "L2": "L2 donggwaru", "L3": "L3 cider", "other": "other shelf"}
ACTIONS = ("take", "return", "none")

# 대조 규칙 (센서 시각 - 정답 시각, 초)
WINDOW = (-2.0, 8.0)   # 이 안이면 같은 행동의 센서 기록 후보. +8 = 늦게 찍히는 것까지 허용
ON_TIME = 1.5          # |dt| <= 1.5 맞음, 넘으면 늦음(+) / 이름(-)
BORDER = 1.0           # | |dt| - 1.5 | < 1.0 이면 경계 — 시각 정밀도(±1초) 안이라 맞음/늦음을 단정 못 함
SENSOR_EVENT = {"take": "REMOVE", "return": "ADD"}

FONT = cv2.FONT_HERSHEY_SIMPLEX
ACTION_COLOR = {"take": (60, 60, 255), "return": (80, 200, 80), "none": (160, 160, 160)}
SENSOR_COLOR = {"REMOVE": (60, 60, 255), "ADD": (80, 200, 80), "RESYNC": (160, 160, 160)}


@dataclass
class TakeLabel:
    pair: str
    seq: int
    time_sec: float | None
    lane: str
    action: str
    qty: int
    checked_by: str = ""
    note: str = ""


@dataclass
class SensorRow:
    time_sec: float
    lane: str
    event: str
    before: int
    after: int
    in_video: bool
    raw_sec: float | None = None
    """보정 전 영상 시각 (split_sensor_events 의 raw_sec). 옛 CSV 는 time_sec 와 같다."""

    @property
    def units(self) -> int:
        return abs(self.after - self.before)


# ---------------------------------------------------------------- 파일 읽고 쓰기

def pair_key(p: str):
    m = re.match(r"([A-Z])(\d+)(.*)", p)
    return (m.group(1), int(m.group(2)), m.group(3)) if m else (p, 0, "")


def load_labels(path: Path = OUT) -> dict[str, list[TakeLabel]]:
    out: dict[str, list[TakeLabel]] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            t = (r.get("time_sec") or "").strip()
            out.setdefault(r["pair"], []).append(TakeLabel(
                pair=r["pair"], seq=int(r.get("seq") or 0), time_sec=float(t) if t else None,
                lane=(r.get("lane") or "").strip(), action=r["action"].strip(), qty=int(r.get("qty") or 0),
                checked_by=r.get("checked_by", ""), note=r.get("note", "")))
    return out


def save_labels(labels: dict[str, list[TakeLabel]], path: Path = OUT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for pair in sorted(labels, key=pair_key):
            for lab in labels[pair]:
                row = asdict(lab)
                row["time_sec"] = "" if lab.time_sec is None else f"{lab.time_sec:.2f}"
                w.writerow(row)


def load_status(path: Path = STATUS) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = {r["pair"]: r for r in csv.DictReader(f)}
    for r in rows.values():   # 옛 파일(mode 열 없음)은 센서를 보며 찍었다 — open 으로 고정
        if r.get("mode") is None:
            r["mode"] = "open"
    return rows


def save_status(status: dict[str, dict], path: Path = STATUS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STATUS_FIELDS)
        w.writeheader()
        for pair in sorted(status, key=pair_key):
            w.writerow({**{k: status[pair].get(k, "") for k in STATUS_FIELDS}, "mode": mode_of(status, pair)})


def set_status(status: dict[str, dict], pair: str, value: str, checked_by: str) -> None:
    status[pair] = {"pair": pair, "status": value, "checked_by": checked_by,
                    "updated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "mode": mode_of(status, pair)}


def mode_of(status: dict[str, dict], pair: str) -> str:
    """blind = 처음부터 센서를 안 보고 찍음. 옛 파일(mode 열 없음)의 줄은 open, 아직 연 적 없는 줄·없는 줄은 ''."""
    row = status.get(pair)
    if row is None:
        return ""
    mode = row.get("mode")
    return "open" if mode is None else mode


def mark_mode(status: dict[str, dict], pair: str, blind: bool, checked_by: str) -> None:
    """영상을 여는 순간 부른다. 센서가 보이는 상태(open)로 한 번이라도 열면 영구히 open."""
    if pair not in status:
        status[pair] = {"pair": pair, "status": "draft", "checked_by": checked_by,
                        "updated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "mode": ""}
    status[pair]["mode"] = "open" if (not blind or mode_of(status, pair) == "open") else "blind"


def is_done(status: dict[str, dict], pair: str) -> bool:
    return status.get(pair, {}).get("status") == "done"


def load_sensor(pair: str, folder: Path = LABELS / "sensor") -> list[SensorRow] | None:
    """영상별 센서 CSV. 파일이 없으면 None (센서 없이 라벨만 찍는다)."""
    path = folder / f"{pair}.csv"
    if not path.exists():
        return None
    with open(path, encoding="utf-8-sig", newline="") as f:
        out = []
        for r in csv.DictReader(f):
            t = float(r["time_sec"])
            raw = (r.get("raw_sec") or "").strip()
            out.append(SensorRow(t, r["lane"], r["event"], int(r["before"]), int(r["after"]),
                                 r.get("in_video", "1").strip() != "0", float(raw) if raw else t))
        return out


def load_clock_offset(path: Path = LABELS / "sensor_index.csv") -> str:
    if not path.exists():
        return "?"
    with open(path, encoding="utf-8-sig", newline="") as f:
        values = {r.get("clock_offset") or "+0.00(기록 없음)" for r in csv.DictReader(f)}
    return ", ".join(sorted(values))


def renumber(rows: list[TakeLabel]) -> list[TakeLabel]:
    """행동은 시각 순 1,2,3… / 행동 없음(none)은 0."""
    timed = sorted((r for r in rows if r.action != "none"), key=lambda r: r.time_sec or 0.0)
    for i, r in enumerate(timed, 1):
        r.seq = i
    return timed + [r for r in rows if r.action == "none"]


# ---------------------------------------------------------------- 대조

@dataclass
class Unit:
    """한 개 단위. 정답 qty=2 나 센서 2→0 은 단위 2개로 펼친다."""
    time_sec: float
    lane: str
    direction: str  # take / return
    in_video: bool = True
    src: int = -1
    """정답 행 번호 (수량으로 펼친 단위가 같은 행인지 — 영상별 보정 표본은 행 단위로 센다)."""


def label_units(rows: list[TakeLabel]) -> list[Unit]:
    return [Unit(r.time_sec, r.lane, r.action, True, k) for k, r in enumerate(rows)
            if r.action in SENSOR_EVENT and r.time_sec is not None and r.lane in ("L1", "L2", "L3")
            for _ in range(max(r.qty, 1))]


def sensor_units(rows: list[SensorRow]) -> list[Unit]:
    back = {v: k for k, v in SENSOR_EVENT.items()}
    return [Unit(r.time_sec, r.lane, back[r.event], r.in_video) for r in rows
            if r.event in back for _ in range(max(r.units, 1))]


def best_matching(labels: list[Unit], sensors: list[Unit]) -> list[tuple[int, int]]:
    """창 안 쌍 중 **매칭 수 최대 → 그 안에서 |dt| 합 최소**. (탐욕은 연결 가능한 쌍을 남긴다 — C05 반례)
    단위 수가 작아(레인당 몇 개) 센서 쪽 비트마스크 DP 로 정확히 푼다."""
    n = len(sensors)

    def ok(i: int, j: int) -> bool:
        d = sensors[j].time_sec - labels[i].time_sec
        return WINDOW[0] <= d <= WINDOW[1]

    @lru_cache(maxsize=None)
    def go(i: int, used: int) -> tuple[int, float, tuple]:
        if i == len(labels):
            return 0, 0.0, ()
        best = go(i + 1, used)  # 이 정답은 짝 없음
        best = (best[0], best[1], best[2])
        for j in range(n):
            if used >> j & 1 or not ok(i, j):
                continue
            cnt, cost, pairs = go(i + 1, used | 1 << j)
            cand = (cnt + 1, cost + abs(sensors[j].time_sec - labels[i].time_sec), ((i, j),) + pairs)
            if cand[0] > best[0] or (cand[0] == best[0] and cand[1] < best[1] - 1e-9):
                best = cand
        return best

    return list(go(0, 0)[2])


def classify(dt_sec: float) -> str:
    if abs(abs(dt_sec) - ON_TIME) < BORDER:
        return "경계"
    if abs(dt_sec) <= ON_TIME:
        return "맞음"
    return "늦음" if dt_sec > 0 else "이름"


def compare_pair(labels: list[TakeLabel], sensor: list[SensorRow]) -> list[dict]:
    """한 영상의 정답과 센서를 단위별로 대조. 결과 kind: 맞음/늦음/이름/경계/레인 다름/놓침/가짜."""
    lab, sen = label_units(labels), sensor_units(sensor)
    out: list[dict] = []
    left_l, left_s = set(range(len(lab))), set(range(len(sen)))
    # 1) 같은 레인·같은 방향
    for lane in ("L1", "L2", "L3"):
        for direction in SENSOR_EVENT:
            li = [i for i in sorted(left_l) if lab[i].lane == lane and lab[i].direction == direction]
            si = [j for j in sorted(left_s) if sen[j].lane == lane and sen[j].direction == direction]
            for a, b in best_matching([lab[i] for i in li], [sen[j] for j in si]):
                i, j = li[a], si[b]
                d = sen[j].time_sec - lab[i].time_sec
                out.append(_row(classify(d), lab[i], sen[j], d))
                left_l.discard(i)
                left_s.discard(j)
    # 2) 방향은 같은데 레인이 다름
    for direction in SENSOR_EVENT:
        li = [i for i in sorted(left_l) if lab[i].direction == direction]
        si = [j for j in sorted(left_s) if sen[j].direction == direction]
        for a, b in best_matching([lab[i] for i in li], [sen[j] for j in si]):
            i, j = li[a], si[b]
            out.append(_row("레인 다름", lab[i], sen[j], sen[j].time_sec - lab[i].time_sec))
            left_l.discard(i)
            left_s.discard(j)
    for i in sorted(left_l):
        out.append(_row("놓침", lab[i], None, None))
    for j in sorted(left_s):
        if sen[j].in_video:  # 여유 구간(영상 밖) 기록은 다른 행동의 응답이거나 다시 채운 것 — 가짜로 세지 않음
            out.append(_row("가짜", None, sen[j], None))
    return sorted(out, key=lambda r: r["sort"])


def _row(kind: str, lab: Unit | None, sen: Unit | None, d: float | None) -> dict:
    base = lab or sen
    return {"kind": kind, "lane": base.lane, "direction": base.direction,
            "label_sec": "" if lab is None else f"{lab.time_sec:.2f}",
            "sensor_lane": "" if sen is None else sen.lane,
            "sensor_sec": "" if sen is None else f"{sen.time_sec:.1f}",
            "dt": "" if d is None else f"{d:+.2f}", "sort": base.time_sec, "_src": None if lab is None else lab.src}


KINDS = ("맞음", "경계", "늦음", "이름", "레인 다름", "놓침", "가짜")
CHECK_FIELDS = ("pair", "kind", "lane", "direction", "label_sec", "sensor_lane", "sensor_sec", "dt")


def compare(labels: dict[str, list[TakeLabel]], status: dict[str, dict], pairs: list[str],
            write: Path | None = CHECK) -> dict[str, int]:
    done = [p for p in pairs if is_done(status, p)]
    print(f"검수 완료 {len(done)}/{len(pairs)} 영상만 대조 (draft·미라벨 {len(pairs) - len(done)} 제외)")
    modes = [mode_of(status, p) for p in done]
    print(f"  센서 안 보고 찍음(blind) {modes.count('blind')} / 센서 보며(open) {modes.count('open')}")
    shifted = load_video_shifts()
    shifted = {p: v for p, v in shifted.items() if p in done and abs(v) > 1e-9}
    if shifted:
        print(f"  영상별 센서 보정 적용 {len(shifted)}편 — 그 보정을 정한 같은 정답으로 다시 대조한 값이라 독립 지표가 아니다")
    print(f"시계 보정 clock_offset: {load_clock_offset()} (휴대폰-센서 PC, 따로 재지 않음 → 시간 판정은 잠정)")
    print(f"규칙: dt = 센서 - 정답, 창 [{WINDOW[0]:+.0f}, {WINDOW[1]:+.0f}]초, |dt|<={ON_TIME} 맞음, "
          f"| |dt|-{ON_TIME} |<{BORDER} 경계")
    total = {k: 0 for k in KINDS}
    rows_all, dts, no_sensor, other = [], [], [], 0
    for p in done:
        sensor = load_sensor(p)
        if sensor is None:
            no_sensor.append(p)
            continue
        other += sum(r.qty for r in labels.get(p, []) if r.lane == "other" and r.action in SENSOR_EVENT)
        rows = compare_pair(labels.get(p, []), sensor)
        resync = sum(1 for s in sensor if s.event == "RESYNC" and s.in_video)
        for r in rows:
            total[r["kind"]] += 1
            if r["dt"] and r["kind"] != "레인 다름":
                dts.append(float(r["dt"]))
            rows_all.append({"pair": p, **{k: v for k, v in r.items() if k in CHECK_FIELDS}})
        desc = ", ".join(f"{r['kind']}({r['lane']} {r['direction']}"
                         + (f" {r['dt']}s" if r["dt"] else "") + ")" for r in rows) or "(정답·센서 모두 없음)"
        print(f"  {p:<6} {desc}" + (f"  RESYNC {resync}" if resync else ""))
    print("합계: " + "  ".join(f"{k} {v}" for k, v in total.items()))
    if dts:
        print(f"짝지어진 dt (센서 지연 + 시계 차이): 중앙값 {statistics.median(dts):+.2f}s, "
              f"범위 {min(dts):+.2f} ~ {max(dts):+.2f}s, {len(dts)}개")
    if other:
        print(f"센서 없는 진열대(other) 행동 {other}개는 대조 제외")
    if no_sensor:
        print(f"센서 파일 없음: {', '.join(no_sensor)}")
    if write is not None:  # 결과가 없어도 헤더만으로 덮어써 이전 결과가 남지 않게
        write.parent.mkdir(parents=True, exist_ok=True)
        with open(write, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CHECK_FIELDS)
            w.writeheader()
            w.writerows(rows_all)
        print(f"저장: {write}")
    return total


def load_video_shifts(path: Path = LABELS / "sensor_index.csv") -> dict[str, float]:
    """분할에 이미 반영된 영상별 보정 (sensor_index.csv video_shift). 없으면 0."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {r["pair"]: float(r.get("video_shift") or 0.0) for r in csv.DictReader(f)}


def raw_sensor(rows: list[SensorRow]) -> list[SensorRow]:
    """보정 전 시각으로 되돌린 센서 (fit 은 늘 같은 원시 사건·같은 후보로 매칭해야 반복해도 같은 답)."""
    return [SensorRow(r.raw_sec if r.raw_sec is not None else r.time_sec, r.lane, r.event, r.before, r.after,
                      r.in_video, r.raw_sec) for r in rows]


def row_dts(labels: list[TakeLabel], sensor: list[SensorRow]) -> list[float]:
    """같은 레인 짝(맞음·경계·늦음·이름)의 dt 를 정답 행 단위로 (한 행의 수량 단위들은 중앙값 하나)."""
    by_src: dict[int, list[float]] = {}
    for r in compare_pair(labels, raw_sensor(sensor)):
        if r["kind"] in ("맞음", "경계", "늦음", "이름") and r["_src"] is not None:
            by_src.setdefault(r["_src"], []).append(float(r["dt"]))
    return [statistics.median(v) for _, v in sorted(by_src.items())]


def fit_offsets(labels: dict[str, list[TakeLabel]], status: dict[str, dict], pairs: list[str],
                checked_by: str = "", write: Path | None = OFFSETS) -> dict[str, dict]:
    """영상별 센서 시각 보정. 기준 = 모든 행 dt 의 중앙값(센서 지연 + 시계 차이 — 건드리지 않음),
    shift = 그 영상 행 dt 중앙값 - 기준. 서로 다른 행 ≥ FIT_MIN_ROWS, 범위 ≤ FIT_MAX_SPREAD, |shift| ≤ FIT_MAX_SHIFT 일 때만 적용."""
    per: dict[str, list[float]] = {}
    for p in pairs:
        if not is_done(status, p):
            continue
        sensor = load_sensor(p)
        if sensor is None:
            continue
        per[p] = row_dts(labels.get(p, []), sensor)
    every = [d for v in per.values() for d in v]
    if not every:
        print("짝지어진 정답이 없다 — 보정표를 비운다 (검수 완료 영상을 먼저)")
    base = statistics.median(every) if every else 0.0
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    out: dict[str, dict] = {}
    for p, v in per.items():
        n = len(v)
        spread = (max(v) - min(v)) if v else 0.0
        shift = (statistics.median(v) - base) if v else 0.0
        if n < FIT_MIN_ROWS:
            reason = f"짝지어진 행동 {n}개 < {FIT_MIN_ROWS}"
        elif spread > FIT_MAX_SPREAD:
            reason = f"영상 안 dt 범위 {spread:.2f}s > {FIT_MAX_SPREAD} (들쭉날쭉 — 센서 지연·놓침)"
        elif abs(shift) > FIT_MAX_SHIFT:
            reason = f"|shift| {abs(shift):.2f}s > {FIT_MAX_SHIFT} (시각 오차로 보기엔 큼)"
        else:
            reason = ""
        applied = not reason
        out[p] = {"pair": p, "shift_sec": f"{shift if applied else 0.0:+.2f}", "n": n, "spread_sec": f"{spread:.2f}",
                  "applied": int(applied), "reason": reason or f"dt 중앙 {shift + base:+.2f}s", "base_sec": f"{base:+.2f}",
                  "checked_by": checked_by, "updated": now}
        print(f"  {p:<6} {'적용' if applied else '안 함'} shift {shift:+.2f}s  (행동 {n}, 범위 {spread:.2f}s)"
              + (f"  — {reason}" if reason else ""))
    print(f"기준 dt (센서 지연 + 시계 차이, 그대로 둠): {base:+.2f}s, 적용 {sum(r['applied'] for r in out.values())}/{len(out)}편")
    if write is not None:
        write.parent.mkdir(parents=True, exist_ok=True)
        with open(write, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=OFFSET_FIELDS)
            w.writeheader()
            for p in sorted(out, key=pair_key):
                w.writerow(out[p])
        print(f"저장: {write}\n다음: python scripts/split_sensor_events.py \"../../source/2(angle1)\" 로 다시 나누고 --compare")
    return out


# ---------------------------------------------------------------- 화면

def fit(frame, width: int):
    if frame is None:
        return np.zeros((int(width * 9 / 16), width, 3), np.uint8)
    h, w = frame.shape[:2]
    return frame if w == width else cv2.resize(frame, (width, int(h * width / w)))


def load_side_offsets(path: Path = SYNC) -> dict[str, float]:
    """짝별 카메라 시차 (ms). 공통 시각(angle1) = angle0 시각 + offset_ms."""
    if not path.exists():
        return {}
    return {p: r.offset_ms for p, r in SyncTable.load(path).rows.items()}


def load_applied_offsets(path: Path = LABELS / "sensor_index.csv") -> dict[str, float]:
    """센서 분할에 실제로 쓴 시차 (sensor_index.csv sync_offset_ms). 옆 화면도 이 값을 써야 센서와 같은 축."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {r["pair"]: float(r["sync_offset_ms"]) for r in csv.DictReader(f)
                if (r.get("sync_offset_ms") or "").strip()}


def stale_offsets(applied: dict[str, float], table: dict[str, float], tol_ms: float = 0.05) -> list[str]:
    """분할에 쓴 시차와 지금 sync.csv 가 다른 짝 (sync.csv 를 고친 뒤 split 을 안 돌림)."""
    return sorted((p for p in applied if p not in table or abs(applied[p] - table[p]) > tol_ms), key=pair_key)


def side_time(now_sec: float, offset_ms: float) -> float:
    """angle0 시각 → 같은 순간의 angle1 시각 (초). angle1 = angle0 + offset."""
    return now_sec + offset_ms / 1000.0


def sensor_axis(path: Path = LABELS / "sensor_index.csv") -> str | None:
    """센서 분할의 시각 축 (sensor_index.csv axis 열). 파일·열이 없으면 None (옛 angle1 분할)."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8-sig", newline="") as f:
        axes = {(r.get("axis") or "").strip() for r in csv.DictReader(f)}
    return axes.pop() if len(axes) == 1 and "" not in axes else None


def side_view(side: FrameCache | None, offset_ms: float | None, now_sec: float, width: int, height: int):
    """angle0 의 now_sec 와 같은 순간의 angle1 장면. 영상 밖·시차 없음은 검은 화면에 이유를 적는다."""
    img = np.zeros((height, width, 3), np.uint8)
    if side is None or offset_ms is None:
        msg = "angle1: no video" if side is None else "angle1: no sync offset"
        cv2.putText(img, msg, (20, height // 2), FONT, 0.8, (150, 150, 150), 2)
        return img
    t0 = side_time(now_sec, offset_ms)
    if 0.0 <= t0 < side.duration_sec:
        f = fit(side.get(side.index_at(t0)), width)
        img[:min(height, f.shape[0])] = f[:height]
    else:
        cv2.putText(img, "angle1: out of video", (20, height // 2), FONT, 0.8, (150, 150, 150), 2)
    cv2.putText(img, f"angle1 {t0:6.2f}s (sync {offset_ms:+.0f}ms)", (10, height - 14), FONT, 0.6, (0, 0, 0), 4)
    cv2.putText(img, f"angle1 {t0:6.2f}s (sync {offset_ms:+.0f}ms)", (10, height - 14), FONT, 0.6, (0, 255, 255), 1)
    return img


def screen(cache: FrameCache, idx: int, now_sec: float, width: int, side: FrameCache | None,
           side_offset: float | None, dual: bool):
    """화면 위쪽: dual 이면 [angle0 | angle1] 를 반씩, 아니면 angle0 만 전체 폭."""
    if not dual:
        return fit(cache.get(idx), width)
    half = width // 2
    main = fit(cache.get(idx), half)
    cv2.putText(main, "angle0 (label/sensor time)", (10, main.shape[0] - 14), FONT, 0.6, (0, 0, 0), 4)
    cv2.putText(main, "angle0 (label/sensor time)", (10, main.shape[0] - 14), FONT, 0.6, (255, 255, 255), 1)
    return np.hstack([main, side_view(side, side_offset, now_sec, width - half, main.shape[0])])


def draw(img, pair, idx, fps, n, progress, rows, sensor, lane, qty, playing, done_mark, done, total,
         blind: bool = False, t_sec: float | None = None, duration: float | None = None):
    """t_sec·duration 을 주면 실제 장면 시각·영상 길이로 그린다 (3차 angle1 은 장면을 빠뜨려 idx/fps 가 틀림)."""
    t = t_sec if t_sec is not None else idx / fps
    if blind:
        sensor = []   # 배너·목록·타임라인 모두 그리지 않는다
    if sensor:
        near = [s for s in sensor if s.in_video and abs(s.time_sec - t) <= 0.5]
        for k, s in enumerate(near):
            txt = f"SENSOR {s.lane} {s.event} {s.before}->{s.after} @{s.time_sec:.1f}s"
            cv2.rectangle(img, (10, 10 + 40 * k), (20 + 16 * len(txt), 46 + 40 * k), (0, 0, 0), -1)
            cv2.putText(img, txt, (18, 38 + 40 * k), FONT, 0.8, SENSOR_COLOR.get(s.event, (255, 255, 255)), 2)
    bar = np.full((240, img.shape[1], 3), 30, np.uint8)
    put = lambda s, x, y, c=(230, 230, 230), sc=0.6: cv2.putText(bar, s, (x, y), FONT, sc, c, 1, cv2.LINE_AA)
    put(f"{'BLIND  ' if blind else ''}{pair}  [{done}/{total} done]  {'DONE' if done_mark else 'draft'}   t={t:6.2f}s  frame {idx}/{n}   "
        f"{'PLAY' if playing else 'PAUSE'}   lane={LANE_NAME[lane]}  qty={qty}", 10, 24,
        (120, 255, 120) if done_mark else (255, 255, 255), 0.6)
    if progress < 1.0:
        put(f"loading {progress * 100:3.0f}%", img.shape[1] - 170, 24, (0, 200, 255))
    put("SPACE play | <- -> 1 frame | up/down -1s/+1s | g/b next/prev sensor | 1 2 3 lane  4 other | + - qty",
        10, 48, (170, 170, 170), 0.45)
    put("t take  r return  n none  u undo | c mark done | TAB next  BKSP prev | s save  q quit",
        10, 68, (170, 170, 170), 0.45)
    y = 94
    put("label:", 10, y, (200, 200, 200), 0.5)
    if not rows:
        put("(no label yet)", 80, y, (120, 120, 255), 0.5)
    for r in rows[:6]:
        txt = (f"#{r.seq} {r.action:<6} {r.lane:<5} x{r.qty} at {r.time_sec:.2f}s" if r.time_sec is not None
               else "none (no shelf action)")
        put(txt, 80, y, ACTION_COLOR.get(r.action, (255, 255, 255)), 0.5)
        y += 20
    y = 94
    x = img.shape[1] // 2
    put("sensor:", x, y, (200, 200, 200), 0.5)
    if blind:
        put("(hidden - blind)", x + 80, y, (150, 150, 150), 0.5)
    elif sensor is None:
        put("(no sensor file)", x + 80, y, (120, 120, 255), 0.5)
    elif not [s for s in sensor if s.in_video]:
        put("(no event in video)", x + 80, y, (120, 120, 255), 0.5)
    for s in [s for s in (sensor or []) if s.in_video][:6]:
        put(f"{s.time_sec:5.1f}s {s.lane} {s.event} {s.before}->{s.after}", x + 80, y,
            SENSOR_COLOR.get(s.event, (255, 255, 255)), 0.5)
        y += 20
    # 타임라인: 위 줄 정답, 아래 줄 센서
    x0, x1, ty = 10, img.shape[1] - 10, 222
    if duration:
        pos = lambda sec: int(x0 + (x1 - x0) * min(max(sec / duration, 0.0), 1.0))
    else:
        pos = lambda sec: int(x0 + (x1 - x0) * sec * fps / max(n - 1, 1))
    cv2.line(bar, (x0, ty), (x1, ty), (90, 90, 90), 2)
    cx = pos(t)
    cv2.line(bar, (cx, ty - 22), (cx, ty + 10), (255, 255, 255), 2)
    for r in rows:
        if r.time_sec is not None:
            cv2.circle(bar, (pos(r.time_sec), ty - 12), 6, ACTION_COLOR[r.action], -1)
    for s in sensor or []:
        if s.in_video:
            cv2.circle(bar, (pos(s.time_sec), ty + 6), 5, SENSOR_COLOR.get(s.event, (255, 255, 255)), -1)
    put("label", x1 - 50, ty - 26, (150, 150, 150), 0.4)
    if not blind:
        put("sensor", x1 - 55, ty + 18, (150, 150, 150), 0.4)
    return np.vstack([img, bar])


def label_video(pair, cache: FrameCache, labels, status, checked_by, done, total, width, blind: bool = False,
                side: FrameCache | None = None, side_offset: float | None = None, dual: bool = False,
                readonly: bool = False, start_sec: float | None = None, note: str | None = None) -> str:
    """한 영상을 라벨링한다. 돌려주는 값: 'next' / 'prev' / 'quit'.
    dual 이면 옆에 angle1 을 시차만큼 맞춰 같이 보여준다 (찍는 시각은 늘 angle0 시각).
    readonly 면 보기 전용 — 라벨·상태를 바꾸지도 저장하지도 않는다 (센서가 보여도 blind 이력 유지:
    센서를 본 뒤 라벨을 고치려면 일반 모드로 열어야 하고, 그때 open 이 된다). start_sec·note 는 처음 위치·화면 위 안내."""
    rows = labels.get(pair, []) if readonly else labels.setdefault(pair, [])
    sensor = load_sensor(pair)
    if not readonly:
        mark_mode(status, pair, blind, checked_by)   # 센서가 보이는 상태로 열면 이 순간 open 고정
        save_status(status)
    idx, playing, lane, qty = 0, False, "L1", 1
    if start_sec is not None:
        idx = cache.index_at(max(start_sec, 0.0))
    win = "label_takes"
    marks = [] if blind else sorted({s.time_sec for s in sensor or [] if s.in_video})
    print(f"\n== {pair}" + ("  [blind — 센서 숨김]" if blind else f"  센서 기록 {len(marks)}개"
                              + ("" if sensor is not None else " (센서 파일 없음)"))
          + ("  [검수 완료]" if is_done(status, pair) else ""))
    if dual:
        print("   옆 화면 angle1: " + ("영상 없음" if side is None else
                                       "시차 없음 (sync.csv)" if side_offset is None else f"시차 {side_offset:+.0f}ms"))

    def changed():
        if is_done(status, pair):
            print("   라벨을 바꿔 draft 로 돌아감 (끝까지 보고 c 로 다시 완료)")
        set_status(status, pair, "draft", checked_by)

    try:
        while True:
            idx = max(0, min(idx, cache.n - 1))
            now = cache.time_sec(idx)   # 실제 장면 시각 (번호 ÷ fps 아님)
            img = screen(cache, idx, now, width, side, side_offset, dual)
            if note:
                draw_note(img, note)
            cv2.imshow(win, draw(img, pair, idx, cache.fps, cache.n, cache.progress, rows, sensor, lane, qty,
                                 playing, is_done(status, pair), done, total, blind, now, cache.duration_sec))
            gap = (cache.time_sec(idx + 1) - now) if idx + 1 < cache.n else 1.0 / cache.fps
            key = normalize_key(cv2.waitKeyEx(max(1, int(gap * 1000)) if playing else 0))
            t = round(now, 2)
            if playing and key == 255:
                idx += 1
                if idx >= cache.n - 1:
                    playing = False
                continue
            if readonly and key in EDIT_KEYS:
                print("   보기 전용 (--review) — 고치려면: python scripts/label_takes.py " + pair)
                continue
            if key == ord(" "):
                playing = not playing
            elif key == ord("d"):
                idx += 1
            elif key == ord("a"):
                idx -= 1
            elif key == KEY_DOWN:
                idx = cache.index_at(now + 1.0)
            elif key == KEY_UP:
                idx = cache.index_at(max(now - 1.0, 0.0))
            elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
                lane = LANES[key - ord("1")]
            elif key in (ord("+"), ord("=")):
                qty = min(qty + 1, 9)
            elif key in (ord("-"), ord("_")):
                qty = max(qty - 1, 1)
            elif key in (ord("t"), ord("r")):
                action = "take" if key == ord("t") else "return"
                rows[:] = [r for r in rows if r.action != "none"]
                rows.append(TakeLabel(pair, 0, t, lane, action, qty, checked_by))
                rows[:] = renumber(rows)
                print(f"   {action} {lane} x{qty} at {t:.2f}s")
                qty = 1
                changed()
            elif key == ord("n"):
                rows[:] = [TakeLabel(pair, 0, None, "", "none", 0, checked_by)]
                print("   선반 행동 없음")
                changed()
            elif key == ord("u") and rows:
                print(f"   삭제: {rows.pop()}")
                rows[:] = renumber(rows)
                changed()
            elif key in (ord("g"), ord("b")) and blind:
                print("   blind: 센서 이동 꺼짐")
            elif key in (ord("g"), ord("b")) and marks:
                if key == ord("g"):
                    nxt = [m for m in marks if m > t + 0.05]
                    target = nxt[0] if nxt else marks[-1]
                else:
                    prv = [m for m in marks if m < t - 0.05]
                    target = prv[-1] if prv else marks[0]
                idx = cache.index_at(target)
            elif key == ord("c"):
                if is_done(status, pair):
                    set_status(status, pair, "draft", checked_by)
                    print("   검수 완료 취소")
                elif not rows:
                    print("   아직 찍은 게 없다 — 행동이 없으면 n 을 먼저")
                else:
                    set_status(status, pair, "done", checked_by)
                    print("   검수 완료")
            elif key == ord("s"):
                save_labels(labels)
                save_status(status)
                print(f"   저장: {OUT}")
            elif key == 9:
                return "next"
            elif key == 8:
                return "prev"
            elif key in (ord("q"), 27):
                return "quit"
    finally:
        if not readonly:
            if not rows:
                labels.pop(pair, None)
            save_labels(labels)
            save_status(status)


EDIT_KEYS = {ord(c) for c in "trnucs"}
PROBLEM_KINDS = ("놓침", "가짜", "늦음")
LEAD_SEC = 2.0   # 문제 시각보다 이만큼 앞에서 시작


def load_problems(path: Path = CHECK, kinds=PROBLEM_KINDS, pairs: list[str] | None = None) -> list[dict]:
    """sensor_check.csv 에서 문제 줄만. at = 정답 시각(없으면 센서 시각). 영상·시각 순."""
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["kind"] not in kinds or (pairs and r["pair"] not in pairs):
                continue
            at = r.get("label_sec") or r.get("sensor_sec") or ""
            out.append({**r, "at": float(at) if at.strip() else 0.0})
    return sorted(out, key=lambda r: (pair_key(r["pair"]), r["at"]))


def problem_note(r: dict, k: int, total: int) -> str:
    """화면 위 안내 한 줄 (ASCII — cv2 글꼴)."""
    kind = {"놓침": "MISSED", "가짜": "FAKE", "늦음": "LATE"}.get(r["kind"], r["kind"])
    lab = f"label {r['direction']} {r['lane']} @{r['label_sec']}s" if r.get("label_sec") else "label -"
    sen = f"sensor {r['sensor_lane']} @{r['sensor_sec']}s" if r.get("sensor_sec") else "sensor -"
    dt_txt = f" dt {r['dt']}s" if r.get("dt") else ""
    return f"[{k + 1}/{total}] {kind}: {lab} | {sen}{dt_txt}"


def draw_note(img, text: str) -> None:
    y = img.shape[0] - 50
    cv2.rectangle(img, (0, y - 26), (min(img.shape[1], 14 + 12 * len(text)), y + 10), (0, 0, 0), -1)
    cv2.putText(img, text, (8, y), FONT, 0.65, (0, 220, 255), 2)


def review(problems: list[dict], folder: Path, side_folder: Path | None, width: int, labels, status) -> None:
    """문제 줄을 차례로 연다 (TAB 다음 · BACKSPACE 이전 · q 끝). 센서 보임, 보기 전용."""
    dual = side_folder is not None
    pane = width // 2 if dual else width
    offsets = load_applied_offsets() if dual else {}
    caches: dict[str, FrameCache | None] = {}
    sides: dict[str, FrameCache | None] = {}
    k, step = 0, 1
    while 0 <= k < len(problems):
        r = problems[k]
        pair = r["pair"]
        keep = {pair} | ({problems[k + 1]["pair"]} if k + 1 < len(problems) else set())
        for p in keep:
            if p not in caches:
                caches[p] = open_cache(folder / f"{p}.mp4", pane)
            if dual and p not in sides:
                path = side_folder / f"{p}.mp4"
                sides[p] = open_cache(path, width - pane) if path.exists() else None
        for d in (caches, sides):
            for p in list(d):
                if p not in keep:
                    d.pop(p)
        if caches[pair] is None:
            k += step
            continue
        note = problem_note(r, k, len(problems))
        print(f"\n{note}")
        action = label_video(pair, caches[pair], labels, status, "", 0, 0, width, False, sides.get(pair),
                             offsets.get(pair), dual, readonly=True, start_sec=r["at"] - LEAD_SEC, note=note)
        if action == "quit":
            break
        step = 1 if action == "next" else -1
        k = max(k + step, 0)
    cv2.destroyAllWindows()


def summary(labels, status, pairs) -> None:
    done = [p for p in pairs if is_done(status, p)]
    draft = [p for p in pairs if p in labels and not is_done(status, p)]
    print(f"검수 완료 {len(done)}/{len(pairs)}, draft {len(draft)} ({', '.join(draft) or '-'})")
    for p in pairs:
        if p not in labels:
            continue
        rows = labels[p]
        desc = " / ".join(f"{r.action} {r.lane}x{r.qty}@{r.time_sec:.1f}s" if r.time_sec is not None else "none"
                          for r in rows)
        print(f"  {p:<6} {'done ' if is_done(status, p) else 'draft'} {mode_of(status, p) or '-':<5} | {desc}")


def snapshot(pair: str, folder: Path, labels, status, width: int, at: float, out: Path, blind: bool = False,
             side_folder: Path | None = None) -> None:
    """키 입력 없이 한 장면을 그려 저장 (화면 확인용). side_folder 를 주면 angle1 도 옆에."""
    cache = FrameCache(folder / f"{pair}.mp4", width)
    idx = cache.index_at(at)
    side = open_cache(side_folder / f"{pair}.mp4", width) if side_folder else None
    img = screen(cache, idx, cache.time_sec(idx), width, side, load_applied_offsets().get(pair), side_folder is not None)
    rows = labels.get(pair, [])
    if not blind:   # 센서가 보이는 그림 — label_video 처럼 이 영상은 영구히 open
        mark_mode(status, pair, False, "")
        save_status(status)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), draw(img, pair, idx, cache.fps, cache.n, 1.0, rows, load_sensor(pair), "L2", 1,
                               False, is_done(status, pair), 0, 0, blind, cache.time_sec(idx), cache.duration_sec))
    print(f"저장: {out}")


def open_cache(path: Path, width: int) -> FrameCache | None:
    """열리지 않는 영상은 경고하고 None — 그 영상만 건너뛴다."""
    try:
        return FrameCache(path, width)
    except (FileNotFoundError, OSError) as e:
        print(f"   영상을 열 수 없어 건너뜀: {path.name} ({e})")
        return None


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", nargs="*", help="영상 이름 (예: T01 N05). 비우면 검수 완료 안 된 영상 전부")
    ap.add_argument("--folder", type=Path, default=SOURCE, help="기준 영상 폴더 (기본 source/2(angle0))")
    ap.add_argument("--all", action="store_true", help="검수 완료 영상도 다시 돌아본다")
    ap.add_argument("--summary", action="store_true", help="요약만 출력")
    ap.add_argument("--compare", action="store_true", help="검수 완료 영상을 센서와 대조")
    ap.add_argument("--by", default=os.environ.get("USERNAME", ""), help="checked_by 에 남길 이름")
    ap.add_argument("--width", type=int, default=1280, help="화면 폭(px)")
    ap.add_argument("--snapshot", type=float, metavar="초", help="첫 영상의 이 시각 화면을 그림으로 저장하고 끝")
    ap.add_argument("--review", action="store_true",
                    help="대조에서 문제 난 곳(놓침·가짜·늦음)만 센서 보이게 차례로 (보기 전용, blind 이력 유지). 영상 이름을 주면 그 영상만")
    ap.add_argument("--kinds", default=",".join(PROBLEM_KINDS), help="--review 에서 볼 종류 (쉼표, 예: 놓침,가짜,늦음,경계)")
    ap.add_argument("--blind", action="store_true", help="센서 표시를 숨기고 찍는다 (정답이 센서에 끌리지 않게)")
    ap.add_argument("--side-folder", type=Path, default=SIDE_SOURCE,
                    help="옆에 같이 볼 angle1 영상 폴더 (기본 source/2(angle1))")
    ap.add_argument("--single", action="store_true", help="angle0 만 보여준다 (옆 화면 끔)")
    ap.add_argument("--fit-offsets", action="store_true",
                    help="검수 완료 영상으로 영상별 센서 시각 보정을 계산해 sensor_offsets.csv 에 저장")
    args = ap.parse_args()

    all_pairs = sorted((p.stem for p in args.folder.glob("*.mp4") if p.stem != "area_setting"), key=pair_key)
    if not all_pairs:
        sys.exit(f"영상 없음: {args.folder}")
    labels, status = load_labels(), load_status()
    if args.summary:
        summary(labels, status, all_pairs)
        return
    axis = sensor_axis()
    if axis != AXIS:
        sys.exit(f"센서 분할 시각 축이 {axis or 'angle1(옛 분할)'} — 정답은 {AXIS} 시각이라 섞인다.\n"
                 f"먼저: python scripts/split_sensor_events.py \"../../source/2(angle1)\"  (기본 --axis angle0)")
    stale = stale_offsets(load_applied_offsets(), load_side_offsets())
    if stale:
        sys.exit(f"sync.csv 시차가 센서 분할 때와 다르다: {', '.join(stale)}\n"
                 f"먼저: python scripts/split_sensor_events.py \"../../source/2(angle1)\"  로 다시 나눈다")
    if args.compare:
        compare(labels, status, all_pairs)
        return
    if args.fit_offsets:
        fit_offsets(labels, status, all_pairs, args.by)
        return
    unknown = [p for p in args.pairs if p not in all_pairs]
    if unknown:
        sys.exit(f"영상 없음: {unknown}")
    if args.review:
        problems = load_problems(kinds=tuple(k.strip() for k in args.kinds.split(",") if k.strip()),
                                 pairs=args.pairs or None)
        if not problems:
            sys.exit(f"볼 문제가 없다 — 먼저 --compare ({CHECK})")
        print(f"문제 {len(problems)}곳: TAB 다음 · BACKSPACE 이전 · q 끝 (보기 전용)")
        review(problems, args.folder, None if args.single else args.side_folder, args.width, labels, status)
        return
    todo = args.pairs or [p for p in all_pairs if args.all or not is_done(status, p)]
    if args.snapshot is not None:
        snapshot(todo[0], args.folder, labels, status, args.width, args.snapshot,
                 ROOT / "outputs" / "label_takes" / f"{todo[0]}_{args.snapshot:.1f}s{'_blind' if args.blind else ''}.png",
                 args.blind, None if args.single else args.side_folder)
        return
    if not todo:
        print("모든 영상을 검수했다. 다시 보려면 --all 또는 영상 이름을 준다.")
        summary(labels, status, all_pairs)
        return

    dual = not args.single
    offsets = load_applied_offsets() if dual else {}   # 센서 분할에 쓴 시차 그대로
    pane = args.width // 2 if dual else args.width
    i, step = 0, 1
    caches: dict[str, FrameCache | None] = {}
    sides: dict[str, FrameCache | None] = {}
    while 0 <= i < len(todo):
        keep = todo[i:i + 2]  # 지금 영상 + 다음 영상을 미리 읽어 둔다
        for p in keep:
            if p not in caches:
                caches[p] = open_cache(args.folder / f"{p}.mp4", pane)
            if dual and p not in sides:
                path = args.side_folder / f"{p}.mp4"
                sides[p] = open_cache(path, args.width - pane) if path.exists() else None
        for d in (caches, sides):
            for p in list(d):
                if p not in keep:
                    d.pop(p)
        if caches[todo[i]] is None:  # 열기 실패 — 가던 방향으로 넘어간다
            i += step
            continue
        done = sum(1 for p in all_pairs if is_done(status, p))
        action = label_video(todo[i], caches[todo[i]], labels, status, args.by, done, len(all_pairs), args.width,
                             args.blind, sides.get(todo[i]), offsets.get(todo[i]), dual)
        if action == "quit":
            break
        step = 1 if action == "next" else -1
        i = max(i + step, 0)
    cv2.destroyAllWindows()
    summary(labels, status, all_pairs)


if __name__ == "__main__":
    main()
