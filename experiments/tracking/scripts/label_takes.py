"""3차 촬영(카메라 1대 + 아두이노 레인 센서) 선반 행동 정답 라벨링 도구.

센서는 놓치고(T05), 늦고, 가짜로 찍기도 했다(T01 11초 ADD — 선반 앞에 아무도 없음).
그래서 **정답은 영상을 보고 사람이 찍는다**: 언제, 어느 레인에서, 집었는지/되돌렸는지, 몇 개.
화면에 그 영상의 센서 기록을 같이 띄워 찍으면서 대조하고, --compare 로 센서 정확도를 센다.

    python scripts/label_takes.py               # 검수 완료 안 된 영상부터 차례로
    python scripts/label_takes.py T01 T05       # 고른 영상만
    python scripts/label_takes.py --summary     # 찍은 결과 요약
    python scripts/label_takes.py --compare     # 검수 완료 영상만 센서와 대조

먼저 센서를 영상별로 나눠 둔다: python scripts/split_sensor_events.py "../../source/2(angle1)"

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

SOURCE = ROOT.parent.parent / "source" / "2(angle1)"
LABELS = ROOT / "labels" / "store01_shoot3"
OUT = LABELS / "takes.csv"
STATUS = LABELS / "takes_status.csv"
CHECK = LABELS / "sensor_check.csv"
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
        return {r["pair"]: r for r in csv.DictReader(f)}


def save_status(status: dict[str, dict], path: Path = STATUS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=("pair", "status", "checked_by", "updated"))
        w.writeheader()
        for pair in sorted(status, key=pair_key):
            w.writerow({k: status[pair].get(k, "") for k in ("pair", "status", "checked_by", "updated")})


def set_status(status: dict[str, dict], pair: str, value: str, checked_by: str) -> None:
    status[pair] = {"pair": pair, "status": value, "checked_by": checked_by,
                    "updated": dt.datetime.now().strftime("%Y-%m-%d %H:%M")}


def is_done(status: dict[str, dict], pair: str) -> bool:
    return status.get(pair, {}).get("status") == "done"


def load_sensor(pair: str, folder: Path = LABELS / "sensor") -> list[SensorRow] | None:
    """영상별 센서 CSV. 파일이 없으면 None (센서 없이 라벨만 찍는다)."""
    path = folder / f"{pair}.csv"
    if not path.exists():
        return None
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [SensorRow(float(r["time_sec"]), r["lane"], r["event"], int(r["before"]), int(r["after"]),
                          r.get("in_video", "1").strip() != "0") for r in csv.DictReader(f)]


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


def label_units(rows: list[TakeLabel]) -> list[Unit]:
    return [Unit(r.time_sec, r.lane, r.action) for r in rows
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
            "dt": "" if d is None else f"{d:+.2f}", "sort": base.time_sec}


KINDS = ("맞음", "경계", "늦음", "이름", "레인 다름", "놓침", "가짜")
CHECK_FIELDS = ("pair", "kind", "lane", "direction", "label_sec", "sensor_lane", "sensor_sec", "dt")


def compare(labels: dict[str, list[TakeLabel]], status: dict[str, dict], pairs: list[str],
            write: Path | None = CHECK) -> dict[str, int]:
    done = [p for p in pairs if is_done(status, p)]
    print(f"검수 완료 {len(done)}/{len(pairs)} 영상만 대조 (draft·미라벨 {len(pairs) - len(done)} 제외)")
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
            rows_all.append({"pair": p, **{k: v for k, v in r.items() if k != "sort"}})
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


# ---------------------------------------------------------------- 화면

def fit(frame, width: int):
    if frame is None:
        return np.zeros((int(width * 9 / 16), width, 3), np.uint8)
    h, w = frame.shape[:2]
    return frame if w == width else cv2.resize(frame, (width, int(h * width / w)))


def draw(img, pair, idx, fps, n, progress, rows, sensor, lane, qty, playing, done_mark, done, total):
    t = idx / fps
    if sensor:
        near = [s for s in sensor if s.in_video and abs(s.time_sec - t) <= 0.5]
        for k, s in enumerate(near):
            txt = f"SENSOR {s.lane} {s.event} {s.before}->{s.after} @{s.time_sec:.1f}s"
            cv2.rectangle(img, (10, 10 + 40 * k), (20 + 16 * len(txt), 46 + 40 * k), (0, 0, 0), -1)
            cv2.putText(img, txt, (18, 38 + 40 * k), FONT, 0.8, SENSOR_COLOR.get(s.event, (255, 255, 255)), 2)
    bar = np.full((240, img.shape[1], 3), 30, np.uint8)
    put = lambda s, x, y, c=(230, 230, 230), sc=0.6: cv2.putText(bar, s, (x, y), FONT, sc, c, 1, cv2.LINE_AA)
    put(f"{pair}  [{done}/{total} done]  {'DONE' if done_mark else 'draft'}   t={t:6.2f}s  frame {idx}/{n}   "
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
    if sensor is None:
        put("(no sensor file)", x + 80, y, (120, 120, 255), 0.5)
    elif not [s for s in sensor if s.in_video]:
        put("(no event in video)", x + 80, y, (120, 120, 255), 0.5)
    for s in [s for s in (sensor or []) if s.in_video][:6]:
        put(f"{s.time_sec:5.1f}s {s.lane} {s.event} {s.before}->{s.after}", x + 80, y,
            SENSOR_COLOR.get(s.event, (255, 255, 255)), 0.5)
        y += 20
    # 타임라인: 위 줄 정답, 아래 줄 센서
    x0, x1, ty = 10, img.shape[1] - 10, 222
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
    put("sensor", x1 - 55, ty + 18, (150, 150, 150), 0.4)
    return np.vstack([img, bar])


def label_video(pair, cache: FrameCache, labels, status, checked_by, done, total, width) -> str:
    """한 영상을 라벨링한다. 돌려주는 값: 'next' / 'prev' / 'quit'."""
    rows = labels.setdefault(pair, [])
    sensor = load_sensor(pair)
    idx, playing, lane, qty = 0, False, "L1", 1
    win = "label_takes"
    step_1s = int(round(cache.fps))
    marks = sorted({s.time_sec for s in sensor or [] if s.in_video})
    print(f"\n== {pair}  센서 기록 {len(marks)}개" + ("" if sensor is not None else " (센서 파일 없음)")
          + ("  [검수 완료]" if is_done(status, pair) else ""))

    def changed():
        if is_done(status, pair):
            print("   라벨을 바꿔 draft 로 돌아감 (끝까지 보고 c 로 다시 완료)")
        set_status(status, pair, "draft", checked_by)

    try:
        while True:
            idx = max(0, min(idx, cache.n - 1))
            img = fit(cache.get(idx), width)
            cv2.imshow(win, draw(img, pair, idx, cache.fps, cache.n, cache.progress, rows, sensor, lane, qty,
                                 playing, is_done(status, pair), done, total))
            key = normalize_key(cv2.waitKeyEx(int(1000 / cache.fps) if playing else 0))
            t = round(idx / cache.fps, 2)
            if playing and key == 255:
                idx += 1
                if idx >= cache.n - 1:
                    playing = False
                continue
            if key == ord(" "):
                playing = not playing
            elif key == ord("d"):
                idx += 1
            elif key == ord("a"):
                idx -= 1
            elif key == KEY_DOWN:
                idx += step_1s
            elif key == KEY_UP:
                idx -= step_1s
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
            elif key in (ord("g"), ord("b")) and marks:
                if key == ord("g"):
                    nxt = [m for m in marks if m > t + 0.05]
                    target = nxt[0] if nxt else marks[-1]
                else:
                    prv = [m for m in marks if m < t - 0.05]
                    target = prv[-1] if prv else marks[0]
                idx = int(round(target * cache.fps))
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
        if not rows:
            labels.pop(pair, None)
        save_labels(labels)
        save_status(status)


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
        print(f"  {p:<6} {'done ' if is_done(status, p) else 'draft'} | {desc}")


def snapshot(pair: str, folder: Path, labels, status, width: int, at: float, out: Path) -> None:
    """키 입력 없이 한 장면을 그려 저장 (화면 확인용)."""
    cache = FrameCache(folder / f"{pair}.mp4", width)
    idx = int(round(at * cache.fps))
    img = fit(cache.get(idx), width)
    rows = labels.get(pair, [])
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), draw(img, pair, idx, cache.fps, cache.n, 1.0, rows, load_sensor(pair), "L2", 1,
                               False, is_done(status, pair), 0, 0))
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
    ap.add_argument("--folder", type=Path, default=SOURCE, help="영상 폴더 (기본 source/2(angle1))")
    ap.add_argument("--all", action="store_true", help="검수 완료 영상도 다시 돌아본다")
    ap.add_argument("--summary", action="store_true", help="요약만 출력")
    ap.add_argument("--compare", action="store_true", help="검수 완료 영상을 센서와 대조")
    ap.add_argument("--by", default=os.environ.get("USERNAME", ""), help="checked_by 에 남길 이름")
    ap.add_argument("--width", type=int, default=1280, help="화면 폭(px)")
    ap.add_argument("--snapshot", type=float, metavar="초", help="첫 영상의 이 시각 화면을 그림으로 저장하고 끝")
    args = ap.parse_args()

    all_pairs = sorted((p.stem for p in args.folder.glob("*.mp4") if p.stem != "area_setting"), key=pair_key)
    if not all_pairs:
        sys.exit(f"영상 없음: {args.folder}")
    labels, status = load_labels(), load_status()
    if args.summary:
        summary(labels, status, all_pairs)
        return
    if args.compare:
        compare(labels, status, all_pairs)
        return
    unknown = [p for p in args.pairs if p not in all_pairs]
    if unknown:
        sys.exit(f"영상 없음: {unknown}")
    todo = args.pairs or [p for p in all_pairs if args.all or not is_done(status, p)]
    if args.snapshot is not None:
        snapshot(todo[0], args.folder, labels, status, args.width, args.snapshot,
                 ROOT / "outputs" / "label_takes" / f"{todo[0]}_{args.snapshot:.1f}s.png")
        return
    if not todo:
        print("모든 영상을 검수했다. 다시 보려면 --all 또는 영상 이름을 준다.")
        summary(labels, status, all_pairs)
        return

    i, step = 0, 1
    caches: dict[str, FrameCache | None] = {}
    while 0 <= i < len(todo):
        keep = todo[i:i + 2]  # 지금 영상 + 다음 영상을 미리 읽어 둔다
        for p in keep:
            if p not in caches:
                caches[p] = open_cache(args.folder / f"{p}.mp4", args.width)
        for p in list(caches):
            if p not in keep:
                caches.pop(p)
        if caches[todo[i]] is None:  # 열기 실패 — 가던 방향으로 넘어간다
            i += step
            continue
        done = sum(1 for p in all_pairs if is_done(status, p))
        action = label_video(todo[i], caches[todo[i]], labels, status, args.by, done, len(all_pairs), args.width)
        if action == "quit":
            break
        step = 1 if action == "next" else -1
        i = max(i + step, 0)
    cv2.destroyAllWindows()
    summary(labels, status, all_pairs)


if __name__ == "__main__":
    main()
