"""3차 촬영(카메라 1대 + 아두이노 거리 센서) 센서 로그를 영상별로 나눈다.

센서 PC(shelf_monitor)는 레인별 확정 이벤트를 **실제 시각(초 단위)** 으로 남기고,
휴대폰 영상은 mp4 안에 기록 시각을 남긴다. 이 휴대폰은 그 시각이 **촬영이 끝난 시각**이다
(3차 39편 모두 REMOVE 가 영상 5~12초에, 다시 채운 ADD 는 영상이 끝난 뒤에 온다 — 시작 시각으로 보면 안 맞음).
그래서 영상 시작 = 기록 시각 - 영상 길이, 이벤트의 영상 시각 = 이벤트 시각 - 영상 시작.

    python scripts/split_sensor_events.py "../../source/2(angle1)"
    python scripts/split_sensor_events.py "../../source/2(angle1)" --clock-offset 1.5   # 휴대폰 시계가 센서 PC 보다 1.5초 빠를 때

**시각 축 (기본 angle0)**: 시계 연결은 angle1 mp4 기록 시각으로만 한다 (angle0 의 기록 시각은 초 단위·시계 미검증).
그 뒤 카메라 시차(labels/store01_shoot3/sync.csv, angle1 = angle0 + offset_ms)를 빼서 **angle0 영상 시각**으로 쓴다 —
선반을 정면에서 보는 카메라라 사람이 정답을 찍기 쉽다 (label_takes.py 도 angle0 기준). 옛 축은 --axis angle1.
시차나 angle0 영상이 없는 짝이 하나라도 있으면 아무 파일도 쓰지 않고 멈춘다 (축이 섞이지 않게).

입력: 영상 폴더 안의 mp4 와 events_*.txt (shelf_monitor 이벤트 추출본)
결과: labels/store01_shoot3/sensor/<영상>.csv, labels/store01_shoot3/sensor_index.csv

    time_sec,wall_time,lane,product,event,before,after,delta,distance_mm,in_video,raw_sec,angle1_sec
    7.99,15:08:19,L2,동과루,REMOVE,3,2,-1,276,1,7.99,8.00

time_sec·raw_sec 는 축 영상 시각, angle1_sec 는 보정 전 angle1 시각(추적용).

영상별 보정: labels/store01_shoot3/sensor_offsets.csv (label_takes.py --fit-offsets 가 정답으로 계산) 에서
applied=1 인 영상은 time_sec = raw_sec - shift_sec. raw_sec 는 보정 전 시각 — 보정을 다시 계산할 때 쓴다.
여유 구간은 보정과 무관하게 raw 기준으로 FIT 여유(±1.2초)만큼 넓혀 늘 같은 사건을 남긴다. 끄려면 --no-video-offsets.

영상 앞 2초·뒤 8초(여유 구간)의 이벤트도 in_video=0 으로 남긴다 — 영상 끝 무렵 행동에 센서가 늦게 응답한 것을
정답과 대조할 때 쓴다(label_takes.py --compare). 화면 목록·index 개수는 in_video=1 만 센다.

정밀도: 센서 시각과 mp4 기록 시각이 모두 초 단위라 영상 시각은 ±1초 정도 어긋날 수 있다.
두 기기 시계 차이는 따로 재지 않았다 (--clock-offset 으로 보정 → 이 스크립트를 다시 돌려 여유 구간·in_video 도 그 값으로).
쓴 보정값은 sensor_index.csv 의 clock_offset 열에 남는다.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "labels" / "store01_shoot3"
MARGIN_BEFORE = 2.0  # 영상 시작 전 여유 (초)
MARGIN_AFTER = 8.0   # 영상 끝 뒤 여유 — label_takes 의 허용 지연과 같다
SHIFT_ROOM = 1.2     # 영상별 보정 최대값(label_takes FIT_MAX_SHIFT) — 여유 구간을 이만큼 넓혀 보정해도 사건이 빠지지 않게
OFFSETS = OUT / "sensor_offsets.csv"
SENSOR_FIELDS = ["time_sec", "wall_time", "lane", "product", "event", "before", "after", "delta", "distance_mm",
                 "in_video", "raw_sec", "angle1_sec"]
KST = dt.timezone(dt.timedelta(hours=9))
MP4_EPOCH = dt.datetime(1904, 1, 1, tzinfo=dt.timezone.utc)

LINE = re.compile(
    r"^(?P<time>\d{2}:\d{2}:\d{2})\s+L(?P<lane>\d+)\s+(?P<event>[A-Z_]+)\s+"
    r"(?P<before>\d+)→(?P<after>\d+)\s+(?P<dist>[\d.]+|-)\s+(?P<text>.*)$")


@dataclass
class SensorLine:
    wall: dt.time
    lane: str
    event: str
    before: int
    after: int
    distance_mm: str
    product: str


@dataclass
class VideoClock:
    name: str
    end: dt.datetime
    duration: float

    @property
    def start(self) -> dt.datetime:
        return self.end - dt.timedelta(seconds=self.duration)


def parse_events(text: str) -> list[SensorLine]:
    """events_*.txt 표 줄만 읽는다. 머리말·구분선은 건너뜀."""
    rows = []
    for raw in text.splitlines():
        m = LINE.match(raw.strip())
        if not m:
            continue
        h, mi, s = map(int, m["time"].split(":"))
        parts = [p.strip() for p in m["text"].split("/")]
        rows.append(SensorLine(dt.time(h, mi, s), f"L{m['lane']}", m["event"], int(m["before"]),
                               int(m["after"]), m["dist"], parts[1] if len(parts) > 1 else ""))
    return rows


def read_mvhd(path: Path) -> tuple[dt.datetime, float]:
    """mp4 moov/mvhd 의 기록 시각(UTC)과 길이(초)."""
    with path.open("rb") as f:
        size_total = path.stat().st_size

        def boxes(start: int, end: int):
            pos = start
            while pos + 8 <= end:
                f.seek(pos)
                size, kind = struct.unpack(">I4s", f.read(8))
                header = 8
                if size == 1:
                    size, header = struct.unpack(">Q", f.read(8))[0], 16
                elif size == 0:
                    size = end - pos
                if size < header:
                    return
                yield kind, pos + header, pos + size
                pos += size

        for kind, body, end in boxes(0, size_total):
            if kind != b"moov":
                continue
            for sub, sbody, _ in boxes(body, end):
                if sub != b"mvhd":
                    continue
                f.seek(sbody)
                version = f.read(1)[0]
                f.read(3)
                if version == 1:
                    created, _, scale, length = struct.unpack(">QQIQ", f.read(28))
                else:
                    created, _, scale, length = struct.unpack(">IIII", f.read(16))
                return MP4_EPOCH + dt.timedelta(seconds=created), length / scale
    raise ValueError(f"{path.name}: moov/mvhd 없음")


def assign(events: list[SensorLine], video: VideoClock, day: dt.date, clock_offset: float,
           shift: float = 0.0, axis_offset: float = 0.0, axis_duration: float | None = None) -> list[dict]:
    """영상 구간 [-앞 여유, 길이+뒤 여유] (보정 여유 SHIFT_ROOM 만큼 더 넓힘, raw 기준) 안의 이벤트를 영상 시각과 함께.
    video = 시계 연결용 영상(angle1). angle1_sec = 그 영상 시각.
    axis_offset = 축 영상으로 옮길 시차 (초, angle1 = 축 + axis_offset), axis_duration = 축 영상 길이 (없으면 video 길이).
    raw_sec = 보정 전 축 시각 = angle1_sec - axis_offset, time_sec = raw_sec - shift (영상별 보정),
    in_video = time_sec 가 축 영상 [0, 길이] 안. clock_offset = 휴대폰 시계 - 센서 PC 시계 (초)."""
    start = video.start.astimezone(KST) - dt.timedelta(seconds=clock_offset)
    length = video.duration if axis_duration is None else axis_duration
    out = []
    for e in events:
        wall = dt.datetime.combine(day, e.wall, KST)
        a1 = (wall - start).total_seconds()
        raw = a1 - axis_offset
        if -MARGIN_BEFORE - SHIFT_ROOM <= raw <= length + MARGIN_AFTER + SHIFT_ROOM:
            t = raw - shift
            out.append({"time_sec": f"{t:.2f}", "wall_time": e.wall.strftime("%H:%M:%S"), "lane": e.lane,
                        "product": e.product, "event": e.event, "before": e.before, "after": e.after,
                        "delta": e.after - e.before, "distance_mm": e.distance_mm,
                        "in_video": int(0.0 <= t <= length), "raw_sec": f"{raw:.2f}", "angle1_sec": f"{a1:.2f}"})
    return out


def load_sync(path: Path) -> dict[str, float]:
    """카메라 시차 (ms) — angle1 = angle0 + offset_ms."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {r["pair"]: float(r["offset_ms"]) for r in csv.DictReader(f) if (r.get("offset_ms") or "").strip()}


def load_offsets(path: Path = OFFSETS) -> dict[str, float]:
    """영상별 보정 (applied=1 만). 파일이 없으면 빈 표."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {r["pair"]: float(r["shift_sec"]) for r in csv.DictReader(f) if (r.get("applied") or "0").strip() == "1"}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder", type=Path, help="영상과 events_*.txt 가 있는 폴더")
    ap.add_argument("--events", type=Path, help="이벤트 파일 (기본: 폴더의 events_*.txt)")
    ap.add_argument("--clock-offset", type=float, default=0.0, metavar="초",
                    help="휴대폰 시계 - 센서 PC 시계 (기본 0, 재지 않음)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--no-video-offsets", action="store_true", help="영상별 보정(sensor_offsets.csv)을 쓰지 않음")
    ap.add_argument("--axis", choices=["angle0", "angle1"], default="angle0",
                    help="시각 축 (기본 angle0 — 카메라 시차로 옮김. angle1 은 옛 방식)")
    ap.add_argument("--sync", type=Path, help="카메라 시차 표 (기본 <out>/sync.csv)")
    ap.add_argument("--axis-folder", type=Path, help="angle0 영상 폴더 (기본 <folder>/../2(angle0))")
    args = ap.parse_args()
    shifts = {} if args.no_video_offsets else load_offsets(args.out / "sensor_offsets.csv")
    if shifts:
        print(f"영상별 보정 {len(shifts)}편: " + ", ".join(f"{k} {v:+.2f}s" for k, v in sorted(shifts.items())))

    events_path = args.events or next(iter(sorted(args.folder.glob("events_*.txt"))), None)
    if events_path is None:
        print("events_*.txt 없음")
        return 1
    events = parse_events(events_path.read_text(encoding="utf-8-sig"))
    print(f"센서 이벤트 {len(events)}건 ({events_path.name})")

    videos = [p for p in sorted(args.folder.glob("*.mp4")) if p.stem != "area_setting"]
    axis_offsets: dict[str, float] = {}
    axis_lengths: dict[str, float] = {}
    if args.axis == "angle0":
        sync_path = args.sync or args.out / "sync.csv"
        sync = load_sync(sync_path)
        axis_folder = args.axis_folder or args.folder.parent / "2(angle0)"
        missing = [p.stem for p in videos if p.stem not in sync]
        no_video = [p.stem for p in videos if not (axis_folder / p.name).exists()]
        if missing or no_video:
            print("angle0 축으로 옮길 수 없음 — 아무 파일도 쓰지 않음")
            if missing:
                print(f"  시차 없음 ({sync_path}): {', '.join(missing)}")
            if no_video:
                print(f"  angle0 영상 없음 ({axis_folder}): {', '.join(no_video)}")
            return 1
        for p in videos:
            axis_offsets[p.stem] = sync[p.stem] / 1000.0
            axis_lengths[p.stem] = read_mvhd(axis_folder / p.name)[1]
        print(f"시각 축 angle0 (시차 {sync_path}, 영상 {axis_folder})")

    index = []
    files: dict[str, list[dict]] = {}
    used = 0
    spans: dict[str, tuple[float, float]] = {}
    for path in videos:
        end, duration = read_mvhd(path)
        video = VideoClock(path.stem, end, duration)
        off = axis_offsets.get(path.stem, 0.0)
        length = axis_lengths.get(path.stem, duration)
        rows = assign(events, video, video.end.astimezone(KST).date(), args.clock_offset, shifts.get(path.stem, 0.0),
                      off, length)
        start = video.start.timestamp() - args.clock_offset + off   # 축 영상 0초의 센서 PC 시각
        spans[path.stem] = (start - MARGIN_BEFORE - SHIFT_ROOM, start + length + MARGIN_AFTER + SHIFT_ROOM)
        inside = [r for r in rows if r["in_video"]]
        used += len(inside)
        files[path.stem] = rows
        kinds = [r["event"] for r in inside]
        note = "센서 이벤트 없음" if not inside else ("RESYNC 있음" if "RESYNC" in kinds else "")
        s_axis = video.start + dt.timedelta(seconds=off)
        index.append({"pair": path.stem, "start": s_axis.astimezone(KST).strftime("%H:%M:%S"),
                      "end": (s_axis + dt.timedelta(seconds=length)).astimezone(KST).strftime("%H:%M:%S"),
                      "duration_sec": f"{length:.2f}",
                      "events": len(inside), "remove": kinds.count("REMOVE"), "add": kinds.count("ADD"),
                      "removed_items": -sum(r["delta"] for r in inside if r["event"] == "REMOVE"),
                      "clock_offset": f"{args.clock_offset:+.2f}", "video_shift": f"{shifts.get(path.stem, 0.0):+.2f}",
                      "axis": args.axis, "sync_offset_ms": f"{off * 1000:+.1f}",
                      "note": note})
        print(f"  {path.stem:6s} {index[-1]['start']}~{index[-1]['end']}  "
              + (", ".join(f"{r['time_sec']}s {r['lane']} {r['event']} {r['before']}→{r['after']}" for r in inside)
                 or "(없음)"))
    (args.out / "sensor").mkdir(parents=True, exist_ok=True)   # 모든 짝 계산이 끝난 뒤에 쓴다
    for stem, rows in files.items():
        with (args.out / "sensor" / f"{stem}.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SENSOR_FIELDS)
            w.writeheader()
            w.writerows(rows)
    for a, (s0, e0) in spans.items():  # 여유 구간이 이웃 영상과 겹치면 같은 이벤트가 두 영상에 들어갈 수 있다
        near = [b for b, (s1, e1) in spans.items() if b != a and s0 < e1 and s1 < e0]
        if near:
            row = next(r for r in index if r["pair"] == a)
            row["note"] = "; ".join(x for x in (row["note"], "여유 구간이 겹침: " + ",".join(near)) if x)
    with (args.out / "sensor_index.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0]))
        w.writeheader()
        w.writerows(index)
    print(f"영상 {len(index)}편 안에 {used}건, 영상 밖 {len(events) - used}건 (주로 다시 채운 ADD) → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
