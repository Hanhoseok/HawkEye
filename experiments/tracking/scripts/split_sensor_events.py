"""3차 촬영(카메라 1대 + 아두이노 거리 센서) 센서 로그를 영상별로 나눈다.

센서 PC(shelf_monitor)는 레인별 확정 이벤트를 **실제 시각(초 단위)** 으로 남기고,
휴대폰 영상은 mp4 안에 기록 시각을 남긴다. 이 휴대폰은 그 시각이 **촬영이 끝난 시각**이다
(3차 39편 모두 REMOVE 가 영상 5~12초에, 다시 채운 ADD 는 영상이 끝난 뒤에 온다 — 시작 시각으로 보면 안 맞음).
그래서 영상 시작 = 기록 시각 - 영상 길이, 이벤트의 영상 시각 = 이벤트 시각 - 영상 시작.

    python scripts/split_sensor_events.py "../../source/2(angle1)"
    python scripts/split_sensor_events.py "../../source/2(angle1)" --clock-offset 1.5   # 휴대폰 시계가 센서 PC 보다 1.5초 빠를 때

입력: 영상 폴더 안의 mp4 와 events_*.txt (shelf_monitor 이벤트 추출본)
결과: labels/store01_shoot3/sensor/<영상>.csv, labels/store01_shoot3/sensor_index.csv

    time_sec,wall_time,lane,product,event,before,after,delta,distance_mm,in_video
    8.0,15:08:19,L2,동과루,REMOVE,3,2,-1,276,1

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


def assign(events: list[SensorLine], video: VideoClock, day: dt.date, clock_offset: float) -> list[dict]:
    """영상 구간 [-앞 여유, 길이+뒤 여유] 안의 이벤트를 영상 시각과 함께. in_video = 영상 [0, 길이] 안.
    clock_offset = 휴대폰 시계 - 센서 PC 시계 (초)."""
    start = video.start.astimezone(KST) - dt.timedelta(seconds=clock_offset)
    out = []
    for e in events:
        wall = dt.datetime.combine(day, e.wall, KST)
        t = (wall - start).total_seconds()
        if -MARGIN_BEFORE <= t <= video.duration + MARGIN_AFTER:
            out.append({"time_sec": f"{t:.1f}", "wall_time": e.wall.strftime("%H:%M:%S"), "lane": e.lane,
                        "product": e.product, "event": e.event, "before": e.before, "after": e.after,
                        "delta": e.after - e.before, "distance_mm": e.distance_mm,
                        "in_video": int(0.0 <= t <= video.duration)})
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder", type=Path, help="영상과 events_*.txt 가 있는 폴더")
    ap.add_argument("--events", type=Path, help="이벤트 파일 (기본: 폴더의 events_*.txt)")
    ap.add_argument("--clock-offset", type=float, default=0.0, metavar="초",
                    help="휴대폰 시계 - 센서 PC 시계 (기본 0, 재지 않음)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    events_path = args.events or next(iter(sorted(args.folder.glob("events_*.txt"))), None)
    if events_path is None:
        print("events_*.txt 없음")
        return 1
    events = parse_events(events_path.read_text(encoding="utf-8-sig"))
    print(f"센서 이벤트 {len(events)}건 ({events_path.name})")

    (args.out / "sensor").mkdir(parents=True, exist_ok=True)
    index = []
    used = 0
    spans: dict[str, tuple[float, float]] = {}
    for path in sorted(args.folder.glob("*.mp4")):
        if path.stem == "area_setting":
            continue
        end, duration = read_mvhd(path)
        video = VideoClock(path.stem, end, duration)
        rows = assign(events, video, video.end.astimezone(KST).date(), args.clock_offset)
        start = video.start.timestamp() - args.clock_offset
        spans[path.stem] = (start - MARGIN_BEFORE, start + duration + MARGIN_AFTER)
        inside = [r for r in rows if r["in_video"]]
        used += len(inside)
        with (args.out / "sensor" / f"{path.stem}.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["time_sec", "wall_time", "lane", "product", "event",
                                              "before", "after", "delta", "distance_mm", "in_video"])
            w.writeheader()
            w.writerows(rows)
        kinds = [r["event"] for r in inside]
        note = "센서 이벤트 없음" if not inside else ("RESYNC 있음" if "RESYNC" in kinds else "")
        index.append({"pair": path.stem, "start": video.start.astimezone(KST).strftime("%H:%M:%S"),
                      "end": video.end.astimezone(KST).strftime("%H:%M:%S"), "duration_sec": f"{duration:.2f}",
                      "events": len(inside), "remove": kinds.count("REMOVE"), "add": kinds.count("ADD"),
                      "removed_items": -sum(r["delta"] for r in inside if r["event"] == "REMOVE"),
                      "clock_offset": f"{args.clock_offset:+.2f}", "note": note})
        print(f"  {path.stem:6s} {index[-1]['start']}~{index[-1]['end']}  "
              + (", ".join(f"{r['time_sec']}s {r['lane']} {r['event']} {r['before']}→{r['after']}" for r in inside)
                 or "(없음)"))
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
