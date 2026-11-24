"""logs/events.jsonl → 사람이 읽는 이벤트 표(events_*.txt). HawkEye scripts/split_sensor_events.py 가 이 표를 읽는다.

  python tools/export_events.py --since "2026-11-24 21:00"
  python tools/export_events.py --since "2026-11-24 21:00" --until "2026-11-24 23:00" --out logs/events_촬영5.txt
  python tools/export_events.py --events ~/shelf_relay/2026-11-24/events.jsonl   # 서버 중계본도 된다

표 한 줄 (시각 = 확정 시각, 초 단위):
  21:52:55  L1   ADD           0→1            315  1번 레인 / 칸타타 / 1개 추가 / 현재 1개 / 시작 21:52:53.673
마지막 '시작 HH:MM:SS.mmm' 은 변화가 시작된 시각 (모니터 1.3 이상, change_start_time). 없는 이벤트는 붙지 않는다.
ADD · REMOVE · RESYNC 만 남긴다 (시스템 메시지, OUT_OF_RANGE 등 제외).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelf_monitor.sinks import describe  # noqa: E402

KEEP = ("ADD", "REMOVE", "RESYNC")


def _local(iso: str) -> datetime:
    return datetime.fromisoformat(iso).astimezone()


def load(path: Path, since: datetime | None, until: datetime | None) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r.get("data"), dict):          # 서버 중계본: {"recv", ..., "data": 이벤트}
            r = r["data"]
        if r.get("message_type") != "EVENT" or r.get("event") not in KEEP or not r.get("pc_time"):
            continue
        if r.get("previous_count") is None or r.get("current_count") is None:
            continue
        t = _local(r["pc_time"])
        if (since and t < since) or (until and t > until):
            continue
        out.append(r)
    seen, uniq = set(), []                           # 중계본은 같은 이벤트가 두 번 올 수 있다 (서버가 대부분 지우지만)
    for r in out:
        key = (r.get("pc_event_seq"), r["pc_time"], r.get("lane_id"), r["event"])
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    return sorted(uniq, key=lambda r: r["pc_time"])


def row(r: dict) -> str:
    t = _local(r["pc_time"]).strftime("%H:%M:%S")
    dist = "-" if r.get("distance_mm") is None else f"{float(r['distance_mm']):.0f}"
    text = describe(r)
    if r.get("change_start_time"):
        text += f" / 시작 {_local(r['change_start_time']).strftime('%H:%M:%S.%f')[:12]}"
    arrow = f"{r['previous_count']}→{r['current_count']}"
    return f"{t}  L{r['lane_id']:<3} {r['event']:<13} {arrow:<14} {dist:>4}  {text}"


def render(rows: list[dict], title: str, source: str) -> str:
    kinds = Counter(r["event"] for r in rows)
    lanes = Counter((r["lane_id"], r["event"]) for r in rows)
    with_start = sum(1 for r in rows if r.get("change_start_time"))
    span = (f"{_local(rows[0]['pc_time']):%H:%M:%S} ~ {_local(rows[-1]['pc_time']):%H:%M:%S}" if rows else "-")
    lines = [
        f"센서 이벤트 추출: {title} (총 {len(rows)}건, {span})",
        f"원본: {source} · 시스템 메시지와 수량 범위 밖(OUT_OF_RANGE)은 제외",
        "",
        "종류별: " + ", ".join(f"{k} {kinds[k]}" for k in KEEP if kinds[k]),
        "레인별: " + " | ".join(f"L{l} {e} {n}" for (l, e), n in sorted(lanes.items())),
        f"변화 시작 시각 있음: {with_start}건 (내용 끝 '시작 HH:MM:SS.mmm' — 사람 매칭은 이 시각으로)",
        "",
        f"{'시각':<10}{'레인':<5}{'이벤트':<12}{'이전→현재':<12}{'거리(mm)':<8}내용",
        "-" * 100,
    ]
    return "\n".join(lines + [row(r) for r in rows]) + "\n"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description="센서 이벤트 표 추출 (HawkEye 입력용)")
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--events", type=Path, default=root / "logs" / "events.jsonl")
    ap.add_argument("--since", help='예: "2026-11-24 21:00" (이 시각 이후, PC 시간대)')
    ap.add_argument("--until", help="이 시각까지")
    ap.add_argument("--out", type=Path, help="기본: logs/events_<since>_이후.txt")
    args = ap.parse_args()

    since = datetime.fromisoformat(args.since).astimezone() if args.since else None
    until = datetime.fromisoformat(args.until).astimezone() if args.until else None
    rows = load(args.events, since, until)
    title = (f"{since:%Y-%m-%d %H:%M} 이후" if since else "전체") + (f" ~ {until:%H:%M}" if until else "")
    out = args.out or root / "logs" / (f"events_{since:%Y-%m-%d_%H%M}_이후.txt" if since else "events_전체.txt")
    out.write_text(render(rows, title, str(args.events)), encoding="utf-8-sig")
    print(f"{len(rows)}건 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
