import datetime as dt
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import split_sensor_events as S  # noqa: E402

SAMPLE = """﻿센서 이벤트 추출: 2026-11-11 15:08 이후 (총 3건)
시각        레인   이벤트           이전→현재       거리(mm)  내용
----------------------------------------------------------------------------------------------------
15:08:19  L2   REMOVE        3→2            276  2번 레인 / 동과루 / 1개 제거 / 현재 2개
15:08:22  L3   ADD           0→1            309  3번 레인 / 애플사이다 / 1개 추가 / 현재 1개
15:13:38  L2   RESYNC        3→3            203  2번 레인 / 동과루 / 재동기화 3개 → 3개 (순변화 +0)
"""


def test_parse_events_reads_table_rows_only():
    rows = S.parse_events(SAMPLE)
    assert [(r.lane, r.event, r.before, r.after, r.product) for r in rows] == [
        ("L2", "REMOVE", 3, 2, "동과루"), ("L3", "ADD", 0, 1, "애플사이다"), ("L2", "RESYNC", 3, 3, "동과루")]
    assert rows[0].wall == dt.time(15, 8, 19)


def test_assign_uses_recording_time_as_video_end():
    rows = S.parse_events(SAMPLE)
    end = dt.datetime(2026, 11, 11, 6, 8, 25, tzinfo=dt.timezone.utc)  # 15:08:25 KST
    video = S.VideoClock("T01", end, 14.0)  # 15:08:11 ~ 15:08:25
    out = S.assign(rows, video, dt.date(2026, 11, 11), 0.0)
    assert [(r["time_sec"], r["lane"], r["delta"]) for r in out] == [("8.00", "L2", -1), ("11.00", "L3", 1)]
    # 휴대폰 시계가 2초 빠르면 영상 시각은 2초 늦어진다
    shifted = S.assign(rows, video, dt.date(2026, 11, 11), 2.0)
    assert [r["time_sec"] for r in shifted] == ["10.00", "13.00"]


def test_read_mvhd(tmp_path):
    created = int((dt.datetime(2026, 11, 11, 6, 8, 25, tzinfo=dt.timezone.utc) - S.MP4_EPOCH).total_seconds())
    mvhd_body = bytes([0, 0, 0, 0]) + struct.pack(">IIII", created, created, 600, 600 * 14)
    mvhd = struct.pack(">I4s", 8 + len(mvhd_body), b"mvhd") + mvhd_body
    moov = struct.pack(">I4s", 8 + len(mvhd), b"moov") + mvhd
    ftyp = struct.pack(">I4s", 16, b"ftyp") + b"qt  \0\0\0\0"
    path = tmp_path / "x.mp4"
    path.write_bytes(ftyp + moov)
    end, duration = S.read_mvhd(path)
    assert end == dt.datetime(2026, 11, 11, 6, 8, 25, tzinfo=dt.timezone.utc)
    assert duration == 14.0


def test_assign_keeps_margin_events_with_in_video_flag():
    rows = S.parse_events("15:08:27  L2   REMOVE        3→2            276  2번 레인 / 동과루 / 1개 제거 / 현재 2개\n"
                          "15:08:40  L2   ADD           2→3            203  2번 레인 / 동과루 / 1개 추가 / 현재 3개\n"
                          "15:08:08  L1   REMOVE        2→1            300  1번 레인 / 칸타타 / 1개 제거 / 현재 1개\n"
                          "15:08:07  L1   ADD           1→2            300  1번 레인 / 칸타타 / 1개 추가 / 현재 2개\n")
    end = dt.datetime(2026, 11, 11, 6, 8, 25, tzinfo=dt.timezone.utc)  # 영상 15:08:11 ~ 15:08:25
    out = S.assign(rows, S.VideoClock("T01", end, 14.0), dt.date(2026, 11, 11), 0.0)
    # 여유 = 앞 2초·뒤 8초 + 보정 여유 1.2초 (raw 기준): 끝 뒤 2초(16초)·시작 3초 전(-3.0) 은 in_video=0 으로 남고,
    # 끝 뒤 15초 ADD·시작 4초 전 ADD 는 밖
    assert [(r["time_sec"], r["in_video"], r["raw_sec"]) for r in out] == [("16.00", 0, "16.00"), ("-3.00", 0, "-3.00")]
    # 시계 보정 +1.5초면 L1 ADD 가 -2.5초가 되어 여유 구간으로 들어온다 → 다시 분할해야 하는 이유
    shifted = S.assign(rows, S.VideoClock("T01", end, 14.0), dt.date(2026, 11, 11), 1.5)
    assert [(r["time_sec"], r["lane"]) for r in shifted] == [("17.50", "L2"), ("-1.50", "L1"), ("-2.50", "L1")]


def test_video_shift_moves_time_but_keeps_raw_and_event_set(tmp_path):
    rows = S.parse_events("15:08:19  L2   REMOVE        3→2            276  2번 레인 / 동과루 / 1개 제거 / 현재 2개\n"
                          "15:08:10  L1   REMOVE        2→1            300  1번 레인 / 칸타타 / 1개 제거 / 현재 1개\n")
    end = dt.datetime(2026, 11, 11, 6, 8, 25, tzinfo=dt.timezone.utc)
    clip = S.VideoClock("T01", end, 14.0)
    plain = S.assign(rows, clip, dt.date(2026, 11, 11), 0.0)
    moved = S.assign(rows, clip, dt.date(2026, 11, 11), 0.0, shift=-1.1)
    assert [r["raw_sec"] for r in plain] == [r["raw_sec"] for r in moved] == ["8.00", "-1.00"]
    assert [(r["time_sec"], r["in_video"]) for r in moved] == [("9.10", 1), ("0.10", 1)]   # -1.0 → 0.1 영상 안으로
    assert [r["in_video"] for r in plain] == [1, 0]
    path = tmp_path / "o.csv"
    path.write_text("pair,shift_sec,n,spread_sec,applied,reason,base_sec,checked_by,updated\n"
                    "T01,-1.10,2,0.3,1,,+0.5,h,\nT02,+0.00,1,0,0,표본 1,+0.5,h,\n", encoding="utf-8")
    assert S.load_offsets(path) == {"T01": -1.1}
    assert S.load_offsets(tmp_path / "none.csv") == {}


# ---------------------------------------------------------------- angle0 축

def _mp4(path, end_utc, seconds):
    created = int((end_utc - S.MP4_EPOCH).total_seconds())
    body = bytes([0, 0, 0, 0]) + struct.pack(">IIII", created, created, 600, int(600 * seconds))
    mvhd = struct.pack(">I4s", 8 + len(body), b"mvhd") + body
    moov = struct.pack(">I4s", 8 + len(mvhd), b"moov") + mvhd
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack(">I4s", 16, b"ftyp") + b"qt  \0\0\0\0" + moov)


def test_assign_moves_to_axis_video_time():
    rows = S.parse_events(SAMPLE)
    end = dt.datetime(2026, 11, 11, 6, 8, 25, tzinfo=dt.timezone.utc)
    video = S.VideoClock("T01", end, 14.0)
    # angle1 = angle0 + 2.012초 → angle0 시각 = angle1 시각 - 2.012
    out = S.assign(rows, video, dt.date(2026, 11, 11), 0.0, axis_offset=2.012, axis_duration=9.5)
    assert [(r["time_sec"], r["raw_sec"], r["angle1_sec"], r["in_video"]) for r in out] == [
        ("5.99", "5.99", "8.00", 1), ("8.99", "8.99", "11.00", 1)]
    short = S.assign(rows, video, dt.date(2026, 11, 11), 0.0, axis_offset=2.012, axis_duration=7.0)
    assert [r["in_video"] for r in short] == [1, 0]   # angle0 영상 길이로 판단


def _setup(tmp_path, sync_rows):
    src = tmp_path / "source"
    end = dt.datetime(2026, 11, 11, 6, 8, 25, tzinfo=dt.timezone.utc)
    for cam in ("2(angle1)", "2(angle0)"):
        _mp4(src / cam / "T01.mp4", end, 14.0)
    (src / "2(angle1)" / "events_x.txt").write_text(SAMPLE, encoding="utf-8")
    out = tmp_path / "labels"
    out.mkdir()
    (out / "sync.csv").write_text("pair,video,reference,offset_ms,method,checked_by,note\n" + sync_rows,
                                  encoding="utf-8")
    return src / "2(angle1)", out


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["split_sensor_events.py", *map(str, argv)])
    return S.main()


def test_main_angle0_axis_writes_axis_columns(tmp_path, monkeypatch):
    folder, out = _setup(tmp_path, "T01,a,b,1000.0,manual,h,\n")
    assert _run(monkeypatch, folder, "--out", out) == 0
    import csv
    rows = list(csv.DictReader((out / "sensor" / "T01.csv").open(encoding="utf-8")))
    assert [(r["time_sec"], r["angle1_sec"]) for r in rows] == [("7.00", "8.00"), ("10.00", "11.00")]
    idx = list(csv.DictReader((out / "sensor_index.csv").open(encoding="utf-8")))[0]
    assert idx["axis"] == "angle0" and idx["sync_offset_ms"] == "+1000.0"
    assert idx["start"] == "15:08:12"   # angle1 시작 15:08:11 + 1초


def test_main_angle0_axis_missing_sync_writes_nothing(tmp_path, monkeypatch):
    folder, out = _setup(tmp_path, "")
    assert _run(monkeypatch, folder, "--out", out) == 1
    assert not (out / "sensor").exists() and not (out / "sensor_index.csv").exists()
