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
    assert [(r["time_sec"], r["lane"], r["delta"]) for r in out] == [("8.0", "L2", -1), ("11.0", "L3", 1)]
    # 휴대폰 시계가 2초 빠르면 영상 시각은 2초 늦어진다
    shifted = S.assign(rows, video, dt.date(2026, 11, 11), 2.0)
    assert [r["time_sec"] for r in shifted] == ["10.0", "13.0"]


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
                          "15:08:08  L1   REMOVE        2→1            300  1번 레인 / 칸타타 / 1개 제거 / 현재 1개\n")
    end = dt.datetime(2026, 11, 11, 6, 8, 25, tzinfo=dt.timezone.utc)  # 영상 15:08:11 ~ 15:08:25
    out = S.assign(rows, S.VideoClock("T01", end, 14.0), dt.date(2026, 11, 11), 0.0)
    # 끝 뒤 2초(16초) 응답은 in_video=0 으로 남고, 끝 뒤 15초 ADD 와 시작 3초 전 L1 은 여유 밖
    assert [(r["time_sec"], r["in_video"]) for r in out] == [("16.0", 0)]
    # 시계 보정 +1.5초면 L1 이 -1.5초가 되어 여유 구간으로 들어온다 → 다시 분할해야 하는 이유
    shifted = S.assign(rows, S.VideoClock("T01", end, 14.0), dt.date(2026, 11, 11), 1.5)
    assert [(r["time_sec"], r["lane"], r["in_video"]) for r in shifted] == [("17.5", "L2", 0), ("-1.5", "L1", 0)]
