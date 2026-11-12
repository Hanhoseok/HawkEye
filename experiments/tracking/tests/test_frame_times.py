"""실제 장면 시각 (3차 angle1 장면 빠짐) — frame_times, FrameCache, sync_pair.Clip, label_takes, sync_floor."""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import label_payments as LP  # noqa: E402
import label_takes as L  # noqa: E402
import sync_floor as SF  # noqa: E402
import sync_pair as SP  # noqa: E402
from src.inputs import frame_times as FT  # noqa: E402

# 16.7ms 간격 중간에 233ms 빈틈
TIMES = [0.0, 16.7, 33.3, 266.6, 283.3, 300.0]


def test_index_at_nearest_and_clamped():
    assert FT.index_at(TIMES, 34.0) == 2
    assert FT.index_at(TIMES, 140.0) == 2 and FT.index_at(TIMES, 160.0) == 3   # 빈틈 가운데 기준 가까운 쪽
    assert FT.index_at(TIMES, -50.0) == 0 and FT.index_at(TIMES, 999.0) == 5
    assert FT.index_at([], 10.0) == 0


def test_first_ms_and_end():
    assert FT.first_ms(-133.6, 33.32, 30.0) == 0.0
    assert FT.first_ms(0.0, 16.7, 60.0) == 0.0
    assert abs(FT.first_ms(50.0, 40.0, 60.0) - (40.0 - 1000 / 60)) < 1e-9
    assert abs(FT.end_ms(TIMES, 60.0) - 316.7) < 1e-9
    assert abs(FT.drift_s(TIMES, 60.0) - (300.0 - 5 * 1000 / 60) / 1000) < 1e-9


def clip(times):
    c = object.__new__(SP.Clip)
    c.fps = 60.0
    c.frames = [np.full((2, 2, 3), i + 1, np.uint8) for i in range(len(times))]
    c.times = list(times)
    c.duration_ms = FT.end_ms(times, 60.0)
    return c


def test_clip_at_uses_real_times_and_black_outside():
    c = clip(TIMES)
    assert c.at(270.0)[0, 0, 0] == 4               # 번호÷fps 라면 270ms = 16번(없음) — 실제 시각으로 4번째 장면
    assert not c.at(-2012.0).any() and not c.at(400.0).any()   # 영상 밖 = 검은 화면
    assert c.at(0.0)[0, 0, 0] == 1


def fake_cache(times):
    c = object.__new__(LP.FrameCache)
    c.fps, c.expected, c.done = 60.0, len(times), True
    c.frames, c.times = [b"x"] * len(times), list(times)
    return c


def test_frame_cache_time_and_index():
    c = fake_cache(TIMES)
    assert c.time_sec(3) == 0.2666 and c.time_sec(99) == 0.3
    assert c.index_at(0.27) == 3
    assert abs(c.duration_sec - 0.3167) < 1e-9


def test_label_takes_draw_uses_given_time():
    img = np.zeros((90, 160, 3), np.uint8)
    a = L.draw(img.copy(), "X", 3, 60.0, 6, 1.0, [], None, "L1", 1, False, False, 0, 1, t_sec=0.2666, duration=0.3167)
    b = L.draw(img.copy(), "X", 3, 60.0, 6, 1.0, [], None, "L1", 1, False, False, 0, 1)
    assert not np.array_equal(a, b)   # 시각 글자·타임라인 위치가 실제 시각으로 그려짐


def test_retime_rows_old_and_new_cache():
    times = np.array([0.0, 16.7, 33.3, 266.6, 283.3, 300.0])
    old = [{"ms": "50.0", "person": "0"}]                  # 옛 캐시: 번호÷fps(60) = 3번 장면
    assert SF.retime_rows(old, times, 60.0) == [{"ms": "266.600", "person": "0", "frame": 3}]
    new = [{"ms": "999.000", "person": "0", "frame": "4"}]  # frame 열이 있으면 그 번호
    assert SF.retime_rows(new, times, 60.0)[0]["ms"] == "283.300"


def test_frame_at_nearest_with_rounded_request(tmp_path):
    path = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (32, 24))
    for i in range(10):
        w.write(np.full((24, 32, 3), i * 20, np.uint8))
    w.release()
    times = [i * 1000 / 30 for i in range(10)]
    out = SF.frame_at(path, [66.7, 400.0, -5.0], width=32, times=times)   # 66.666… 을 66.7 로 반올림해 줘도 2번
    assert abs(int(out[0][0, 0, 0]) - 40) <= 6
    assert out[1] is None and out[2] is None
