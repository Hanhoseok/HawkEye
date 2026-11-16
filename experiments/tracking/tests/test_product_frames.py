"""상품 탐지 학습용 장면 고르기 (scripts/product_frames.py) — 영상을 읽지 않는 부분."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import product_frames as pf  # noqa: E402

TIMES = [i * 100.0 for i in range(101)]   # 0~10초, 초당 10장


def test_base_is_one_frame_per_second_by_real_time():
    assert pf.base_frames(TIMES) == list(range(0, 101, 10))
    # 장면을 빠뜨린 영상(3차 angle1): 번호가 아니라 시각으로 고른다 — 빠진 구간은 그 뒤 첫 장면
    gappy = [0.0, 100.0, 200.0, 2500.0, 2600.0, 3100.0]
    assert pf.base_frames(gappy) == [0, 3, 5]


def test_near_frames_skip_times_outside_the_video():
    assert pf.near_frames(TIMES, 5000.0, (0, 750, 1500)) == [50, 57, 65]      # 5.75초는 두 장면의 한가운데 — 앞 장면
    assert pf.near_frames(TIMES, 9500.0, (0, 750, 1500)) == [95]          # 10.25초 · 11초는 영상 밖
    assert pf.near_frames(TIMES, 5000.0, (0, 20)) == [50]                 # 같은 장면은 한 번만


def test_take_moments_come_first_and_keep_their_lane():
    rows = pf.choose(TIMES, takes=[(5000.0, "L2", "take")], scans=[8000.0])
    assert rows[50] == {"reason": "take", "lane": "L2", "action": "take", "batch": 1}
    assert rows[57]["batch"] == 1 and rows[65]["batch"] == 1
    assert rows[55]["reason"] == "take" and rows[55]["batch"] == 0        # 후보에는 있지만 먼저 찍지는 않는다
    assert rows[80] == {"reason": "scan", "lane": "", "action": "", "batch": 1}
    assert rows[60]["reason"] == "take"                                   # 6.0초 base 장면이 take 구간 안 — take 가 이긴다
    first_base = [i for i, r in rows.items() if r["reason"] == "base" and r["batch"] == 1]
    assert len(first_base) == pf.BASE_FIRST


def test_first_base_frames_prefer_frames_with_people():
    people = {i: (1 if i in (20, 30) else 0) for i in range(0, 101, 10)}
    rows = pf.choose(TIMES, takes=[], scans=[], people=people)
    assert sorted(i for i, r in rows.items() if r["batch"] == 1) == [20, 30]


def test_split_is_by_video_and_per_set():
    pairs = [f"C{i:02d}" for i in range(1, 11)] + [f"N{i:02d}" for i in range(1, 6)] + ["T01", "T02"]
    split = pf.split_of(pairs)
    assert [split[f"C{i:02d}"] for i in range(1, 11)] == ["train", "train", "test", "train", "val"] * 2
    assert split["N03"] == "test" and split["N05"] == "val"
    assert split["T01"] == split["T02"] == "train"                        # 2편뿐인 세트는 전부 학습


def test_committed_plan_keeps_each_video_on_one_side():
    path = pf.DATASET / "frames.csv"
    if not path.exists():
        return
    rows = pf.read_plan(path)
    side: dict[str, set] = {}
    for r in rows:
        side.setdefault(r["pair"], set()).add(r["split"])
    assert all(len(s) == 1 for s in side.values())
    assert len({r["image"] for r in rows}) == len(rows)
