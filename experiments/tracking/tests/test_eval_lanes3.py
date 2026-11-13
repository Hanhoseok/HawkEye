import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import eval_lanes3 as E  # noqa: E402
from src.shelf.slot_diff import SlotDiff, SlotDiffConfig  # noqa: E402

OK = {"A": True}


def ep(lane, s, e, changed=True, direction="줄어듦", state="compared", pair="A"):
    return E.Episode(pair, lane, s * 1000.0, e * 1000.0, state, changed, direction)


def lb(t, lane="L1", action="take", pair="A", seq=1):
    return E.Label(pair, seq, t * 1000.0, lane, action)


def test_found_inside_episode_and_direction():
    res, ch = E.evaluate([lb(5.0), lb(9.0, action="return", seq=2)],
                         [ep("L1", 4, 6), ep("L1", 8, 10, direction="줄어듦")], OK)
    assert [r.video for r in res] == ["찾음", "찾음"]
    assert [r.direction for r in res] == ["맞음", "틀림"]
    assert [len(c["labels"]) for c in ch] == [1, 1]


def test_window_edges_and_lane_must_match():
    eps = [ep("L1", 4, 6)]
    assert E.evaluate([lb(3.0)], eps, OK)[0][0].video == "찾음"        # 시작 1초 전 = 안
    assert E.evaluate([lb(2.9)], eps, OK)[0][0].video == "놓침"
    assert E.evaluate([lb(7.0)], eps, OK)[0][0].video == "찾음"        # 끝 1초 뒤 = 안
    assert E.evaluate([lb(5.0, lane="L2")], eps, OK)[0][0].video == "놓침"


def test_one_label_gets_one_change_inside_beats_near():
    eps = [ep("L1", 2, 4.5), ep("L1", 5, 7)]
    res, ch = E.evaluate([lb(5.2)], eps, OK)
    assert res[0].episode == 1                                        # 품은 구간
    s = E.lane_stats(res, ch, "L1")
    assert s["changes"] == 2 and s["fake"] == 1 and s["precision"] == 0.5 and s["recall"] == 1.0


def test_many_labels_share_one_change_precision_by_change():
    # N05 꼴: 같은 레인 세 번, 변화 하나 + 따로 가짜 하나
    eps = [ep("L2", 6, 15), ep("L2", 20, 21)]
    res, ch = E.evaluate([lb(7, "L2"), lb(10, "L2", seq=2), lb(13.9, "L2", seq=3)], eps, OK)
    s = E.lane_stats(res, ch, "L2")
    assert s["found"] == 3 and s["recall"] == 1.0
    assert s["changes"] == 2 and s["fake"] == 1 and s["precision"] == 0.5 and s["multi"] == 1


def test_direction_unknown_and_missed_causes():
    eps = [ep("L1", 4, 6, direction="바뀜"), ep("L2", 4, 6, changed=False),
           ep("L3", 0, 3, state="no_reference", changed=False), ep("L1", 10, 12, state="unfinished", changed=False)]
    res, _ = E.evaluate([lb(5), lb(5, "L2", seq=2), lb(2, "L3", seq=3), lb(11, seq=4), lb(20, seq=5)], eps, OK)
    assert res[0].direction == "모름"
    assert [r.cause for r in res[1:]] == ["비교했지만 변화 없음", "기준 없음(첫 가림)", "끝까지 가림", "가림 구간 없음"]


def test_take_return_pair_in_unchanged_episode_is_excluded():
    # N10 꼴: 꺼냈다 4.5초 만에 되돌림, 그림 그대로
    eps = [ep("L1", 10, 15.5, changed=False)]
    res, ch = E.evaluate([lb(10.3), lb(14.9, action="return", seq=2)], eps, OK)
    assert [r.video for r in res] == ["되돌림 쌍", "되돌림 쌍"]
    assert E.lane_stats(res, ch, "L1")["excluded"] == 2
    # 같은 구간이 바뀌었으면 일반 규칙
    res2, _ = E.evaluate([lb(10.3), lb(14.9, action="return", seq=2)], [ep("L1", 10, 15.5)], OK)
    assert [r.video for r in res2] == ["찾음", "찾음"]


def test_failed_video_excluded_and_its_changes_not_counted():
    res, ch = E.evaluate([lb(5)], [ep("L1", 4, 6)], {"A": False})
    assert res[0].video == "처리 실패" and ch == []


def test_sensor_success_kinds_and_crosstab():
    sensor = {("A", "L1", "take", 5000): "늦음", ("A", "L2", "take", 5000): "경계", ("A", "L3", "take", 5000): "놓침"}
    res, _ = E.evaluate([lb(5), lb(5, "L2", seq=2), lb(5, "L3", seq=3)], [ep("L1", 4, 6), ep("L3", 4, 6, direction="늘어남")],
                        OK, sensor)
    assert [r.sensor_ok for r in res] == [True, True, False]
    a = E.crosstab(res, None, lambda r: r.video == "찾음")
    assert a == {"both": 1, "video_only": 1, "sensor_only": 1, "neither": 0, "n": 3}
    b = E.crosstab(res, None, lambda r: r.video == "찾음" and r.direction == "맞음")
    assert b["video_only"] == 0 and b["neither"] == 1


def test_parse_set():
    cfg = E.parse_set(["min_ratio=0.1", "shift_px=6"])
    assert cfg.min_ratio == 0.1 and cfg.shift_px == 6 and isinstance(cfg.shift_px, int)
    with pytest.raises(SystemExit):
        E.parse_set(["nope=1"])


def test_load_pose(tmp_path):
    p = tmp_path / "p.csv"
    p.write_text("ms,frame,person,x1,y1,x2,y2,conf,lwx,lwy,lwc,rwx,rwy,rwc\n"
                 "0.000,0,-1,,,,,,,,,,,\n"
                 "100.000,3,0,1,2,3,4,0.9,0,0,0,0,0,0\n100.000,3,1,5,6,7,8,0.8,0,0,0,0,0,0\n", encoding="utf-8")
    assert E.load_pose(p) == {0: (0.0, []), 3: (100.0, [(1, 2, 3, 4), (5, 6, 7, 8)])}


# ---------------------------------------------------------------- SlotDiff 가림 기록

def _img(v):
    img = np.full((40, 60, 3), 128, np.uint8)
    img[10:30, 5:25] = v          # 칸 a 안 무늬
    img[10:30, 35:55] = 60        # 칸 b
    return img


def test_slot_diff_episodes_record_compared_no_reference_and_unfinished():
    sd = SlotDiff([("a", (0, 5, 30, 35)), ("b", (30, 5, 60, 35))], SlotDiffConfig(stable_seconds=0.2))
    person = [(0, 0, 15, 40)]
    t = 0
    for _ in range(3):                               # 처음부터 가림 → 첫 가림은 기준 없음
        sd.update(t, _img(200), person); t += 100
    for _ in range(4):
        sd.update(t, _img(200), []); t += 100
    assert [(e["slot"], e["state"]) for e in sd.episodes] == [("a", "no_reference")]
    for _ in range(3):                               # 가렸다가 그림이 바뀜
        sd.update(t, _img(200), person); t += 100
    out = []
    for _ in range(4):
        out += sd.update(t, _img(20), []); t += 100
    comp = [e for e in sd.episodes if e["state"] == "compared"]
    assert len(out) == 1 and comp and comp[-1]["changed"] == 1 and comp[-1]["slot"] == "a"
    sd.update(t, _img(20), person)                   # 끝에 가린 채
    sd.finish(t + 100)
    assert sd.episodes[-1]["state"] == "unfinished" and sd.episodes[-1]["slot"] == "a"


class _Cap:
    def __init__(self, n, bad_retrieve=None):
        self.n, self.i, self.bad = n, -1, bad_retrieve

    def isOpened(self):
        return True

    def grab(self):
        if self.i + 1 >= self.n:
            return False
        self.i += 1
        return True

    def retrieve(self):
        if self.i == self.bad:
            return False, None
        return True, np.zeros((40, 60, 3), np.uint8)

    def release(self):
        pass

    def get(self, prop):
        return {E.cv2.CAP_PROP_FRAME_COUNT: self.n, E.cv2.CAP_PROP_FPS: 30.0}.get(prop, 0.0)


BOXES = [("L1", (0, 0, 20, 20))]


def test_diff_one_fails_on_empty_cache_short_video_or_bad_frame(monkeypatch):
    poses = {0: (0.0, []), 5: (500.0, [])}
    monkeypatch.setattr(E.cv2, "VideoCapture", lambda p: _Cap(8))
    eps, used = E.diff_one(Path("x.mp4"), BOXES, poses, SlotDiffConfig())
    assert used == 2
    with pytest.raises(ValueError):
        E.diff_one(Path("x.mp4"), BOXES, {}, SlotDiffConfig())
    monkeypatch.setattr(E.cv2, "VideoCapture", lambda p: _Cap(40))         # 캐시가 영상 끝까지 못 감 (부분 캐시)
    with pytest.raises(ValueError):
        E.diff_one(Path("x.mp4"), BOXES, poses, SlotDiffConfig())
    monkeypatch.setattr(E.cv2, "VideoCapture", lambda p: _Cap(3))          # 캐시보다 짧음
    with pytest.raises(ValueError):
        E.diff_one(Path("x.mp4"), BOXES, poses, SlotDiffConfig())
    monkeypatch.setattr(E.cv2, "VideoCapture", lambda p: _Cap(8, bad_retrieve=5))
    with pytest.raises(ValueError):
        E.diff_one(Path("x.mp4"), BOXES, poses, SlotDiffConfig())


class _ShortCap(_Cap):
    """장면 수는 30 이라 하고 6장에서 grab 이 끊김."""

    def grab(self):
        if self.i + 1 >= 6:
            return False
        self.i += 1
        return True

    def get(self, prop):
        return {E.cv2.CAP_PROP_FRAME_COUNT: 30, E.cv2.CAP_PROP_FPS: 30.0, E.cv2.CAP_PROP_POS_MSEC: self.i * 33.3}.get(prop, 0.0)


def test_pose_one_raises_when_reading_stops_early(monkeypatch):
    class Model:
        def predict(self, *a, **k):
            class R:
                boxes = None
                keypoints = None
            return [R()]

    monkeypatch.setattr(E.cv2, "VideoCapture", lambda p: _ShortCap(30))
    with pytest.raises(ValueError):
        E.pose_one(Model(), Path("x.mp4"))
