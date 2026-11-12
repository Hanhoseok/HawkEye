import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import label_takes as L  # noqa: E402
from label_takes import SensorRow, TakeLabel  # noqa: E402


def take(t, lane="L2", qty=1, action="take"):
    return TakeLabel("X", 0, t, lane, action, qty)


def remove(t, lane="L2", before=3, after=2, in_video=True):
    return SensorRow(t, lane, "REMOVE", before, after, in_video)


def kinds(rows):
    return [(r["kind"], r["lane"], r["dt"]) for r in rows]


def test_best_matching_beats_greedy_c05():
    # 실제 C05 L2 REMOVE 5초·9초. 정답 0.5초·6초 → 탐욕(6→5 먼저)은 1쌍, 최적은 2쌍
    rows = L.compare_pair([take(0.5), take(6.0)], [remove(5.0), remove(9.0, before=2, after=1)])
    assert sorted(r["kind"] for r in rows) == ["늦음", "늦음"]
    assert [r["dt"] for r in rows] == ["+4.50", "+3.00"]


def test_quantity_units_expand_both_sides():
    rows = L.compare_pair([take(5.0, lane="L1", qty=2)], [remove(5.5, lane="L1", before=2, after=0)])
    assert kinds(rows) == [("맞음", "L1", "+0.50"), ("맞음", "L1", "+0.50")]


def test_missed_fake_lane_mismatch_and_resync():
    labels = [take(8.0, "L2"), take(12.0, "L1"), take(20.0, "L3")]
    sensor = [remove(8.3, "L2"), remove(12.4, "L3", 1, 0), remove(30.0, "L1", 2, 1),
              SensorRow(1.0, "L2", "RESYNC", 3, 3, True)]
    rows = L.compare_pair(labels, sensor)
    assert sorted(r["kind"] for r in rows) == ["가짜", "놓침", "레인 다름", "맞음"]
    mismatch = next(r for r in rows if r["kind"] == "레인 다름")
    assert (mismatch["lane"], mismatch["sensor_lane"]) == ("L1", "L3")


def test_return_matches_add_not_remove():
    add = SensorRow(11.0, "L3", "ADD", 0, 1, True)
    rows = L.compare_pair([take(8.0, "L3", action="return")], [add, remove(8.0, "L3", 1, 0)])
    assert sorted(r["kind"] for r in rows) == ["가짜", "늦음"]


def test_margin_events_match_but_never_count_as_fake():
    # 14초 영상, 13초 행동에 센서가 15초(영상 밖)에 응답 → 짝지음. 짝 없는 영상 밖 ADD(다시 채움)는 가짜 아님
    rows = L.compare_pair([take(13.0)], [remove(15.0, in_video=False),
                                         SensorRow(30.0, "L2", "ADD", 2, 3, False)])
    assert kinds(rows) == [("경계", "L2", "+2.00")]


def test_other_lane_not_compared():
    assert L.compare_pair([take(5.0, "other")], []) == []


def test_classify_border_both_sides():
    assert L.classify(0.2) == "맞음"
    assert L.classify(-0.4) == "맞음"
    assert L.classify(1.6) == "경계"
    assert L.classify(-1.6) == "경계"
    assert L.classify(0.6) == "경계"
    assert L.classify(2.6) == "늦음"
    assert L.classify(-2.6) == "이름"


def test_window_limits():
    assert sorted(r["kind"] for r in L.compare_pair([take(5.0)], [remove(13.5)])) == ["가짜", "놓침"]  # +8.5 창 밖
    assert sorted(r["kind"] for r in L.compare_pair([take(5.0)], [remove(2.5)])) == ["가짜", "놓침"]  # -2.5 창 밖


def test_save_load_roundtrip_keeps_other_videos(tmp_path):
    path = tmp_path / "takes.csv"
    labels = {"T01": [TakeLabel("T01", 1, 8.0, "L2", "take", 1, "h")],
              "T05": [TakeLabel("T05", 0, None, "", "none", 0, "h")]}
    L.save_labels(labels, path)
    back = L.load_labels(path)
    assert back["T01"][0].time_sec == 8.0 and back["T05"][0].action == "none"
    back["T01"].append(TakeLabel("T01", 0, 3.0, "L1", "return", 2, "h"))
    back["T01"] = L.renumber(back["T01"])
    L.save_labels(back, path)
    again = L.load_labels(path)
    assert [(r.seq, r.time_sec) for r in again["T01"]] == [(1, 3.0), (2, 8.0)]
    assert "T05" in again


def test_compare_counts_only_done_videos(tmp_path, monkeypatch):
    sensor_dir = tmp_path / "sensor"
    sensor_dir.mkdir()
    header = "time_sec,wall_time,lane,product,event,before,after,delta,distance_mm,in_video\n"
    (sensor_dir / "A.csv").write_text(header + "7.6,,L2,,REMOVE,3,2,-1,,1\n22.6,,L1,,REMOVE,2,1,-1,,1\n",
                                      encoding="utf-8")
    original = L.load_sensor
    monkeypatch.setattr(L, "load_sensor", lambda p: original(p, sensor_dir))
    monkeypatch.setattr(L, "load_clock_offset", lambda: "+0.00")
    labels = {"A": [TakeLabel("A", 1, 7.5, "L2", "take", 1)]}
    status = {}
    L.set_status(status, "A", "draft", "h")
    total = L.compare(labels, status, ["A"], write=None)
    assert sum(total.values()) == 0  # 초안은 안 센다 (22.6초를 가짜로 세지 않음)
    L.set_status(status, "A", "done", "h")
    total = L.compare(labels, status, ["A"], write=None)
    assert total["맞음"] == 1 and total["가짜"] == 1


def test_status_roundtrip(tmp_path):
    path = tmp_path / "s.csv"
    status = {}
    L.set_status(status, "T01", "done", "h")
    L.save_status(status, path)
    assert L.is_done(L.load_status(path), "T01")


def test_compare_overwrites_old_result_when_nothing_done(tmp_path, monkeypatch):
    out = tmp_path / "check.csv"
    out.write_text("old,result\n1,2\n", encoding="utf-8")
    monkeypatch.setattr(L, "load_clock_offset", lambda: "+0.00")
    L.compare({}, {}, ["A"], write=out)
    assert out.read_text(encoding="utf-8").strip() == ",".join(L.CHECK_FIELDS)


def test_open_cache_skips_unreadable_video(tmp_path):
    assert L.open_cache(tmp_path / "missing.mp4", 640) is None


# ---------------------------------------------------------------- blind · 영상별 보정

import csv  # noqa: E402

import numpy as np  # noqa: E402

ORIGINAL_LOAD_SENSOR = L.load_sensor


def test_blind_draw_hides_sensor():
    img = np.zeros((360, 640, 3), np.uint8)
    sensor = [SensorRow(5.0, "L2", "REMOVE", 3, 2, True)]
    shown = L.draw(img.copy(), "T01", 150, 30.0, 600, 1.0, [], sensor, "L1", 1, False, False, 0, 39)
    hidden = L.draw(img.copy(), "T01", 150, 30.0, 600, 1.0, [], sensor, "L1", 1, False, False, 0, 39, blind=True)
    assert shown[:60].any() and not hidden[:60].any()          # 배너(영상 위쪽) 없음
    assert not np.array_equal(shown[360:], hidden[360:])         # 아래 막대도 다름 (목록·타임라인 점)


def test_mode_open_once_stays_open():
    status = {}
    L.mark_mode(status, "A", True, "h")
    L.set_status(status, "A", "draft", "h")
    assert L.mode_of(status, "A") == "blind"
    L.mark_mode(status, "A", False, "h")       # 센서 보이는 상태로 열고 아무것도 안 고침
    L.mark_mode(status, "A", True, "h")        # 다시 blind 로 열어 완료
    L.set_status(status, "A", "done", "h")
    assert L.mode_of(status, "A") == "open"
    assert L.mode_of({"B": {"pair": "B", "status": "done"}}, "B") == "open"   # 옛 파일(mode 없음)


def write_sensor(folder, pair, rows):
    folder.mkdir(exist_ok=True)
    with open(folder / f"{pair}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_sec", "wall_time", "lane", "product", "event", "before", "after", "delta",
                    "distance_mm", "in_video", "raw_sec"])
        for t, raw, lane, before, after in rows:
            w.writerow([t, "", lane, "", "REMOVE", before, after, after - before, "", 1, raw])


def fit_env(tmp_path, monkeypatch, sensors):
    folder = tmp_path / "sensor"
    for pair, rows in sensors.items():
        write_sensor(folder, pair, rows)
    monkeypatch.setattr(L, "load_sensor", lambda p: ORIGINAL_LOAD_SENSOR(p, folder))


def test_fit_offsets_rules(tmp_path, monkeypatch):
    # A: 두 행동 모두 +1.1 (기준보다 +0.6) → 적용 / B·C: +0.5 기준 / D: qty=2 한 행동 → 표본 1 / E: 들쭉날쭉
    fit_env(tmp_path, monkeypatch, {
        "A": [(6.1, 6.1, "L2", 3, 2), (11.1, 11.1, "L1", 2, 1)],
        "B": [(5.5, 5.5, "L2", 3, 2), (9.5, 9.5, "L1", 2, 1)],
        "C": [(4.5, 4.5, "L2", 3, 2), (8.5, 8.5, "L1", 2, 1)],
        "D": [(5.5, 5.5, "L1", 2, 0)],
        "E": [(5.0, 5.0, "L2", 3, 2), (12.0, 12.0, "L1", 2, 1)],
    })
    labels = {"A": [take(5.0), take(10.0, "L1")], "B": [take(5.0), take(9.0, "L1")], "C": [take(4.0), take(8.0, "L1")],
              "D": [take(5.0, "L1", qty=2)], "E": [take(5.0), take(9.0, "L1")]}
    status = {}
    for p in labels:
        L.set_status(status, p, "done", "h")
    out = L.fit_offsets(labels, status, list(labels), "h", write=None)
    assert out["A"]["applied"] == 1 and out["A"]["shift_sec"] == "+0.60"
    assert out["B"]["applied"] == 1 and out["B"]["shift_sec"] == "+0.00"
    assert out["D"]["applied"] == 0 and "1개" in out["D"]["reason"]
    assert out["E"]["applied"] == 0 and "범위" in out["E"]["reason"]


def test_fit_offsets_uses_raw_time_so_refit_is_same(tmp_path, monkeypatch):
    # 원시 dt [-2.4, -1.4, -0.8] 같은 경계 사례: 보정 후 time_sec 로 매칭하면 짝이 바뀐다 → raw 로 다시 맞춰 같은 답
    raw = [(7.6, 7.6, "L2", 3, 2), (12.6, 12.6, "L2", 2, 1), (19.2, 19.2, "L2", 1, 0)]
    other = {"B": [(5.5, 5.5, "L2", 3, 2), (9.5, 9.5, "L2", 2, 1)], "C": [(4.5, 4.5, "L2", 3, 2), (8.5, 8.5, "L2", 2, 1)]}
    labels = {"A": [take(10.0), take(14.0), take(20.0)], "B": [take(5.0), take(9.0)], "C": [take(4.0), take(8.0)]}
    status = {}
    for p in labels:
        L.set_status(status, p, "done", "h")
    fit_env(tmp_path, monkeypatch, {"A": raw, **other})
    first = L.fit_offsets(labels, status, list(labels), "h", write=None)
    shift = float(first["A"]["shift_sec"])
    moved = [(round(t - shift, 1), r, lane, b, a) for t, r, lane, b, a in raw]   # 분할이 보정을 반영한 CSV
    fit_env(tmp_path, monkeypatch, {"A": moved, **other})
    again = L.fit_offsets(labels, status, list(labels), "h", write=None)
    assert {k: (v["shift_sec"], v["applied"], v["n"]) for k, v in first.items()} == \
           {k: (v["shift_sec"], v["applied"], v["n"]) for k, v in again.items()}


def test_old_status_rows_stay_open_after_save_and_reload(tmp_path):
    path = tmp_path / "s.csv"
    path.write_text("pair,status,checked_by,updated\nA,done,h,\nB,draft,h,\n", encoding="utf-8")
    status = L.load_status(path)
    L.mark_mode(status, "A", True, "h")
    L.save_status(status, path)
    again = L.load_status(path)
    L.mark_mode(again, "B", True, "h")
    assert L.mode_of(again, "A") == "open" and L.mode_of(again, "B") == "open"


def test_fit_offsets_overwrites_old_table_when_no_samples(tmp_path, monkeypatch):
    fit_env(tmp_path, monkeypatch, {"A": [(30.0, 30.0, "L2", 3, 2)]})
    out = tmp_path / "offsets.csv"
    out.write_text("pair,shift_sec,n,spread_sec,applied,reason,base_sec,checked_by,updated\nA,+0.80,2,0.1,1,,0,h,\n",
                   encoding="utf-8")
    status = {}
    L.set_status(status, "A", "done", "h")
    L.fit_offsets({"A": [take(5.0)]}, status, ["A"], "h", write=out)   # 짝 없음
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert [(r["pair"], r["applied"]) for r in rows] == [("A", "0")]
    L.fit_offsets({}, {}, ["A"], "h", write=out)                         # 완료 영상 없음 → 머리줄만
    assert open(out, encoding="utf-8").read().strip() == ",".join(L.OFFSET_FIELDS)


# ---------------------------------------------------------------- 옆 화면 (angle0)
import numpy as np  # noqa: E402
import pytest  # noqa: E402


class _FakeSide:
    """FrameCache 대신: 0.1초 간격 장면, 장면 i 는 밝기 i 로 채운 그림."""

    def __init__(self, n=50, width=64):
        self.n, self.width = n, width
        self.asked = []

    @property
    def duration_sec(self):
        return self.n * 0.1

    def index_at(self, sec):
        self.asked.append(sec)
        return max(0, min(int(round(sec / 0.1)), self.n - 1))

    def get(self, i):
        return np.full((36, self.width, 3), i, np.uint8)


def test_side_time_uses_sync_direction():
    # angle1 = angle0 + offset → 기준 angle0 에서 옆 angle1 시각
    assert L.side_time(6.70, 2012.0) == pytest.approx(8.712)
    assert L.side_time(1.5, -500.0) == pytest.approx(1.0)


def test_side_view_picks_synced_frame_and_blanks_outside():
    side = _FakeSide()
    img = L.side_view(side, 1000.0, 1.0, 64, 36)        # angle1 = 2.0초
    assert side.asked == [pytest.approx(2.0)]
    assert img.shape == (36, 64, 3) and img[0, 0, 0] == 20
    side.asked.clear()
    L.side_view(side, -1000.0, 0.5, 64, 36)    # angle1 -0.5초: 영상 밖
    L.side_view(side, 1000.0, 4.5, 64, 36)     # angle1 5.5초: 끝 넘음
    assert side.asked == []


def test_sensor_axis(tmp_path):
    p = tmp_path / "sensor_index.csv"
    assert L.sensor_axis(p) is None
    p.write_text("pair,start,video_shift\nC01,15:00:00,+0.00\n", encoding="utf-8")
    assert L.sensor_axis(p) is None                       # 옛 분할 (axis 열 없음)
    p.write_text("pair,axis\nC01,angle0\nC02,angle0\n", encoding="utf-8")
    assert L.sensor_axis(p) == "angle0"
    p.write_text("pair,axis\nC01,angle0\nC02,angle1\n", encoding="utf-8")
    assert L.sensor_axis(p) is None                       # 섞임


def test_side_view_without_video_or_offset_is_black_panel():
    assert L.side_view(None, 10.0, 1.0, 64, 36).shape == (36, 64, 3)
    side = _FakeSide()
    L.side_view(side, None, 1.0, 64, 36)
    assert side.asked == []


def test_applied_offsets_and_stale_check(tmp_path):
    p = tmp_path / "sensor_index.csv"
    p.write_text("pair,axis,sync_offset_ms\nC01,angle0,+14.9\nC02,angle0,-255.0\n", encoding="utf-8")
    applied = L.load_applied_offsets(p)
    assert applied == {"C01": pytest.approx(14.9), "C02": pytest.approx(-255.0)}
    assert L.stale_offsets(applied, {"C01": 14.9, "C02": -255.0}) == []
    # --sync 로 다른 표를 썼거나 split 뒤 sync.csv 를 고침 → 다른 짝을 알려준다
    assert L.stale_offsets(applied, {"C01": 14.9, "C02": -163.2}) == ["C02"]
    assert L.stale_offsets(applied, {"C01": 14.9}) == ["C02"]
    assert L.load_applied_offsets(tmp_path / "none.csv") == {}


def test_load_side_offsets(tmp_path):
    p = tmp_path / "sync.csv"
    p.write_text("pair,video,reference,offset_ms,method,checked_by,note\n"
                 "C16,a,b,2012.2,manual,hoseok,\n", encoding="utf-8")
    assert L.load_side_offsets(p) == {"C16": pytest.approx(2012.2)}
    assert L.load_side_offsets(tmp_path / "none.csv") == {}


def _fake_snapshot_env(monkeypatch, status_saves):
    class FakeCache:
        fps, n, duration_sec = 10.0, 5, 0.5

        def __init__(self, path, width):
            pass

        def index_at(self, sec):
            return 0

        def time_sec(self, i):
            return 0.0

        def get(self, i):
            return np.zeros((36, 64, 3), np.uint8)

    monkeypatch.setattr(L, "FrameCache", FakeCache)
    monkeypatch.setattr(L, "load_sensor", lambda pair: [])
    monkeypatch.setattr(L, "save_status", lambda status: status_saves.append(L.mode_of(status, "C02")))


def test_open_snapshot_marks_blind_pair_open(monkeypatch, tmp_path):
    saves = []
    _fake_snapshot_env(monkeypatch, saves)
    status = {"C02": {"pair": "C02", "status": "draft", "checked_by": "h", "updated": "", "mode": "blind"}}
    L.snapshot("C02", tmp_path, {}, status, 64, 0.0, tmp_path / "a.png", blind=False)
    assert L.mode_of(status, "C02") == "open" and saves == ["open"]
    L.mark_mode(status, "C02", True, "h")   # 다시 blind 로 열어도 open 유지
    assert L.mode_of(status, "C02") == "open"


def test_blind_snapshot_keeps_blind(monkeypatch, tmp_path):
    saves = []
    _fake_snapshot_env(monkeypatch, saves)
    status = {"C02": {"pair": "C02", "status": "draft", "checked_by": "h", "updated": "", "mode": "blind"}}
    L.snapshot("C02", tmp_path, {}, status, 64, 0.0, tmp_path / "a.png", blind=True)
    assert L.mode_of(status, "C02") == "blind" and saves == []


# ---------------------------------------------------------------- 문제 다시 보기 (--review)

CHECK_CSV = ("pair,kind,lane,direction,label_sec,sensor_lane,sensor_sec,dt\n"
             "C04,가짜,L3,take,,L3,4.9,\n"
             "C04,맞음,L2,take,6.53,L2,6.9,+0.40\n"
             "C04,놓침,L2,take,14.06,,,\n"
             "C19,늦음,L1,take,7.86,L1,11.3,+3.43\n"
             "C01,경계,L1,take,6.5,L1,7.0,+0.50\n")


def test_load_problems_filters_and_sorts(tmp_path):
    p = tmp_path / "check.csv"
    p.write_text(CHECK_CSV, encoding="utf-8")
    probs = L.load_problems(p)
    assert [(r["pair"], r["kind"], r["at"]) for r in probs] == [
        ("C04", "가짜", 4.9), ("C04", "놓침", 14.06), ("C19", "늦음", 7.86)]
    assert [r["pair"] for r in L.load_problems(p, kinds=("경계",))] == ["C01"]
    assert [r["kind"] for r in L.load_problems(p, pairs=["C19"])] == ["늦음"]
    assert L.load_problems(tmp_path / "none.csv") == []
    note = L.problem_note(probs[2], 2, 3)
    assert note.startswith("[3/3] LATE") and "dt +3.43s" in note and note.isascii()


def test_readonly_review_does_not_edit_or_save(monkeypatch):
    class Cache:
        n, fps, progress, duration_sec = 10, 10.0, 1.0, 1.0

        def time_sec(self, i):
            return i / 10

        def index_at(self, sec):
            return int(sec * 10)

        def get(self, i):
            return np.zeros((36, 64, 3), np.uint8)

    keys = iter([ord("t"), ord("u"), ord("c"), ord("s"), ord("q")])
    monkeypatch.setattr(L.cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(L.cv2, "waitKeyEx", lambda *a: next(keys))
    monkeypatch.setattr(L, "load_sensor", lambda pair: [])
    saved = []
    monkeypatch.setattr(L, "save_labels", lambda *a: saved.append("labels"))
    monkeypatch.setattr(L, "save_status", lambda *a: saved.append("status"))
    rows = [L.TakeLabel("C04", 1, 6.5, "L2", "take", 1, "h")]
    labels = {"C04": list(rows)}
    status = {"C04": {"pair": "C04", "status": "done", "checked_by": "h", "updated": "", "mode": "blind"}}
    out = L.label_video("C04", Cache(), labels, status, "h", 0, 0, 64, readonly=True, start_sec=0.3, note="x")
    assert out == "quit" and saved == []
    assert labels == {"C04": rows} and status["C04"]["mode"] == "blind" and status["C04"]["status"] == "done"
