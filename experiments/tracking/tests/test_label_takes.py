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
