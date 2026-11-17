import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import shoot3_eval as S  # noqa: E402
from src.fusion.shelf_events import TimeAxis  # noqa: E402

SHIFT = np.array([[1, 0, 80], [0, 1, 5], [0, 0, 1]], float)   # 영상 → area_setting: +80px, +5px


def test_time_axis_uses_real_times_when_given():
    times = (0.0, 16.7, 50.0, 66.7, 300.0)            # 장면을 빠뜨린 영상
    ax = TimeAxis(60.0, 100.0, times)
    assert ax.to_ms(4) == 400.0 and ax.to_ms(99) == 400.0
    assert ax.to_frame(100.0 + 200.0) == 4 and ax.to_frame(100.0 + 40.0) == 2
    plain = TimeAxis(60.0, 100.0)                     # 없으면 번호 ÷ fps (기존)
    assert plain.to_ms(4) == pytest.approx(4 * 1000 / 60 + 100) and plain.to_frame(100 + 1000 / 60 * 3) == 3


def test_pair_floor_moves_image_coords_only():
    base = {"calibration": {"image": [[0.5, 0.5]], "floor": [[0, 0]]}, "zones": [{"name": "door", "polygon": [[1, 2]]}],
            "ignore_image": [[0.1, 0.1, 0.2, 0.2]]}
    out = S.pair_floor(base, SHIFT)
    assert out["calibration"]["image"][0] == pytest.approx([(960 - 80) / 1920, (540 - 5) / 1080], abs=1e-5)
    assert out["zones"] == base["zones"] and out["calibration"]["floor"] == [[0, 0]]
    assert out["ignore_image"][0][0] == pytest.approx((192 - 80) / 1920, abs=1e-5)
    assert base["calibration"]["image"] == [[0.5, 0.5]]   # 원본 그대로


def test_pair_detail_moves_shelf_and_lanes_and_sets_band_map():
    base = {"reference": "area_setting", "shelves": [{"name": "shelf", "box": [0.5, 0.3, 0.6, 0.4], "zone": "shelf"}],
            "zones": [{"name": "L1", "type": "SHELF", "polygon": [[0.6, 0.3], [0.61, 0.3], [0.61, 0.4], [0.6, 0.4]]}]}
    out = S.pair_detail(base, SHIFT, [1.5, -0.2])
    dx = 80 / 1920
    assert out["shelves"][0]["box"][0] == pytest.approx(0.5 - dx, abs=1e-5)
    assert out["zones"][0]["polygon"][0][0] == pytest.approx(0.6 - dx, abs=1e-5)
    assert out["band_map"] == [1.5, -0.2] and "reference" not in out


def test_detail_point_same_place_same_value_under_camera_motion():
    # 같은 기준 자리(area_setting x=1150)를 카메라 위치가 다른 두 영상이 각자 화면에서 본다
    h_a = np.eye(3)
    h_b = SHIFT
    box_a = (1100, 300, 1200, 700)                    # 영상 A 화면 (기준과 같음)
    box_b = (1020, 295, 1120, 695)                    # 영상 B 화면 (= 기준 − 80, −5)
    assert S.detail_point(h_a)(box_a, 1920) == pytest.approx(S.detail_point(h_b)(box_b, 1920))
    assert S.detail_point(h_a)(box_a, 1920) == pytest.approx(1150 / 1920)


def test_band_map_file_roundtrip_and_default(tmp_path):
    p = tmp_path / "band_map.yaml"
    assert S.load_band_map(p) == [1.0, 0.0]
    S.save_band_map(1.234567, -0.5, "메모", p)
    assert S.load_band_map(p) == [1.2346, -0.5]


def test_make_pair_zones_merges_band_map(tmp_path, monkeypatch):
    base = tmp_path / "zones"
    base.mkdir()
    (base / "floor.yaml").write_text(yaml.safe_dump({"calibration": {"image": [[0.5, 0.5]], "floor": [[0, 0]]},
                                                     "zones": []}), encoding="utf-8")
    (base / "angle0.yaml").write_text(yaml.safe_dump({"shelves": [], "zones": []}), encoding="utf-8")
    S.save_band_map(2.0, 0.1, "", base / "band_map.yaml")
    monkeypatch.setattr(S, "homographies", lambda cam: {"N01": np.eye(3)})
    d = S.make_pair_zones("N01", tmp_path / "out", base)
    assert yaml.safe_load((d / "angle0.yaml").read_text(encoding="utf-8"))["band_map"] == [2.0, 0.1]
    with pytest.raises(KeyError):
        S.make_pair_zones("T99", tmp_path / "out", base)


def test_cached_times_checks_frame_count(tmp_path):
    cache = tmp_path / "t.json"
    cache.write_text(json.dumps([0.0, 16.7, 50.0]), encoding="utf-8")
    assert S.cached_times(Path("x.mp4"), cache, 3) == [0.0, 16.7, 50.0]
    assert S.cached_times(Path("x.mp4"), cache, 4) == [0.0, 16.7, 50.0, 83.3]   # 1장 많게 센 것은 늘려 맞춤
    with pytest.raises(ValueError):
        S.cached_times(Path("x.mp4"), cache, 6)                              # 3장 차이는 오류
    with pytest.raises(ValueError):
        S.cached_times(Path("x.mp4"), cache, 2)                              # 기록이 더 적으면 오류


def test_dual_eval_profile_paths():
    import dual_eval as D
    try:
        D.set_profile(3)
        assert D.OUT.name == "dual_eval_shoot3" and D.LABELS.name == "store01_shoot3" and D.SHOOT == 3
    finally:
        D.set_profile(2)
    assert D.OUT.name == "dual_eval_pose3" and D.ZONES.name == "store01_dual"
    assert D.zones_dir("N1") == D.ZONES and D.detail_point_for("N1") is None


def test_judge_records_failed_pair_and_keeps_going(tmp_path, monkeypatch):
    import dual_eval as D
    try:
        D.set_profile(3)
        monkeypatch.setattr(D, "OUT", tmp_path)
        for p in ("A01", "B01"):
            (tmp_path / p).mkdir()
            (tmp_path / p / ".done").write_text("ok")
        monkeypatch.setattr(D, "pairs", lambda pattern: ["A01", "B01"])
        monkeypatch.setattr(D, "load_band_map", lambda: (1.0, 0.0))
        monkeypatch.setattr(D, "load_truth", lambda: {})
        monkeypatch.setattr(D.AppConfig, "load", classmethod(lambda cls, p: None))
        monkeypatch.setattr(D.SyncTable, "load", classmethod(lambda cls, p: D.SyncTable([])))

        def fake(pair, *a, **k):
            if pair == "A01":
                raise KeyError(pair)
            return {"pair": pair, "kind": "정상", "verdicts": [], "expected": "CLEAR", "expected_source": "truth",
                    "outcome": "no_verdict", "correct": True, "correct_last": True, "no_verdict": True,
                    "last_verdicts": {}, "people": 1, "people_expected": 1, "people_ok": True, "links": 0,
                    "payments": 0, "unmatched_payments": 0, "cancels": 0, "unowned_takes": 0, "reasons": [], "actions": []}
        monkeypatch.setattr(D, "judge_pair", fake)
        rows = D.judge(None, True, False)
        assert [r["pair"] for r in rows] == ["B01"]
        assert json.loads((tmp_path / "failed.json").read_text(encoding="utf-8"))[0]["pair"] == "A01"
    finally:
        D.set_profile(2)


def test_load_sensor_axis_lanes_and_filters(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("time_sec,wall_time,lane,product,event,before,after,delta,distance_mm,in_video,raw_sec,angle1_sec\n"
                 "5.00,x,L2,a,REMOVE,3,2,-1,1,1,5.00,7.00\n"
                 "6.00,x,L3,a,REMOVE,1,0,-1,1,1,6.00,8.00\n"
                 "7.00,x,L1,a,RESYNC,2,2,0,1,1,7.00,9.00\n"
                 "40.00,x,L2,a,ADD,2,3,1,1,0,40.00,42.00\n", encoding="utf-8")
    ev = S.load_sensor(p, 2000.0)
    assert [(e.time_ms, e.slot, e.delta_g) for e in ev] == [(7000.0, "L2", -1.0), (8000.0, "L3", -1.0)]   # 영상 밖 ADD·RESYNC 뺌, +시차
    assert [e.slot for e in S.load_sensor(p, 0.0, ("L1", "L2"))] == ["L2"]
    assert S.load_sensor(tmp_path / "none.csv", 0.0) == []


def test_pair_detail_carries_the_floor_map_moved_to_the_clip():
    # 선반 카메라 바닥 지도(area_setting 기준)의 보정점을 그 영상 화면으로 옮겨 싣는다 — 두 화면 짝을 바닥 위 거리로 맺을 때 쓴다
    from src.zones.floor import detail_floor_map

    base = {"shelves": [], "zones": []}
    floor0 = {"calibration": {"image": [[0.2, 0.7], [0.2, 0.9], [0.6, 0.7], [0.6, 0.9]],
                              "floor": [[0, 0], [64, 0], [0, 74], [64, 74]]}}
    out = S.pair_detail(base, SHIFT, [1.0, 0.0], floor0)
    cal = out["floor_calibration"]
    assert cal["image"][0] == pytest.approx([0.2 - 80 / 1920, 0.7 - 5 / 1080], abs=1e-5) and cal["floor"][3] == [64.0, 74.0]
    # 같은 바닥 자리를 영상 화면에서 보면(기준 − 80, −5 px) 같은 바닥 cm 가 나온다
    ref = detail_floor_map({"floor_calibration": floor0["calibration"]}).to_floor(0.45, 0.8)
    moved = detail_floor_map(out).to_floor(0.45 - 80 / 1920, 0.8 - 5 / 1080)
    assert moved == pytest.approx(ref, abs=1e-3)
    assert "floor_calibration" not in S.pair_detail(base, SHIFT, [1.0, 0.0])   # 바닥 지도가 없으면 싣지 않는다
