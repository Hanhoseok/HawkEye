"""분석 도구의 촬영 회차 나누기 (2차·3차) — 회차별 자리, 견줘 보기(상품 모델), 실제 장면 시각, 바닥 위 거리.

영상·기록 없이 만든 조각으로 시험한다.
"""

import dataclasses
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from workbench import intake  # noqa: E402
from workbench.analyze import _duration, lens_link, locate, presence  # noqa: E402
from workbench import data as wb_data  # noqa: E402
from workbench.data import MAX_VIEWS, PROFILES, SHOOT3_PREV, Builder, check_views, main_views, stale_outputs, views_of  # noqa: E402
from workbench.tune import changed_values, knobs_for  # noqa: E402


def track(frames: dict, fps: float = 15.0, stride: int = 1, total: int = 300, **more) -> dict:
    return {"fps": fps, "stride": stride, "frames_total": total, "w": 1920, "h": 1080, "frames": frames, **more}


def row(pid: int, floor=None) -> list:
    return [pid, pid, pid, 100, 100, 200, 400, 0, None, [], floor]


# ---------------------------------------------------------------------- 회차별 자리

def test_each_shoot_keeps_its_own_records_labels_and_output():
    two, three = PROFILES[2], PROFILES[3]
    for name in ("records", "labels", "zones", "out", "judge_file"):
        assert getattr(two, name) != getattr(three, name), name
    # 짝 이름이 겹친다(C12 는 양쪽에 있다) — 만든 것도, 사람이 적은 정답도 한 폴더에 섞이면 안 된다
    assert Builder(2).events_dir() != Builder(3).events_dir()
    assert Builder(3).out == three.out and Builder(3).records_dir() == three.records
    assert two.can_add and not three.can_add      # 3차는 영상마다 카메라 맞추기가 필요해 도구에서 추가하지 못한다


def test_set_profile_switches_the_applied_values_file():
    import dual_eval as de
    try:
        de.set_profile(3)
        assert de.JUDGE_FILE == PROFILES[3].judge_file and de.OUT == PROFILES[3].records and de.LABELS == PROFILES[3].labels
        de.set_profile(2)
        assert de.JUDGE_FILE == PROFILES[2].judge_file and de.OUT == PROFILES[2].records
    finally:
        de.set_profile(2)


def test_knobs_differ_by_how_each_shoot_links_the_two_views():
    two, three = {k[0] for k in knobs_for(2)}, {k[0] for k in knobs_for(3)}
    assert "fusion.max_band_gap" in two and "fusion.max_band_gap" not in three          # 2차: 선반 앞 띠 한 줄
    assert {"fusion.max_floor_gap_cm", "fusion.ray_slack_cm"} <= three                  # 3차: 바닥 위 거리
    assert not {"fusion.max_floor_gap_cm", "fusion.ray_slack_cm"} & two
    assert two - three == {"fusion.max_band_gap"}                                        # 나머지는 같다
    with pytest.raises(ValueError):
        changed_values({"fusion.ray_slack_cm": 10}, {"fusion.min_score": 0.5})           # 이 촬영에서 안 쓰는 값


def test_pair_zone_files_are_left_alone_when_nothing_changed(tmp_path):
    import shoot3_eval
    path = tmp_path / "floor.yaml"
    shoot3_eval._write_if_changed(path, "a: 1\n")
    stamp = path.stat().st_mtime_ns
    shoot3_eval._write_if_changed(path, "a: 1\n")
    assert path.stat().st_mtime_ns == stamp and not list(tmp_path.glob("*.part"))
    shoot3_eval._write_if_changed(path, "a: 2\n")
    assert path.read_text(encoding="utf-8") == "a: 2\n" and not list(tmp_path.glob("*.part"))


# ---------------------------------------------------------------------- 견줘 보기 (같은 회차의 다른 기록)

def test_shoot3_reads_the_product_model_records_like_the_judge_script():
    import dual_eval as de
    assert PROFILES[3].records == de.ROOT / "outputs" / de.RECORDS[3] and de.RECORDS[3] == "dual_eval_shoot3_prod"
    assert PROFILES[2].records == de.ROOT / "outputs" / de.RECORDS[2]
    assert PROFILES[3].overrides == () and PROFILES[3].can_tune           # 3차 기본 판정 값은 판정 코드(dual_config)에 있다


OTHER = 39
"""시험용 보기 번호 — 같은 회차의 '다른 기록'을 보는 견줘 보기. 지금은 그런 보기가 없어 (바로 전 판정은 같은 기록) 시험에서만 만든다."""


@pytest.fixture
def other_records(monkeypatch):
    """3차의 다른 기록 폴더를 보는 견줘 보기 하나를 잠깐 더한다 (바로 전 판정 자리에 — 규칙은 회차마다 2개)."""
    monkeypatch.delitem(PROFILES, SHOOT3_PREV)
    monkeypatch.setitem(PROFILES, OTHER, dataclasses.replace(
        PROFILES[3], name="다른 기록", records=PROFILES[3].records.with_name("dual_eval_shoot3"), out=PROFILES[3].out.with_name("shoot3_other"),
        can_tune=False, thumbs=PROFILES[3].out / "thumbs", overrides=("fusion.overview_count=false",), note="시험"))
    check_views()
    return PROFILES[OTHER]


def test_previous_view_judges_the_same_records_without_the_instant_count():
    three, prev = PROFILES[3], PROFILES[SHOOT3_PREV]
    assert SHOOT3_PREV == 31 and prev.shoot == 3 and Builder(SHOOT3_PREV).shoot == 3 and Builder(SHOOT3_PREV).key == SHOOT3_PREV
    assert prev.records == three.records and prev.out != three.out          # 같은 기록, 만든 것은 따로
    assert prev.overrides == ("fusion.overview_instant=false",)
    for name in ("labels", "zones", "judge_file", "signature_files", "payment"):
        assert getattr(prev, name) == getattr(three, name), name            # 정답표 · 구역 · 적용한 값은 3차 것
    assert three.can_tune and not prev.can_tune and not prev.can_add        # 적용한 값을 함께 쓰므로 여기서는 못 바꾼다
    assert prev.note and not three.note and prev.thumbs == three.out / "thumbs"
    assert Builder(SHOOT3_PREV).signature("C01") != Builder(3).signature("C01")       # 얹는 값이 다르면 다시 만든다
    assert main_views() == [2, 3]


def test_each_shoot_keeps_at_most_two_views(monkeypatch):
    """사용자 규칙 (2026-11-20): 분석 도구의 보기는 회차마다 최신 2개 — 지금 기준과 바로 전 판정."""
    assert MAX_VIEWS == 2 and set(PROFILES) == {2, 3, SHOOT3_PREV}
    assert views_of(3) == [3, SHOOT3_PREV] and views_of(2) == [2]
    check_views()
    monkeypatch.setitem(PROFILES, 32, dataclasses.replace(PROFILES[SHOOT3_PREV], out=PROFILES[3].out.with_name("shoot3_third")))
    with pytest.raises(RuntimeError, match="최신 2개"):
        check_views()                                                       # 세 번째를 더하면 멈춘다 — 가장 옛 것을 지워야 한다
    monkeypatch.setitem(PROFILES, 32, dataclasses.replace(PROFILES[2], shoot=2, out=PROFILES[2].out))
    with pytest.raises(RuntimeError, match="같은 폴더"):
        check_views()


def test_stale_outputs_are_the_folders_no_view_uses(tmp_path, monkeypatch):
    monkeypatch.setattr(wb_data, "OUT", tmp_path)
    for name in ("shoot3", "shoot3_prev", "shoot3_generic", "shoot3_owner", "pairs", "thumbs"):
        (tmp_path / name).mkdir()
    (tmp_path / "shoot_note.txt").write_text("x", encoding="utf-8")         # 폴더만 본다
    monkeypatch.setitem(PROFILES, 3, dataclasses.replace(PROFILES[3], out=tmp_path / "shoot3"))
    monkeypatch.setitem(PROFILES, SHOOT3_PREV, dataclasses.replace(PROFILES[SHOOT3_PREV], out=tmp_path / "shoot3_prev"))
    assert [d.name for d in stale_outputs()] == ["shoot3_generic", "shoot3_owner"]   # 2차의 pairs · thumbs 는 안 건드린다


def test_engine_points_the_judge_at_the_views_records(other_records):
    import dual_eval as de
    assert other_records.records.name == "dual_eval_shoot3" and other_records.records != PROFILES[3].records
    try:
        with Builder(OTHER).engine():
            assert de.SHOOT == 3 and de.OUT == other_records.records and de.LABELS == PROFILES[3].labels
        with Builder(3).engine():
            assert de.SHOOT == 3 and de.OUT == PROFILES[3].records          # 같은 회차라도 기록 폴더가 다르면 다시 맞춘다
        with Builder(OTHER).engine():
            assert de.OUT == other_records.records
    finally:
        de.set_profile(2)


def test_thumbnail_is_copied_from_the_base_view(tmp_path):
    builder = Builder(SHOOT3_PREV)
    ready = tmp_path / "base"
    ready.mkdir()
    (ready / "C01.jpg").write_bytes(b"jpg")
    builder.profile = dataclasses.replace(builder.profile, records=tmp_path / "records", thumbs=ready)
    builder.out = tmp_path / "out"
    assert builder._thumb("C01", {}) == "thumbs/C01.jpg"                    # 영상을 읽지 않는다 (data 가 비어도 된다)
    assert (tmp_path / "out" / "thumbs" / "C01.jpg").read_bytes() == b"jpg"
    assert builder._thumb("C02", {}) is None                                # 가져올 것이 없으면 예전 길 (여기서는 영상이 없어 None)


def test_server_never_opens_on_a_compare_view_and_has_no_tuner_for_it(monkeypatch):
    from workbench import server
    assert SHOOT3_PREV in server.BUILDERS and SHOOT3_PREV not in server.TUNERS
    assert set(server.TUNERS) == {2, 3}
    for key, b in server.BUILDERS.items():
        monkeypatch.setattr(b, "pairs", lambda key=key: ["C01"])            # 모든 보기에 기록이 있다고 치고
    assert server.shoot_of(None, None) == 3                                 # 처음 여는 화면은 가장 최근 회차
    assert server.shoot_of(str(SHOOT3_PREV)) == SHOOT3_PREV           # 골라서 여는 것은 된다
    with pytest.raises(ValueError):
        server.shoot_of("30")


# ---------------------------------------------------------------------- 실제 장면 시각

def test_real_frame_times_replace_index_over_fps():
    """3차 전체 카메라는 장면을 빠뜨린다 — 30번 장면이 2초가 아니라 4.5초일 수 있다."""
    t = track({"0": [row(1)], "30": [row(1)], "31": [row(1)]}, ms={"0": 0.0, "30": 4500.0, "31": 4566.7}, end_ms=20500)
    segs = presence(t)
    assert [s[0] for s in segs[1]] == [0, 4500]
    assert locate(t, [0.08, 0.2], 4500.0) == 1          # 그 시각에 그 점의 사람
    assert locate(t, [0.08, 0.2], 2000.0) is None       # 번호 ÷ fps 로는 여기(2초)였을 자리 — 실제로는 아무도 없다
    plain = track({"0": [row(1)], "30": [row(1)]})
    assert locate(plain, [0.08, 0.2], 2000.0) == 1      # 시각 표가 없으면(2차) 예전대로


def test_duration_uses_the_real_end_time():
    data = {"offset_ms": 100.0, "tracks": {"angle1": track({}, total=300, end_ms=25000), "angle0": track({}, total=300)}}
    assert _duration(data) == 25000
    data["tracks"]["angle1"].pop("end_ms")
    assert _duration(data) == 20100                      # 300장 ÷ 15 + 시차


# ---------------------------------------------------------------------- 바닥 위 거리

def test_link_gaps_measure_floor_distance_with_the_judges_formula():
    import review_export as rx
    tracks = {
        "angle1": track({0: [row(7, [100.0, 0.0])], 8: [row(7, [100.0, 0.0])]}),
        # 선반 카메라: 발목이 보이는 장면(점)과 가려서 방향만 아는 장면(반직선)
        "angle0": track({0: [row(1, [103.0, 4.0])], 8: [row(1, [100.0, -50.0, 0.0, 1.0])]}),
    }
    links = [{"start": 0, "end": 300, "detail": 1, "overview": 7}, {"start": 400, "end": 700, "detail": 1, "overview": 7},
             {"start": 5000, "end": 5500, "detail": 1, "overview": 7}]
    rx._link_gaps(links, tracks, 0.0, 250.0)
    assert links[0]["gap"] == 5.0 and links[0]["hidden"] == 0.0     # 점과 점
    assert links[1]["gap"] == 0.0 and links[1]["hidden"] == 1.0     # 반직선이 그 사람을 지난다
    assert "gap" not in links[2]                                    # 같은 때 표본이 없으면 적지 않는다


def test_floor_row_keeps_direction_only_for_hidden_feet():
    import review_export as rx
    assert rx._floor_row(((12.34, 5.67), None)) == [12.3, 5.7]
    assert rx._floor_row(((12.34, 5.67), (0.6, 0.8))) == [12.3, 5.7, 0.6, 0.8]
    assert rx._floor_row((None, None)) is None
    assert rx._snap(-0.9, 0.3) == [-1, 0] and rx._snap(0.2, 0.7) == [0, 1]


def link_data(links: list[dict]) -> dict:
    return {"offset_ms": 0.0, "links": links, "floor": {"max_gap": 60.0, "slack": 25.0},
            "tracks": {"angle0": track({str(i): [row(1)] for i in range(0, 30)})}}


def test_link_lens_reports_floor_distance_and_flags_appearance_only_links():
    near = [{"start": 0, "end": 500, "detail": 1, "overview": 1, "score": 0.9, "pos": 0.9, "gap": 6.0, "hidden": 0.0},
            {"start": 500, "end": 1000, "detail": 1, "overview": 1, "score": 0.9, "pos": 1.0, "gap": 18.0, "hidden": 1.0}]
    lens = lens_link(link_data(near), {1})
    text = " ".join(f["text"] for f in lens["facts"])
    assert "발목이 보인 조각 1개는 중앙 6cm" in text and "방향으로만 잰 조각 1개는 중앙 18cm" in text
    assert lens["status"] == "info"
    assert "바닥 위 6cm" in lens["rows"][0]["segs"][1]["tip"] and "방향으로만" in lens["rows"][0]["segs"][2]["tip"]

    far = near + [{"start": 1000, "end": 1500, "detail": 1, "overview": 1, "score": 0.5, "pos": 0.0, "gap": 140.0, "hidden": 0.0}]
    lens = lens_link(link_data(far), {1})
    warn = [f for f in lens["facts"] if f["level"] == "warn"]
    assert len(warn) == 1 and "140cm" in warn[0]["text"] and warn[0]["ms"] == 1000 and lens["status"] == "warn"


def test_link_lens_says_nothing_about_floor_for_the_band_shoot():
    data = link_data([{"start": 0, "end": 500, "detail": 1, "overview": 1, "score": 0.9}])
    data.pop("floor")
    lens = lens_link(data, {1})
    assert not any("바닥" in f["text"] for f in lens["facts"])
    assert "바닥" not in lens["rows"][0]["segs"][1]["tip"]


# ---------------------------------------------------------------------- 시차 저장

def test_set_offset_writes_that_shoots_table_and_keeps_the_previous_manual_value(tmp_path):
    builder = Builder(3)
    builder.profile = dataclasses.replace(PROFILES[3], labels=tmp_path)
    (tmp_path / "sync.csv").write_text(
        "pair,video,reference,offset_ms,method,checked_by,note\n"
        "C01,2(angle0)/C01.mp4,2(angle1)/C01.mp4,14.9,manual,teammate,\n"
        "C02,2(angle0)/C02.mp4,2(angle1)/C02.mp4,-255.0,manual,teammate,\n", encoding="utf-8")
    intake.set_offset(builder, "C01", 48.0, "me")
    rows = {r.split(",")[0]: r for r in (tmp_path / "sync.csv").read_text(encoding="utf-8").splitlines()[1:]}
    assert ",48.0,manual,me," in rows["C01"] and "+15ms" in rows["C01"] and "teammate" in rows["C01"]   # 덮어쓴 값이 메모에 남는다
    assert rows["C01"].startswith("C01,2(angle0)/C01.mp4,2(angle1)/C01.mp4,")
    assert rows["C02"] == "C02,2(angle0)/C02.mp4,2(angle1)/C02.mp4,-255.0,manual,teammate,"                # 다른 줄은 그대로
    intake.set_offset(builder, "C01", 50.0, "me")
    again = (tmp_path / "sync.csv").read_text(encoding="utf-8")
    assert ",50.0,manual,me," in again
    assert again.count("분석 도구에서 다시 맞춤") == 1 and "+15ms" in again                                # 처음 값이 남고, 메모는 쌓이지 않는다
