"""4차 촬영(2026-11-19, source/3) 을 3차와 같은 길로 돌리는 회차 표 — 경로·세트 이름·입력 검사·띠 축 맞추기·엄격 회귀 비교.

판정 규칙은 건드리지 않는다. 2·3차 경로가 그대로인지도 함께 본다.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import dual_eval as D  # noqa: E402
import regress  # noqa: E402
import shoot3_zones as Z  # noqa: E402
import shoot4_setup as S4  # noqa: E402
import sync_floor as SF  # noqa: E402
import sync_pair as SP  # noqa: E402
from src.inputs.sync import SyncRow, SyncTable  # noqa: E402


@pytest.fixture
def profile():
    """시험이 바꾼 회차·경로를 되돌린다."""
    yield
    D.set_profile(2)
    SF.set_shoot(3)


# ---------------------------------------------------------------------- 경로

def test_pair_files_by_shoot():
    v4, r4 = SP.pair_files("T01", 4)
    assert (SP.rel(v4), SP.rel(r4)) == ("3(angle0)/T01.mp4", "3(angle1)/T01.mp4")
    v3, r3 = SP.pair_files("T01", 3)
    assert (SP.rel(v3), SP.rel(r3)) == ("2(angle0)/T01.mp4", "2(angle1)/T01.mp4")
    assert SP.SHOOT3 is SP.PHONE_SHOOTS[3]
    assert SP.PHONE_SHOOTS[4]["table"].parent.name == "store01_shoot4"


def test_dual_eval_profiles(profile):
    D.set_profile(4)
    assert (D.OUT.name, D.ZONES.name, D.LABELS.name, D.SHOOT) == ("dual_eval_shoot4", "store01_shoot4", "store01_shoot4", 4)
    assert D.phone_shoot()
    D.set_profile(3)
    assert (D.OUT.name, D.ZONES.name, D.LABELS.name) == ("dual_eval_shoot3", "store01_shoot3", "store01_shoot3")
    assert D.phone_shoot()
    D.set_profile(2)
    assert (D.OUT.name, D.ZONES.name, D.LABELS.name) == ("dual_eval_pose3", "store01_dual", "store01_dual")
    assert not D.phone_shoot()


def test_register_and_sync_floor_paths(profile):
    assert Z.register_path(3) == Z.REGISTER and Z.register_path(3).parent.name == "sync_floor"
    assert Z.register_path(4).parent.name == "sync_floor4"
    SF.set_shoot(4)
    assert SF.CAMS["angle0"].name == "3(angle0)" and SF.REGISTER == Z.register_path(4)
    assert SF.LABELS.name == "store01_shoot4" and SF.rel("angle1", "N01") == "3(angle1)/N01.mp4"
    SF.set_shoot(3)
    assert SF.CAMS["angle1"].name == "2(angle1)" and SF.REGISTER == Z.register_path(3)
    assert SF.rel("angle0", "N01") == "2(angle0)/N01.mp4"


# ---------------------------------------------------------------------- 세트 이름·채점

def test_kind_and_summary_cover_long_and_short_sets():
    assert D.kind("L1") == "긴 영상" and D.kind("S1") == "짧은 영상"
    assert [D.kind(p) for p in ("N01", "T01", "C01")] == ["정상", "절도", "복합 정상"]
    truth = {"L1": {"expected": "HIGH_RISK", "people": 3, "set": "theft"},
             "S1": {"expected": "HIGH_RISK", "people": 1, "set": "theft"},
             "N01": {"expected": "CLEAR", "people": 1, "set": "normal"}}

    def row(pair, finals, people):
        return {"pair": pair, "kind": D.kind(pair), **D.score(pair, finals, people, truth)}

    s = D.summarize([row("L1", [(1, "CLEAR"), (2, "HIGH_RISK"), (3, "CLEAR")], 3), row("S1", [(1, "CLEAR")], 1),
                     row("N01", [(1, "CLEAR")], 1)])
    assert (s["긴 영상"]["correct"], s["긴 영상"]["n"]) == (1, 1)
    assert (s["짧은 영상"]["correct"], s["짧은 영상"]["n"]) == (0, 1)
    assert (s["정상"]["correct"], s["절도"]["n"], s["복합 정상"]["n"]) == (1, 0, 0)
    assert (s["전체"]["correct"], s["전체"]["n"]) == (2, 3)


# ---------------------------------------------------------------------- 입력 검사 (R3)

def _fake_shoot(tmp_path, monkeypatch, angle0, angle1, truth):
    a0, a1, labels = tmp_path / "3(angle0)", tmp_path / "3(angle1)", tmp_path / "labels"
    for folder, names in ((a0, angle0), (a1, angle1)):
        folder.mkdir()
        for n in [*names, "area_setting"]:
            (folder / f"{n}.mp4").write_bytes(b"")
    labels.mkdir()
    with (labels / "truth.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "set", "people", "expected", "notes"])
        for p in truth:
            w.writerow([p, "normal", 1, "CLEAR", ""])
    monkeypatch.setitem(SP.PHONE_SHOOTS, 4, {"video": a0, "ref": a1, "table": labels / "sync.csv"})
    D.set_profile(4)
    monkeypatch.setattr(D, "LABELS", labels)


def test_inputs_match_truth(tmp_path, monkeypatch, profile):
    _fake_shoot(tmp_path, monkeypatch, ["N01", "T01"], ["N01", "T01"], ["N01", "T01"])
    assert D.input_problems(None) == []
    D.require_inputs(None)


def test_inputs_catch_a_missing_camera_file_and_truth_mismatch(tmp_path, monkeypatch, profile):
    """한쪽 카메라 파일이 없으면 그 짝은 목록에서 조용히 빠진다 (sync_pair.all_pairs) — 4차는 그 상태로 돌리지 않는다."""
    _fake_shoot(tmp_path, monkeypatch, ["N01", "T01", "C09"], ["N01", "C09", "S1"], ["N01", "T01"])
    problems = D.input_problems(None)
    assert any(p.startswith("T01: angle0 영상만") for p in problems)
    assert any(p.startswith("S1: angle1 영상만") for p in problems)
    assert any(p.startswith("C09: 영상은 있는데 정답표에 없음") for p in problems)
    assert any(p.startswith("T01: 정답표에 있는데 영상이 없음") for p in problems)
    with pytest.raises(SystemExit):
        D.require_inputs(None)
    assert D.input_problems("N01") == []                     # 고른 짝만 볼 때
    assert D.input_problems("T*") == ["T01: angle1 영상 없음"]
    D.set_profile(3)
    D.require_inputs(None)                                   # 3차는 검사하지 않는다 (지금까지 그대로)


def test_shoot4_truth_table_matches_the_videos_on_disk(profile):
    """실제 자료가 있을 때만: 두 카메라 42편 = 정답표 42짝."""
    if not SP.PHONE_SHOOTS[4]["ref"].exists():
        pytest.skip("4차 영상 없음")
    D.set_profile(4)
    assert D.input_problems(None) == []
    truth = D.load_truth()
    assert len(truth) == 42 and truth["L1"]["people"] == 3
    assert {p for p, r in truth.items() if r["expected"] == "HIGH_RISK"} == {
        "T01", "T02", "T03", "T04", "T05", "T06", "T08", "T09", "T10", "L1", "S1"}


# ---------------------------------------------------------------------- 띠 축 맞추기 (R1)

def _fake_tracks(n):
    xs = np.linspace(0.2, 0.8, n)
    a0 = [(i * 100.0, float(x)) for i, x in enumerate(xs)]
    a1 = [(i * 100.0, float(0.5 * x + 0.1)) for i, x in enumerate(xs)]
    return a0, a1


def _calibrate_setup(tmp_path, monkeypatch, method):
    out, labels, zones = tmp_path / "out", tmp_path / "labels", tmp_path / "zones"
    for d in (out / "N01", labels, zones):
        d.mkdir(parents=True)
    (out / "N01" / ".done").write_text("ok", encoding="utf-8")
    table = SyncTable({})
    table.upsert(SyncRow("N01", "3(angle0)/N01.mp4", "3(angle1)/N01.mp4", 0.0, method, "", ""))
    table.save(labels / "sync.csv")
    D.set_profile(4)
    monkeypatch.setattr(D, "OUT", out)
    monkeypatch.setattr(D, "LABELS", labels)
    monkeypatch.setattr(D, "ZONES", zones)
    monkeypatch.setattr(D, "pairs", lambda pattern: ["N01"])
    monkeypatch.setattr(D, "PairLog", lambda pair: pair)
    monkeypatch.setattr(D, "_tracks", lambda log: _fake_tracks(300))
    return zones


def test_calibrate_accepts_audio_sync_rows(tmp_path, monkeypatch, profile):
    """4차 시차는 소리(audio) — 예전 허용 목록(trajectory·manual)이면 표본 0개로 polyfit 이 죽었다."""
    zones = _calibrate_setup(tmp_path, monkeypatch, "audio")
    D.calibrate("N*")
    import yaml
    a, b = yaml.safe_load((zones / "band_map.yaml").read_text(encoding="utf-8"))["band_map"]
    assert (a, b) == pytest.approx((0.5, 0.1), abs=1e-3)


def test_calibrate_stops_cleanly_when_only_fallback_rows(tmp_path, monkeypatch, profile):
    zones = _calibrate_setup(tmp_path, monkeypatch, "audio-fallback")
    with pytest.raises(SystemExit) as e:
        D.calibrate("N*")
    assert "표본이 모자람" in str(e.value) and not (zones / "band_map.yaml").exists()


# ---------------------------------------------------------------------- 소리 시차 표

def test_audio_rows_mark_weak_peaks_as_fallback(profile):
    SF.set_shoot(4)
    sounds = {"N01": (40.0, 1.9), "N02": (60.0, 1.5), "T01": (900.0, 1.05), "C01": (float("nan"), 0.0)}
    rows = {r.pair: r for r in SF.audio_rows(sounds, ["N01", "N02", "T01", "C01", "S1"])}
    assert (rows["N01"].method, rows["N01"].offset_ms) == ("audio", 40.0)
    assert rows["N01"].video == "3(angle0)/N01.mp4" and rows["N01"].reference == "3(angle1)/N01.mp4"
    for weak in ("T01", "C01", "S1"):        # 피크 약함 · 무음 · 소리 없음 → 쓸 만한 짝의 중앙값
        assert (rows[weak].method, rows[weak].offset_ms) == ("audio-fallback", 50.0)
    assert "audio-fallback" not in D.CALIBRATE_METHODS and "audio" in D.CALIBRATE_METHODS


def test_audio_sync_table_keeps_manual_rows(tmp_path, monkeypatch, profile):
    SF.set_shoot(4)
    monkeypatch.setattr(SF, "LABELS", tmp_path)
    monkeypatch.setattr(SF, "pairs", lambda only=None: ["N01", "N02"])
    monkeypatch.setattr(SF, "audio", lambda only=None, force=False: {"N01": (40.0, 1.9), "N02": (60.0, 1.5)})
    table = SyncTable({})
    table.upsert(SyncRow("N01", "3(angle0)/N01.mp4", "3(angle1)/N01.mp4", 7.0, "manual", "hoseok", ""))
    table.save(tmp_path / "sync.csv")
    SF.audio_sync_table(None)
    back = SyncTable.load(tmp_path / "sync.csv")
    assert (back.rows["N01"].method, back.rows["N01"].offset_ms) == ("manual", 7.0)
    assert (back.rows["N02"].method, back.rows["N02"].offset_ms) == ("audio", 60.0)


# ---------------------------------------------------------------------- 엄격 회귀 비교 (R2)·처음 만드는 기준 (R3)

def _row(pair, correct=True, verdicts=("CLEAR",), people=1, people_ok=True, no_verdict=False):
    return {"pair": pair, "correct": correct, "verdicts": list(verdicts), "people": people, "people_ok": people_ok,
            "no_verdict": no_verdict}


BASE = [_row("N1"), _row("C1", correct=False, verdicts=("HIGH_RISK",)), _row("T1", verdicts=("HIGH_RISK",)),
        _row("C2", people=3, people_ok=False)]


def test_strict_passes_only_when_every_stored_field_is_the_same():
    d = regress.compare(BASE, [dict(r) for r in BASE])
    assert d["differs"] == {} and regress.exit_code([d], False, strict=True) == 0


@pytest.mark.parametrize("change, fields", [
    ({"pair": "T1", "verdicts": ["HIGH_RISK", "CLEAR"]}, ["verdicts"]),            # 판정만 바뀜
    ({"pair": "C1", "correct": True, "verdicts": ["CLEAR"]}, ["verdicts", "correct"]),   # 좋아짐
    ({"pair": "C2", "people": 4}, ["people"]),                                      # 인원 오답 → 다른 오답
    ({"pair": "C2", "people": 2, "people_ok": True}, ["people", "people_ok"]),      # 손님 수 고쳐짐
    ({"pair": "N1", "verdicts": [], "no_verdict": True}, ["verdicts", "no_verdict"]),
])
def test_strict_fails_on_any_field_change_that_the_default_check_lets_through(change, fields):
    new = [{**r, **change} if r["pair"] == change["pair"] else dict(r) for r in BASE]
    d = regress.compare(BASE, new)
    assert d["differs"] == {change["pair"]: fields}
    assert regress.exit_code([d], False) == 0                    # 기본 check 는 통과시킨다
    assert regress.exit_code([d], False, strict=True) == 1
    assert change["pair"] in regress.describe(3, "v", d, strict=True)


def test_strict_still_reports_incomplete_for_missing_or_new_pairs():
    assert regress.exit_code([regress.compare(BASE, BASE[:3])], False, strict=True) == 2
    assert regress.exit_code([regress.compare(BASE, [*BASE, _row("N9")])], False, strict=True) == 2


def test_first_baseline_needs_every_pair_in_the_truth_table(tmp_path):
    expected = {r["pair"] for r in BASE}
    why = regress.write_baseline(4, "v", BASE[:3], root=tmp_path, expected=expected)
    assert why and "C2" in why and not regress.baseline_path(4, "v", tmp_path).exists()
    why = regress.write_baseline(4, "v", [*BASE, _row("X9")], root=tmp_path, expected=expected)
    assert why and "X9" in why
    assert regress.write_baseline(4, "v", BASE, root=tmp_path, expected=expected) is None
    assert regress.baseline_path(4, "v", tmp_path).exists()


def test_regress_modes_and_default_shoots(tmp_path):
    assert set(regress.MODES[4]) == {"v", "s", "p", "ps"} and regress.MODES[4]["s"]["lanes"] == ("L1", "L2")
    assert regress.default_shoots(tmp_path) == []
    regress.write_baseline(2, "v", BASE, root=tmp_path)
    regress.write_baseline(3, "ps", BASE, root=tmp_path)
    assert regress.default_shoots(tmp_path) == [2, 3]            # 4차 기준이 없으면 4차는 기본 검사에서 빠진다


# ---------------------------------------------------------------------- 바닥 옮기기·결제 시계

import datetime as _dt  # noqa: E402

import pos_payments as P  # noqa: E402


def _at(text: str) -> _dt.datetime:
    return _dt.datetime.strptime(f"2026-11-19 {text}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=P.KST)


def test_pay_clock_reassigns_orders_for_every_candidate():
    """이웃 영상이 3초 간격일 때: 주문을 한 영상에 미리 고정하면 앞 영상의 계산대 기록으로 세어 엉뚱한 시계 차이를 고른다."""
    spans = [P.Span("N01", _at("17:00:00"), 20.0, 20.0), P.Span("N02", _at("17:00:23"), 20.0, 20.0)]
    orders = [P.Order(k, _at("17:00:24"), 1, ["a"]) for k in (1, 2, 3)]          # 실제로는 N02 의 1초
    presence = {"N01": [(18.0, 20.0)], "N02": [(0.5, 6.0)]}
    table = S4.clock_table(orders, spans, presence, [0.0, 4.5])
    assert table == [(0.0, 3, 3), (4.5, 3, 3)]      # 0초: N02 의 1초에 계산대 / 4.5초: N01 의 19.5초로 옮겨 감 — 둘 다 셈
    assert S4.pick_offset(table) == 0.0             # 0초가 최댓값 → 그대로 (N01 로 옮기지 않는다)
    assert [r["pair"] for r in P.assign(orders, spans, 0.0)] == ["N02"] * 3


def test_update_writes_no_baseline_when_a_later_mode_is_incomplete(tmp_path):
    """4차 첫 점수는 네 방식이 모두 온전할 때만 고정 — 앞 방식만 써지고 뒤 방식이 실패한 채 남지 않는다."""
    expected = {r["pair"] for r in BASE}
    ok = {"rows": BASE, "failed": [], "skipped": []}
    broken = {"rows": [], "failed": [{"pair": "*", "error": "결제 라벨 없는 짝"}], "skipped": []}
    whys = regress.update_shoot(4, {"v": ok, "s": ok, "p": broken, "ps": ok}, root=tmp_path, expected=expected)
    assert all(whys.values()) and "결제 라벨" in regress.MODE_NAMES["p"] and not list(tmp_path.glob("*.json"))
    short = {"rows": BASE[:3], "failed": [], "skipped": []}
    assert all(regress.update_shoot(4, {"v": ok, "s": short}, root=tmp_path, expected=expected).values())
    assert not list(tmp_path.glob("*.json"))
    whys = regress.update_shoot(4, {"v": ok, "s": ok, "p": ok, "ps": ok}, root=tmp_path, expected=expected)
    assert not any(whys.values()) and len(list(tmp_path.glob("shoot4_*.json"))) == 4

def test_move_points_follows_the_floor_homography():
    shift = np.array([[1.0, 0.0, 96.0], [0.0, 1.0, -54.0], [0.0, 0.0, 1.0]])   # 4차 → 3차: 오른쪽 96px, 위 54px
    assert S4.move_points([[0.5, 0.5]], shift) == [[0.45, 0.55]]


def test_floor_doc_changes_only_the_image_points():
    base = {"camera": "angle1", "note": "3차", "notes": {"door": "…"}, "band": {"start": -10.0, "end": 90.0},
            "calibration": {"image": [[0.1, 0.1]] * 4, "floor": [[0, 0], [64, 0], [0, 74], [64, 74]]},
            "zones": [{"name": "counter", "type": "CHECKOUT", "polygon": [[0, 0], [1, 0], [1, 1]]}], "ignore_image": []}
    doc = S4.floor_doc(base, [[0.2, 0.2]] * 4, 55, 83)
    assert doc["calibration"] == {"image": [[0.2, 0.2]] * 4, "floor": base["calibration"]["floor"]}
    assert (doc["zones"], doc["band"], doc["ignore_image"]) == (base["zones"], base["band"], [])
    assert "4차" in doc["note"] and "55/83" in doc["note"]


def test_pay_clock_counts_orders_with_someone_at_the_counter():
    presence = {"N01": [(10.0, 14.0)], "N02": [(20.0, 22.0)]}
    assert S4.merge_presence([(1.0, True), (1.2, True), (3.0, True), (3.1, False), (3.2, True)]) == [(1.0, 1.2), (3.0, 3.2)]
    spans = [P.Span("N01", _at("17:00:00"), 20.0, 20.1), P.Span("N02", _at("17:01:00"), 30.0, 30.1)]
    orders = [P.Order(1, _at("17:00:15"), 1, ["a"]), P.Order(2, _at("17:01:24"), 1, ["b"])]   # 영상 시각 15초·24초
    table = S4.clock_table(orders, spans, presence, [0.0, 1.0, 2.0, 3.0])
    assert table == [(0.0, 0, 2), (1.0, 1, 2), (2.0, 2, 2), (3.0, 2, 2)]   # 결제가 2~3초 늦게 찍힌 시계
    assert S4.pick_offset(table) == 2.0                          # 최댓값 후보(2.0, 3.0)의 가운데 — 둘이면 0 에 가까운 쪽
    assert S4.pick_offset([(-3.0, 4), (-2.0, 9), (-1.0, 9), (0.0, 9), (1.0, 2)]) == 0.0
    assert S4.pick_offset([(-3.0, 9), (-2.0, 9), (-1.0, 9), (0.0, 5), (1.0, 2)]) == -2.0
    assert S4.pick_offset([(-1.0, 5), (0.0, 9), (1.0, 10)]) == 0.0   # 0초가 최댓값과 1건 이내면 그대로
