"""3차 결제 초안 (scripts/beep_payments.py) — 삑 소리 찾기·계산 묶기·시나리오 규칙, 3차 라벨 경로."""
import numpy as np
import pytest

import scripts.beep_payments as B
import scripts.label_payments as L
from scripts.label_payments import PaymentLabel


@pytest.fixture(autouse=True)
def _shoot2_after():
    yield
    L.set_shoot(2)


def _signal(beeps, seconds=10.0, sr=16000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.01, int(seconds * sr))
    for start, dur in beeps:
        i, n = int(start * sr), int(dur * sr)
        x[i:i + n] += 0.3 * np.sin(2 * np.pi * 3400 * np.arange(n) / sr)
    return x


def test_find_beeps_scan_and_short_click():
    found = B.find_beeps(_signal([(2.0, 0.3), (6.0, 0.07)]))
    assert [b.kind for b in found] == ["scan", "short"]
    assert abs(found[0].start - 2.0) < 0.05 and abs(found[0].dur - 0.3) < 0.06


def test_find_beeps_ignores_other_tones_and_silence():
    sr = 16000
    t = np.arange(sr * 3) / sr
    assert B.find_beeps(0.3 * np.sin(2 * np.pi * 600 * t)) == []   # 말소리 대역
    assert B.find_beeps(np.zeros(100)) == []


def test_group_scans_by_gap():
    assert B.group_scans([21.6, 24.8, 27.8]) == [[21.6, 24.8, 27.8]]
    assert B.group_scans([11.4, 23.8]) == [[11.4], [23.8]]


def test_draft_paid_groups_and_notes():
    rows = B.draft("N04", [(21.62, "scan"), (21.86, "short"), (24.78, "scan"), (27.79, "scan")])
    assert [(r.result, r.items, r.time_sec) for r in rows] == [("paid", 3, 27.79)]
    assert rows[0].checked_by == L.DRAFT_BY and "21.86" in rows[0].note
    two = B.draft("N08-1", [(11.37, "scan"), (23.82, "scan")])
    assert [(r.attempt, r.items) for r in two] == [(1, 1), (2, 1)] and "2번" in two[0].note


def test_draft_scenario_overrides():
    assert [(r.result, r.time_sec) for r in B.draft("T06", [])] == [("incomplete", None)]
    assert B.draft("T06", [(9.0, "scan")])[0].time_sec == 9.0
    assert B.draft("N06", [])[0].result == "cancelled"
    assert [(r.result, r.attempt) for r in B.draft("T01", [])] == [("none", 0)]
    assert "대본 스캔 2 / 삑 1" in B.draft("T04", [(18.3, "scan")])[0].note


def test_merge_keeps_verified_unless_forced():
    labels = {"N01": [PaymentLabel("N01", 1, 12.8, 1, "paid", "hoseok")]}
    draft = B.draft("N01", [(13.0, "scan")])
    assert not B.merge(labels, "N01", draft, force=False) and labels["N01"][0].checked_by == "hoseok"
    assert B.merge(labels, "N01", draft, force=True) and labels["N01"][0].checked_by == L.DRAFT_BY
    assert B.merge(labels, "N02", B.draft("N02", []), force=False)


def test_label_state_missing_draft_verified():
    labels = {"T01": [PaymentLabel("T01", 0, None, 0, "none", "hoseok")],
              "T02": [PaymentLabel("T02", 0, None, 0, "none", L.DRAFT_BY)]}
    assert L.label_state(labels, "T01") == "verified"     # 확인된 none 은 '없음'과 다르다
    assert L.label_state(labels, "T02") == "draft"
    assert L.label_state(labels, "T03") == "missing"


def test_shoot3_paths_leave_shoot2_file_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "ROOT", tmp_path)
    L.set_shoot(2)
    two = tmp_path / "labels" / "store01_dual" / "payments.csv"
    L.save_labels({"N1": [PaymentLabel("N1", 1, 11.2, 1, "paid", "me")]})
    before = two.read_bytes()
    L.set_shoot(3)
    L.save_labels({"N01": [PaymentLabel("N01", 1, 12.8, 1, "paid", L.DRAFT_BY)]})
    assert two.read_bytes() == before
    assert list(L.load_labels()) == ["N01"]
    assert L.load_feed("N01")._events[0].items == 1


def test_scenario_hint_zero_padded_names():
    assert "1개 결제" in L.scenario_hint("N01")
    assert "결제 2번" in L.scenario_hint("N08-2")
    assert L.scenario_id("T10") == "T10"


def test_jump_marks():
    marks = L.marks([PaymentLabel("x", 1, 12.8, 1, "paid")], [(10.0, "scan"), (12.8, "scan")])
    assert marks == [10.0, 12.8]
    assert L.jump_target(10.0, marks, True) == 12.8 and L.jump_target(10.0, marks, False) is None


def test_step_mark_walks_marks_inside_vfr_gap():
    from src.inputs.frame_times import index_at
    frames = [0.0, 33.3, 266.6]                       # 233ms 빈 구간 (장면 빠짐)
    find = lambda sec: index_at(frames, sec * 1000.0)
    marks = [0.1, 0.2, 0.25]
    idx, at, seen = 0, None, []
    for _ in range(3):
        idx, at = L.step_mark(frames[idx] / 1000.0, at, idx, find, marks, True)
        seen.append(at)
    assert seen == [0.1, 0.2, 0.25]
    assert L.step_mark(frames[idx] / 1000.0, at, idx, find, marks, False)[1] == 0.2


def test_merge_keeps_mixed_pair_with_human_row():
    labels = {"N01": [PaymentLabel("N01", 1, 12.8, 1, "paid", L.DRAFT_BY),
                      PaymentLabel("N01", 2, 20.0, 1, "paid", "hoseok")]}
    assert not B.merge(labels, "N01", B.draft("N01", [(13.0, "scan")]), force=False)
    assert len(labels["N01"]) == 2
    assert B.merge(labels, "N01", B.draft("N01", [(12.8, "scan")]), force=False) is False
    only_draft = {"N02": [PaymentLabel("N02", 1, 14.4, 1, "paid", L.DRAFT_BY)]}
    assert B.merge(only_draft, "N02", B.draft("N02", [(14.0, "scan")]), force=False)
