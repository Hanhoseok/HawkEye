"""회귀 검사 스크립트의 비교·기준 쓰기 규칙 (scripts/regress.py) — 판정은 돌리지 않는다."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import regress  # noqa: E402


def row(pair, correct=True, verdicts=("CLEAR",), people=1, people_ok=True, no_verdict=False):
    return {"pair": pair, "correct": correct, "verdicts": list(verdicts), "people": people, "people_ok": people_ok,
            "no_verdict": no_verdict}


BASE = [row("N1"), row("C1", correct=False, verdicts=("HIGH_RISK",)), row("T1", verdicts=("HIGH_RISK",)),
        row("C2", people=3, people_ok=False)]


def test_same_result_passes():
    d = regress.compare(BASE, [dict(r) for r in BASE])
    assert not any(d[k] for k in ("better", "worse", "people_worse", "changed", "missing", "extra"))
    assert d["score"] == (3, 3) and regress.exit_code([d], False) == 0


def test_better_and_changed_pass_worse_fails():
    new = [row("N1"), row("C1"), row("T1", verdicts=("HIGH_RISK", "CLEAR")), row("C2", people=2, people_ok=True)]
    d = regress.compare(BASE, new)
    assert d["better"] == ["C1"] and d["changed"] == ["T1"] and d["people_better"] == ["C2"]
    assert regress.exit_code([d], False) == 0
    worse = regress.compare(BASE, [row("N1"), row("C1", correct=False, verdicts=("HIGH_RISK",)),
                                   row("T1", correct=False, verdicts=("CLEAR",)), row("C2", people=3, people_ok=False)])
    assert worse["worse"] == ["T1"] and regress.exit_code([worse], False) == 1
    people = regress.compare(BASE, [row("N1", people=2, people_ok=False), *BASE[1:]])
    assert people["people_worse"] == ["N1"] and regress.exit_code([people], False) == 1


def test_missing_or_failed_pairs_make_the_check_incomplete():
    """빠진 짝이 있으면 점수가 같아 보여도 통과가 아니다 (판정이 실패한 짝은 결과에서 빠진다)."""
    d = regress.compare(BASE, BASE[:3])
    assert d["missing"] == ["C2"] and regress.exit_code([d], False) == 2
    assert regress.exit_code([regress.compare(BASE, BASE)], True) == 2
    assert regress.exit_code([regress.compare(BASE, [*BASE, row("N9")])], False) == 2


def test_update_refuses_incomplete_results(tmp_path):
    assert regress.write_baseline(3, "v", BASE, root=tmp_path) is None
    path = regress.baseline_path(3, "v", tmp_path)
    saved = path.read_text(encoding="utf-8")
    assert [r["pair"] for r in json.loads(saved)] == ["N1", "C1", "T1", "C2"]
    # 영상이 없어 대상 목록에서 빠진 짝·빈 결과·실패·건너뜀 — 기존 기준을 그대로 둔다
    assert "없음" in regress.write_baseline(3, "v", BASE[:3], root=tmp_path)
    assert "비었음" in regress.write_baseline(3, "v", [], root=tmp_path)
    assert regress.write_baseline(3, "v", BASE, failed=[{"pair": "C2", "error": "x"}], root=tmp_path)
    assert regress.write_baseline(3, "v", BASE, skipped=["C2"], root=tmp_path)
    assert path.read_text(encoding="utf-8") == saved
    assert regress.write_baseline(3, "v", BASE[:3], allow_remove=True, root=tmp_path) is None
    assert len(json.loads(path.read_text(encoding="utf-8"))) == 3


def test_settings_that_would_be_silently_ignored_are_refused():
    assert regress.check_sets(["risk.exit_hold_seconds=0", "fusion.release_delay_seconds=1"], False, False) is None
    assert "--replay" in regress.check_sets(["identity.late_merge_threshold=0.9"], False, False)
    assert regress.check_sets(["identity.late_merge_threshold=0.9"], True, False) is None
    assert "직접" in regress.check_sets(["tracker.name=bytetrack"], True, False)
    assert "--identity-check" in regress.check_sets(["risk.swap_threshold=0.5"], False, False)
    assert regress.check_sets(["risk.swap_threshold=0.5"], False, True) is None
