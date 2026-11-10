"""분석 도구의 수치 조정(workbench/tune.py) — 값 다루기·전후 비교와, 적용한 판정 값 파일 읽기(src/core/config.py)."""

import pytest

from src.core.config import AppConfig, apply_judge_overrides
from workbench.tune import KNOBS, changed_values, coerce, compare, describe, draft_id, to_set

CURRENT = {"risk.dwell_pay_seconds": 1.5, "risk.inside_seconds": 1.0, "zones.foot_point": "ankle",
           "fusion.unchanged_clears_all": True, "fusion.solo_appearance_min": 0.85}


def card(pair, set_, correct, **over):
    c = {"pair": pair, "set": set_, "set_name": {"normal": "정상", "theft": "절도", "complex": "복합"}[set_],
         "headline": "정상 통과" if correct else "정상인데 경보", "outcome": "pass" if correct else "false_alarm",
         "correct": correct, "correct_last": correct, "no_verdict": False, "verdicts": ["CLEAR"] if correct else ["HIGH_RISK"],
         "people": 1, "people_ok": True, "taken": 1, "link_pct": 60, "reason": ""}
    c.update(over)
    return c


def test_every_knob_is_a_judge_stage_setting_that_exists():
    config = AppConfig()
    for key, *_ in KNOBS:
        section, name = key.split(".")
        assert section in ("risk", "fusion", "zones"), key      # 영상을 다시 돌려야 하는 절은 여기 두지 않는다
        assert hasattr(getattr(config, section), name), key


def test_coerce_clamps_numbers_and_rejects_wrong_kinds():
    assert coerce("risk.dwell_pay_seconds", "2.5") == 2.5
    assert coerce("risk.dwell_pay_seconds", 99) == 6.0 and coerce("risk.dwell_pay_seconds", -1) == 0.5
    assert coerce("zones.foot_point", "bbox") == "bbox"
    assert coerce("fusion.unchanged_clears_all", False) is False
    for key, value in (("risk.dwell_pay_seconds", "많이"), ("zones.foot_point", "knee"),
                       ("fusion.unchanged_clears_all", "yes"), ("identity.similarity_threshold", 0.5),
                       ("risk.dwell_pay_seconds", float("nan"))):
        with pytest.raises(ValueError):
            coerce(key, value)


def test_changed_values_keeps_only_real_changes_in_knob_order():
    values = {"zones.foot_point": "bbox", "risk.inside_seconds": 1.0, "risk.dwell_pay_seconds": 2}
    assert list(changed_values(values, CURRENT).items()) == [("risk.dwell_pay_seconds", 2.0), ("zones.foot_point", "bbox")]
    assert changed_values({}, CURRENT) == {}


def test_draft_id_depends_on_values_and_on_the_baseline():
    a = draft_id({"risk.dwell_pay_seconds": 2.0}, "base1")
    assert a == draft_id({"risk.dwell_pay_seconds": 2.0}, "base1")
    assert a != draft_id({"risk.dwell_pay_seconds": 2.5}, "base1")
    assert a != draft_id({"risk.dwell_pay_seconds": 2.0}, "base2")     # 기준(적용한 값·구역·기록)이 바뀌면 다른 초안


def test_to_set_matches_dual_eval_set_format():
    import yaml
    items = to_set({"risk.dwell_pay_seconds": 2.0, "zones.foot_point": "bbox", "fusion.unchanged_clears_all": False})
    assert items == ["risk.dwell_pay_seconds=2.0", 'zones.foot_point="bbox"', "fusion.unchanged_clears_all=false"]
    assert [yaml.safe_load(i.split("=", 1)[1]) for i in items] == [2.0, "bbox", False]   # dual_eval 이 이렇게 읽는다


def test_describe_uses_plain_words():
    text = describe({"risk.dwell_pay_seconds": 2.0, "zones.foot_point": "bbox", "fusion.solo_appearance_min": 0.0,
                     "fusion.unchanged_clears_all": False}, CURRENT)
    assert text == ["계산대에 서 있어야 하는 시간 1.5초 → 2초", "발 위치를 무엇으로 볼까 발목 → 상자 아래 가운데",
                    "후보가 한 명일 때 외형만으로 잇는 기준 0.85 → 0", "선반이 그대로면 옆 사람 손 뻗기도 지우기 켬 → 끔"]


def test_compare_reports_scores_and_only_changed_pairs():
    before = [card("N1", "normal", True), card("N2", "normal", True), card("T1", "theft", False), card("C1", "complex", True)]
    after = [card("N1", "normal", True),
             card("N2", "normal", False),                                  # 나빠짐
             card("T1", "theft", True, headline="절도를 잡음", outcome="hit"),   # 좋아짐
             card("C1", "complex", True, people=2, people_ok=False)]       # 판정은 같고 손님 수만 틀려짐
    r = compare(before, after)
    assert (r["before"]["correct"], r["after"]["correct"], r["n"]) == (3, 3, 4)
    assert [(x["pair"], x["effect"]) for x in r["rows"]] == [("N2", "worse"), ("C1", "worse"), ("T1", "better")]
    assert r["rows"][1]["diff"] == ["people"]
    assert r["after"]["by_set"]["theft"]["correct"] == 1 and r["after"]["people_ok"] == 3


def test_compare_ignores_pairs_missing_from_baseline():
    r = compare([card("N1", "normal", True)], [card("N1", "normal", True), card("N2", "normal", False)])
    assert r["n"] == 1 and r["rows"] == []


def test_apply_judge_overrides_sets_values_and_rejects_unknown(tmp_path):
    config = AppConfig()
    assert apply_judge_overrides(config, tmp_path / "없음.yaml") == {} and apply_judge_overrides(config, None) == {}
    path = tmp_path / "judge.yaml"
    path.write_text("# 수치 조정에서 적용\nrisk.dwell_pay_seconds: 2.0\nzones.foot_point: bbox\n", encoding="utf-8")
    assert apply_judge_overrides(config, path) == {"risk.dwell_pay_seconds": 2.0, "zones.foot_point": "bbox"}
    assert config.risk.dwell_pay_seconds == 2.0 and config.zones.foot_point == "bbox"
    for bad in ("identity.edge_sides: [left]\n", "risk.no_such_value: 1\n"):      # 다시 돌려야 하는 절·없는 칸
        path.write_text(bad, encoding="utf-8")
        with pytest.raises(ValueError):
            apply_judge_overrides(AppConfig(), path)
