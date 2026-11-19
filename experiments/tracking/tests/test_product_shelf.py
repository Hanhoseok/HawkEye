"""상품 모델을 선반 지도에 끼워 보는 도구 (scripts/product_shelf_cache.py · product_shelf_report.py) — 영상 · 모델을 안 읽는 부분."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import product_shelf_cache as pc  # noqa: E402
import product_shelf_report as pr  # noqa: E402

OLD = {"conf": 0.05, "idle_every": 3, "ignore": ["person"], "imgsz": 1280, "log_items": True, "map_score": 0.15,
       "model_path": "yolov8n.pt", "stable_seconds": 0.6, "weak_presence": 0.3}


def test_shelf_settings_swap_the_model_and_keep_the_rest():
    new = pc.shelf_settings(OLD, "handoff/n.pt", [])
    assert new["model_path"] == "handoff/n.pt" and OLD["model_path"] == "yolov8n.pt"      # 원본은 안 바뀐다
    assert {k: v for k, v in new.items() if k != "model_path"} == {k: v for k, v in OLD.items() if k != "model_path"}


def test_shelf_settings_take_extra_values_and_refuse_unknown_keys():
    new = pc.shelf_settings(OLD, "n.pt", ["map_score=0.25", "idle_every=1"])
    assert new["map_score"] == 0.25 and new["idle_every"] == 1
    with pytest.raises(SystemExit):
        pc.shelf_settings(OLD, "n.pt", ["mapscore=0.25"])


def test_detections_are_reused_only_when_the_detector_settings_match():
    a = pc.shelf_settings(OLD, "n.pt", [])
    assert pc.same_detection(a, pc.shelf_settings(OLD, "n.pt", ["map_score=0.3"]))            # 지도 기준만 바뀜 — 다시 안 찾는다
    assert not pc.same_detection(a, pc.shelf_settings(OLD, "n.pt", ["conf=0.25"]))
    assert not pc.same_detection(a, pc.shelf_settings(OLD, "s.pt", []))


def test_parse_detail_reads_removed_and_added_names():
    text = "[shelf] 사라짐: 1단 1번째 donggwaru, 1단 3번째 cantata / 생김: 1단 2번째 cantata -> 정하지 못함 (손님 1, 2 이 함께 있었음)"
    assert pr.parse_detail(text) == (["donggwaru", "cantata"], ["cantata"])
    assert pr.parse_detail("[shelf] 변화 없음 -> 손님 3") == ([], [])
    assert pr.parse_detail("[shelf] 사라짐: 2단 1번째 cell phone -> 손님 1") == (["cell phone"], [])      # 이름에 빈칸


def test_truth_counts_by_class_of_the_lane():
    rows = [{"pair": "C01", "lane": "L1", "action": "take", "qty": "1"}, {"pair": "C01", "lane": "L2", "action": "take", "qty": ""},
            {"pair": "C01", "lane": "L2", "action": "return", "qty": "1"}]
    t = pr.truth_of(rows)["C01"]
    assert t["take"] == Counter({"cantata": 1, "donggwaru": 1}) and t["ret"] == Counter({"donggwaru": 1})


def _visit(removed: list[str], added: list[str], person: int = -1, source: str = "vision") -> dict:
    parts = []
    if removed:
        parts.append("사라짐: " + ", ".join(f"1단 {i + 1}번째 {n}" for i, n in enumerate(removed)))
    if added:
        parts.append("생김: " + ", ".join(f"1단 {i + 1}번째 {n}" for i, n in enumerate(added)))
    return {"person_id": person, "removed": len(removed), "added": len(added), "source": source,
            "detail": f"[shelf] {' / '.join(parts) or '변화 없음'} -> 손님"}


def test_compare_checks_count_class_and_owner():
    truth = {"take": Counter({"cantata": 1, "donggwaru": 1}), "ret": Counter()}
    good = pr.compare(truth, pr.seen_of([_visit(["donggwaru", "cantata"], [], person=3), _visit([], [])]))
    assert good["count_ok"] and good["class_ok"] and (good["changed"], good["owned"]) == (1, 1)
    wrong_class = pr.compare(truth, pr.seen_of([_visit(["donggwaru", "donggwaru"], [])]))
    assert wrong_class["count_ok"] and not wrong_class["class_ok"] and wrong_class["owned"] == 0
    generic = pr.compare(truth, pr.seen_of([_visit(["bottle", "bottle"], [])]))
    assert generic["count_ok"] and not generic["class_ok"]                    # 우리 종류가 아닌 이름은 종류 맞음이 아니다
    short = pr.compare(truth, pr.seen_of([_visit(["cantata"], [])]))
    assert not short["count_ok"] and short["seen_net"] == 1


def test_take_then_return_nets_out_and_sensor_visits_are_ignored():
    truth = {"take": Counter({"donggwaru": 2}), "ret": Counter({"donggwaru": 2})}
    seen = pr.seen_of([_visit(["donggwaru"], []), _visit([], ["donggwaru"]), _visit(["cantata"], [], source="sensor")])
    r = pr.compare(truth, seen)
    assert r["count_ok"] and r["class_ok"] and r["seen_removed"] == 1 and r["seen_added"] == 1
