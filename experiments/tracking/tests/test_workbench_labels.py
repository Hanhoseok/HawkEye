"""분석 도구의 정답 적기 — 정답 파일 읽고 쓰기(workbench/labels.py)와 그 정답으로 채점하기(workbench/analyze.py)."""

import json

import pytest

from workbench import labels as store
from workbench.analyze import analyze, card, locate, score_labels
from workbench.data import _labeled


def track(frames: dict, fps: float = 15.0) -> dict:
    return {"fps": fps, "stride": 1, "frames_total": 300, "w": 1000, "h": 1000, "frames": frames}


def row(pid: int, box: tuple) -> list:
    return [pid, pid, pid, *box, 0, None, []]


LEFT, RIGHT = (100, 100, 300, 900), (600, 100, 800, 900)       # 손님 1 은 왼쪽, 손님 2 는 오른쪽
AT_LEFT, AT_RIGHT = [0.2, 0.5], [0.7, 0.5]


def make(labels: dict | None = None, **over) -> dict:
    """손님 둘이 10초 동안 나란히 보인다. 선반 화면 1 = 손님 1 (왼쪽), 선반 화면 2 는 짝이 틀려 손님 1 에 붙어 있다."""
    both = {str(i): [row(1, LEFT), row(2, RIGHT)] for i in range(0, 150)}
    data = {
        "pair": "C2", "offset_ms": 0.0, "truth": {"set": "complex", "people": 2, "expected": "HIGH_RISK", "notes": ""},
        "tracks": {"angle1": track(both), "angle0": track({str(i): [row(1, LEFT), row(2, RIGHT)] for i in range(0, 150)})},
        "customers": {"1": {"inside": True}, "2": {"inside": True}},
        "births": {"angle1": [], "angle0": []}, "revivals": {"angle1": [], "angle0": []}, "merges": {"angle1": [], "angle0": []},
        "links": [{"start": 0, "end": 9000, "detail": 1, "overview": 1, "score": 0.9},
                  {"start": 0, "end": 9000, "detail": 2, "overview": 1, "score": 0.6}],
        "takes0": [], "takes1": [], "visits": [], "shelf_events": [], "slot_changes": [],
        "actions": [{"pid": 1, "kind": "TAKE", "name": "집음", "count": 1, "evidence": "선반+동작", "start": 3000.0, "end": 4000.0,
                     "text": "손님 1: 집음 1개"}],
        "payments": [],
        "events": [{"pid": 1, "level": "HIGH_RISK", "ms": 8000.0, "reason": "물건 1개, 결제 없음 — 문 밖으로 나감"},
                   {"pid": 2, "level": "CLEAR", "ms": 9000.0, "reason": "집은 물건 없음 — 출구 쪽에서 사라짐"}],
        "labels": {**store.empty("C2"), **(labels or {})},
    }
    data.update(over)
    return data


RESULT = {"expected": "HIGH_RISK", "outcome": "hit", "correct": True, "correct_last": True, "no_verdict": False,
          "verdicts": ["HIGH_RISK", "CLEAR"], "people": 2, "people_expected": 2, "people_ok": True, "unowned_takes": 0}


def take(t, who, kind="take", count=1, slot=""):
    return {"time_sec": t, "kind": kind, "count": count, "slot": slot, "who": who, "note": ""}


# ---------------------------------------------------------------------- 파일

def test_clean_checks_values_and_sorts_by_time():
    raw = {"checked_by": " 규원 \n", "takes_done": 1,
           "takes": [take(8.456, [0.5, 0.25], count=2, slot="bottle"), take(3, [0, 1], kind="put")],
           "culprits": [{"time_sec": "12", "who": [0.1, 0.2]}],
           "links": [{"time_sec": 5, "detail": [0.3, 0.3], "overview": [0.6, 0.6]}]}
    out = store.clean("T2", raw)
    assert [t["time_sec"] for t in out["takes"]] == [3.0, 8.46] and out["takes"][1]["count"] == 2
    assert out["checked_by"] == "규원" and out["takes_done"] is True and out["culprits"][0]["time_sec"] == 12.0
    for bad in ({"takes": [take(1, [1.2, 0.5])]}, {"takes": [take(1, [0.5])]}, {"takes": [take(-1, [0.5, 0.5])]},
                {"takes": [take(1, [0.5, 0.5], kind="steal")]}, {"takes": [take(1, [0.5, 0.5], count=99)]},
                {"links": [{"time_sec": 1, "detail": [0.5, 0.5]}]}, {"takes": [take(1, [0.5, 0.5])] * 201}):
        with pytest.raises(ValueError):
            store.clean("T2", bad)


def test_save_roundtrip_one_item_per_line_and_empty_removes_file(tmp_path):
    saved = store.save("T2", {"takes": [take(8.4, AT_LEFT, slot="bottle")], "takes_done": True}, tmp_path)
    path = tmp_path / "T2.json"
    assert store.load("T2", tmp_path) == saved and saved["updated"]
    assert json.loads(path.read_text(encoding="utf-8"))["takes"][0]["slot"] == "bottle"
    assert sum('"time_sec"' in line for line in path.read_text(encoding="utf-8").splitlines()) == 1   # 항목 하나가 한 줄
    store.save("T2", {"takes": [], "takes_done": False}, tmp_path)
    assert not path.exists() and store.load("T2", tmp_path) == store.empty("T2")


def test_takes_done_alone_is_a_label(tmp_path):
    # '아무도 안 집었다'도 정답이다 — 항목 없이 '다 적었음'만 있어도 저장된다
    store.save("N1", {"takes_done": True}, tmp_path)
    assert store.load("N1", tmp_path)["takes_done"] is True


def test_pair_name_cannot_escape_the_folder(tmp_path):
    for name in ("../x", "a/b", "C:\\x", "", "x.json"):
        with pytest.raises(ValueError):
            store.path_of(name, tmp_path)


def test_save_payments_touches_only_that_pair(tmp_path):
    path = tmp_path / "payments.csv"
    path.write_text("pair,attempt,time_sec,items,result,checked_by,note\n"
                    "N1,1,11.20,1,paid,hoseok,\nT2,0,,0,none,hoseok,\nT4,1,9.50,2,paid,hoseok,\n", encoding="utf-8")
    assert store.load_payments("T2", path) == [{"time_sec": None, "items": 0, "result": "none", "checked_by": "hoseok", "note": ""}]
    same = store.load_payments("N1", path)
    assert store.save_payments("N1", same, path) is False                      # 안 바뀌면 파일을 건드리지 않는다
    rows = [{"time_sec": 14.0, "items": 1, "result": "failed"}, {"time_sec": 9.0, "items": 2, "result": "paid", "checked_by": "me"}]
    assert store.save_payments("T2", rows, path) is True
    text = path.read_text(encoding="utf-8")
    assert "N1,1,11.20,1,paid,hoseok," in text and "T4,1,9.50,2,paid,hoseok," in text
    assert "T2,1,9.00,2,paid,me," in text and "T2,2,14.00,1,failed,," in text   # 시각 순으로 번호를 다시 매긴다
    assert store.save_payments("T2", [], path) is True and "T2" not in path.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        store.save_payments("T2", [{"result": "none"}, {"time_sec": 1, "items": 1, "result": "paid"}], path)


# ---------------------------------------------------------------------- 채점

def test_locate_picks_the_box_under_the_point_near_that_time():
    t = track({"0": [row(1, LEFT), row(2, RIGHT)], "30": [row(3, (0, 0, 1000, 1000)), row(1, LEFT)]})
    assert locate(t, AT_LEFT, 0) == 1 and locate(t, AT_RIGHT, 0) == 2
    assert locate(t, AT_LEFT, 2000) == 1                 # 겹치면 작은 상자 (큰 가짜 상자가 아니라)
    assert locate(t, [0.45, 0.5], 0) is None             # 아무 상자에도 안 든 점
    assert locate(t, AT_RIGHT, 1000) is None             # 앞뒤 0.7초 안에 장면이 없다 (0초와 2초 장면뿐)
    assert locate(t, AT_RIGHT, 1500) == 3                # 0.5초 뒤 장면에서 그 점을 품은 상자
    assert locate(track({"30": [row(1, LEFT)]}), AT_LEFT, 2100, offset_ms=100.0) == 1   # 선반 카메라는 시차만큼 민다


def test_no_labels_means_no_scoring():
    data = make()
    assert score_labels(data, {1, 2}) is None
    s = analyze(data, RESULT)
    assert s["labeled"] is None and s["lenses"]["take"]["status"] != "ok"


def test_take_counts_compared_per_person_only_when_done():
    labels = {"takes": [take(3.5, AT_LEFT)]}
    s = analyze(make(labels), RESULT)
    assert s["labeled"]["take_ok"] is None                                    # '다 적었음' 전에는 채점하지 않는다
    assert any("적는 중" in f["text"] for f in s["lenses"]["take"]["facts"])
    s = analyze(make({**labels, "takes_done": True}), RESULT)
    assert s["labeled"]["take_ok"] is True and s["lenses"]["take"]["status"] == "ok"
    assert s["lenses"]["take"]["rows"][0]["label"] == "정답: 집음·놓음"
    # 정답은 손님 2 가 집었다 → 손님 1 은 프로그램이 1개 더, 손님 2 는 1개 덜
    s = analyze(make({"takes": [take(3.5, AT_RIGHT)], "takes_done": True}), RESULT)
    facts = [f["text"] for f in s["lenses"]["take"]["facts"]]
    assert s["labeled"]["take_ok"] is False and s["lenses"]["take"]["status"] == "bad"
    assert "가져간 개수가 정답과 다름 — 2명 중 0명 맞음" in facts
    assert "손님 1: 정답 0개 · 프로그램 1개" in facts and "손님 2: 정답 1개 · 프로그램 0개" in facts


def test_put_back_cancels_a_take():
    labels = {"takes": [take(3.5, AT_LEFT, count=2), take(6.0, AT_LEFT, kind="put")], "takes_done": True}
    assert analyze(make(labels), RESULT)["labeled"]["take_ok"] is True        # 2개 집고 1개 놓음 = 1개


def test_nobody_took_anything_is_a_valid_answer():
    s = analyze(make({"takes_done": True}, actions=[]), RESULT)
    assert s["labeled"]["take_ok"] is True
    assert s["lenses"]["take"]["facts"][0]["text"] == "아무도 가져가지 않음 — 정답과 같음"


def test_labeled_person_the_program_never_saw_counts_as_wrong():
    s = analyze(make({"takes": [take(3.5, [0.45, 0.5])], "takes_done": True}), RESULT)
    assert s["labeled"]["take_ok"] is False
    assert any("손님으로 보지 못함" in f["text"] for f in s["lenses"]["take"]["facts"])


def test_link_anchors_are_scored_against_the_programs_link():
    right = {"time_sec": 4.0, "detail": AT_LEFT, "overview": AT_LEFT, "note": ""}
    wrong = {"time_sec": 5.0, "detail": AT_RIGHT, "overview": AT_RIGHT, "note": ""}   # 선반 화면 2 는 손님 1 에 붙어 있다
    s = analyze(make({"links": [right]}), RESULT)
    assert s["labeled"]["link"] == [1, 1] and s["lenses"]["link"]["status"] == "ok"
    s = analyze(make({"links": [right, wrong]}), RESULT)
    link = s["lenses"]["link"]
    assert s["labeled"]["link"] == [1, 2] and link["status"] == "bad"
    assert any("선반 화면 2 의 짝: 정답 손님 2 · 프로그램 손님 1" in f["text"] for f in link["facts"])
    assert [m["tone"] for m in link["rows"][0]["marks"]] == ["ok", "bad"]
    late = {"time_sec": 9.6, "detail": AT_LEFT, "overview": AT_LEFT, "note": ""}       # 짝 기록이 끝난 뒤
    assert analyze(make({"links": [late]}), RESULT)["labeled"]["link"] == [0, 1]


def test_alarm_on_the_wrong_person_makes_the_verdict_bad_even_if_the_pair_counts_as_correct():
    # 경보는 손님 1 에게 났다. 쌍 단위 채점은 '경보가 났다'로 맞음이지만, 정답의 사람이 손님 2 면 틀린 것이다 (한계 A-14)
    ok = analyze(make({"culprits": [{"time_sec": 7.0, "who": AT_LEFT, "note": ""}]}), RESULT)
    assert ok["labeled"]["culprit_ok"] is True and ok["lenses"]["verdict"]["status"] == "ok"
    bad = analyze(make({"culprits": [{"time_sec": 7.0, "who": AT_RIGHT, "note": ""}]}), RESULT)
    assert bad["correct"] is True and bad["labeled"]["culprit_ok"] is False and bad["lenses"]["verdict"]["status"] == "bad"
    assert any("경보가 다른 사람에게 남 — 정답 손님 2 · 경보 손님 1" in f["text"] for f in bad["lenses"]["verdict"]["facts"])
    quiet = make({"culprits": [{"time_sec": 7.0, "who": AT_RIGHT, "note": ""}]}, events=[])
    assert analyze(quiet, {**RESULT, "outcome": "miss", "correct": False, "no_verdict": True, "verdicts": []})["labeled"]["culprit_ok"] is None


def test_gallery_totals_count_only_labeled_videos():
    cards = [card(analyze(make({"takes": [take(3.5, AT_LEFT)], "takes_done": True,
                                "links": [{"time_sec": 4.0, "detail": AT_LEFT, "overview": AT_LEFT, "note": ""}]}), RESULT)),
             card(analyze(make({"takes": [take(3.5, AT_RIGHT)], "takes_done": True}), RESULT)),
             card(analyze(make(), RESULT))]
    assert _labeled(cards) == {"n": 2, "take_n": 2, "take_ok": 1, "link_n": 1, "link_ok": 1, "culprit_n": 0, "culprit_ok": 0}
    assert cards[0]["status"]["take"] == "ok" and cards[1]["status"]["take"] == "bad"
