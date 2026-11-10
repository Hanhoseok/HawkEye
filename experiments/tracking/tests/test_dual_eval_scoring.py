"""카메라 2대 38쌍 채점(scripts/dual_eval.py score·summarize) — 정답표를 읽고, 판정 없음을 따로 센다 (limitations A-15)."""

from scripts.dual_eval import load_truth, score, summarize

TRUTH = {
    "N1": {"expected": "CLEAR", "people": 1, "set": "normal"},
    "T4": {"expected": "HIGH_RISK", "people": 1, "set": "theft"},
    "C2": {"expected": "HIGH_RISK", "people": 2, "set": "complex"},
    "C12": {"expected": "CLEAR", "people": 2, "set": "complex"},
}


def test_truth_file_overrides_pair_name():
    # C2 는 이름은 복합 정상(C)이지만 정답표에서 미결제 이탈 — 이름 첫 글자로 채점하면 오답이 된다
    s = score("C2", [(1, "HIGH_RISK"), (2, "CLEAR")], 2, TRUTH)
    assert s["expected"] == "HIGH_RISK" and s["expected_source"] == "truth"
    assert s["correct"] and s["outcome"] == "hit"
    assert not score("C2", [(1, "CLEAR"), (2, "CLEAR")], 2, TRUTH)["correct"]


def test_real_truth_file_has_c2_as_theft():
    truth = load_truth()
    assert truth["C2"]["expected"] == "HIGH_RISK" and truth["C2"]["set"] == "complex"
    assert truth["N1"] == {"expected": "CLEAR", "people": 1, "set": "normal"}
    assert truth["T9"]["people"] == 2


def test_pair_missing_from_truth_falls_back_to_name_and_says_so():
    assert score("T99", [], 1, TRUTH)["expected_source"] == "name"
    assert score("T99", [(1, "HIGH_RISK")], 1, TRUTH)["correct"]
    assert score("N99", [(1, "CLEAR")], 1, {})["expected"] == "CLEAR"


def test_no_verdict_is_flagged_even_when_counted_correct():
    normal = score("N1", [], 1, TRUTH)
    assert normal["correct"] and normal["no_verdict"] and normal["outcome"] == "no_verdict"
    passed = score("N1", [(1, "CLEAR")], 1, TRUTH)
    assert passed["correct"] and not passed["no_verdict"] and passed["outcome"] == "pass"
    theft = score("T4", [], 1, TRUTH)
    assert not theft["correct"] and theft["no_verdict"] and theft["outcome"] == "miss"


def test_review_is_not_an_alarm():
    assert score("T4", [(1, "REVIEW")], 1, TRUTH)["outcome"] == "miss"
    assert score("N1", [(1, "REVIEW")], 1, TRUTH)["outcome"] == "pass"


def test_last_verdict_per_person_reflects_customer_who_came_back():
    # C12: 손님 2 가 문 쪽에서 HIGH_RISK 확정 뒤 돌아와 결제하고 CLEAR 로 나감 (E-15)
    s = score("C12", [(2, "HIGH_RISK"), (1, "CLEAR"), (2, "CLEAR")], 2, TRUTH)
    assert not s["correct"] and s["outcome"] == "false_alarm"   # 경보는 이미 울렸다
    assert s["correct_last"] and s["last_verdicts"] == {"2": "CLEAR", "1": "CLEAR"}
    # 다른 손님의 CLEAR 는 경보를 지우지 않는다
    assert not score("C12", [(2, "HIGH_RISK"), (1, "CLEAR")], 2, TRUTH)["correct_last"]


def test_people_count_compared_with_truth():
    assert score("C12", [], 3, TRUTH)["people_ok"] is False
    assert score("C12", [], 2, TRUTH)["people_ok"] is True
    assert score("N99", [], 1, TRUTH)["people_ok"] is None


def test_summarize_separates_no_verdict_from_decided():
    def row(pair, kind, finals, people):
        return {"pair": pair, "kind": kind, **score(pair, finals, people, TRUTH)}
    rows = [
        row("N1", "정상", [], 1),                                             # 판정 없이 맞음
        row("T4", "절도", [], 1),                                             # 판정 없어 놓침
        row("C2", "복합 정상", [(1, "HIGH_RISK")], 2),                         # 정답표로 맞음
        row("C12", "복합 정상", [(2, "HIGH_RISK"), (2, "CLEAR")], 3),          # 거짓 경보, 마지막 판정으로는 맞음
    ]
    s = summarize(rows)
    assert (s["정상"]["correct"], s["정상"]["correct_no_verdict"]) == (1, 1)
    assert s["절도"]["correct"] == 0 and s["절도"]["wrong"] == ["T4"]
    assert s["복합 정상"]["correct"] == 1
    total = s["전체"]
    assert (total["n"], total["correct"], total["correct_decided"], total["correct_no_verdict"]) == (4, 2, 1, 1)
    assert total["no_verdict"] == 2 and total["wrong"] == ["T4", "C12"] and total["correct_last"] == 3
    assert s["people"] == {"n": 4, "ok": 3, "wrong": ["C12"]}
    assert s["by_name"] == []
