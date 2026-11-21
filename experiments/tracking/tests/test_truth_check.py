"""정답표 영상 확인표 (scripts/truth_check.py)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import truth_check as T  # noqa: E402
from label_payments import PaymentLabel  # noqa: E402

PLAN = "정답·인원은 시나리오 표(사용자, 2026-11-20) 기준 — 영상 확인 전"
TRUTH = [
    {"pair": "N01", "set": "normal", "people": "1", "expected": "CLEAR", "notes": f"시나리오: 1개 구매. {PLAN}"},
    {"pair": "T05", "set": "theft", "people": "1", "expected": "HIGH_RISK", "notes": f"시나리오: 같은 상품 3개 중 1개만 결제. {PLAN}"},
    {"pair": "C21", "set": "complex", "people": "2", "expected": "CLEAR",
     "notes": f"시나리오: 2명 팔 겹침: 한 명은 만지기만, 한 명은 가져감. {PLAN}"},
]
PAYMENTS = {
    "N01": [PaymentLabel("N01", 1, 12.05, 1, "paid", "pos-log", "주문 9 (17:21:30) 칸타타×1")],
    "T05": [PaymentLabel("T05", 0, None, 0, "none", "pos-log", "결제 기록에 주문 없음")],
    "C21": [PaymentLabel("C21", 2, 28.97, 1, "paid", "pos-log", "주문 32 (17:44:52) 칸타타×1"),
            PaymentLabel("C21", 1, 10.97, 2, "paid", "pos-log", "주문 31 (17:44:34) 동과루×1 + 애플사이다×1")],
}


def filled(rows, **by_pair):
    return [{**r, **by_pair.get(r["pair"], {})} for r in rows]


def test_sheet_has_scenario_and_payments_but_no_verdict_columns():
    rows = T.sheet_rows(TRUTH, PAYMENTS, [])
    assert [r["pair"] for r in rows] == ["N01", "T05", "C21"]
    assert rows[0]["scenario"] == "1개 구매" and rows[0]["pos_paid"] == "12초 1개 (칸타타×1)"
    assert rows[1]["pos_paid"] == "결제 없음"
    assert rows[2]["scenario"] == "2명 팔 겹침: 한 명은 만지기만, 한 명은 가져감"
    assert rows[2]["pos_paid"] == "11초 2개 (동과루×1 + 애플사이다×1) / 29초 1개 (칸타타×1)"     # 시각 순
    assert T.sheet_rows(TRUTH, {}, [])[0]["pos_paid"] == ""                                       # 결제 라벨이 없는 짝
    assert all(r[k] == "" for r in rows for k in T.FILLED)
    assert not any("verdict" in k or "judge" in k for k in T.SHEET_FIELDS)                         # 판정 결과는 싣지 않는다


def test_apply_updates_only_checked_rows():
    sheet = filled(T.sheet_rows(TRUTH, PAYMENTS, []),
                   N01={"seen_people": "2", "seen_expected": "clear", "checked_by": "hoseok", "note": "촬영자가 같이 찍힘"},
                   T05={"seen_people": "1", "seen_expected": "HIGH_RISK", "checked_by": "hoseok"})
    new, changes, problems = T.apply(TRUTH, sheet, "2026-11-21")
    assert problems == []
    assert changes == ["N01: 인원 1→2  (촬영자가 같이 찍힘)"]
    by = {r["pair"]: r for r in new}
    assert (by["N01"]["people"], by["N01"]["expected"]) == ("2", "CLEAR")                         # 소문자 입력도 받는다
    assert by["N01"]["notes"] == "시나리오: 1개 구매. 영상 확인 (hoseok, 2026-11-21) — 시나리오 표와 다름: 인원 1→2 · 촬영자가 같이 찍힘"
    assert by["T05"]["notes"] == "시나리오: 같은 상품 3개 중 1개만 결제. 영상 확인 (hoseok, 2026-11-21)"
    assert by["C21"] == TRUTH[2]                                                                   # 안 본 영상은 그대로
    assert by["N01"]["set"] == "normal"                                                            # 시나리오 분류는 안 바꾼다


def test_apply_again_after_write_keeps_the_scenario_difference():
    """반영한 정답표로 확인표를 다시 만들고 다시 반영해도 결과가 같다 — plan_* 는 처음 값 그대로."""
    sheet = filled(T.sheet_rows(TRUTH, PAYMENTS, []),
                   T05={"seen_people": "1", "seen_expected": "CLEAR", "checked_by": "hoseok", "note": "3개 다 결제함"})
    new, changes, _ = T.apply(TRUTH, sheet, "2026-11-21")
    again_sheet = T.sheet_rows(new, PAYMENTS, sheet)
    t05 = next(r for r in again_sheet if r["pair"] == "T05")
    assert (t05["plan_expected"], t05["seen_expected"], t05["scenario"]) == ("HIGH_RISK", "CLEAR", "같은 상품 3개 중 1개만 결제")
    again, again_changes, _ = T.apply(new, again_sheet, "2026-11-21")
    assert again == new and again_changes == changes == ["T05: 정답 HIGH_RISK→CLEAR  (3개 다 결제함)"]


def test_half_filled_or_invalid_rows_are_reported_and_not_applied():
    sheet = filled(T.sheet_rows(TRUTH, PAYMENTS, []),
                   N01={"seen_people": "1"},                                                       # 일부만 채움
                   T05={"seen_people": "0", "seen_expected": "REVIEW", "checked_by": "h"},         # 틀린 값
                   C21={"note": "나중에 다시"})                                                    # 메모만
    sheet.append({"pair": "Z99", "seen_people": "1", "seen_expected": "CLEAR", "checked_by": "h"})
    new, changes, problems = T.apply(TRUTH, sheet, "2026-11-21")
    assert new == TRUTH and changes == []
    assert len(problems) == 4 and problems[0] == "Z99: 정답표에 없는 짝"
    assert "seen_expected 비어 있음" in problems[1] and "checked_by 비어 있음" in problems[1]
    assert "REVIEW" in problems[2] and "1 이상의 숫자" in problems[2]
    assert problems[3].startswith("C21: seen_people 비어 있음")


def test_cli_round_trip_and_excel_encoding(tmp_path, capsys):
    labels = tmp_path / "labels"
    T.write_csv(labels / "truth.csv", T.TRUTH_FIELDS, TRUTH, "utf-8")
    before = (labels / "truth.csv").read_bytes()
    assert T.main(["sheet", "--labels", str(labels)]) == 0
    sheet_path = labels / "truth_check.csv"
    assert sheet_path.read_bytes().startswith(b"\xef\xbb\xbf")                                     # 엑셀용 BOM
    assert T.main(["status", "--labels", str(labels)]) == 0 and "확인 0/3" in capsys.readouterr().out

    rows = filled(T.read_csv(sheet_path), N01={"seen_people": "2", "seen_expected": "CLEAR", "checked_by": "호석"})
    T.write_csv(sheet_path, T.SHEET_FIELDS, rows, "cp949")                                         # 엑셀이 'CSV(쉼표로 분리)'로 다시 저장한 경우
    assert T.main(["apply", "--labels", str(labels)]) == 0
    assert (labels / "truth.csv").read_bytes() == before                                           # --write 없이는 쓰지 않는다
    assert "정답·인원이 바뀌는 짝 1편: N01" in capsys.readouterr().out
    assert T.main(["apply", "--labels", str(labels), "--write"]) == 0
    back = {r["pair"]: r for r in T.read_csv(labels / "truth.csv")}
    assert back["N01"]["people"] == "2" and "영상 확인 (호석" in back["N01"]["notes"] and back["T05"] == TRUTH[1]
    assert T.main(["sheet", "--labels", str(labels)]) == 0                                         # 다시 만들어도 채운 칸 유지
    assert next(r for r in T.read_csv(sheet_path) if r["pair"] == "N01")["checked_by"] == "호석"


def test_write_refuses_when_a_row_has_problems(tmp_path):
    labels = tmp_path / "labels"
    T.write_csv(labels / "truth.csv", T.TRUTH_FIELDS, TRUTH, "utf-8")
    rows = filled(T.sheet_rows(TRUTH, {}, []), N01={"seen_people": "2", "seen_expected": "CLEAR", "checked_by": "h"},
                  T05={"seen_expected": "CLEAR"})
    T.write_csv(labels / "truth_check.csv", T.SHEET_FIELDS, rows, "utf-8-sig")
    before = (labels / "truth.csv").read_bytes()
    try:
        T.main(["apply", "--labels", str(labels), "--write"])
    except SystemExit as e:
        assert "아무것도 쓰지 않았다" in str(e)
    else:
        raise AssertionError("문제가 있는데 썼다")
    assert (labels / "truth.csv").read_bytes() == before


def test_write_keeps_the_truth_files_line_endings(tmp_path):
    """한 줄만 확인해 반영하면 그 줄만 달라진다 — 줄바꿈(CRLF)과 나머지 줄은 바이트 그대로."""
    labels = tmp_path / "labels"
    T.write_csv(labels / "truth.csv", T.TRUTH_FIELDS, TRUTH, "utf-8", "\r\n")
    before = (labels / "truth.csv").read_bytes().split(b"\r\n")
    rows = filled(T.sheet_rows(TRUTH, {}, []), N01={"seen_people": "1", "seen_expected": "CLEAR", "checked_by": "h"})
    T.write_csv(labels / "truth_check.csv", T.SHEET_FIELDS, rows, "utf-8-sig")
    assert T.main(["apply", "--labels", str(labels), "--write"]) == 0
    after = (labels / "truth.csv").read_bytes().split(b"\r\n")
    assert len(after) == len(before) and [i for i, (a, b) in enumerate(zip(before, after)) if a != b] == [1]


def test_duplicate_pair_rows_block_both_regeneration_and_write(tmp_path):
    """같은 짝이 두 줄이면(엑셀에서 줄 복사) 어느 줄이 맞는지 모른다 — 순서와 상관없이 다시 만들지도 반영하지도 않는다."""
    good = {"seen_people": "1", "seen_expected": "CLEAR", "checked_by": "h", "note": "확인함"}
    base = T.sheet_rows(TRUTH, PAYMENTS, [])
    n01 = base[0]
    cases = {
        "채운 줄 뒤에 빈 줄": [{**n01, **good}, dict(n01)],
        "틀린 줄 뒤에 맞는 줄": [{**n01, "seen_people": "0", "seen_expected": "INVALID", "checked_by": "h"}, {**n01, **good}],
        "맞는 줄 뒤에 틀린 줄": [{**n01, **good}, {**n01, "seen_people": "0", "seen_expected": "INVALID", "checked_by": "h"}],
    }
    for name, pair_rows in cases.items():
        sheet = pair_rows + base[1:]
        new, changes, problems = T.apply(TRUTH, sheet, "2026-11-21")
        assert new == TRUTH and changes == [], name
        assert problems[0] == "N01: 확인표에 2줄 — 한 줄만 남긴다", name

        labels = tmp_path / name
        T.write_csv(labels / "truth.csv", T.TRUTH_FIELDS, TRUTH, "utf-8")
        T.write_csv(labels / "truth_check.csv", T.SHEET_FIELDS, sheet, "utf-8-sig")
        before = [(labels / f).read_bytes() for f in ("truth.csv", "truth_check.csv")]
        for argv in (["sheet"], ["apply", "--write"]):
            try:
                T.main([*argv, "--labels", str(labels)])
            except SystemExit as e:
                assert "N01" in str(e) or "아무것도 쓰지 않았다" in str(e), name
            else:
                raise AssertionError(f"{name}: {argv} 가 막히지 않았다")
        assert [(labels / f).read_bytes() for f in ("truth.csv", "truth_check.csv")] == before, name


def test_rows_without_a_pair_name():
    base = T.sheet_rows(TRUTH, PAYMENTS, [])
    blank = {k: "" for k in T.SHEET_FIELDS}
    assert T.table_problems(base + [blank, blank], {"N01", "T05", "C21"}) == []           # 엑셀이 붙인 빈 줄은 문제가 아니다
    assert T.sheet_rows(TRUTH, PAYMENTS, base + [blank]) == base
    lost = {**blank, "seen_people": "1", "seen_expected": "CLEAR", "checked_by": "h"}       # 짝 이름이 지워진 줄
    assert T.table_problems(base + [lost], {"N01", "T05", "C21"}) == ["짝 이름이 빈 줄 1개에 입력이 있음"]
    new, _, problems = T.apply(TRUTH, base + [lost], "2026-11-21")
    assert new == TRUTH and problems
