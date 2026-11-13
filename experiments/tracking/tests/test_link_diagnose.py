"""두 화면 짝 진단(scripts/link_diagnose.py) 의 '맞는 짝' 기준 — 기록 없이 만든 조각으로 시험한다."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import link_diagnose as L  # noqa: E402


def cand(o, fgap, pgap=None, n_point=0, n_all=8):
    return {"o": o, "fgap": fgap, "pgap": pgap, "n_point": n_point, "n_all": n_all}


def row(start, linked, cands, d=1, pair="C01"):
    return {"pair": pair, "d": d, "start": start, "o": linked, "cands": cands, "overs": [c["o"] for c in cands]}


def test_sure_pairs_needs_visible_ankles_and_everyone_else_far():
    rows = [
        row(0.0, 5, [cand(5, 4.0, 4.0, 8), cand(7, 80.0)]),                 # 발목 보임 4cm, 다른 후보 80cm → 확실
        row(500.0, 5, [cand(5, 4.0, 4.0, 8), cand(7, 12.0)]),               # 다른 후보도 가까움 → 모름
        row(1000.0, 5, [cand(5, 3.0), cand(7, 80.0)]),                      # 발이 가려 반직선으로만 → 모름
        row(1500.0, 5, [cand(5, 4.0, 4.0, 8), cand(7, None, n_all=0)]),     # 같은 때 보인 후보가 한 명뿐 → 두 명 장면이 아님
    ]
    assert L.sure_pairs(rows) == {("C01", 1, 0.0): 5}


def test_carried_pairs_borrow_the_nearest_sure_window_while_that_person_is_still_seen():
    rows = [
        row(0.0, 5, [cand(5, 4.0, 4.0, 8), cand(7, 80.0)]),
        row(500.0, 7, [cand(5, 10.0), cand(7, 0.0)]),                       # 발 가림: 가까운 확실한 조각의 짝(5)이 기준
        row(1000.0, 7, [cand(7, 0.0), cand(9, 20.0)]),                      # 5 가 더는 안 보임 → 기준 없음
    ]
    sure = L.sure_pairs(rows)
    carried = L.carried_pairs(rows, sure)
    assert carried == {("C01", 1, 500.0): 5}
    assert L.score_links(rows, sure) == (1, 0, 0)
    assert L.score_links(rows, carried) == (0, 1, 0)                        # 다른 사람(7)과 맺힘
    rows[1]["o"] = None
    assert L.score_links(rows, carried) == (0, 0, 1)


def test_real_switches_tell_a_swap_from_a_partner_who_left():
    rows = [
        row(0.0, 5, [cand(5, 4.0), cand(7, 80.0)]),
        row(500.0, 7, [cand(5, 30.0), cand(7, 2.0)]),                       # 5 가 아직 보이는데 7 로 → 바뀜
        row(1000.0, None, [cand(7, 90.0)]),
        row(1500.0, 9, [cand(9, 3.0)]),                                     # 7 이 사라지고 9 → 새 짝
    ]
    assert L.real_switches(rows) == (1, 1)
