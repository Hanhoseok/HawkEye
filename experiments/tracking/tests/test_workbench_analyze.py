"""분석 도구의 질문별 요약(workbench/analyze.py) — 기록만으로 계산하는 부분."""

from workbench.analyze import analyze, card, presence, short_reason
from workbench.data import natural


def track(frames: dict, fps: float = 15.0, stride: int = 1, total: int = 300) -> dict:
    return {"fps": fps, "stride": stride, "frames_total": total, "w": 1920, "h": 1080, "frames": frames}


def row(pid: int, rid: int | None = None) -> list:
    return [pid, pid if rid is None else rid, pid, 100, 100, 200, 400, 0, None, []]


def make(**over) -> dict:
    """손님 1 이 선반 화면 1 로 이어지고, 손을 댔지만 선반이 그대로라 '만지기만'이 된 절도 영상."""
    data = {
        "pair": "T2", "offset_ms": 0.0, "truth": {"set": "theft", "people": 1, "expected": "HIGH_RISK", "notes": "절도"},
        "tracks": {
            "angle1": track({str(i): [row(1)] + ([row(2)] if 30 <= i < 36 else []) for i in range(0, 150)}),
            "angle0": track({str(i): [row(1)] for i in range(30, 120)}),
        },
        "customers": {"1": {"inside": True}, "2": {"inside": False}},
        "births": {"angle1": [{"pid": 1, "ms": 0.0, "edge": "T"}, {"pid": 2, "ms": 2000.0, "edge": ""}], "angle0": []},
        "revivals": {"angle1": [], "angle0": []},
        "merges": {"angle1": [], "angle0": []},
        "links": [{"start": 2000, "end": 2500, "detail": 1, "overview": 1, "score": 0.9},
                  {"start": 2500, "end": 3000, "detail": 1, "overview": 1, "score": 0.8}],
        "takes0": [{"pid": 1, "zone": "bottle", "start": 4000, "end": 4400}],
        "takes1": [{"pid": 1, "zone": "shelf", "start": 3000, "end": 6000}],
        "visits": [{"pid": 1, "start": 3000, "end": 6000, "removed": 0, "added": 0, "detail": "", "slot_change": None}],
        "shelf_events": [{"kind": "take", "start_ms": 4000.2, "end_ms": 4400.0, "owner": 1, "how": "짝", "detail": ""},
                         {"kind": "visit", "start_ms": 3000.0, "end_ms": 6000.0, "owner": 1, "how": "가린 사람의 짝", "detail": ""}],
        "slot_changes": [],
        "actions": [{"pid": 1, "kind": "TOUCH", "name": "만지기만", "count": 0, "evidence": "선반+동작",
                     "start": 3000.0, "end": 6000.0, "text": "손님 1: 만지기만 (선반+동작)"}],
        "payments": [],
        "events": [{"pid": 1, "level": "CLEAR", "ms": 9000.0, "reason": "가져간 물건 없음 [만지기만 1번] — 출구 쪽에서 사라짐"}],
    }
    data.update(over)
    return data


def result(**over) -> dict:
    r = {"expected": "HIGH_RISK", "outcome": "miss", "correct": False, "correct_last": False, "no_verdict": False,
         "verdicts": ["CLEAR"], "people": 1, "people_expected": 1, "people_ok": True, "unowned_takes": 0}
    r.update(over)
    return r


def test_presence_splits_on_gaps_and_applies_offset():
    t = track({"0": [row(1)], "1": [row(1)], "30": [row(1)], "31": [row(-1)]})
    segs = presence(t, offset_ms=100.0)
    assert list(segs) == [1]                      # 신원이 정해지지 않은 관측(-1)은 뺀다
    assert len(segs[1]) == 2                      # 2초 가까이 안 보였으니 두 구간
    assert segs[1][0][0] == 100 and segs[1][1][0] == 2100


def test_presence_uses_merged_identity():
    assert list(presence(track({"0": [row(5, rid=1)]}))) == [1]


def test_missed_theft_points_at_the_take_lens():
    s = analyze(make(), result())
    assert s["headline"] == "절도인데 놓침" and s["reason"] == "가져간 물건 없음 [만지기만 1번]"
    assert s["lenses"]["take"]["status"] == "bad" and s["taken"] == 0
    assert s["lenses"]["take"]["facts"][0]["level"] == "bad"
    touch = next(f for f in s["lenses"]["take"]["facts"] if "만지기만" in f["text"] and f["level"] == "warn")
    assert "기록에 남지 않아" in touch["text"]      # 기준에 못 미친 칸 변화율은 볼 수 없다고 말한다
    assert s["lenses"]["verdict"]["status"] == "bad" and s["lenses"]["track"]["status"] == "ok"


def test_non_customers_are_separated_from_customers():
    s = analyze(make(), result())
    labels = [r["label"] for r in s["lenses"]["track"]["rows"]]
    assert labels[:2] == ["손님 1", "손님 아님 #2"] and "선반 화면 1" in labels
    assert s["customers"] == [1]
    assert any("손님이 아닌 것으로 걸러짐" in f["text"] for f in s["lenses"]["track"]["facts"])


def test_wrong_people_count_is_bad():
    s = analyze(make(), result(people=2, people_ok=False))
    assert s["lenses"]["track"]["status"] == "bad"
    assert "프로그램 2명 · 정답 1명" in s["lenses"]["track"]["facts"][0]["text"]


def test_link_coverage_and_partner_change():
    s = analyze(make(), result())
    link = s["lenses"]["link"]
    assert link["pct"] == 17 and s["link_pct"] == 17          # 6초 보인 중 1초에 짝
    assert link["status"] == "info"                            # 정답이 없어 맞음·틀림을 말하지 않는다
    swapped = make(links=[{"start": 2000, "end": 2500, "detail": 1, "overview": 1, "score": 0.9},
                          {"start": 2500, "end": 3000, "detail": 1, "overview": 2, "score": 0.7}])
    link = analyze(swapped, result())["lenses"]["link"]
    assert link["status"] == "warn"
    assert any("짝이 바뀜" in f["text"] for f in link["facts"])
    assert any("손님이 아닌 신원(#2)" in f["text"] for f in link["facts"])


def test_person_never_linked_is_flagged():
    link = analyze(make(links=[]), result())["lenses"]["link"]
    assert link["pct"] == 0 and any("한 번도 짝을 맺지 못함" in f["text"] for f in link["facts"])


def test_hand_contact_owner_comes_from_shelf_events():
    take = analyze(make(), result())["lenses"]["take"]
    contact = next(r for r in take["rows"] if r["label"] == "손 접촉")["segs"][0]
    assert contact["who"] == 1 and "주인: 손님 1 (짝)" in contact["tip"]
    orphan = analyze(make(shelf_events=[]), result())["lenses"]["take"]
    seg = next(r for r in orphan["rows"] if r["label"] == "손 접촉")["segs"][0]
    assert seg["who"] is None and seg["tone"] == "warn"


def test_no_verdict_normal_pair_is_a_warning_not_a_pass():
    data = make(events=[], truth={"set": "normal", "people": 1, "expected": "CLEAR", "notes": ""})
    s = analyze(data, result(expected="CLEAR", outcome="no_verdict", correct=True, correct_last=True, no_verdict=True, verdicts=[]))
    assert s["headline"] == "판정 없음" and s["reason"] == "영상 끝까지 나감이 확정되지 않음"
    assert s["lenses"]["verdict"]["status"] == "warn"
    assert any("통과까지 확인된 것은 아님" in f["text"] for f in s["lenses"]["verdict"]["facts"])


def test_alarm_replaced_by_later_verdict_is_explained():
    data = make(events=[{"pid": 1, "level": "HIGH_RISK", "ms": 5000.0, "reason": "집기 행동 2번, 결제 없음 — 문 밖으로 나감"},
                        {"pid": 1, "level": "CLEAR", "ms": 9000.0, "reason": "모두 결제함 — 출구 쪽에서 사라짐"}],
                payments=[{"ms": 7000, "items": 2, "pid": 1, "method": "dwell"}],
                truth={"set": "complex", "people": 1, "expected": "CLEAR", "notes": ""})
    r = result(expected="CLEAR", outcome="false_alarm", correct=False, correct_last=True, verdicts=["HIGH_RISK", "CLEAR"])
    s = analyze(data, r)
    assert short_reason(data, r) == "집기 행동 2번, 결제 없음"          # 카드에는 경보가 난 이유를 쓴다
    facts = [f["text"] for f in s["lenses"]["verdict"]["facts"]]
    assert any("다시 판정받아 '통과'" in t for t in facts)
    assert any("손님별 마지막 판정만 보면 맞음" in t for t in facts)
    assert any("결제 1건" in t and "1.5초" in t for t in facts)


def test_card_keeps_only_what_the_gallery_needs():
    c = card(analyze(make(), result()))
    assert c["pair"] == "T2" and c["set_name"] == "절도" and "lenses" not in c
    assert c["status"] == {"track": "ok", "link": "info", "take": "bad", "verdict": "bad"}


def test_natural_order_groups_sets_and_sorts_numbers():
    names = ["C10", "N10", "T2", "N2", "C2", "N8-2", "N8-1", "T10"]
    assert sorted(names, key=natural) == ["N2", "N8-1", "N8-2", "N10", "T2", "T10", "C2", "C10"]


# ---------------------------------------------------------------- 센서 로그 · 결제 로그 표시 (후속 63)

SENSOR_CSV = """time_sec,wall_time,lane,product,event,before,after,delta,distance_mm,in_video,raw_sec,angle1_sec
7.17,17:25:57,L2,동과루,REMOVE,3,1,-2,333,1,7.17,7.15
10.17,17:26:00,L3,애플사이다,RESYNC,2,2,0,264,1,10.17,10.15
12.00,17:26:02,L3,애플사이다,REMOVE,2,1,-1,264,1,12.00,11.98
20.00,17:26:10,L2,동과루,ADD,1,2,1,300,1,20.00,19.98
95.00,17:27:25,L1,칸타타,ADD,0,2,2,300,0,95.00,94.98
"""


def test_sensor_log_keeps_unused_records_with_the_reason(tmp_path):
    from workbench.analyze import sensor_log

    path = tmp_path / "N05.csv"
    path.write_text(SENSOR_CSV, encoding="utf-8")
    log = sensor_log(path, -20.0, ("L1", "L2"))
    assert [(e["lane"], e["event"], e["count"], e["skip"]) for e in log] == [
        ("L2", "REMOVE", 2, ""), ("L3", "REMOVE", 1, "판정에 안 쓰는 레인"), ("L2", "ADD", 1, ""), ("L1", "ADD", 2, "영상 밖의 기록")]
    assert log[0]["ms"] == 7150.0                       # 선반 카메라 시각 + 시차 — 판정이 읽는 시각과 같다
    assert sensor_log(tmp_path / "없음.csv", 0.0, ("L1",)) == []


def test_take_lens_draws_the_sensor_log_row():
    log = [{"ms": 7150.0, "lane": "L2", "product": "동과루", "event": "REMOVE", "count": 2, "before": "3", "after": "1", "skip": ""},
           {"ms": 11980.0, "lane": "L3", "product": "애플사이다", "event": "REMOVE", "count": 1, "before": "2", "after": "1",
            "skip": "판정에 안 쓰는 레인"}]
    take = analyze(make(sensor_log=log), result())["lenses"]["take"]
    row_ = next(r for r in take["rows"] if r["label"] == "센서 로그")
    assert [m["tone"] for m in row_["marks"]] == ["take", "quiet"]
    assert row_["marks"][0]["tip"] == "L2 동과루 2개 빠짐 (3 → 1)" and "판정에 안 씀" in row_["marks"][1]["tip"]
    assert any(f["text"] == "센서 로그: 빠짐 2개 · 놓임 0개를 판정에 넣음 (안 쓴 기록 1건)" for f in take["facts"])
    assert any(line["text"].startswith("센서: L2 동과루") for line in take["log"])


def test_no_sensor_row_without_a_sensor_log():
    take = analyze(make(), result())["lenses"]["take"]
    assert all(r["label"] != "센서 로그" for r in take["rows"])


def test_payment_log_view_does_not_say_payment_was_guessed():
    paid = [{"ms": 5000, "items": 1, "pid": 1, "method": "checkout"}]
    said = lambda dwell: " ".join(f["text"] for f in analyze(make(payments=paid), result(), dwell)["lenses"]["verdict"]["facts"])  # noqa: E731
    assert "실제 결제가 아님" in said(1.5)
    assert "결제 기록 사용" in said(None) and "실제 결제가 아님" not in said(None)
