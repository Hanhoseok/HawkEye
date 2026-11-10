"""결제 라벨 파일(scripts/label_payments.py) 읽기·쓰기와 판정용 결제 기록 변환."""

from scripts.label_payments import PaymentLabel, load_feed, load_labels, renumber, save_labels, scenario_hint


def test_roundtrip_and_feed_keeps_only_paid(tmp_path):
    path = tmp_path / "payments.csv"
    labels = {
        "T4": [PaymentLabel("T4", 1, 9.5, 2, "paid", "me")],
        "T6": [PaymentLabel("T6", 1, 8.0, 0, "incomplete", "me")],
        "T1": [PaymentLabel("T1", 0, None, 0, "none", "me")],
        "N8-1": [PaymentLabel("N8-1", 1, 8.0, 1, "paid"), PaymentLabel("N8-1", 2, 20.0, 1, "paid")],
    }
    save_labels(labels, path)
    back = load_labels(path)
    assert back["T4"][0].time_sec == 9.5 and back["T4"][0].items == 2
    assert back["T1"][0].time_sec is None and back["T1"][0].result == "none"

    # 실제 POS 처럼 완료된 결제만 결제 기록이 된다
    assert len(load_feed("T4", path)) == 1
    assert len(load_feed("T6", path)) == 0
    assert len(load_feed("T1", path)) == 0
    assert len(load_feed("N8-1", path)) == 2
    assert len(load_feed("없는짝", path)) == 0
    feed = load_feed("N8-1", path)
    assert [e.time_sec for e in feed.due(10_000)] == [8.0]


def test_renumber_orders_attempts_by_time():
    rows = renumber([PaymentLabel("N7", 0, 12.0, 1, "paid"), PaymentLabel("N7", 0, 9.0, 0, "failed")])
    assert [(r.attempt, r.result) for r in rows] == [(1, "failed"), (2, "paid")]


def test_scenario_hint_uses_scenario_prefix():
    assert "2개만" in scenario_hint("T4")
    assert "결제 2번" in scenario_hint("N8-2")
    assert "대본 없음" in scenario_hint("C17")


def test_arrow_keys_map_to_frame_and_second_steps():
    import scripts.label_payments as L
    assert L.normalize_key(2424832) == ord("a") and L.normalize_key(2555904) == ord("d")   # ← →
    assert L.normalize_key(2490368) == L.KEY_UP and L.normalize_key(2621440) == L.KEY_DOWN  # ↑ ↓
    assert L.normalize_key(ord("[")) == ord("[") and L.normalize_key(-1) == 255


def test_save_offset_writes_manual_row_only_when_changed(tmp_path, monkeypatch):
    import scripts.label_payments as L
    from src.inputs.sync import SyncRow, SyncTable
    monkeypatch.setattr(L, "LABELS", tmp_path)
    row = SyncRow("N1", "1(angle0)/a.mp4", "1(angle1)/b.mp4", -200.0, "trajectory")
    table = SyncTable([row])
    L.save_offset(table, row, -200.0, "me")
    assert not (tmp_path / "sync.csv").exists()          # 안 바뀌면 파일을 건드리지 않는다
    L.save_offset(table, row, -133.3, "me")
    back = SyncTable.load(tmp_path / "sync.csv").rows["N1"]
    assert back.offset_ms == -133.3 and back.method == "manual" and back.checked_by == "me"
    assert "-200" in back.note
    L.save_offset(table, back, -100.0, "me")             # 두 번째 저장에도 처음 값이 남는다
    assert "-200" in SyncTable.load(tmp_path / "sync.csv").rows["N1"].note
