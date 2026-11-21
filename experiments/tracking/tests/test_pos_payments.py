"""4차 결제 기록(payment-log.csv) → 영상별 결제 라벨 (scripts/pos_payments.py)."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pos_payments as P  # noqa: E402
from label_payments import PaymentLabel, label_state, load_feed, load_labels, save_labels  # noqa: E402

LOG = """주문번호,결제시각,계산대ID,품목순번,바코드,상품명,단가,수량,품목금액,주문총액,주문수량합
5,2026-11-05 15:45:49,VS7TpMSX,1,="8801056018900",칸타타,2400,1,2400,2400,1
9,2026-11-19 17:21:30,nmjeB6Qa,1,="8801056018900",칸타타,2400,1,2400,2400,1
12,2026-11-19 17:24:56,nmjeB6Qa,1,="8801056018900",칸타타,2400,1,2400,6000,3
12,2026-11-19 17:24:56,nmjeB6Qa,2,="4710587185675",동과루,1800,1,1800,6000,3
12,2026-11-19 17:24:56,nmjeB6Qa,3,="4710128023022",애플사이다,1800,1,1800,6000,3
13,2026-11-19 17:26:14,nmjeB6Qa,1,="4710587185675",동과루,1800,3,5400,5400,3
"""


def at(text: str) -> dt.datetime:
    return dt.datetime.strptime(f"2026-11-19 {text}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=P.KST)


def span(pair: str, start: str, last: float, length: float | None = None) -> P.Span:
    return P.Span(pair, at(start), last, length if length is not None else last + 0.05)


def test_read_orders_groups_lines_and_sums_quantities(tmp_path):
    path = tmp_path / "payment-log.csv"
    path.write_text(LOG, encoding="utf-8-sig")
    orders = P.read_orders(path)
    assert [(o.number, o.items) for o in orders] == [(5, 1), (9, 1), (12, 3), (13, 3)]
    assert orders[2].lines == ["칸타타×1", "동과루×1", "애플사이다×1"]          # 품목 3줄 = 주문 하나
    assert orders[3].lines == ["동과루×3"] and orders[1].paid_at == at("17:21:30")   # 한 줄에 수량 3


def test_last_processed_frame_time():
    times = [i * 16.0 for i in range(10)]            # 장면 10장 → stride 4 로 0·4·8 처리
    assert P.last_processed_sec(times) == 0.128
    assert P.last_processed_sec(times[:9]) == 0.128 and P.last_processed_sec(times[:8]) == 0.064
    assert P.last_processed_sec([]) == 0.0


def test_assign_keeps_only_orders_the_judge_will_consume(tmp_path):
    path = tmp_path / "payment-log.csv"
    path.write_text(LOG, encoding="utf-8-sig")
    spans = [span("N01", "17:21:18", 19.9), span("N04", "17:24:30", 26.0, 26.1), span("N05", "17:25:40", 30.0)]
    rows = {r["order"]: r for r in P.assign(P.read_orders(path), spans)}
    assert (rows[9]["pair"], rows[9]["time_sec"], rows[9]["status"]) == ("N01", "12.00", "assigned")
    assert (rows[12]["pair"], rows[12]["time_sec"], rows[12]["status"]) == ("N04", "26.00", "near-end")   # 경계: 마지막 처리 장면과 같은 시각
    assert rows[13]["status"] == "after-end" and rows[13]["pair"] == ""      # 34초 > 마지막 처리 장면 30초 — 붙이지 않는다
    assert rows[5]["status"] == "outside"                                      # 다른 날 주문
    # 시계 차이를 주면 그만큼 당겨진다 (결제 서버가 4초 빠름 → 영상 시각 −4초)
    rows = {r["order"]: r for r in P.assign(P.read_orders(path), spans, clock_offset=4.0)}
    assert (rows[13]["pair"], rows[13]["time_sec"]) == ("N05", "30.00")
    assert (rows[9]["time_sec"], rows[12]["time_sec"]) == ("8.00", "22.00")


def test_order_between_videos_is_not_attached():
    spans = [span("N01", "17:21:18", 19.9), span("N02", "17:23:00", 20.0)]
    rows = P.assign([P.Order(1, at("17:22:20"), 1, ["칸타타×1"])], spans)       # N01 끝 뒤 42초, N02 시작 전
    assert (rows[0]["status"], rows[0]["pair"]) == ("between", "")


def test_labels_feed_and_boundary_consumption(tmp_path):
    """배정한 결제는 판정이 실제로 소비한다 — 마지막 처리 시각에 due() 로 나온다. none 줄은 '라벨 있음'으로 읽힌다."""
    spans = [span("N04", "17:24:30", 26.0), span("T01", "17:30:00", 10.0)]
    orders = [P.Order(12, at("17:24:56"), 3, ["칸타타×1", "동과루×1", "애플사이다×1"])]
    labels = P.to_labels(P.assign(orders, spans), [s.pair for s in spans])
    assert [(r.attempt, r.time_sec, r.items, r.result, r.checked_by) for r in labels["N04"]] == [(1, 26.0, 3, "paid", "pos-log")]
    assert [(r.attempt, r.time_sec, r.result) for r in labels["T01"]] == [(0, None, "none")]
    path = tmp_path / "payments.csv"
    save_labels(labels, path)
    back = load_labels(path)
    assert label_state(back, "N04") == "verified" and label_state(back, "T01") == "verified"   # 초안 경고·'라벨 없음'에 안 걸림
    feed = load_feed("N04", path)
    assert [e.items for e in feed.due(spans[0].last_sec * 1000.0)] == [3]      # 마지막 처리 시각에 소비됨
    assert len(load_feed("T01", path)) == 0
    late = P.to_labels(P.assign([P.Order(13, at("17:24:57"), 1, ["동과루×1"])], spans), ["N04", "T01"])
    assert late["N04"][0].result == "none"                                     # 마지막 처리 장면 뒤 1초 — 결제로 적지 않는다


def test_merge_keeps_pairs_a_person_checked():
    old = {"N04": [PaymentLabel("N04", 1, 25.0, 3, "paid", "hoseok", "눈으로 확인")],
           "T01": [PaymentLabel("T01", 0, None, 0, "none", "pos-log", "")],
           "N99": [PaymentLabel("N99", 1, 3.0, 1, "paid", "pos-log", "")]}
    new = {"N04": [PaymentLabel("N04", 1, 26.0, 3, "paid", "pos-log", "")],
           "T01": [PaymentLabel("T01", 1, 5.0, 1, "paid", "pos-log", "")]}
    merged, kept = P.merge(old, new)
    assert kept == ["N04"] and merged["N04"][0].time_sec == 25.0             # 사람이 확인한 짝은 그대로
    assert merged["T01"][0].result == "paid"                                  # pos-log 줄은 새 값으로
    assert "N99" in merged                                                    # 이번 목록에 없는 짝의 줄도 지우지 않는다
