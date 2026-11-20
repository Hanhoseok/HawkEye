"""전체 카메라에서 찾은 상품 (scripts/product_overview_items.py) — 영상 · 모델을 안 읽는 부분."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import product_overview_items as po  # noqa: E402
import review_export as rx  # noqa: E402

TIMES = [i * 1000 / 60 for i in range(2000)]          # 초당 60장 영상의 장면 시각


def item(name: str, x: float, y: float, score: float = 0.9) -> list:
    return [name, x, y, x + 40, y + 60, score]


SHELF = [item("cantata", 1500, 670), item("cantata", 1540, 670), item("donggwaru", 1490, 650),
         item("donggwaru", 1530, 650), item("donggwaru", 1570, 650), item("applecider", 1475, 630)]


def rows(extra: dict[int, list] | None = None, last: int = 1200) -> list[dict]:
    """4장에 1장 처리한 기록. extra = {장면 번호: 더 보인 상자들}."""
    extra = extra or {}
    return [{"frame": f, "items": SHELF + extra.get(f, [])} for f in range(0, last, 4)]


def test_row_keeps_name_box_and_score():
    d = SimpleNamespace(class_name="cantata", bbox=(10.04, 20.0, 30.06, 40.0), score=0.87654)
    assert po.row(8, [d]) == {"frame": 8, "items": [["cantata", 10.0, 20.0, 30.1, 40.0, 0.877]]}


def test_made_only_when_settings_and_frame_count_match(tmp_path):
    want = po.settings("n.pt", 1280, 0.1)
    assert not po.made(tmp_path, want, 3)
    po.write(tmp_path, [po.row(f, []) for f in (0, 4, 8)], want)
    assert po.made(tmp_path, want, 3)
    assert not po.made(tmp_path, want, 4)                                   # 장면 수가 다르다 (영상이나 처리 간격이 바뀜)
    assert not po.made(tmp_path, po.settings("s.pt", 1280, 0.1), 3)         # 다른 모델
    assert not po.made(tmp_path, po.settings("n.pt", 1280, 0.25), 3)        # 다른 점수 기준
    assert json.loads((tmp_path / "product_items.meta.json").read_text(encoding="utf-8"))["boxes"] == 0


def test_start_count_is_the_median_of_the_first_second():
    r = rows()
    r[1]["items"] = SHELF[:5]                                               # 한 장면에서 하나 놓쳐도 가운데값은 그대로
    assert po.start_count(r, TIMES) == po.STOCK
    weak = rows()
    for x in weak:
        x["items"] = SHELF[:5] + [item("applecider", 1475, 630, 0.2)]       # 점수 0.25 미만은 안 센다
    assert po.start_count(weak, TIMES)["applecider"] == 0


def test_shelf_region_wraps_the_first_second_and_away_is_outside_it():
    region = po.shelf_region(rows(), TIMES)
    assert region == (1475 - po.PAD, 630 - po.PAD, 1610 + po.PAD, 730 + po.PAD)
    hand = item("cantata", 900, 500)
    assert po.away(SHELF + [hand], region) == [hand]
    assert po.away([item("cantata", 1450, 600)], region) == []              # 여유 안 (가운데 점 기준)
    assert po.away([item("cantata", 900, 500, 0.2)], region) == []          # 점수가 낮으면 안 센다
    assert po.shelf_region([], TIMES) is None and po.away([hand], None) == []


def test_shelf_region_ignores_a_stray_box_in_the_first_second():
    """3차 N05: 처음 1초에 문 쪽에서 잘못 찾은 상자 하나가 선반 자리를 화면 전체로 키웠다 — 꺼낸 상품이 전부 '선반 안'이 됐다."""
    stray = item("cantata", 360, 200, 0.3)
    r = rows({0: [stray]})
    assert po.shelf_region(r, TIMES) == po.shelf_region(rows(), TIMES)
    assert po.away(SHELF + [stray], po.shelf_region(r, TIMES)) == [stray]


def test_score_threshold_can_be_raised_for_counting():
    region = po.shelf_region(rows(), TIMES)
    weak = item("donggwaru", 420, 300, 0.3)
    assert po.away([weak], region) == [weak] and po.away([weak], region, 0.5) == []
    r = rows({600: [weak]})
    assert po.seen_after(r, TIMES, region, 9000.0, "donggwaru") == 1
    assert po.seen_after(r, TIMES, region, 9000.0, "donggwaru", 0.5) == 0


def test_seen_after_counts_only_that_kind_inside_the_window():
    hand = item("donggwaru", 900, 500)
    r = rows({600: [hand], 604: [hand], 1196: [hand]})                      # 10초 · 10.07초 · 19.9초
    region = po.shelf_region(r, TIMES)
    assert po.seen_after(r, TIMES, region, 9000.0, "donggwaru") == 2
    assert po.seen_after(r, TIMES, region, 9000.0, "cantata") == 0          # 다른 종류
    assert po.seen_after(r, TIMES, region, 9000.0, None) == 2               # 종류 무관
    assert po.seen_after(r, TIMES, region, 0.0, "donggwaru") == 1           # 0~10초 창에는 10초 장면 하나


def test_item_rows_drop_weak_boxes_and_carry_the_frame_time(tmp_path):
    path = tmp_path / "product_items.jsonl"
    path.write_text(json.dumps({"frame": 4, "items": [["cantata", 10.4, 20.6, 30.0, 40.0, 0.9],
                                                      ["donggwaru", 1.0, 2.0, 3.0, 4.0, 0.2]]}) + "\n", encoding="utf-8")
    assert rx._item_rows(path, lambda f: f * 16.666) == [[4, [["cantata", 10, 21, 30, 40, 90]], 66.7]]
    assert rx._item_rows(tmp_path / "none.jsonl", None) == []               # 기록이 없으면 빈 목록 (일반 YOLO 기록)
