"""센서 + 카메라를 상품 종류별로 합치기 (fusion.sensor_kinds, 후속 61 · D-24)."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import FusionConfig  # noqa: E402
from src.fusion.cross_view import CrossViewLinker  # noqa: E402
from src.fusion.shelf_events import SensorEvent, ShelfEventFuser, TimeAxis, visit_kinds  # noqa: E402
from src.risk.customers import ShelfVisit  # noqa: E402
from src.shelf.watcher import ShelfWatcher  # noqa: E402

LANES = {"L1": "shelf", "L2": "shelf", "L3": "shelf"}
KINDS = {"L1": "cantata", "L2": "donggwaru"}


def fuser(sensors, **cfg):
    config = FusionConfig(sensor_kinds=dict(KINDS), **cfg)
    return ShelfEventFuser(config, CrossViewLinker(config), TimeAxis(30.0), TimeAxis(30.0), dict(LANES), sensors)


def count(ms, lane, n):
    return SensorEvent(ms, lane, float(n), 1.0, unit="count")


def visit(removed=(), added=(), pid=1):
    return ShelfVisit(pid, "shelf", 0, 0, len(removed), len(added), "shelf", "[shelf] …", removed_kinds=tuple(removed),
                      added_kinds=tuple(added))


def test_visit_kinds_reads_fields_and_falls_back_to_the_detail_text():
    assert visit_kinds(visit(["cantata", "cantata"], ["donggwaru"])) == (Counter(cantata=2), Counter(donggwaru=1))
    old = ShelfVisit(1, "shelf", 0, 0, 2, 1, "shelf",
                     "[shelf] 사라짐: 1단 2번째 donggwaru, 1단 4번째 cantata / 생김: 2단 1번째 applecider -> 손님 1")
    assert visit_kinds(old) == (Counter(donggwaru=1, cantata=1), Counter(applecider=1))
    assert visit_kinds(ShelfVisit(1, "shelf", 0, 0, 0, 0, "shelf", "[shelf] 변화 없음 -> 손님 1")) == (Counter(), Counter())


def test_sensor_counts_its_lanes_and_camera_counts_the_rest():
    # 칸타타(L1 센서) 1개 + 애플사이다(센서 없음) 1개를 함께 꺼냄 — 예전에는 센서 개수 1로 바뀌어 애플사이다가 빠졌다
    f = fuser([count(3000.0, "L1", -1)])
    out = f.detail_visit(visit(["cantata", "applecider"]), 2000.0, 5000.0)
    assert (out.removed, out.added, out.source) == (2, 0, "sensor+vision")
    assert "센서 L1 -1개" in out.detail and "카메라 applecider -1" in out.detail


def test_same_item_seen_by_both_is_counted_once():
    f = fuser([count(3000.0, "L1", -1), count(3500.0, "L2", -2)])
    out = f.detail_visit(visit(["cantata", "donggwaru", "donggwaru"]), 2000.0, 5000.0)
    assert out.removed == 3


def test_camera_fills_what_the_sensor_missed_only_under_the_max_rule():
    # 3차 T05: 센서가 동과루 3개를 하나도 못 적었다
    out = fuser([]).detail_visit(visit(["donggwaru"] * 3), 2000.0, 5000.0)
    assert out.removed == 3 and "카메라 개수로 셈" in out.detail
    out = fuser([], sensor_kind_rule="sensor").detail_visit(visit(["donggwaru"] * 3), 2000.0, 5000.0)
    assert out.removed == 0 and "안 셈" in out.detail
    # 센서 2 · 카메라 3 → 3 (남는 1개만 카메라 개수)
    out = fuser([count(3000.0, "L2", -2)]).detail_visit(visit(["donggwaru"] * 3), 2000.0, 5000.0)
    assert out.removed == 3


def test_late_sensor_record_joins_the_visit_that_saw_that_kind():
    # 3차 N11: 방문 1(3.1~7.3초)에서 칸타타를 꺼냈는데 센서는 9.07초에 적었다 — 방문 2(8.3~11.9초, 동과루) 시간 안이다
    events = [count(9070.0, "L1", -1), count(11070.0, "L2", -1)]
    f = fuser(events, sensor_lag_seconds=3.0)
    first = f.detail_visit(visit(["cantata"]), 3100.0, 7300.0)
    second = f.detail_visit(visit(["donggwaru"]), 8300.0, 11900.0)
    assert (first.removed, second.removed) == (1, 1)
    assert first.source == "sensor+vision"
    # 늦음을 안 봐주면 칸타타가 두 번 세어진다 (방문 1 은 카메라 개수, 방문 2 는 센서 2건)
    f = fuser(events)
    assert [f.detail_visit(visit(["cantata"]), 3100.0, 7300.0).removed,
            f.detail_visit(visit(["donggwaru"]), 8300.0, 11900.0).removed] == [1, 2]


def test_late_record_of_another_kind_is_left_for_later():
    f = fuser([count(8000.0, "L2", -1)], sensor_lag_seconds=3.0)
    out = f.detail_visit(visit(["cantata"]), 3000.0, 6000.0)     # 카메라는 칸타타만 봤다 — 동과루 센서 기록은 이 방문 것이 아니다
    assert out.removed == 1
    f.note_band(6500.0, 7)
    late = f.sensor_only(20000.0)
    assert len(late) == 1 and late[0].removed == 1 and late[0].person_id == 7    # 기록 앞 늦음 시간 동안 선반 앞에 있던 사람


def test_sensor_only_keeps_records_inside_a_held_visit():
    f = fuser([count(5000.0, "L1", -1)])
    assert f.sensor_only(20000.0, [(3000.0, float("inf"), "shelf")]) == []      # 아직 안 끝난 방문 — 붙들어 둔다
    assert len(f.sensor_only(20000.0)) == 1


def test_returns_are_combined_by_kind_too():
    f = fuser([count(4000.0, "L1", 1)])
    out = f.detail_visit(visit([], ["cantata", "applecider"]), 3000.0, 6000.0)
    assert (out.removed, out.added) == (0, 2)


def test_without_sensor_kinds_the_old_rule_still_replaces_the_whole_count():
    config = FusionConfig()
    f = ShelfEventFuser(config, CrossViewLinker(config), TimeAxis(30.0), TimeAxis(30.0), dict(LANES), [count(3000.0, "L1", -1)])
    out = f.detail_visit(visit(["cantata", "applecider"]), 2000.0, 5000.0)
    assert out.removed == 1 and "센서와 다름" in out.detail


def test_slot_picture_change_is_left_out_when_slot_evidence_is_off():
    # 4차 C16: 앞 상품 둘의 자리만 맞바꿨다 — 선반 지도는 '변화 없음', 칸 그림은 '바뀜'. 상품 모델이 상품을 다 알면 칸 그림은 안 붙인다
    class Change:
        slot, start_ms, end_ms, ratio, direction = "L2", 2500.0, 4500.0, 0.13, "바뀜"

    def run(**cfg):
        f = fuser([], **cfg)
        f.note_slot_change(Change())
        return f.detail_visit(visit(), 2000.0, 5000.0)

    assert run().slot_change == "바뀜"
    assert run(slot_evidence=False).slot_change is None


def test_watcher_reports_open_visits():
    class Monitor:
        def __init__(self, name, since):
            self.name, self._occluded_since = name, since

    watcher = ShelfWatcher.__new__(ShelfWatcher)
    watcher.monitors = [({}, Monitor("shelf", 1200.0)), ({}, Monitor("side", None))]
    assert watcher.open_visits() == [(1200.0, "shelf")]
