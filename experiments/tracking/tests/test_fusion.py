"""⑦ 합치기 층 — 교차 화면 연결, 선반 사건 묶기, 칸 그림 비교, 바닥 구역 어댑터 (설계 M-3·M-4·M-5·M-7·M-8)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from src.core.config import FusionConfig  # noqa: E402
from src.core.types import TakeCandidate, ZoneType  # noqa: E402
from src.fusion.cross_view import CrossViewLinker  # noqa: E402
from src.fusion.shelf_events import SensorEvent, ShelfEventFuser, TimeAxis  # noqa: E402
from src.risk.customers import ShelfVisit, classify_visit  # noqa: E402
from src.shelf.slot_diff import SlotDiff, SlotDiffConfig, changed_ratio  # noqa: E402
from src.zones.floor import FloorMap, FloorZone, FloorZoneMap  # noqa: E402


def feed(linker, moves, t0=0, t1=3000, step=66):
    """moves: {(view, pid): 시각 → 띠 위치 함수}."""
    for t in range(t0, t1, step):
        for (view, pid), fn in moves.items():
            linker.observe(view, pid, float(t), fn(t))
        linker.update(float(t))


def test_linker_pairs_people_by_position_not_by_number():
    # 선반 카메라 7번은 선반 끝(0.2)에, 9번은 계산대 쪽(0.8)에. 전체 카메라 2번은 0.8, 5번은 0.2.
    linker = CrossViewLinker(FusionConfig())
    feed(linker, {("detail", 7): lambda t: 0.2, ("detail", 9): lambda t: 0.8,
                  ("overview", 2): lambda t: 0.82, ("overview", 5): lambda t: 0.18})
    assert linker.current(7) == 5 and linker.current(9) == 2
    assert linker.owner(7, 1000, 2000) == 5


def test_linker_leaves_unmatched_when_positions_disagree():
    linker = CrossViewLinker(FusionConfig())
    feed(linker, {("detail", 1): lambda t: 0.0, ("overview", 3): lambda t: 0.9})
    assert linker.current(1) is None     # 띠에서 한참 떨어진 사람과는 잇지 않는다 (짝 없음 허용)


def test_linker_trusts_appearance_only_when_one_candidate():
    # 선반 카메라 띠 위치가 0.25 넘게 어긋나도(계산대·팔 뻗기), 비교할 사람이 한 명이고 외형이 높으면 잇는다
    cfg = FusionConfig()
    a, b = np.array([1.0, 0.0]), np.array([0.6, 0.8])
    linker = CrossViewLinker(cfg)
    for t in range(0, 3000, 66):
        linker.observe("detail", 1, float(t), 0.70, a)
        linker.observe("overview", 3, float(t), 0.95, a)
        linker.update(float(t))
    assert linker.current(1) == 3
    # 외형이 낮으면(0.6) 그대로 짝 없음
    linker = CrossViewLinker(cfg)
    for t in range(0, 3000, 66):
        linker.observe("detail", 1, float(t), 0.70, a)
        linker.observe("overview", 3, float(t), 0.95, b)
        linker.update(float(t))
    assert linker.current(1) is None
    # 후보가 둘이면 외형만으로 잇지 않는다 — 위치가 가까운 사람이 없으면 짝 없음
    linker = CrossViewLinker(cfg)
    for t in range(0, 3000, 66):
        linker.observe("detail", 1, float(t), 0.70, a)
        linker.observe("overview", 3, float(t), 0.95, a)
        linker.observe("overview", 4, float(t), 0.20, a)
        linker.update(float(t))
    assert linker.current(1) is None
    # 끄면(0) 위치만 본다
    linker = CrossViewLinker(FusionConfig(solo_appearance_min=0.0))
    for t in range(0, 3000, 66):
        linker.observe("detail", 1, float(t), 0.70, a)
        linker.observe("overview", 3, float(t), 0.95, a)
        linker.update(float(t))
    assert linker.current(1) is None


def test_linker_follows_crossing_people_with_appearance_help():
    # 두 사람이 자리를 바꾼다 — 위치만으로 따라가고, 외형이 같은 쪽을 받쳐 준다
    cfg = FusionConfig()
    linker = CrossViewLinker(cfg)
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    for t in range(0, 4000, 66):
        x = 0.2 + 0.6 * t / 4000
        linker.observe("detail", 1, t, x, a)
        linker.observe("detail", 2, t, 1.0 - x, b)
        linker.observe("overview", 10, t, x + 0.02, a)
        linker.observe("overview", 20, t, 1.0 - x - 0.02, b)
        linker.update(float(t))
    assert linker.owner(1, 3000, 4000) == 10 and linker.owner(2, 3000, 4000) == 20


def test_fuser_assigns_unowned_visit_through_hand_touch_and_converts_frames():
    cfg = FusionConfig()
    linker = CrossViewLinker(cfg)
    feed(linker, {("detail", 4): lambda t: 0.3, ("detail", 6): lambda t: 0.35,
                  ("overview", 11): lambda t: 0.3, ("overview", 12): lambda t: 0.36}, 0, 6000)
    detail_axis, overview_axis = TimeAxis(30.0, -300.0), TimeAxis(60.0)
    fuser = ShelfEventFuser(cfg, linker, detail_axis, overview_axis, {"card": "shelf"})
    # 선반 카메라 4번이 card 칸에 손을 댐 (선반 카메라 장면 90~105 = 3.0~3.5초 → 시차 -0.3초 → 공통 2.7~3.2초)
    take = TakeCandidate(4, "card", 90, 105, "", "", 0.5, 8, 0.0, 0.0)
    fused_take = fuser.detail_take(take)
    assert fused_take.person_id == 11 and fused_take.zone == "shelf"
    assert fused_take.start_frame == overview_axis.to_frame(3000.0 - 300.0) == 162
    # 두 사람이 함께 가려 선반 카메라가 주인을 못 정한 방문 → 손을 댄 사람에게
    visit = ShelfVisit(-1, "shelf", 60, 150, 1, 0, "shelf", "[shelf] 사라짐: 1단 1번째")
    fused = fuser.detail_visit(visit, 2000.0, 4700.0)
    assert fused.person_id == 11
    assert fuser.records[-1].how == "손-칸 접촉"


def test_fuser_sensor_overrides_count_and_reports_disagreement():
    cfg = FusionConfig(sensor_unit_g={"card": 45.0})
    linker = CrossViewLinker(cfg)
    feed(linker, {("detail", 1): lambda t: 0.3, ("overview", 2): lambda t: 0.3}, 0, 6000)
    sensors = [SensorEvent(3000.0, "card", -92.0)]
    fuser = ShelfEventFuser(cfg, linker, TimeAxis(30.0), TimeAxis(60.0), {"card": "shelf"}, sensors)
    visit = ShelfVisit(1, "shelf", 60, 120, 0, 0, "shelf", "[shelf] 변화 없음")
    fused = fuser.detail_visit(visit, 2500.0, 3800.0)
    assert (fused.removed, fused.added, fused.source) == (2, 0, "sensor+vision")   # 무게로 2개
    assert "센서와 다름" in fused.detail
    action = classify_visit(fused, [])
    assert action.kind == "TAKE" and action.count == 2 and action.evidence.startswith("센서+선반")


def test_sensor_only_event_becomes_visit_after_waiting():
    cfg = FusionConfig(sensor_unit_g={"card": 45.0})
    linker = CrossViewLinker(cfg)
    fuser = ShelfEventFuser(cfg, linker, TimeAxis(30.0), TimeAxis(60.0), {"card": "shelf"},
                            [SensorEvent(1000.0, "card", -44.0)])
    fuser.note_band(900.0, 8)
    assert fuser.sensor_only(2000.0) == []           # 아직 카메라 방문과 묶일 수 있다
    out = fuser.sensor_only(10000.0)
    assert len(out) == 1 and out[0].person_id == 8 and out[0].source == "sensor" and out[0].removed == 1


# ---------------------------------------------------------------- 센서 레인 ↔ 손-칸 접촉 레인 (3차 L1·L2·L3)

LANE_SHELF = {"L1": "shelf", "L2": "shelf", "L3": "shelf"}


def lane_fuser(cfg: FusionConfig, sensors, touches):
    """선반 카메라 4번 ↔ 전체 11번, 6번 ↔ 12번으로 짝지은 뒤 touches=[(선반 카메라 신원, 레인), ...] 손 접촉."""
    linker = CrossViewLinker(cfg)
    feed(linker, {("detail", 4): lambda t: 0.3, ("detail", 6): lambda t: 0.7,
                  ("overview", 11): lambda t: 0.3, ("overview", 12): lambda t: 0.7}, 0, 6000)
    fuser = ShelfEventFuser(cfg, linker, TimeAxis(30.0), TimeAxis(60.0), LANE_SHELF, sensors)
    for pid, lane in touches:
        fuser.detail_take(TakeCandidate(pid, lane, 85, 100, "", "", 0.5, 8, 0.0, 0.0))   # 2.8~3.3초
    for t in (2800.0, 3000.0, 3200.0):
        fuser.note_band(t, 11)
        fuser.note_band(t, 12)
    return fuser


def two_person_visit(fuser):
    return fuser.detail_visit(ShelfVisit(-1, "shelf", 60, 120, 0, 0, "shelf", "[shelf] 두 사람이 가림"), 2500.0, 3800.0)


def test_sensor_lane_picks_the_person_who_touched_that_lane():
    # 두 사람이 다른 레인을 만졌고 센서는 L2 에서 1개 꺼냄 → L2 를 만진 사람(12)
    fuser = lane_fuser(FusionConfig(), [SensorEvent(3000.0, "L2", -1.0, unit="count")], [(4, "L1"), (6, "L2")])
    fused = two_person_visit(fuser)
    assert fused.person_id == 12 and fused.removed == 1
    assert "센서 레인 L2" in fuser.records[-1].how


def test_sensor_lane_match_off_keeps_old_unknown():
    cfg = FusionConfig(sensor_lane_match=False)
    fuser = lane_fuser(cfg, [SensorEvent(3000.0, "L2", -1.0, unit="count")], [(4, "L1"), (6, "L2")])
    fused = two_person_visit(fuser)
    assert fused.person_id == -1 and "모름" in fuser.records[-1].how


def test_same_lane_touched_by_two_people_stays_unknown():
    fuser = lane_fuser(FusionConfig(), [SensorEvent(3000.0, "L2", -1.0, unit="count")], [(4, "L2"), (6, "L2")])
    assert two_person_visit(fuser).person_id == -1


def test_two_lanes_at_once_need_one_person():
    # 센서가 L1·L2 둘 다 바뀌었는데 각 레인을 다른 사람이 만짐 → 한 방문의 주인으로는 못 정함
    sensors = [SensorEvent(3000.0, "L1", -1.0, unit="count"), SensorEvent(3100.0, "L2", -1.0, unit="count")]
    fuser = lane_fuser(FusionConfig(), sensors, [(4, "L1"), (6, "L2")])
    assert two_person_visit(fuser).person_id == -1


def test_sensor_only_event_uses_lane_touch():
    # 카메라 방문과 안 묶인 센서 기록도 그 레인을 만진 사람에게 (선반 앞 2명이라 예전엔 모름)
    fuser = lane_fuser(FusionConfig(), [SensorEvent(3000.0, "L1", -1.0, unit="count")], [(4, "L1"), (6, "L2")])
    out = fuser.sensor_only(20000.0)
    assert len(out) == 1 and out[0].person_id == 11 and out[0].source == "sensor"
    assert "센서 레인 L1" in fuser.records[-1].how


def test_lane_sensor_counts_multiple_items():
    # 레인 센서는 개수를 보낸다: REMOVE 2 = 2개 (예전엔 칸 무게가 없으면 기록 하나 = 1개로 셈)
    fuser = lane_fuser(FusionConfig(), [SensorEvent(3000.0, "L2", -2.0, unit="count")], [(6, "L2")])
    fused = two_person_visit(fuser)
    assert (fused.person_id, fused.removed) == (12, 2)


def test_lane_first_overrides_single_reach_only_when_enabled():
    sensors = [SensorEvent(3000.0, "L2", -1.0, unit="count")]
    for first, expect in ((False, 11), (True, 12)):
        fuser = lane_fuser(FusionConfig(sensor_lane_first=first), sensors, [(4, "L1"), (6, "L2")])
        fuser.note_reach(2700.0, 3300.0, 11)                    # 전체 카메라에선 11 만 팔을 뻗음
        assert two_person_visit(fuser).person_id == expect


# ---------------------------------------------------------------- 위쪽(전체) 카메라로 센서 레인 주인 (3차 C02)

def hands(fuser, pid, lane, times):
    for t in times:
        fuser.note_overview_lane(t, pid, lane)


def test_overview_hand_in_lane_picks_owner():
    # 선반 카메라 손 접촉 없음, 선반 앞 2명 — 위쪽 카메라에서 L2 에 손을 넣은 사람(12)
    fuser = lane_fuser(FusionConfig(), [SensorEvent(3000.0, "L2", -1.0, unit="count")], [])
    hands(fuser, 12, "L2", (2000.0, 2066.0, 2133.0))
    hands(fuser, 11, "L1", (2000.0, 2066.0, 2133.0))
    assert two_person_visit(fuser).person_id == 12
    assert "위쪽 카메라 센서 레인 L2 손" in fuser.records[-1].how


def test_overview_hand_uses_longest_stay_lane():
    # 11 은 L2 를 잠깐 지나갔지만 L3 에 오래 머묾 — 레인 폭이 좁아 손이 옆 레인 위를 지나가는 것은 가장 오래 머문 레인으로 거른다
    fuser = lane_fuser(FusionConfig(), [SensorEvent(3000.0, "L2", -1.0, unit="count")], [])
    hands(fuser, 11, "L2", (2000.0, 2066.0))
    hands(fuser, 11, "L3", (2100.0, 2166.0, 2233.0, 2300.0, 2366.0))
    hands(fuser, 12, "L2", (2400.0, 2466.0))
    assert two_person_visit(fuser).person_id == 12


def test_overview_hand_window_is_before_sensor():
    # 센서는 손보다 늦게 찍힌다 — 센서 3초 전까지 보고, 그보다 이른 손은 안 본다
    sensors = [SensorEvent(3000.0, "L2", -1.0, unit="count")]
    fuser = lane_fuser(FusionConfig(), sensors, [])
    hands(fuser, 12, "L2", (100.0, 166.0))
    assert two_person_visit(fuser).person_id == 12
    fuser = lane_fuser(FusionConfig(), sensors, [])
    hands(fuser, 12, "L2", (-500.0, -434.0))
    assert two_person_visit(fuser).person_id == -1


def test_tray_cover_when_hand_is_hidden():
    # C02 14초: 패딩 손님이 받침 앞에 서서 손이 안 보임 — 그 시각 받침을 몸으로 덮은 사람(11)
    fuser = lane_fuser(FusionConfig(), [SensorEvent(3000.0, "L1", -1.0, unit="count")], [])
    for t in (2500.0, 2600.0, 2700.0, 2800.0):
        fuser.note_tray_cover(t, 11, 0.95)
        fuser.note_tray_cover(t, 12, 0.2)
    assert two_person_visit(fuser).person_id == 11
    assert "받침을 가린 사람" in fuser.records[-1].how


def test_tray_cover_by_two_people_stays_unknown():
    fuser = lane_fuser(FusionConfig(), [SensorEvent(3000.0, "L1", -1.0, unit="count")], [])
    for t in (2500.0, 2600.0, 2700.0):
        fuser.note_tray_cover(t, 11, 0.9)
        fuser.note_tray_cover(t, 12, 0.7)
    assert two_person_visit(fuser).person_id == -1


def test_overview_lane_match_off_ignores_overview():
    cfg = FusionConfig(overview_lane_match=False)
    fuser = lane_fuser(cfg, [SensorEvent(3000.0, "L2", -1.0, unit="count")], [])
    hands(fuser, 12, "L2", (2000.0, 2066.0, 2133.0))
    for t in (2500.0, 2600.0, 2700.0):
        fuser.note_tray_cover(t, 11, 0.9)
    assert two_person_visit(fuser).person_id == -1


def test_sensor_only_uses_overview_hand():
    fuser = lane_fuser(FusionConfig(), [SensorEvent(3000.0, "L2", -1.0, unit="count")], [])
    hands(fuser, 11, "L2", (2200.0, 2266.0))
    out = fuser.sensor_only(20000.0)
    assert len(out) == 1 and out[0].person_id == 11


def test_pair_floor_moves_lanes_from_reference_clip():
    import numpy as np
    import shoot3_eval as E

    base = {"calibration": {"image": [[0, 0]], "floor": [[0, 0]]}, "lanes_image_ref": "C02",
            "lanes_image": {"L1": [[0.5, 0.5], [0.6, 0.5], [0.6, 0.6], [0.5, 0.6]]}}
    shift = np.array([[1, 0, 19.2], [0, 1, 0], [0, 0, 1]], float)       # 기준 영상 → area_setting: 오른쪽 1%
    out = E.pair_floor(base, np.eye(3), shift)
    assert "lanes_image_ref" not in out
    assert out["lanes_image"]["L1"][0] == [0.51, 0.5]
    assert "lanes_image" not in E.pair_floor(base, np.eye(3), None)     # 기준 영상 맞추기가 없으면 뺀다


def test_lanes_image_zones_reads_named_lanes():
    from src.zones.floor import lanes_image_zones

    zm = lanes_image_zones({"lanes_image": {"L1": [[0, 0], [1, 0], [1, 1]], "L2": [[0, 0], [1, 1], [0, 1]]}})
    assert [z.name for z in zm.zones] == ["L1", "L2"]
    assert lanes_image_zones({}) is None


def test_slot_change_keeps_take_when_detector_sees_nothing():
    # 물건 탐지 '변화 없음' + 칸 줄어듦 → 집음 1개 / 칸 바뀜(방향 모름) + 손 뻗기 → 확인 못 함(+1) / 칸 그대로 → 만지기만
    take = TakeCandidate(1, "shelf", 10, 40, "", "", 1.0, 5, 0.0, 0.0)
    base = dict(person_id=1, shelf="shelf", start_frame=0, end_frame=50, removed=0, added=0, zone="shelf")
    assert classify_visit(ShelfVisit(**base, slot_change="줄어듦"), [take]).kind == "TAKE"
    unsure = classify_visit(ShelfVisit(**base, slot_change="바뀜"), [take])
    assert unsure.kind == "UNVERIFIED" and unsure.count == 1
    assert classify_visit(ShelfVisit(**base), [take]).kind == "TOUCH"


def shelf_image(bottles: int, light: float = 1.0) -> np.ndarray:
    img = np.full((200, 300, 3), 170, np.uint8)
    img[:, :, 0] = 120
    for i in range(bottles):
        x = 20 + i * 30
        img[40:140, x:x + 18] = (40, 60, 200)          # 병: 진한 직사각형
    img[150:190, 200:280] = (230, 230, 230)            # 옆 칸의 흰 상자 (안 바뀜)
    return np.clip(img.astype(np.float32) * light, 0, 255).astype(np.uint8)


def test_slot_diff_detects_removed_item_and_ignores_lighting():
    slots = [("bottle", (10, 30, 190, 150)), ("box", (190, 145, 290, 195))]
    diff = SlotDiff(slots, SlotDiffConfig(stable_seconds=0.2))
    t = 0
    for _ in range(5):                       # 기준 그림
        diff.update(t, shelf_image(5), [])
        t += 100
    for _ in range(5):                       # 손님이 칸을 가림
        diff.update(t, shelf_image(5), [(50, 0, 200, 200)])
        t += 100
    out = []
    for _ in range(5):                       # 떠난 뒤: 병 둘이 빠짐
        out += diff.update(t, shelf_image(3), [])
        t += 100
    assert [c.slot for c in out] == ["bottle"] and out[0].direction == "줄어듦"

    # 조명만 어두워진 경우: 밝기를 맞춰 비교하므로 변화 아님
    assert changed_ratio(shelf_image(3), shelf_image(3, 0.7), 0.6, 4) < 0.02


def test_floor_zone_adapter_reports_foot_zone():
    img = [(0.4, 1.0), (0.6, 1.0), (0.45, 0.4), (0.6, 0.4)]
    flr = [(0, 0), (100, 0), (0, 200), (100, 200)]
    counter = FloorZone("counter", ZoneType.CHECKOUT, ((-50, 100), (150, 100), (150, 250), (-50, 250)))
    shelf = FloorZone("shelf", ZoneType.SHELF, ((-50, -50), (150, -50), (150, 100), (-50, 100)))
    fz = FloorZoneMap(FloorMap("angle1", img, flr, [counter, shelf], (0.0, 200.0)))
    fz.resolve(1000, 1000)
    foot_near = (450, 500, 550, 990)     # 화면 아래쪽 = 선반 앞
    assert [h.zone for h in fz.evaluate(foot_near)] == ["shelf"]
    assert fz.evaluate(foot_near)[0].overlap == 1.0 and fz.evaluate(foot_near)[0].contains_foot
    hits = fz.evaluate((480, 100, 560, 520))   # 화면 위쪽 = 계산대 쪽
    assert [h.zone for h in hits] == ["counter"]
    assert len(fz.polygon_of("counter")) == 4


def test_overview_reach_names_the_owner_over_the_band_link():
    # C2 같은 경우: 비슷한 옷 2명이 선반 앞 띠에서 거의 같은 자리 — 띠 위치로는 선반 카메라 4번이 11번과 짝.
    # 전체 카메라에서는 12번만 선반으로 팔을 뻗었다 → 그 시각 선반 변화의 주인은 12번
    cfg = FusionConfig()
    linker = CrossViewLinker(cfg)
    feed(linker, {("detail", 4): lambda t: 0.3, ("overview", 11): lambda t: 0.3,
                  ("overview", 12): lambda t: 0.45}, 0, 6000)
    fuser = ShelfEventFuser(cfg, linker, TimeAxis(30.0), TimeAxis(60.0), {"card": "shelf"})
    visit = ShelfVisit(4, "shelf", 60, 150, 1, 0, "shelf", "[shelf] 사라짐: 1단 1번째")
    assert fuser.detail_visit(visit, 2000.0, 4700.0).person_id == 11          # 팔 뻗기 기록이 없으면 예전처럼 짝
    fuser.note_reach(2500.0, 3200.0, 12)
    fused = fuser.detail_visit(visit, 2000.0, 4700.0)
    assert fused.person_id == 12 and fuser.records[-1].how == "전체 카메라 팔 뻗기"
    fuser.note_reach(2600.0, 3000.0, 11)                                      # 둘 다 뻗었으면 팔 뻗기로는 못 정함
    assert fuser.detail_visit(visit, 2000.0, 4700.0).person_id == 11


def kp_with_wrist(x: float, y: float, conf: float = 0.9):
    """관절 17점 — 왼손목(9)만 (x, y), 나머지는 안 보임."""
    pts = [(0.0, 0.0, 0.0)] * 17
    pts[9] = (x, y, conf)
    return tuple(pts)


class RecordingRisk:
    def __init__(self):
        self.takes = []

    def reset(self):
        self.takes = []

    def update(self, frame, identities, takes, merges, visits):
        self.takes += takes
        return [], []

    def finish(self, frame, size):
        return []


TEST_SHELF_IMAGE = [[0.0, 0.22], [0.2, 0.2], [0.3, 0.33], [0.375, 0.36], [0.375, 1.0], [0.0, 1.0]]
"""시험용 전체 카메라 선반 영역 — 11-04 에 그린 모양(앞 모서리 x 0.375). 적용본은 구역 편집 페이지에서 바뀌므로 시험은 이것에 묶는다."""


def _with_test_shelf(raw: dict) -> dict:
    return {**raw, "shelf_image": [TEST_SHELF_IMAGE]}


def test_dual_judge_turns_overview_wrist_in_shelf_into_reach():
    from src.core.types import Frame, IdentityObservation
    from src.fusion.judge import DualJudge
    from src.zones.floor import shelf_image_zones

    root = Path(__file__).resolve().parents[1]
    floor = FloorMap.load(root / "zones" / "store01_dual" / "floor.yaml")
    import yaml
    raw = _with_test_shelf(yaml.safe_load((root / "zones" / "store01_dual" / "floor.yaml").read_text(encoding="utf-8")))
    w, h = 1920, 1080
    bbox = (0.30 * w, 0.20 * h, 0.45 * w, 0.85 * h)

    def run(reach_seconds, wrist, as_take=True):
        risk = RecordingRisk()
        judge = DualJudge(FusionConfig(overview_reach_seconds=reach_seconds, overview_reach_takes=as_take), risk,
                          floor, (1.0, 0.0),
                          TimeAxis(30.0), TimeAxis(60.0), w, (w, h), {}, None, shelf_image_zones(raw))
        judge.reset()
        last = None
        for i in range(0, 120, 4):                       # 2초, 초당 15장
            frame = Frame(index=i, timestamp="", pts_ms=i * 1000.0 / 60, image=None, camera_id="angle1")
            x = wrist if 0.5 <= frame.pts_ms / 1000 <= 1.2 else 0.43 * w   # 0.5~1.2초 동안 손목이 선반 안
            obs = IdentityObservation(person_id=3, track_id=3, bbox=bbox, frame=i, timestamp="",
                                      keypoints=kp_with_wrist(x, 0.5 * h))
            judge.overview_step(frame, [obs], [], [])
            last = frame
        judge.finish(last)
        return risk.takes, judge

    takes, judge = run(0.3, 0.33 * w)
    assert len(takes) == 1 and takes[0].person_id == 3 and takes[0].zone == "shelf"
    assert 0.6 <= takes[0].duration_sec <= 0.8
    assert any(r.kind == "reach" for r in judge.fuser.records)
    assert run(0.3, 0.40 * w)[0] == []                  # 손목이 선반 앞 모서리(0.375) 밖 — 몸 옆
    assert run(0.0, 0.33 * w)[0] == []                  # 기본은 꺼짐 (전체 카메라에 손목이 없던 예전 실행과 같음)
    takes, judge = run(0.3, 0.33 * w, as_take=False)    # 기본: 주인 정하기 근거로만 — 집기 행동으로 넘기지 않음
    assert takes == [] and any(r.kind == "reach" for r in judge.fuser.records)


def test_overview_reach_ignores_a_person_standing_inside_the_shelf():
    # 상품 포장의 사람 그림에 관절이 잡히면 '발'도 선반 영역 안에 있다 — 바닥에 선 사람이 아니므로 팔 뻗기로 안 봄
    from src.core.types import Frame, IdentityObservation
    from src.fusion.judge import DualJudge
    from src.zones.floor import shelf_image_zones
    import yaml

    root = Path(__file__).resolve().parents[1]
    raw = _with_test_shelf(yaml.safe_load((root / "zones" / "store01_dual" / "floor.yaml").read_text(encoding="utf-8")))
    floor = FloorMap.load(root / "zones" / "store01_dual" / "floor.yaml")
    w, h = 1920, 1080
    risk = RecordingRisk()
    judge = DualJudge(FusionConfig(overview_reach_seconds=0.3, overview_reach_takes=True), risk, floor, (1.0, 0.0),
                      TimeAxis(30.0), TimeAxis(60.0), w, (w, h), {}, None, shelf_image_zones(raw))
    judge.reset()
    ghost = (0.15 * w, 0.40 * h, 0.25 * w, 0.70 * h)          # 선반 한가운데 작은 '사람'
    for i in range(0, 120, 4):
        frame = Frame(index=i, timestamp="", pts_ms=i * 1000.0 / 60, image=None, camera_id="angle1")
        obs = IdentityObservation(person_id=9, track_id=9, bbox=ghost, frame=i, timestamp="",
                                  keypoints=kp_with_wrist(0.2 * w, 0.55 * h))
        judge.overview_step(frame, [obs], [], [])
        last = frame
    judge.finish(last)
    assert risk.takes == [] and not any(r.kind == "reach" for r in judge.fuser.records)


def test_dual_judge_notes_overview_lane_hand_and_tray_cover():
    # 3차 위쪽 카메라 레인(floor.yaml lanes_image, C02 화면 기준): 손이 L2 안이면 레인 손, 몸이 받침을 덮으면 가림
    from src.core.types import Frame, IdentityObservation
    from src.fusion.judge import DualJudge
    from src.zones.floor import lanes_image_zones
    import yaml

    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "zones" / "store01_shoot3" / "floor.yaml").read_text(encoding="utf-8"))
    floor = FloorMap.load(root / "zones" / "store01_shoot3" / "floor.yaml")
    w, h = 1920, 1080
    judge = DualJudge(FusionConfig(), RecordingRisk(), floor, (1.0, 0.0), TimeAxis(30.0), TimeAxis(60.0), w, (w, h),
                      {}, None, None, lanes_image=lanes_image_zones(raw))
    judge.reset()
    l2 = raw["lanes_image"]["L2"]
    cx = sum(x for x, _ in l2) / 4 * w
    cy = sum(y for _, y in l2) / 4 * h
    beside = (1200.0, 300.0, 1450.0, 1000.0)          # 받침 옆에 선 사람 — 손만 레인 안
    front = (1350.0, 450.0, 1700.0, 1080.0)           # 받침 앞에 선 사람 — 몸이 받침을 다 덮음
    for i in range(0, 12, 4):
        frame = Frame(index=i, timestamp="", pts_ms=i * 1000.0 / 60, image=None, camera_id="angle1")
        judge.overview_step(frame, [
            IdentityObservation(person_id=3, track_id=3, bbox=beside, frame=i, timestamp="", keypoints=kp_with_wrist(cx, cy)),
            IdentityObservation(person_id=4, track_id=4, bbox=front, frame=i, timestamp="", keypoints=kp_with_wrist(0, 0, 0.0)),
        ], [], [])
    lanes = {(pid, lane) for _, pid, lane in judge.fuser._ov_lane}
    assert lanes == {(3, "L2")}
    cover = {pid: r for _, pid, r in judge.fuser._cover}
    assert cover[4] == 1.0 and cover.get(3, 0.0) < 0.6


def test_c02_pose_trace_picks_both_owners():
    # 3차 C02 위쪽 카메라를 yolov8n-pose 로 돌린 실제 기록 (요약). 센서: L2 6.67초, L1 14.67초 (위쪽 카메라 시각)
    # 니트 손님(1): 손이 L2 에 4.34~5.21초 연속, 들어오고 나가며 L3 위를 지나감. 패딩 손님(2): 12.1~15.9초 받침을 90% 넘게 덮음,
    # 가려진 손목은 L1·L2·L3 에 흩어져 찍힘. 둘 다 선반 앞에 있어 선반 앞 한 명으로는 못 정함.
    sensors = [SensorEvent(6670.0, "L2", -1.0, unit="count"), SensorEvent(14670.0, "L1", -1.0, unit="count")]
    fuser = lane_fuser(FusionConfig(), sensors, [])
    step = 1000.0 / 15
    frames = lambda a, b: [a + k * step for k in range(int(round((b - a) / step)) + 1)]
    hands(fuser, 11, "L3", frames(4070.0, 4270.0) + frames(5270.0, 5410.0) + frames(5540.0, 5670.0))
    hands(fuser, 11, "L2", frames(4340.0, 5210.0))
    for t in frames(4270.0, 5610.0):
        fuser.note_tray_cover(t, 11, 0.75)
    hands(fuser, 12, "L2", (13020.0, 13150.0, 13220.0, 13750.0, 13880.0, 14480.0))
    hands(fuser, 12, "L3", (13080.0, 13280.0, 13420.0, 13550.0, 13620.0, 13680.0, 13820.0))
    hands(fuser, 12, "L1", (13950.0, 14020.0, 14080.0, 14150.0, 14280.0, 14350.0, 14420.0))
    for t in frames(12100.0, 15900.0):
        fuser.note_tray_cover(t, 12, 0.93)
    for t in frames(14820.0, 15020.0):
        fuser.note_tray_cover(t, 11, 0.2)
    for t in frames(3000.0, 17000.0):
        fuser.note_band(t, 11)
        fuser.note_band(t, 12)
    out = fuser.sensor_only(60000.0)
    assert [(v.detail, v.person_id) for v in out] == [("센서 L2 -1개", 11), ("센서 L1 -1개", 12)]
    hows = [r.how for r in fuser.records if r.kind == "sensor"]
    assert "위쪽 카메라 센서 레인 L2 손" in hows[0]
