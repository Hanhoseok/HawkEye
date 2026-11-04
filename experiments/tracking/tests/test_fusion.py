"""⑦ 합치기 층 — 교차 화면 연결, 선반 사건 묶기, 칸 그림 비교, 바닥 구역 어댑터 (설계 M-3·M-4·M-5·M-7·M-8)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def test_dual_judge_turns_overview_wrist_in_shelf_into_reach():
    from src.core.types import Frame, IdentityObservation
    from src.fusion.judge import DualJudge
    from src.zones.floor import shelf_image_zones

    root = Path(__file__).resolve().parents[1]
    floor = FloorMap.load(root / "zones" / "store01_dual" / "floor.yaml")
    import yaml
    raw = yaml.safe_load((root / "zones" / "store01_dual" / "floor.yaml").read_text(encoding="utf-8"))
    w, h = 1920, 1080
    bbox = (0.30 * w, 0.20 * h, 0.45 * w, 0.85 * h)

    def run(reach_seconds, wrist):
        risk = RecordingRisk()
        judge = DualJudge(FusionConfig(overview_reach_seconds=reach_seconds), risk, floor, (1.0, 0.0),
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


def test_overview_reach_ignores_a_person_standing_inside_the_shelf():
    # 상품 포장의 사람 그림에 관절이 잡히면 '발'도 선반 영역 안에 있다 — 바닥에 선 사람이 아니므로 팔 뻗기로 안 봄
    from src.core.types import Frame, IdentityObservation
    from src.fusion.judge import DualJudge
    from src.zones.floor import shelf_image_zones
    import yaml

    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "zones" / "store01_dual" / "floor.yaml").read_text(encoding="utf-8"))
    floor = FloorMap.load(root / "zones" / "store01_dual" / "floor.yaml")
    w, h = 1920, 1080
    risk = RecordingRisk()
    judge = DualJudge(FusionConfig(overview_reach_seconds=0.3), risk, floor, (1.0, 0.0), TimeAxis(30.0),
                      TimeAxis(60.0), w, (w, h), {}, None, shelf_image_zones(raw))
    judge.reset()
    ghost = (0.15 * w, 0.40 * h, 0.25 * w, 0.70 * h)          # 선반 한가운데 작은 '사람'
    for i in range(0, 120, 4):
        frame = Frame(index=i, timestamp="", pts_ms=i * 1000.0 / 60, image=None, camera_id="angle1")
        obs = IdentityObservation(person_id=9, track_id=9, bbox=ghost, frame=i, timestamp="",
                                  keypoints=kp_with_wrist(0.2 * w, 0.55 * h))
        judge.overview_step(frame, [obs], [], [])
        last = frame
    judge.finish(last)
    assert risk.takes == []
