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


# ---------------------------------------------------------------- 바닥 위 거리로 잇기 (3차 촬영, position_mode "floor")


def feed_floor(linker, moves, t0=0, t1=3000, step=66):
    """moves: {(view, pid): (띠 위치, 바닥 cm 또는 None[, 발목이 안 보일 때의 방향])}."""
    for t in range(t0, t1, step):
        for (view, pid), (band, floor, *ray) in moves.items():
            linker.observe(view, pid, float(t), band, None, floor, ray[0] if ray else None)
        linker.update(float(t))


def test_floor_mode_separates_two_people_at_the_same_band_position():
    # 선반 앞 같은 좌우 위치(띠 0.5)에 앞뒤로 선 두 사람 — 한 줄 좌표로는 못 가르고(B-14), 바닥 위 거리로는 가른다.
    people = {("detail", 7): (0.5, (10.0, 40.0)), ("detail", 9): (0.5, (70.0, 40.0)),
              ("overview", 2): (0.5, (68.0, 42.0)), ("overview", 5): (0.5, (12.0, 38.0))}
    linker = CrossViewLinker(FusionConfig(position_mode="floor"))
    feed_floor(linker, people)
    assert linker.current(7) == 5 and linker.current(9) == 2
    scores = {l.detail_id: l.position for l in linker.links}
    assert scores[7] > 0.9 and scores[9] > 0.9
    band_only = CrossViewLinker(FusionConfig())            # 기본(band)은 바닥 좌표를 줘도 쓰지 않는다
    feed_floor(band_only, people)
    assert {l.position for l in band_only.links} == {1.0}  # 네 조합 모두 위치 점수 1 — 누가 누구인지 위치로는 모른다


def test_floor_mode_refuses_a_far_person_even_with_the_same_band_position():
    linker = CrossViewLinker(FusionConfig(position_mode="floor"))
    feed_floor(linker, {("detail", 1): (0.5, (0.0, 40.0)), ("overview", 3): (0.5, (90.0, 40.0))})
    assert linker.current(1) is None                       # 90cm 떨어짐 (기준 60cm 에서 점수 0)


def test_floor_mode_falls_back_to_band_when_there_is_no_floor_point():
    # 전체 카메라 사람이 지평선 너머라 바닥 좌표가 없다 — 그 표본은 띠 위치로 견준다.
    linker = CrossViewLinker(FusionConfig(position_mode="floor"))
    feed_floor(linker, {("detail", 1): (0.60, (20.0, 60.0)), ("overview", 3): (0.62, None)})
    assert linker.current(1) == 3


def test_floor_mode_treats_a_hidden_foot_as_a_ray_not_a_point():
    # 전체 카메라에서 선반 앞 사람의 발이 화면 아래로 잘렸다: 상자 아래는 (20, 75) 인데 실제 발은 그 아래쪽(+y) 어딘가.
    # 선반 카메라는 발목을 (22, 115) 로 본다 — 점으로 견주면 40cm 라 못 잇고, 반직선으로 견주면 2cm.
    hidden = {("detail", 1): (0.9, (22.0, 115.0)), ("overview", 3): (0.85, (20.0, 75.0), (0.0, 1.0))}
    linker = CrossViewLinker(FusionConfig(position_mode="floor"))
    feed_floor(linker, hidden)
    assert linker.current(1) == 3 and linker.links[-1].position > 0.9
    as_point = CrossViewLinker(FusionConfig(position_mode="floor"))
    feed_floor(as_point, {k: v[:2] for k, v in hidden.items()})
    assert as_point.current(1) is None
    # 반직선의 반대쪽(화면 위쪽)에 있는 사람은 시작점까지의 거리 그대로 — 발이 상자 아래보다 위일 수는 없다
    behind = CrossViewLinker(FusionConfig(position_mode="floor"))
    feed_floor(behind, {("detail", 1): (0.4, (22.0, 20.0)), ("overview", 3): (0.85, (20.0, 75.0), (0.0, 1.0))})
    assert behind.current(1) is None


def test_floor_distance_between_two_rays():
    from src.fusion.cross_view import _Sample, floor_distance

    crossing = floor_distance(_Sample(0, None, (0.0, 0.0), (1.0, 0.0)), _Sample(0, None, (30.0, -20.0), (0.0, 1.0)))
    apart = floor_distance(_Sample(0, None, (0.0, 0.0), (1.0, 0.0)), _Sample(0, None, (-30.0, 20.0), (0.0, 1.0)))
    assert crossing == 0.0 and abs(apart - (30.0 ** 2 + 20.0 ** 2) ** 0.5) < 1e-6


def test_dual_judge_floor_point_is_a_point_with_ankles_and_a_ray_without():
    from src.fusion.judge import DualJudge

    fm = FloorMap("detail", [(0.2, 0.5), (0.8, 0.5), (0.2, 1.0), (0.8, 1.0)], [(0, 0), (100, 0), (0, 100), (100, 100)])
    ankles = [[0.0, 0.0, 0.0]] * 15 + [[940.0, 800.0, 0.9], [980.0, 820.0, 0.9]]
    (x, y), ray = DualJudge.floor_point(fm, (900.0, 300.0, 1020.0, 830.0), ankles, 1920, 1080)
    ex, ey = fm.to_floor(960 / 1920, 810 / 1080)           # 두 발목의 가운데
    assert abs(x - ex) < 1e-6 and abs(y - ey) < 1e-6 and ray is None
    # 발목이 안 보이면 상자 아래 가운데에서 화면 아래쪽(이 지도에서는 바닥 +y)으로 뻗는 반직선
    (bx, by), ray = DualJudge.floor_point(fm, (900.0, 300.0, 1020.0, 830.0), None, 1920, 1080)
    assert (round(bx, 3), round(by, 3)) == tuple(round(v, 3) for v in fm.to_floor(960 / 1920, 830 / 1080))
    assert abs(ray[0]) < 1e-6 and abs(ray[1] - 1.0) < 1e-6


def test_detail_floor_map_reads_calibration_from_zone_file():
    from src.zones.floor import detail_floor_map

    assert detail_floor_map({}) is None and detail_floor_map(None) is None
    fm = detail_floor_map({"floor_calibration": {"image": [[0.2, 0.7], [0.2, 0.9], [0.6, 0.7], [0.6, 0.9]],
                                                 "floor": [[0, 0], [64, 0], [0, 74], [64, 74]]}})
    fx, fy = fm.to_floor(0.6, 0.9)
    assert abs(fx - 64) < 1e-6 and abs(fy - 74) < 1e-6
