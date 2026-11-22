"""전체 카메라 번호 맞바뀜 되돌리기 (fusion.swap_repair, C-26 '두 카메라 보증').

배경 (3차 C20 16.1초, 후속 47·49): 전체 카메라에서 두 손님의 상자가 겹친 뒤 두 사람의 번호가 서로 바뀌어 붙었다.
선반 카메라는 두 사람을 끊김 없이 따로 보고 있었고, 두 화면의 짝만 (가⇄2, 나⇄1) → (가⇄1, 나⇄2) 로 맞바뀌었다.
짝은 자리를 따라가 맞게 맺혔지만 매장 신원(= 전체 카메라 번호)이 사람을 바꿔 탔다 — 그 앞의 집기 기록이 다른 사람 것이 된다.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import FusionConfig  # noqa: E402
from src.core.types import Frame, IdentityObservation  # noqa: E402
from src.fusion.cross_view import CrossViewLinker  # noqa: E402
from src.fusion.judge import DualJudge  # noqa: E402
from src.fusion.shelf_events import TimeAxis  # noqa: E402

W = H = 1000
LEFT, RIGHT = (100.0, 100.0, 300.0, 900.0), (700.0, 100.0, 900.0, 900.0)   # 떨어진 두 상자
MID_L, MID_R = (400.0, 100.0, 600.0, 900.0), (450.0, 100.0, 650.0, 900.0)  # 겹친 두 상자 (0.75)
BAND = {LEFT: 0.2, MID_L: 0.2, RIGHT: 0.8, MID_R: 0.8}
"""상자 → 선반 앞 띠 위치 (시험용 — 자리는 이 표가 정한다)."""
SWAP_MS = 2112
"""전체 카메라가 번호를 뒤바꿔 붙이는 시각 (조각 경계: 66ms 간격이면 조각이 0·528·1056·1584·2112… 에서 닫힌다)."""


class Risk:
    """손님 장부 대신 — 받은 신원·합침 기록을 적어 둔다."""

    def __init__(self):
        self.book = SimpleNamespace(resolve=lambda pid: {30: 12}.get(pid, pid))
        self.reset()

    def reset(self):
        self.seen, self.merges = [], []

    def update(self, frame, identities, takes, merges, visits):
        self.seen.append((frame.pts_ms, {o.track_id: o.person_id for o in identities}))
        self.merges = list(merges)
        return [], []

    def finish(self, frame, size):
        return []


def person(pid, bbox, track=None):
    return IdentityObservation(person_id=pid, track_id=track or pid, bbox=bbox, frame=0, timestamp="")


def play(repair=True, overview_touch=True, shelf_touch=False, until=4000, merges_at=None, **cfg):
    """선반 카메라 1·2 번은 왼쪽·오른쪽에 그대로 선다. 전체 카메라 11번(추적 번호 101, 왼쪽 사람)·12번(102, 오른쪽 사람)은
    SWAP_MS 부터 번호가 서로 바뀌어 붙는다 — 11 이 오른쪽 사람(102)에게, 12 가 왼쪽 사람(101)에게."""
    floor = SimpleNamespace(foot_mode="bbox", foot_on_floor=lambda bbox, w, h, kp=None: tuple(bbox),
                            band_position=lambda point: BAND[point], zones_at=lambda point: [])
    risk = Risk()
    judge = DualJudge(FusionConfig(swap_repair=repair, **cfg), risk, floor, (1.0, 0.0), TimeAxis(15.0), TimeAxis(15.0),
                      W, (W, H), {}, None, None, detail_point=lambda bbox, width: BAND[tuple(bbox)])
    judge.reset()
    merges = []
    for t in range(0, until, 66):
        near = SWAP_MS - 600 <= t <= SWAP_MS + 600
        s_left, s_right = (MID_L, MID_R) if shelf_touch and near else (LEFT, RIGHT)
        o_left, o_right = (MID_L, MID_R) if overview_touch and near else (LEFT, RIGHT)
        left_id, right_id = (12, 11) if t >= SWAP_MS else (11, 12)
        judge.detail_step(float(t), [person(1, s_left), person(2, s_right)], [], [])
        if merges_at is not None and t >= merges_at and not merges:
            merges.append({"from": 30, "into": 11, "frame": 0})
        frame = Frame(index=t, timestamp="", pts_ms=float(t), image=None, camera_id="angle1")
        judge.overview_step(frame, [person(left_id, o_left, track=101), person(right_id, o_right, track=102)], merges)
    return judge, risk


def test_swapped_overview_numbers_are_read_back():
    judge, risk = play()
    assert len(judge.swaps) == 1
    swap = judge.swaps[0]
    assert sorted(swap["screen"]) == [11, 12] and swap["detail"] == [1, 2]
    assert swap["ms"] == SWAP_MS and swap["overview_overlap"] == 0.75 and swap["detail_overlap"] == 0.0
    # 선반 카메라 1번(왼쪽 사람)의 짝은 처음부터 끝까지 매장 신원 11 — 구간이 끊기지 않는다
    assert judge.linker.owner(1, 500, 1500) == 11 and judge.linker.owner(1, 3000, 3900) == 11
    assert judge.linker.owner(2, 3000, 3900) == 12
    assert [o for _, _, o in judge.linker._history[1]] == [11]
    assert {(l.detail_id, l.overview_id) for l in judge.linker.links} == {(1, 11), (2, 12)}
    # 손님 장부에는 알아챈 뒤부터 매장 신원으로: 왼쪽 사람(추적 101)은 화면 신원 12 를 달고 있어도 11 로 들어간다
    assert risk.seen[-1][1] == {101: 11, 102: 12}
    assert judge.store_id(12) == 11 and judge.screen_id(11) == 12


def test_off_by_default_the_store_identity_changes_person():
    judge, risk = play(repair=False)
    assert not judge.swaps
    assert judge.linker.owner(1, 500, 1500) == 11 and judge.linker.owner(1, 3000, 3900) == 12
    assert risk.seen[-1][1] == {101: 12, 102: 11}


def test_no_repair_when_the_overview_boxes_never_overlapped():
    # 전체 카메라 상자가 겹친 적이 없으면 번호가 바뀔 까닭이 없다 — 짝이 틀렸다가 바로잡힌 것일 수 있다 (2차 C20 11.7초)
    judge, _ = play(overview_touch=False)
    assert not judge.swaps and judge.linker.owner(1, 3000, 3900) == 12


def test_no_repair_when_the_shelf_boxes_overlapped_too():
    # 선반 카메라 상자도 겹쳤으면 어느 카메라의 번호가 바뀌었는지 알 수 없다
    judge, _ = play(shelf_touch=True)
    assert not judge.swaps


def test_overlap_limit_is_the_setting():
    assert not play(swap_overlap=0.8)[0].swaps       # 전체 카메라 겹침 0.75 < 0.8
    assert play(swap_overlap=0.7)[0].swaps


def test_merges_after_a_swap_go_to_the_book_under_store_identities():
    # 번호를 바꿔 읽기 시작한 뒤 화면 신원 11 로 합쳐진 조각은 그 번호가 지금 가리키는 사람(매장 신원 12)에게 간다
    judge, risk = play(merges_at=3000)
    assert risk.merges == [{"from": 30, "into": 12, "frame": 0}]
    assert judge.final_id(30, lambda pid: {30: 11}.get(pid, pid)) == 12      # 판정 번호 풀이는 손님 장부의 합침을 따른다
    untouched, risk = play(repair=False, merges_at=3000)
    assert risk.merges == [{"from": 30, "into": 11, "frame": 0}]
    assert untouched.final_id(30, lambda pid: {30: 11}.get(pid, pid)) == 11  # 바꿔 읽은 적 없으면 화면 신원의 합침 풀이 그대로


def test_a_second_swap_between_the_same_two_goes_back_to_plain_numbers():
    judge, _ = play()
    judge.linker.windows.clear()
    # 같은 두 사람의 번호가 한 번 더 바뀌면 — 화면 신원과 매장 신원이 다시 같아진다
    for t in range(4026, 8000, 66):
        near = 5500 <= t <= 6700
        left_id, right_id = (11, 12) if t >= 6072 else (12, 11)
        judge.detail_step(float(t), [person(1, LEFT), person(2, RIGHT)], [], [])
        frame = Frame(index=t, timestamp="", pts_ms=float(t), image=None, camera_id="angle1")
        judge.overview_step(frame, [person(left_id, MID_L if near else LEFT, track=101),
                                    person(right_id, MID_R if near else RIGHT, track=102)], [])
    assert len(judge.swaps) == 2 and judge._store == {}
    assert [o for _, _, o in judge.linker._history[1]] == [11]


# ---------------------------------------------------------------------- 짝 연결기

def feed(linker, moves, t0, t1, step=66):
    for t in range(t0, t1, step):
        for (view, pid), band in moves.items():
            linker.observe(view, pid, float(t), band)
        linker.update(float(t))


def test_mutual_swap_needs_two_steady_windows_before_it():
    straight = {("detail", 1): 0.2, ("detail", 2): 0.8, ("overview", 11): 0.2, ("overview", 12): 0.8}
    crossed = {**straight, ("overview", 11): 0.8, ("overview", 12): 0.2}
    linker = CrossViewLinker(FusionConfig())
    feed(linker, straight, 0, 1122)          # 조각 2개 (528·1056 에서 닫힘)
    assert linker.mutual_swaps() == []
    feed(linker, crossed, 1122, 1650)        # 세 번째 조각(1056~1584)에서 맞바뀜
    assert linker.mutual_swaps() == [(1, 2, 11, 12)]
    feed(linker, crossed, 1650, 2178)        # 다음 조각은 그대로 — 방금 닫힌 조각만 본다
    assert linker.mutual_swaps() == []
    # 앞 짝이 한 조각뿐이었으면(처음 잘못 맺힌 짝이 바로잡힌 경우) 맞바뀜으로 안 본다
    linker = CrossViewLinker(FusionConfig())
    feed(linker, straight, 0, 594)
    feed(linker, crossed, 594, 1122)
    assert linker.mutual_swaps() == []


def test_swap_overview_rewrites_only_what_came_after():
    straight = {("detail", 1): 0.2, ("detail", 2): 0.8, ("overview", 11): 0.2, ("overview", 12): 0.8}
    crossed = {**straight, ("overview", 11): 0.8, ("overview", 12): 0.2}
    linker = CrossViewLinker(FusionConfig())
    feed(linker, straight, 0, 1056)
    feed(linker, crossed, 1056, 1650)        # 번호가 조각 경계(1056)에서 바뀜
    since = linker.windows[-1][0]
    assert linker.current(1) == 12
    linker.swap_overview(11, 12, since)
    assert linker.current(1) == 11 and linker.current(2) == 12
    assert linker._history[1] == [(0.0, 1584.0, 11)] and linker._history[2] == [(0.0, 1584.0, 12)]
    assert all(l.overview_id == {1: 11, 2: 12}[l.detail_id] for l in linker.links)
    # 위치 표본도 바꿔 든다: 매장 신원 11 은 처음부터 끝까지 왼쪽(0.2)
    assert {round(s.band, 1) for s in linker._overview[11].samples} == {0.2}
    assert {round(s.band, 1) for s in linker._overview[12].samples} == {0.8}
    assert linker.mutual_swaps() == []       # 닫힌 조각 기록도 고쳐져 같은 맞바뀜을 다시 잡지 않는다
