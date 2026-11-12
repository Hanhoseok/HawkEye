"""3차 촬영 발 위치 시차 (src/inputs/floor_sync.py, scripts/sync_floor.py) 와 sync_pair --shoot 3."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import sync_floor as SF  # noqa: E402
import sync_pair as SP  # noqa: E402
from src.inputs import floor_sync as FS  # noqa: E402
from src.zones.floor import FloorMap  # noqa: E402

# 실제 3차 보정 (area_setting 메모지 중심, 화면 비율) — 정규화 뒤 보정점의 w 가 음수가 되는 경우 (검토 F1)
A0 = [(435.8 / 1920, 816.2 / 1080), (351.7 / 1920, 1025.6 / 1080), (1134.7 / 1920, 784.1 / 1080), (1217.8 / 1920, 1036.2 / 1080)]
A1 = [(1124.9 / 1920, 605.6 / 1080), (942.4 / 1920, 603.5 / 1080), (1190.4 / 1920, 1069.7 / 1080), (847.4 / 1920, 1058.2 / 1080)]


def maps():
    return {"angle0": FloorMap("angle0", A0, SF.FLOOR_CM), "angle1": FloorMap("angle1", A1, SF.FLOOR_CM)}


def test_floor_side_keeps_calibration_points_and_drops_beyond_horizon():
    for fm, pts in ((maps()["angle0"], A0), (maps()["angle1"], A1)):
        assert all(fm.on_floor_side(x, y) for x, y in pts)
        assert all(fm.image_side_ok(fx, fy) for fx, fy in SF.FLOOR_CM)
        assert fm.on_floor_side(0.5, 0.95)
        # 화면 한참 위쪽(하늘 방향)은 지평선 너머 → 바닥이 아니다
        assert not fm.on_floor_side(0.5, -3.0)


def test_cross_view_mapping_is_consistent_through_floor():
    m = maps()
    for x, y in A0:
        fx, fy = m["angle0"].to_floor(x, y)
        u, v = m["angle1"].to_image(fx, fy)
        k = A0.index((x, y))
        assert abs(u - A1[k][0]) < 1e-6 and abs(v - A1[k][1]) < 1e-6


def walk(t_ms, phase=0.0):
    """매장 안을 걷는 한 사람 (cm). 주기 아님."""
    s = t_ms / 1000.0
    return np.array([60 + 80 * np.sin(0.7 * s + phase) + 15 * s, 40 + 120 * np.cos(0.4 * s + phase)])


def samples(times, people, shift_ms=0.0, extra=None):
    out = []
    for t in times:
        pts = [p(t + shift_ms) for p in people]
        if extra is not None:
            pts.append(extra(t))
        out.append((float(t), np.array(pts, dtype=float)))
    return out


def test_search_recovers_known_offset_with_fake_person():
    true = 370.0   # 공통 시각 = angle0 시각 + 370ms
    people = [walk, lambda t: walk(t, 1.7)]
    a0 = samples(np.arange(0, 20000, 100), people, shift_ms=true, extra=lambda t: np.array([300.0, -200.0]))
    a1 = samples(np.arange(0, 21000, 50), people)
    res = FS.search(a0, a1)
    assert res.offset_ms is not None and abs(res.offset_ms - true) <= 20
    ok, why = res.judge()
    assert res.sharpness > FS.OK_MIN_SHARPNESS


def test_not_enough_samples_is_low():
    a0 = samples(np.arange(0, 1000, 100), [walk])
    a1 = samples(np.arange(0, 1000, 50), [walk])
    res = FS.search(a0, a1)
    assert res.offset_ms is None
    assert res.judge() == (False, ["두 카메라에 함께 보이는 사람 표본이 부족"])


def test_periodic_motion_is_not_sharp():
    def pace(t):   # 1초 주기로 같은 자리를 왕복
        return np.array([100 + 60 * np.sin(2 * np.pi * t / 1000.0), 50.0])
    a0 = samples(np.arange(0, 15000, 100), [pace], shift_ms=200)
    a1 = samples(np.arange(0, 16000, 50), [pace])
    res = FS.search(a0, a1)
    ok, why = res.judge()
    assert not ok and any("뾰족함" in w for w in why)


def test_chamfer_caps_far_points():
    a = np.array([[0.0, 0.0]])
    b = np.array([[0.0, 0.0], [1000.0, 0.0]])
    assert FS.chamfer(a, b) == FS.CAP_CM / 3


def note_image(cut=False):
    img = np.full((400, 600, 3), 120, np.uint8)
    cv2.rectangle(img, (100, 100), (160, 145), (90, 220, 230), -1)   # 노란 메모지 (BGR)
    cv2.rectangle(img, (300, 360) if cut else (300, 200), (360, 400) if cut else (360, 245), (90, 220, 230), -1)
    cv2.rectangle(img, (420, 80), (520, 180), (60, 150, 200), -1)    # 나무색 가구 (노랑 성분 있음)
    return img


def test_note_centers_pick_nearest_blob_and_fix_cut_note():
    centers, warn = SF.note_centers(note_image(), [(125, 118), (335, 228)])
    assert np.hypot(centers[0][0] - 130, centers[0][1] - 122.5) < 3
    assert np.hypot(centers[1][0] - 330, centers[1][1] - 222.5) < 3
    assert not warn
    # 화면 아래 끝에 잘린 메모지: 위 모서리 + 반 높이 (위 메모지 세로/가로 비 46/61)
    centers, warn = SF.note_centers(note_image(cut=True), [(125, 118), (330, 385)])
    expect_y = 360 + 61 * (46 / 61) / 2
    assert abs(centers[1][1] - expect_y) < 3 and "잘림" in warn[0]


def test_pair_map_moves_calibration_points_by_registration():
    base = maps()["angle0"]
    shift = np.array([[1, 0, 40], [0, 1, -10], [0, 0, 1]], float)   # 영상 → area: +40, -10 px
    moved = SF.pair_map(base, {"h": " ".join(str(v) for v in shift.ravel())})
    for (x, y), (u, v) in zip(base.image_points, moved.image_points):
        assert abs((x - u) * 1920 - 40) < 1e-3 and abs((y - v) * 1080 + 10) < 1e-3


def test_samples_recomputed_from_screen_cache_with_current_calibration(tmp_path, monkeypatch):
    monkeypatch.setattr(SF, "FEET", tmp_path)
    x, y = 0.4 * 1920, 0.85 * 1080
    for cam in SF.CAMS:
        (tmp_path / f"P_{cam}.csv").write_text(
            ",".join(SF.FEET_FIELDS) + f"\n100.0,0,{x},{y},1920,1080,0.9,1,0\n100.0,1,{x},{y},1920,1080,0.9,0,1\n"
            "150.0,-1,,,1920,1080,,,\n", encoding="utf-8")
    m = maps()
    first = SF.load_samples("P", "angle0", m)
    assert [len(p) for _, p in first] == [1, 0]   # 잘린 사람은 빠지고, 사람 없는 표본은 빈 채로 남는다
    other = dict(m)
    other["angle0"] = FloorMap("angle0", A0, [(fx * 2, fy * 2) for fx, fy in SF.FLOOR_CM])
    second = SF.load_samples("P", "angle0", other)
    assert np.allclose(second[0][1], first[0][1] * 2)


def test_sync_pair_shoot3_names_and_relative_paths():
    video, ref = SP.pair_files("N08-1", shoot=3)
    assert (SP.rel(video), SP.rel(ref)) == ("2(angle0)/N08-1.mp4", "2(angle1)/N08-1.mp4")
    v2, r2 = SP.pair_files("N8-1")
    assert SP.rel(v2) == "1(angle0)/20261028_N8-1.mp4" and SP.rel(r2) == "1(angle1)/N08-1.mp4"


# ---------------------------------------------------------------------- 소리 + 판정

from src.inputs import audio_sync as AS  # noqa: E402


def test_gcc_phat_recovers_delay_and_rejects_silence():
    rng = np.random.default_rng(0)
    sr = 16000
    src = rng.normal(size=sr * 6)
    delay = int(0.25 * sr)            # angle0 이 사건을 0.25초 늦게 녹음 → angle0 시각 - 250ms = angle1 시각
    a0 = np.concatenate([np.zeros(delay), src])[: sr * 6] + 0.3 * rng.normal(size=sr * 6)
    a1 = src + 0.3 * rng.normal(size=sr * 6)
    off, peak = AS.gcc_phat(a0, a1, sr)
    assert abs(off + 250) <= 5 and peak >= AS.MIN_PEAK
    off, peak = AS.gcc_phat(np.zeros(sr), a1, sr)
    assert np.isnan(off) and peak == 0.0 and not AS.usable(off, peak)


def floor_result(offset=0.0, cost=5.0, pairs=100, sharp=2.0, curve=None):
    curve = curve if curve is not None else [(float(l), cost + abs(l - offset) / 100.0, pairs) for l in range(-3000, 3001, 20)]
    return FS.FloorSyncResult(offset, cost, pairs, sharp, None, curve)


def test_combine_agree_uses_audio_value():
    d = FS.combine(floor_result(offset=40.0), 10.0, True, 5.3, 100)
    assert (d.method, d.ok, d.offset_ms) == ("floor+audio", True, 10.0)


def test_combine_floor_pass_but_audio_disagrees_is_low():
    d = FS.combine(floor_result(offset=0.0), 1000.0, True, 15.0, 100)
    assert d.method == "floor-low" and not d.ok and d.offset_ms == 0.0
    assert any("불일치" in w for w in d.why)


def test_combine_does_not_promote_flat_curve_few_samples_or_weak_audio():
    flat = [(float(l), 100.0, 20) for l in range(-3000, 3001, 20)]
    res = FS.FloorSyncResult(-3000.0, 100.0, 20, 1.0, None, flat)
    d = FS.combine(res, -2900.0, True, 100.0, 20)
    assert d.method == "floor-low" and any("표본" in w for w in d.why) and any("비용" in w for w in d.why)
    # 표본·비용은 충분해도 곡선이 평평하면 일치는 증거가 아님 → 확인 필요, 시작값은 소리
    flat_ok = [(float(l), 20.0, 100) for l in range(-3000, 3001, 20)]
    d = FS.combine(FS.FloorSyncResult(0.0, 20.0, 100, 1.0, None, flat_ok), 30.0, True, 20.0, 100)
    assert d.method == "floor-low" and d.offset_ms == 30.0 and any("가려내지" in w for w in d.why)
    # 약한 소리 → 발 위치 판정만
    d = FS.combine(floor_result(offset=0.0), 900.0, AS.usable(900.0, 1.05), 14.0, 100)
    assert d.method == "floor" and d.ok


def test_combine_disagree_start_value_depends_on_floor_objection():
    # 곡선이 넓어 소리 시차 비용이 최소와 거의 같음 → 불일치여도 시작값은 소리 (note 에 두 후보)
    wide = [(float(l), 5.0 + abs(l) / 2000.0, 100) for l in range(-3000, 3001, 20)]
    d = FS.combine(FS.FloorSyncResult(0.0, 5.0, 100, 1.1, None, wide), 1000.0, True, 5.5, 100)
    assert d.method == "floor-low" and d.offset_ms == 1000.0 and any("불일치" in w for w in d.why)
    # 발 위치가 반대함(소리 시차 비용이 훨씬 큼) → 시작값은 발 위치
    d = FS.combine(floor_result(offset=0.0), 1000.0, True, 15.0, 100)
    assert d.offset_ms == 0.0


def test_audio_stage_survives_missing_ffmpeg(tmp_path, monkeypatch):
    import subprocess

    def boom(*a, **k):
        raise FileNotFoundError("ffmpeg")
    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(SF, "AUDIO", tmp_path / "audio.csv")
    monkeypatch.setattr(SF, "OUT", tmp_path)
    monkeypatch.setattr(SF, "pairs", lambda only=None: ["X"])
    assert SF.audio(None) == {}


def test_frame_at_skips_negative_times(tmp_path):
    path = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for i in range(30):
        w.write(np.full((48, 64, 3), i * 8, np.uint8))
    w.release()
    out = SF.frame_at(path, [-1000.0, 1000.0, 2000.0], width=64)
    assert out[0] is None and out[1] is not None and out[2] is not None
