"""바닥 지도(호모그래피)와 두 카메라 시차 표 — 1단계 1-0 도구의 계산 부분 (docs/multiview-design.md M-3·M-9)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.types import ZoneType  # noqa: E402
from src.inputs.sync import SyncRow, SyncTable, best_lag, offset_from_events  # noqa: E402
from src.zones.floor import FloorMap, FloorZone, apply_homography, fit_homography  # noqa: E402

# 알고 있는 행렬로 바닥 점을 화면에 투영해 '찍은 점'을 만든다 — 맞춘 행렬이 원래 대응을 되살리는지 본다
TRUE_IMG_FROM_FLOOR = np.array([[0.0009, 0.0002, 0.30], [0.0000, 0.0004, 0.55], [0.0, 0.0006, 1.0]])
FLOOR_PTS = [(0, 0), (240, 0), (240, 300), (0, 300), (120, 150)]


def image_points(pts):
    return [apply_homography(TRUE_IMG_FROM_FLOOR, x, y) for x, y in pts]


def test_homography_recovers_floor_positions():
    fm = FloorMap("angle1", image_points(FLOOR_PTS), FLOOR_PTS)
    assert fm.reprojection_error_cm() < 0.5
    # 보정에 안 쓴 점도 맞아야 한다 (바닥 타일 한 칸 = 60cm 가정)
    for fx, fy in [(60, 60), (200, 250), (30, 280)]:
        x, y = apply_homography(TRUE_IMG_FROM_FLOOR, fx, fy)
        gx, gy = fm.to_floor(x, y)
        assert abs(gx - fx) < 0.5 and abs(gy - fy) < 0.5
        back = fm.to_image(gx, gy)
        assert abs(back[0] - x) < 1e-6 and abs(back[1] - y) < 1e-6


def test_homography_needs_four_points_not_on_a_line():
    with pytest.raises(ValueError):
        fit_homography([(0, 0), (1, 0), (0, 1)], [(0, 0), (1, 0), (0, 1)])
    with pytest.raises(ValueError):
        fit_homography([(0, 0), (1, 0), (2, 0), (3, 0)], [(0, 0), (1, 0), (2, 0), (3, 0)])


def test_foot_position_and_floor_zones(tmp_path):
    counter = FloorZone("counter", ZoneType.CHECKOUT, ((150, 0), (240, 0), (240, 90), (150, 90)))
    fm = FloorMap("angle1", image_points(FLOOR_PTS), FLOOR_PTS, [counter])
    # 발이 (200, 40) cm 에 있는 사람: 화면 상자의 아래 가운데가 그 점의 투영
    fx, fy = apply_homography(TRUE_IMG_FROM_FLOOR, 200, 40)
    w, h = 1920, 1080
    bbox = (fx * w - 50, fy * h - 400, fx * w + 50, fy * h)
    point = fm.foot_on_floor(bbox, w, h)
    assert abs(point[0] - 200) < 1 and abs(point[1] - 40) < 1
    assert [z.name for z in fm.zones_at(point)] == ["counter"]
    assert fm.zones_at((20, 200)) == []

    # 저장했다 읽어도 같은 대응
    path = tmp_path / "floor.yaml"
    fm.save(path)
    again = FloorMap.load(path)
    assert again.camera == "angle1" and again.zones[0].type == ZoneType.CHECKOUT
    assert np.allclose(again.to_floor(fx, fy), fm.to_floor(fx, fy), atol=0.5)


def test_offset_definition_and_table(tmp_path):
    # 박수가 angle0 에서 2.4초, angle1 에서 2.1초에 보임 → angle0 시각에 -300ms 를 더하면 공통 시각
    off = offset_from_events(2400.0, 2100.0)
    assert off == -300.0
    table = SyncTable()
    table.upsert(SyncRow("N1", "1(angle0)/20261028_N1.mp4", "1(angle1)/N01.mp4", off, "manual", "규원"))
    table.upsert(SyncRow("N1", "1(angle0)/20261028_N1.mp4", "1(angle1)/N01.mp4", -280.0, "manual", "규원", "재확인"))
    path = tmp_path / "sync.csv"
    table.save(path)
    loaded = SyncTable.load(path)
    assert loaded.offset_ms("N1") == -280.0      # 같은 짝은 덮어씀
    assert loaded.rows["N1"].note == "재확인"
    assert loaded.offset_ms("T1") is None
    assert SyncTable.load(tmp_path / "없음.csv").rows == {}


def test_best_lag_finds_shift_in_offset_convention():
    hz = 10.0
    rng = np.random.default_rng(0)
    base = rng.random(200)
    # reference 는 signal 보다 0.4초 늦게 같은 일을 본다 → signal 시각 + 400ms = reference 시각
    signal = base
    reference = np.r_[np.zeros(4), base[:-4]]
    off, corr = best_lag(signal, reference, hz)
    assert off == pytest.approx(400.0)
    assert corr > 0.9
