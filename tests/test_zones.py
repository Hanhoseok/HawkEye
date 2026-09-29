"""구역 기하·판정 테스트."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.types import Zone, ZoneType  # noqa: E402
from src.zones.geometry import clip_to_rect, contains_point, overlap_ratio, polygon_area  # noqa: E402
from src.zones.zone_map import ZoneMap  # noqa: E402

SQUARE = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))


def test_point_in_polygon():
    assert contains_point(SQUARE, 5, 5)
    assert not contains_point(SQUARE, 15, 5)
    assert not contains_point(SQUARE, 5, -1)


def test_polygon_area_is_orientation_independent():
    reversed_square = tuple(reversed(SQUARE))
    assert polygon_area(SQUARE) == polygon_area(reversed_square) == 100.0


def test_overlap_ratio_is_relative_to_bbox():
    """사람 bbox 기준으로 나눠야 한다. 구역 기준이면 넓은 선반에서 항상 0 에 가까워진다."""
    assert overlap_ratio(SQUARE, (2, 2, 4, 4)) == 1.0       # bbox 가 완전히 안에
    assert overlap_ratio(SQUARE, (5, 0, 15, 10)) == 0.5     # 절반만
    assert overlap_ratio(SQUARE, (20, 20, 30, 30)) == 0.0   # 바깥


def test_clip_handles_non_overlapping():
    assert clip_to_rect(SQUARE, (20, 20, 30, 30)) == ()


def test_triangle_clip_area():
    tri = ((0.0, 0.0), (10.0, 0.0), (0.0, 10.0))
    assert polygon_area(tri) == 50.0
    assert abs(overlap_ratio(tri, (0, 0, 10, 10)) - 0.5) < 1e-9


def test_normalized_coordinates_scale_with_frame():
    """상대 좌표는 처리 해상도가 바뀌어도 같은 위치를 가리켜야 한다."""
    zone = Zone(
        name="right_half",
        type=ZoneType.SHELF,
        polygon=((0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)),
        normalized=True,
    )
    for w, h in [(720, 404), (360, 202), (1920, 1080)]:
        zm = ZoneMap([zone])
        zm.resolve(w, h)
        # 오른쪽 끝에 있는 사람은 항상 겹쳐야 한다
        bbox = (w * 0.8, h * 0.5, w * 0.9, h * 0.9)
        hits = zm.evaluate(bbox)
        assert len(hits) == 1 and hits[0].overlap == 1.0, f"{w}x{h} 에서 실패"


def test_shelf_uses_overlap_exit_uses_foot():
    """SHELF 는 겹침, EXIT 은 발 위치로 판정한다는 구분이 살아 있어야 한다.

    사람은 선반 '안에' 서지 않으므로 포함 판정으로는 선반 접촉을 잡을 수 없고,
    반대로 원근 때문에 선반의 이미지 영역이 앞쪽 바닥을 덮어 발이 들어오기도 한다.
    """
    zones = [
        Zone("shelf", ZoneType.SHELF, ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))),
        Zone("exit", ZoneType.EXIT, ((20.0, 0.0), (30.0, 0.0), (30.0, 10.0), (20.0, 10.0))),
    ]
    zm = ZoneMap(zones)
    zm.resolve(100, 100)

    # 선반에 몸이 걸쳐 있지만 발은 바깥
    hits = {h.zone: h for h in zm.evaluate((8.0, 2.0, 14.0, 20.0))}
    assert hits["shelf"].overlap > 0
    assert not hits["shelf"].contains_foot

    # 출구 안에 서 있음
    hits = {h.zone: h for h in zm.evaluate((22.0, 0.0, 28.0, 8.0))}
    assert hits["exit"].contains_foot


def test_loads_project_zone_file():
    """실제 zones.yaml 이 읽히고 타입이 제대로 붙는지."""
    path = Path(__file__).resolve().parents[1] / "zones.yaml"
    if not path.exists():
        return
    zm = ZoneMap.load(path)
    assert len(zm) > 0
    assert any(z.type is ZoneType.SHELF for z in zm.zones)
    assert all(z.normalized for z in zm.zones), "상대 좌표 사용을 권장한다"
