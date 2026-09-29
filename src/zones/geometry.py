"""다각형 기하 연산.

외부 라이브러리 없이 numpy 만 쓴다. shapely 를 끌어오지 않은 이유는
여기서 필요한 연산이 두 개뿐이고, 의존성을 늘리면 계층 분리 원칙이 흐려지기 때문이다.

두 가지 판정이 필요하다.

* **안에 있는가** (EXIT / CHECKOUT) — 사람이 그 구역에 서 있는지.
  발 위치(bbox 아래쪽 중앙)로 판정한다. 바닥 평면의 구역이므로 서 있는 지점이 기준이다.
* **얼마나 겹치는가** (SHELF) — 사람이 선반에 얼마나 가까운지.
  사람은 선반 '안에' 서지 않으므로 포함 판정으로는 잡히지 않는다. 겹침 넓이로 본다.
"""

from __future__ import annotations

Point = tuple[float, float]
Polygon = tuple[Point, ...]


def contains_point(polygon: Polygon, x: float, y: float) -> bool:
    """ray casting. 점이 다각형 내부에 있으면 True."""
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        # 점의 y 가 변 사이에 있고, 그 y 에서 변의 x 가 점보다 오른쪽이면 교차 1회
        if (y1 > y) != (y2 > y):
            t = (y - y1) / (y2 - y1)
            if x < x1 + t * (x2 - x1):
                inside = not inside
    return inside


def polygon_area(polygon: Polygon) -> float:
    """shoelace. 꼭짓점 순서와 무관하게 양수 넓이를 돌려준다."""
    total = 0.0
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def clip_to_rect(polygon: Polygon, rect: tuple[float, float, float, float]) -> Polygon:
    """Sutherland-Hodgman. 다각형을 직사각형으로 잘라낸 결과를 돌려준다.

    rect 는 (x1, y1, x2, y2). 잘린 결과가 비면 빈 튜플.
    """
    x1, y1, x2, y2 = rect
    # 각 변에 대해 "안쪽"의 정의와 교점 계산
    edges = (
        ("left", x1),
        ("right", x2),
        ("top", y1),
        ("bottom", y2),
    )

    def inside(p: Point, side: str, value: float) -> bool:
        if side == "left":
            return p[0] >= value
        if side == "right":
            return p[0] <= value
        if side == "top":
            return p[1] >= value
        return p[1] <= value

    def intersect(a: Point, b: Point, side: str, value: float) -> Point:
        if side in ("left", "right"):
            t = (value - a[0]) / (b[0] - a[0])
            return (value, a[1] + t * (b[1] - a[1]))
        t = (value - a[1]) / (b[1] - a[1])
        return (a[0] + t * (b[0] - a[0]), value)

    output: list[Point] = list(polygon)
    for side, value in edges:
        if not output:
            return ()
        current = output
        output = []
        for i in range(len(current)):
            a = current[i]
            b = current[(i + 1) % len(current)]
            a_in = inside(a, side, value)
            b_in = inside(b, side, value)
            if a_in:
                output.append(a)
            if a_in != b_in:
                output.append(intersect(a, b, side, value))
    return tuple(output)


def overlap_ratio(polygon: Polygon, bbox: tuple[float, float, float, float]) -> float:
    """bbox 넓이 중 다각형과 겹치는 비율 (0~1).

    bbox 기준으로 나누는 이유: 선반 구역은 사람보다 훨씬 넓어서
    구역 기준으로 나누면 값이 항상 0 에 가까워 쓸모가 없다.
    """
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    if area <= 0:
        return 0.0
    clipped = clip_to_rect(polygon, (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
    if len(clipped) < 3:
        return 0.0
    return min(1.0, polygon_area(clipped) / area)


def _point_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    """점에서 선분까지의 최단 거리."""
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    qx, qy = ax + t * dx, ay + t * dy
    return ((px - qx) ** 2 + (py - qy) ** 2) ** 0.5


def distance_to_polygon(polygon: Polygon, x: float, y: float) -> float:
    """점에서 다각형까지의 거리. 내부면 0.

    수직 천장 시점에서는 이미지가 거의 평면도이므로, 이 거리가
    "손이 선반에 얼마나 가까운가"를 실제로 뜻한다.
    비스듬한 각도에서는 깊이 모호성 때문에 같은 해석이 성립하지 않는다.
    """
    if contains_point(polygon, x, y):
        return 0.0
    n = len(polygon)
    return min(
        _point_segment_distance(x, y, polygon[i][0], polygon[i][1],
                                polygon[(i + 1) % n][0], polygon[(i + 1) % n][1])
        for i in range(n)
    )
