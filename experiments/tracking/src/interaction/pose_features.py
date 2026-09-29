"""pose keypoint 에서 '손이 선반에 닿았는가'를 계산한다.

몸 전체 bbox 가 선반과 겹치는지는 판별력이 없다는 것이 측정으로 확인됐다 —
집기 · 되돌려놓기 · 구경이 전부 같은 자리에서 일어나기 때문이다
(docs/take-candidates.md §6-E, 오탐 7건 중 기하 오류 0건).

손은 다르다. 구경할 때 손은 몸 옆에 있고, 집을 때만 선반 안으로 들어간다.
그리고 그 동작은 **짧다** — 국소화(현재 9%) 개선도 같이 노린다.
"""

from __future__ import annotations

from ..core.types import (
    KP_LEFT_ELBOW,
    KP_LEFT_HIP,
    KP_LEFT_SHOULDER,
    KP_LEFT_WRIST,
    KP_RIGHT_ELBOW,
    KP_RIGHT_HIP,
    KP_RIGHT_SHOULDER,
    KP_RIGHT_WRIST,
    Keypoints,
)

WRISTS = ((KP_LEFT_WRIST, KP_LEFT_ELBOW, KP_LEFT_SHOULDER, KP_LEFT_HIP),
          (KP_RIGHT_WRIST, KP_RIGHT_ELBOW, KP_RIGHT_SHOULDER, KP_RIGHT_HIP))


def confident_wrists(keypoints: Keypoints | None, min_conf: float) -> list[tuple[float, float]]:
    """신뢰도가 충분한 손목 좌표만 돌려준다.

    pose 모델은 가려진 keypoint 를 (0, 0) 에 낮은 신뢰도로 내보낸다.
    그대로 쓰면 화면 좌상단이 항상 '손'이 되어 버린다.
    """
    if not keypoints:
        return []
    out = []
    for wrist_idx, *_ in WRISTS:
        if wrist_idx >= len(keypoints):
            continue
        x, y, conf = keypoints[wrist_idx]
        if conf >= min_conf and (x > 0 or y > 0):
            out.append((float(x), float(y)))
    return out


def arm_extension(keypoints: Keypoints | None, min_conf: float, body_height: float) -> float | None:
    """팔이 얼마나 뻗었는지. 어깨-손목 거리를 체구 높이로 나눈 값.

    체구로 나누는 이유는 속도와 같다 — 거리에 따라 픽셀 크기가 4~10배 변하므로
    픽셀 거리로는 가까운 사람과 먼 사람을 같은 기준으로 볼 수 없다.
    """
    if not keypoints or body_height <= 0:
        return None
    best = None
    for wrist_idx, _elbow, shoulder_idx, _hip in WRISTS:
        if max(wrist_idx, shoulder_idx) >= len(keypoints):
            continue
        wx, wy, wc = keypoints[wrist_idx]
        sx, sy, sc = keypoints[shoulder_idx]
        if wc < min_conf or sc < min_conf:
            continue
        if (wx == 0 and wy == 0) or (sx == 0 and sy == 0):
            continue
        dist = ((wx - sx) ** 2 + (wy - sy) ** 2) ** 0.5 / body_height
        best = dist if best is None else max(best, dist)
    return best


def hand_height(keypoints: Keypoints | None, min_conf: float, body_height: float) -> float | None:
    """손목이 엉덩이보다 얼마나 위에 있는지. 체구 높이로 나눈 값. 양수면 위.

    라벨 기준 실측: TAKE 중앙값 0.146 / BROWSE 0.036 (약 4배 차이).
    구경할 때는 손이 내려가 있고, 집을 때만 선반 높이로 올라간다.

    참고로 '팔 뻗음'(어깨-손목 거리)은 판별력이 없었다
    (TAKE 0.296 vs BROWSE 0.315 로 오히려 역전). 위치도 자세도 아무거나
    되는 것이 아니라, 실제로 갈리는 것을 골라야 한다.
    """
    if not keypoints or body_height <= 0:
        return None
    best = None
    for wrist_idx, _elbow, _shoulder, hip_idx in WRISTS:
        if max(wrist_idx, hip_idx) >= len(keypoints):
            continue
        wx, wy, wc = keypoints[wrist_idx]
        hx, hy, hc = keypoints[hip_idx]
        if wc < min_conf or hc < min_conf:
            continue
        if (wx == 0 and wy == 0) or (hx == 0 and hy == 0):
            continue
        value = (hy - wy) / body_height  # 이미지 y 는 아래로 증가하므로 뺄셈 방향이 이렇다
        best = value if best is None else max(best, value)
    return best


def hands_in_zones(zone_map, keypoints: Keypoints | None, min_conf: float) -> set[str]:
    """손목이 안에 들어간 구역 이름들.

    사람 bbox 겹침과 달리, 손이 실제로 구역 안으로 들어가야만 잡힌다.
    """
    hits: set[str] = set()
    wrists = confident_wrists(keypoints, min_conf)
    if not wrists:
        return hits
    from ..zones.geometry import contains_point

    for zone in zone_map.zones:
        polygon = zone_map.polygon_of(zone.name)
        if not polygon:
            continue
        for x, y in wrists:
            if contains_point(polygon, x, y):
                hits.add(zone.name)
                break
    return hits


def wrist_zone_distance(
    zone_map, keypoints: Keypoints | None, min_conf: float, body_height: float
) -> float | None:
    """손목에서 가장 가까운 SHELF 구역까지의 거리. 체구 높이로 나눈 값. 안에 있으면 0.

    **수직 천장 시점 전용 신호다.** 그 화각에서는 이미지가 거의 평면도라
    이 거리가 실제 "손이 선반에 닿았는가"를 뜻한다.

    MERL 실측(라벨 기준, 손목의 화면상 위치로 근사):
        손이 선반 안에  0.426   (25~75%: 0.394~0.466)
        선반 보기만    0.646   (25~75%: 0.569~0.706)
        -> 사분위 범위가 겹치지 않는다

    비스듬한 각도(store-aisle)에서는 쓰지 말 것. 손이 선반 구역 안에 있는 관측이
    68% 나와 선택성이 없었다.
    """
    from ..core.types import ZoneType

    if not keypoints or body_height <= 0:
        return None
    wrists = confident_wrists(keypoints, min_conf)
    if not wrists:
        return None
    from ..zones.geometry import distance_to_polygon

    best = None
    for zone in zone_map.zones:
        if zone.type is not ZoneType.SHELF:
            continue
        polygon = zone_map.polygon_of(zone.name)
        if not polygon:
            continue
        for x, y in wrists:
            d = distance_to_polygon(polygon, x, y) / body_height
            best = d if best is None else min(best, d)
    return best
