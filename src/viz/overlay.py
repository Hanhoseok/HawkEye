"""Phase 1~2 눈으로 검증하기 위한 오버레이.

bbox 와 track_id 를 영상에 그려서 ID 유지 여부를 직접 확인한다.
근거: 02 기능 명세 F9(일부), 06 백로그 M3
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Iterable

import cv2
import numpy as np

from ..core.types import TrackObservation, ZoneType

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def color_for(track_id: int) -> tuple[int, int, int]:
    """track_id 마다 항상 같은 색을 돌려준다. ID switch 를 눈으로 잡기 위한 장치."""
    golden = 0.618033988749895
    hue = int(((track_id * golden) % 1.0) * 179)
    hsv = ((hue, 200, 255),)
    bgr = cv2.cvtColor(np.uint8([hsv]), cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


_ZONE_COLORS = {
    ZoneType.SHELF: (60, 170, 255),
    ZoneType.EXIT: (60, 60, 255),
    ZoneType.CHECKOUT: (60, 255, 170),
    ZoneType.ENTRANCE: (255, 200, 60),
    ZoneType.OTHER: (180, 180, 180),
}


def draw_alerts(image: Any, alerts: Iterable[tuple[tuple, str]]) -> Any:
    """위험 판정을 받은 손님을 굵은 빨간 박스와 문구로 표시한다. alerts: (bbox, 문구) 목록."""
    for bbox, text in alerts:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 4)
        org = (x1, max(20, y1 - 10))
        cv2.putText(image, text, org, _FONT, 0.7, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(image, text, org, _FONT, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return image


def draw_zones(image: Any, zone_map, alpha: float = 0.25) -> Any:
    """구역을 반투명하게 채우고 이름을 적는다. 좌표가 맞는지 눈으로 확인하는 용도."""
    if not zone_map:
        return image
    canvas = image.copy()
    fill = image.copy()
    for zone, polygon in zone_map.resolved_polygons():
        if len(polygon) < 3:
            continue
        pts = np.array(polygon, dtype=np.int32)
        color = _ZONE_COLORS.get(zone.type, (180, 180, 180))
        cv2.fillPoly(fill, [pts], color)
        cv2.polylines(canvas, [pts], True, color, 2)
    canvas = cv2.addWeighted(fill, alpha, canvas, 1 - alpha, 0)
    for zone, polygon in zone_map.resolved_polygons():
        if len(polygon) < 3:
            continue
        pts = np.array(polygon, dtype=np.int32)
        cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
        color = _ZONE_COLORS.get(zone.type, (180, 180, 180))
        cv2.putText(canvas, zone.name, (cx - 40, cy), _FONT, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, zone.name, (cx - 40, cy), _FONT, 0.45, color, 1, cv2.LINE_AA)
    return canvas


class TrackOverlay:
    """track 별 중심점 궤적을 유지하면서 프레임에 오버레이를 그린다."""

    def __init__(self, draw_trail: bool = True, trail_length: int = 30) -> None:
        self.draw_trail = draw_trail
        self._trails: dict[int, deque[tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=trail_length)
        )

    def reset(self) -> None:
        self._trails.clear()

    def draw(
        self,
        image: Any,
        observations: Iterable[TrackObservation],
        header: str | None = None,
    ) -> Any:
        canvas = image.copy()

        for obs in observations:
            x1, y1, x2, y2 = (int(v) for v in obs.bbox)
            # 신원이 붙어 있으면 신원 기준으로 색과 라벨을 정한다.
            # track 이 끊겨도 같은 사람이면 색이 유지되어야 눈으로 검증할 수 있다.
            person_id = getattr(obs, "person_id", None)
            key = person_id if person_id is not None and person_id > 0 else obs.track_id
            color = (128, 128, 128) if person_id == -1 else color_for(key)

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            if person_id is None:
                label = f"ID {obs.track_id}"
            elif person_id == -1:
                label = f"? t{obs.track_id}"  # 크기 미달로 신원 보류
            else:
                label = f"P{person_id} t{obs.track_id}"
            if obs.score is not None:
                label += f" {obs.score:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, _FONT, 0.5, 1)
            cv2.rectangle(canvas, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(canvas, label, (x1 + 2, y1 - baseline - 2), _FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            if self.draw_trail:
                center = ((x1 + x2) // 2, y2)  # 발 위치 기준이 이동 궤적을 보기 좋다
                trail = self._trails[obs.track_id]
                trail.append(center)
                for i in range(1, len(trail)):
                    cv2.line(canvas, trail[i - 1], trail[i], color, 2)

        if header:
            cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (0, 0, 0), -1)
            cv2.putText(canvas, header, (8, 19), _FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        return canvas
