"""구역 정의를 읽고, 사람 bbox 와의 관계를 판정한다.

구역 좌표는 카메라 설치 위치에 100% 종속이므로 코드가 아니라 별도 파일(zones.yaml)에 둔다.
매장이 바뀌면 이 파일만 새로 쓰면 된다.

`--scale` 처럼 처리 해상도를 바꾸는 실험을 하면 픽셀 좌표가 전부 어긋나므로
0~1 상대 좌표(`normalized: true`)를 권장한다.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ..core.types import Zone, ZoneHit, ZoneType
from .geometry import contains_point, overlap_ratio


class ZoneMap:
    """구역 묶음. 프레임 크기를 알게 된 시점에 픽셀 좌표로 확정한다."""

    def __init__(self, zones: list[Zone]) -> None:
        self.zones = zones
        self._resolved: dict[str, tuple[tuple[float, float], ...]] = {}
        self._frame_size: tuple[int, int] | None = None

    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "ZoneMap":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        default_norm = bool(raw.get("normalized", False))
        zones: list[Zone] = []
        for item in raw.get("zones") or []:
            polygon = tuple((float(p[0]), float(p[1])) for p in item["polygon"])
            if len(polygon) < 3:
                raise ValueError(f"구역 '{item.get('name')}' 의 꼭짓점이 3개 미만입니다")
            zones.append(
                Zone(
                    name=item["name"],
                    type=ZoneType(item.get("type", "OTHER")),
                    polygon=polygon,
                    normalized=bool(item.get("normalized", default_norm)),
                )
            )
        return cls(zones)

    @classmethod
    def empty(cls) -> "ZoneMap":
        return cls([])

    def __len__(self) -> int:
        return len(self.zones)

    def __bool__(self) -> bool:
        return bool(self.zones)

    # ------------------------------------------------------------------

    def resolve(self, width: int, height: int) -> None:
        """상대 좌표를 이 프레임 크기의 픽셀 좌표로 확정한다."""
        if self._frame_size == (width, height):
            return
        self._frame_size = (width, height)
        self._resolved = {}
        for zone in self.zones:
            if zone.normalized:
                pts = tuple((x * width, y * height) for x, y in zone.polygon)
            else:
                pts = zone.polygon
            self._resolved[zone.name] = pts

    def polygon_of(self, name: str) -> tuple[tuple[float, float], ...]:
        return self._resolved.get(name, ())

    def resolved_polygons(self):
        """시각화용. (Zone, 픽셀 다각형) 목록."""
        return [(z, self._resolved.get(z.name, z.polygon)) for z in self.zones]

    # ------------------------------------------------------------------

    def evaluate(self, bbox: tuple[float, float, float, float]) -> list[ZoneHit]:
        """사람 bbox 와 각 구역의 관계를 계산한다. 관계가 없는 구역은 제외한다."""
        x1, y1, x2, y2 = bbox
        foot = ((x1 + x2) / 2.0, y2)  # 발 위치: 사람이 서 있는 지점
        hits: list[ZoneHit] = []
        for zone in self.zones:
            polygon = self._resolved.get(zone.name)
            if not polygon:
                continue
            overlap = overlap_ratio(polygon, bbox)
            inside = contains_point(polygon, foot[0], foot[1])
            if overlap <= 0.0 and not inside:
                continue
            hits.append(
                ZoneHit(
                    zone=zone.name,
                    zone_type=zone.type,
                    overlap=round(overlap, 4),
                    contains_foot=inside,
                )
            )
        return hits

    def summary(self) -> str:
        if not self.zones:
            return "구역 없음"
        by_type: dict[str, list[str]] = {}
        for z in self.zones:
            by_type.setdefault(z.type.value, []).append(z.name)
        parts = [f"{t} {len(names)}개({', '.join(names)})" for t, names in sorted(by_type.items())]
        return " / ".join(parts)
