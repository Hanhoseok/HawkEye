"""바닥 지도: 높은 카메라(angle1) 화면의 발 위치를 매장 바닥 좌표(cm)로 옮긴다.

화면 구역(zone_map.py)은 카메라 한 대의 화면 좌표라 카메라가 둘이면 서로 맞출 수 없고(limitations B-6),
눈높이 카메라는 발이 안 보여 계산대·출구를 좌우 위치로만 갈랐다(B-9).
바닥(z=0) 위의 점은 3×3 행렬(호모그래피) 하나로 화면과 오간다 (MVDet 식 2, docs/multiview-papers.md P2).
그래서 발이 보이는 카메라 한 대에서 바닥 위 점 4개 이상의 화면 위치와 실제 위치를 짝지어 행렬을 구하고,
계산대·출구·문 밖·선반 앞 구역을 바닥 좌표 다각형으로 정의한다.

발이 안 보이는 카메라(선반 카메라 angle0)의 사람은 여기로 옮기지 않는다 (docs/multiview-design.md M-3).

파일 형식 (zones/<매장>/floor.yaml):

    camera: angle1
    calibration:            # 같은 순서로 짝지은 점들 (4개 이상)
      image: [[0.31, 0.62], ...]   # 화면 비율 0~1
      floor: [[0, 0], ...]         # 바닥 cm
    zones:
      - {name: counter, type: CHECKOUT, polygon: [[...], ...]}   # 바닥 cm
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from ..core.types import Zone, ZoneHit, ZoneType
from .geometry import contains_point

Point = tuple[float, float]


def fit_homography(src: list[Point], dst: list[Point]) -> np.ndarray:
    """src → dst 3×3 호모그래피 (정규화한 DLT, 점 4개 이상이면 최소제곱).

    cv2.findHomography 와 같은 일을 numpy 로 한다 — 구역 계층이 영상 라이브러리에 묶이지 않게.
    """
    if len(src) != len(dst) or len(src) < 4:
        raise ValueError(f"짝지은 점이 4개 이상 필요합니다 (지금 {len(src)}개 / {len(dst)}개)")
    a, ta = _normalize(np.asarray(src, dtype=float))
    b, tb = _normalize(np.asarray(dst, dtype=float))
    rows = []
    for (x, y), (u, v) in zip(a, b):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, s, vt = np.linalg.svd(np.asarray(rows))
    if s[-2] < 1e-9:
        raise ValueError("점들이 한 줄 위에 있어 바닥 평면을 정할 수 없습니다")
    h = vt[-1].reshape(3, 3)
    h = np.linalg.inv(tb) @ h @ ta
    return h / h[2, 2]


def _normalize(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """평균 0, 평균 거리 √2 로 옮긴다 (DLT 수치 안정)."""
    center = pts.mean(axis=0)
    dist = np.sqrt(((pts - center) ** 2).sum(axis=1)).mean()
    scale = np.sqrt(2) / dist if dist > 0 else 1.0
    t = np.array([[scale, 0, -scale * center[0]], [0, scale, -scale * center[1]], [0, 0, 1]])
    moved = (t @ np.c_[pts, np.ones(len(pts))].T).T
    return moved[:, :2], t


def foot_pixel(bbox: tuple[float, float, float, float], keypoints=None, ankle_conf: float = 0.3) -> tuple[Point, bool]:
    """(발 위치 픽셀, 발목에서 얻었나). keypoints 가 있으면 COCO 15·16 발목 중 신뢰도 ankle_conf 이상이고 (0, 0) 이 아니며
    상자 안(높이 5% 여유)인 것의 가운데, 없으면 상자 아래 가운데."""
    x1, y1, x2, y2 = bbox
    if keypoints is not None and len(keypoints) >= 17:
        pad = 0.05 * (y2 - y1)
        pts = [(k[0], k[1]) for k in keypoints[15:17]
               if k[2] >= ankle_conf and not (k[0] == 0 and k[1] == 0)
               and x1 - pad <= k[0] <= x2 + pad and y1 - pad <= k[1] <= y2 + pad]
        if pts:
            return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)), True
    return ((x1 + x2) / 2.0, y2), False


def _w(h: np.ndarray, x: float, y: float) -> float:
    return float((h @ np.array([x, y, 1.0]))[2])


def _side_ref(h: np.ndarray, points) -> tuple[float, float]:
    """보정점들의 w 부호와 |w| 최소값."""
    ws = [_w(h, x, y) for x, y in points]
    return (1.0 if ws[0] > 0 else -1.0), min(abs(w) for w in ws)


def _same_side(h: np.ndarray, ref: tuple[float, float], x: float, y: float) -> bool:
    sign, wmin = ref
    w = _w(h, x, y)
    return w * sign > 0 and abs(w) >= 0.05 * wmin


def apply_homography(h: np.ndarray, x: float, y: float) -> Point:
    u, v, w = h @ np.array([x, y, 1.0])
    if abs(w) < 1e-12:
        raise ValueError("이 점은 바닥 평면 밖(지평선 너머)입니다")
    return float(u / w), float(v / w)


@dataclass(frozen=True)
class FloorZone:
    name: str
    type: ZoneType
    polygon: tuple[Point, ...]
    """바닥 cm."""


class FloorMap:
    """한 카메라의 화면 ↔ 바닥 대응과 바닥 구역."""

    def __init__(self, camera: str, image_points: list[Point], floor_points: list[Point],
                 zones: list[FloorZone] | None = None, band: tuple[float, float] | None = None,
                 note: str = "") -> None:
        self.camera = camera
        self.image_points = [tuple(map(float, p)) for p in image_points]
        """화면 비율 0~1 — 처리 해상도가 바뀌어도 그대로 쓰려고."""
        self.floor_points = [tuple(map(float, p)) for p in floor_points]
        self.zones = list(zones or [])
        self.band = band
        """선반 앞 띠 축: 바닥 y(통로 방향)의 (시작, 끝) cm. 교차 화면 연결의 공통 축 (src/fusion/cross_view.py)."""
        self.note = note
        self.foot_mode = "bbox"
        """구역 판단의 발 점: "bbox"(상자 아래 가운데) 또는 "ankle"(발목, 없으면 상자). config zones.foot_point."""
        self.ankle_conf = 0.3
        self._h = fit_homography(self.image_points, self.floor_points)
        self._h_inv = np.linalg.inv(self._h)
        self._side_img = _side_ref(self._h, self.image_points)
        self._side_floor = _side_ref(self._h_inv, self.floor_points)

    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "FloorMap":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        cal = raw.get("calibration") or {}
        zones = []
        for item in raw.get("zones") or []:
            polygon = tuple((float(p[0]), float(p[1])) for p in item["polygon"])
            if len(polygon) < 3:
                raise ValueError(f"바닥 구역 '{item.get('name')}' 의 꼭짓점이 3개 미만입니다")
            zones.append(FloorZone(item["name"], ZoneType(item.get("type", "OTHER")), polygon))
        band = raw.get("band")
        return cls(raw.get("camera", "cam0"), cal.get("image") or [], cal.get("floor") or [], zones,
                   (float(band["start"]), float(band["end"])) if band else None, raw.get("note", ""))

    def to_dict(self) -> dict:
        out = {"camera": self.camera}
        if self.note:
            out["note"] = self.note
        if self.band:
            out["band"] = {"start": self.band[0], "end": self.band[1]}
        return out | {
            "calibration": {
                "image": [[round(x, 5), round(y, 5)] for x, y in self.image_points],
                "floor": [[round(x, 1), round(y, 1)] for x, y in self.floor_points],
            },
            "zones": [{"name": z.name, "type": z.type.value, "polygon": [list(p) for p in z.polygon]}
                      for z in self.zones],
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8")

    # ------------------------------------------------------------------

    def to_floor(self, x: float, y: float) -> Point:
        """화면 비율 (x, y) → 바닥 cm."""
        return apply_homography(self._h, x, y)

    def to_image(self, fx: float, fy: float) -> Point:
        """바닥 cm → 화면 비율. 확인 그림(격자)을 그릴 때 쓴다."""
        return apply_homography(self._h_inv, fx, fy)

    def foot_on_floor(self, bbox: tuple[float, float, float, float], width: int, height: int,
                      keypoints=None) -> Point:
        """사람의 발 위치를 바닥 cm 로. keypoints 를 주고 foot_mode 가 "ankle" 이면 발목, 아니면 상자 아래 가운데."""
        x, y = self.foot_pixel(bbox, keypoints)
        return self.to_floor(x / width, y / height)

    def foot_pixel(self, bbox: tuple[float, float, float, float], keypoints=None) -> Point:
        """발 위치 (픽셀). 발목: COCO 15·16 중 신뢰도 ankle_conf 이상이고 (0, 0) 이 아니며 상자 안(높이 5% 여유)인 것의 가운데."""
        return foot_pixel(bbox, keypoints if self.foot_mode == "ankle" else None, self.ankle_conf)[0]

    def on_floor_side(self, x: float, y: float) -> bool:
        """화면 비율 (x, y) 가 보정점과 같은 쪽(지평선 이쪽)인가. 호모그래피의 w 부호는 정규화에 따라 아무 값이나
        될 수 있으므로 보정점의 w 부호를 기준으로 삼고, |w| 가 보정점 최소값의 5% 미만(지평선 근처)도 뺀다."""
        return _same_side(self._h, self._side_img, x, y)

    def image_side_ok(self, fx: float, fy: float) -> bool:
        """바닥 cm (fx, fy) 가 이 카메라 앞쪽(화면에 투영 가능한 쪽)인가."""
        return _same_side(self._h_inv, self._side_floor, fx, fy)


    def band_position(self, point: Point) -> float | None:
        """바닥 점의 선반 앞 띠 위치 (0 = 시작, 1 = 끝). 띠 축이 없으면 None."""
        if not self.band:
            return None
        start, end = self.band
        return (point[1] - start) / (end - start)

    def zones_at(self, point: Point) -> list[FloorZone]:
        return [z for z in self.zones if contains_point(z.polygon, point[0], point[1])]

    def image_polygon(self, zone: FloorZone, width: int, height: int) -> tuple[Point, ...]:
        """바닥 구역을 화면 픽셀 다각형으로 (그리기·자세 계산용)."""
        pts = []
        for fx, fy in zone.polygon:
            u, v = self.to_image(fx, fy)
            pts.append((u * width, v * height))
        return tuple(pts)

    def reprojection_error_cm(self) -> float:
        """보정점을 행렬로 옮겼을 때 실제 위치와의 평균 거리 (cm). 4점이면 0 에 가깝고, 점이 많을수록 의미가 있다."""
        errors = [np.hypot(*(np.subtract(self.to_floor(*img), flr)))
                  for img, flr in zip(self.image_points, self.floor_points)]
        return float(np.mean(errors))


def shelf_image_zones(raw: dict) -> "ZoneMap | None":
    """floor.yaml 의 `shelf_image`(전체 카메라 화면의 선반 영역, 화면 비율 다각형들) → 화면 구역. 없으면 None."""
    from .zone_map import ZoneMap

    polys = raw.get("shelf_image") or []
    if not polys:
        return None
    return ZoneMap([Zone(name=f"shelf_image{i + 1}", type=ZoneType.SHELF,
                         polygon=tuple((float(x), float(y)) for x, y in poly), normalized=True)
                    for i, poly in enumerate(polys)])


class FloorZoneMap:
    """바닥 구역을 화면 구역(ZoneMap)과 같은 방식으로 쓰게 하는 어댑터.

    손 뻗기·손님 장부·위험 판정은 `evaluate(bbox)` 가 돌려주는 ZoneHit 만 본다. 그래서 이 어댑터를 끼우면
    판단 규칙을 바꾸지 않고 구역 판단만 '화면 좌표'에서 '발의 바닥 좌표'로 바뀐다 (design M-3, 1-2).

    - 발 위치(상자 아래 가운데)를 바닥으로 옮겨, 그 점을 품은 바닥 구역마다 ZoneHit 를 낸다.
    - 바닥 구역에는 '겹침 비율'이 없다. 발이 구역 안이면 overlap 1.0, 밖이면 내지 않는다.
      선반 구역도 '선반 앞 띠'(바닥)로 정의하므로, 발이 띠 안 = 선반 앞에 섰다.
    """

    def __init__(self, floor: FloorMap) -> None:
        self.floor = floor
        self.zones = [Zone(z.name, z.type, z.polygon, normalized=False) for z in floor.zones]
        self._size: tuple[int, int] | None = None
        self._image: dict[str, tuple[Point, ...]] = {}

    def __len__(self) -> int:
        return len(self.zones)

    def __bool__(self) -> bool:
        return bool(self.zones)

    def resolve(self, width: int, height: int) -> None:
        if self._size == (width, height):
            return
        self._size = (width, height)
        self._image = {z.name: self.floor.image_polygon(z, width, height) for z in self.floor.zones}

    def polygon_of(self, name: str) -> tuple[Point, ...]:
        return self._image.get(name, ())

    def resolved_polygons(self):
        return [(z, self._image.get(z.name, ())) for z in self.zones]

    def evaluate(self, bbox: tuple[float, float, float, float], keypoints=None) -> list[ZoneHit]:
        if self._size is None:
            return []
        point = self.floor.foot_on_floor(bbox, *self._size, keypoints)
        return [ZoneHit(zone=z.name, zone_type=z.type, overlap=1.0, contains_foot=True)
                for z in self.floor.zones_at(point)]

    def summary(self) -> str:
        if not self.zones:
            return "바닥 구역 없음"
        return "바닥 " + " / ".join(f"{z.type.value} {z.name}" for z in self.zones)
