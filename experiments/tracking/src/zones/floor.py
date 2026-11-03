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

from ..core.types import ZoneType
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
                 zones: list[FloorZone] | None = None) -> None:
        self.camera = camera
        self.image_points = [tuple(map(float, p)) for p in image_points]
        """화면 비율 0~1 — 처리 해상도가 바뀌어도 그대로 쓰려고."""
        self.floor_points = [tuple(map(float, p)) for p in floor_points]
        self.zones = list(zones or [])
        self._h = fit_homography(self.image_points, self.floor_points)
        self._h_inv = np.linalg.inv(self._h)

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
        return cls(raw.get("camera", "cam0"), cal.get("image") or [], cal.get("floor") or [], zones)

    def to_dict(self) -> dict:
        return {
            "camera": self.camera,
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

    def foot_on_floor(self, bbox: tuple[float, float, float, float], width: int, height: int) -> Point:
        """사람 상자(픽셀)의 발 위치(아래 가운데)를 바닥 cm 로."""
        x1, _, x2, y2 = bbox
        return self.to_floor((x1 + x2) / 2.0 / width, y2 / height)

    def zones_at(self, point: Point) -> list[FloorZone]:
        return [z for z in self.zones if contains_point(z.polygon, point[0], point[1])]

    def reprojection_error_cm(self) -> float:
        """보정점을 행렬로 옮겼을 때 실제 위치와의 평균 거리 (cm). 4점이면 0 에 가깝고, 점이 많을수록 의미가 있다."""
        errors = [np.hypot(*(np.subtract(self.to_floor(*img), flr)))
                  for img, flr in zip(self.image_points, self.floor_points)]
        return float(np.mean(errors))
