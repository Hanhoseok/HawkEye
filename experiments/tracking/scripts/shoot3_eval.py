"""3차 촬영을 카메라 2대 판정(dual_eval.py --shoot 3)에 붙이는 도움 함수.

- 영상별 구역 파일: 기본 구역(zones/store01_shoot3/floor.yaml·angle0.yaml, area_setting 장면 기준)을
  outputs/sync_floor/register.csv 의 영상→area_setting 호모그래피 역으로 그 영상 화면에 옮긴다.
  바닥 cm 구역은 그대로 두고 화면 좌표(보정점·사람 아님 영역·선반 상자·레인)만 옮긴다. band_map 은 band_map.yaml 에서 합친다.
- 선반 카메라 사람 위치(band_map 입력): 상자 아래 가운데를 area_setting 좌표로 옮긴 뒤 화면 폭으로 나눈다 — 카메라가 영상마다
  움직여도 같은 자리면 같은 값 (2차는 상자 가운데 x 그대로).
- 장면 시각: 3차 angle1 은 장면을 빠뜨린다 → 장면 번호 ↔ 시각은 실제 시각 표(frame_times.read_times)를 캐시해 쓴다.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "zones" / "store01_shoot3"
BAND_MAP = BASE / "band_map.yaml"
SIZE = {"angle0": (1920, 1080), "angle1": (1920, 1080)}


def homographies(camera: str) -> dict[str, np.ndarray]:
    import shoot3_zones as Z

    return Z.load_homographies(camera=camera)


def _to_clip(points_norm, h_clip_to_ref: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """area_setting 화면 비율 점들 → 그 영상 화면 비율."""
    w, h = size
    pts = np.asarray(points_norm, float).reshape(-1, 2) * [w, h]
    out = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), np.linalg.inv(h_clip_to_ref)).reshape(-1, 2)
    return out / [w, h]


def _box(points_norm: np.ndarray) -> list[float]:
    return [round(float(v), 5) for v in (points_norm[:, 0].min(), points_norm[:, 1].min(),
                                         points_norm[:, 0].max(), points_norm[:, 1].max())]


def _rect_points(box) -> list[list[float]]:
    x1, y1, x2, y2 = box
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def load_band_map(path: Path = BAND_MAP) -> list[float]:
    if not path.exists():
        return [1.0, 0.0]
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [float(v) for v in raw.get("band_map", [1.0, 0.0])]


def save_band_map(a: float, b: float, note: str, path: Path = BAND_MAP) -> None:
    path.write_text(yaml.safe_dump({"band_map": [round(float(a), 4), round(float(b), 4)], "note": note}, allow_unicode=True,
                                   sort_keys=False), encoding="utf-8")


def pair_floor(base: dict, h1: np.ndarray) -> dict:
    out = json.loads(json.dumps(base))
    cal = out["calibration"]
    cal["image"] = [[round(float(x), 5), round(float(y), 5)] for x, y in _to_clip(cal["image"], h1, SIZE["angle1"])]
    out["ignore_image"] = [_box(_to_clip(_rect_points(b), h1, SIZE["angle1"])) for b in base.get("ignore_image") or []]
    if base.get("shelf_image"):
        out["shelf_image"] = [[[round(float(x), 5), round(float(y), 5)] for x, y in _to_clip(p, h1, SIZE["angle1"])]
                              for p in base["shelf_image"]]
    return out


def pair_detail(base: dict, h0: np.ndarray, band_map: list[float], floor0: dict | None = None) -> dict:
    """floor0 = 선반 카메라 바닥 지도(floor_angle0.yaml). 주면 보정점을 그 영상 화면으로 옮겨 `floor_calibration` 으로 싣는다 —
    두 화면 짝을 바닥 위 거리로 맺을 때 쓴다 (fusion.position_mode floor)."""
    out = json.loads(json.dumps(base))
    if floor0 and floor0.get("calibration"):
        cal = floor0["calibration"]
        out["floor_calibration"] = {
            "image": [[round(float(x), 5), round(float(y), 5)] for x, y in _to_clip(cal["image"], h0, SIZE["angle0"])],
            "floor": [[float(x), float(y)] for x, y in cal["floor"]]}
    for s in out.get("shelves", []):
        s["box"] = _box(_to_clip(_rect_points(s["box"]), h0, SIZE["angle0"]))
        s.pop("polygon", None)
    for z in out.get("zones", []):
        z["polygon"] = [[round(float(x), 5), round(float(y), 5)] for x, y in _to_clip(z["polygon"], h0, SIZE["angle0"])]
    out["band_map"] = band_map
    out.pop("reference", None)
    return out


def make_pair_zones(pair: str, out_root: Path, base_dir: Path = BASE) -> Path:
    """outputs/.../zones/<짝>/{floor,angle0}.yaml 을 기본 파일에서 늘 새로 만든다 (band_map 최신 반영). 호모그래피 없으면 KeyError."""
    h1, h0 = homographies("angle1")[pair], homographies("angle0")[pair]
    floor = yaml.safe_load((base_dir / "floor.yaml").read_text(encoding="utf-8"))
    detail = yaml.safe_load((base_dir / "angle0.yaml").read_text(encoding="utf-8"))
    floor0_path = base_dir / "floor_angle0.yaml"
    floor0 = yaml.safe_load(floor0_path.read_text(encoding="utf-8")) if floor0_path.exists() else None
    d = out_root / pair
    d.mkdir(parents=True, exist_ok=True)
    (d / "floor.yaml").write_text(yaml.safe_dump(pair_floor(floor, h1), allow_unicode=True, sort_keys=False), encoding="utf-8")
    (d / "angle0.yaml").write_text(yaml.safe_dump(pair_detail(detail, h0, load_band_map(base_dir / "band_map.yaml"), floor0),
                                                  allow_unicode=True, sort_keys=False), encoding="utf-8")
    return d


def detail_point(h0: np.ndarray, size: tuple[int, int] = SIZE["angle0"]):
    """선반 카메라 사람 상자(픽셀, 처리 해상도 기준 width) → area_setting 화면 비율 x."""
    def point(bbox, width: float) -> float:
        scale = size[0] / float(width)
        x = (bbox[0] + bbox[2]) / 2.0 * scale
        y = bbox[3] * scale
        p = cv2.perspectiveTransform(np.array([[[x, y]]], float), h0).reshape(2)
        return float(p[0] / size[0])
    return point


def cached_times(video: Path, cache: Path, frames: int | None = None) -> list[float]:
    """실제 장면 시각(ms) — 한 번 읽어 cache(json)에 둔다. frames 를 주면 장면 수가 같은지 확인."""
    if cache.exists():
        times = json.loads(cache.read_text(encoding="utf-8"))
    else:
        from src.inputs.frame_times import read_times

        times = [round(float(t), 3) for t in read_times(video)]
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(times), encoding="utf-8")
    if frames is not None and len(times) != frames:
        # 기록 장면 수(meta frames)는 OpenCV CAP_PROP_FRAME_COUNT — 실제로 읽히는 장면보다 1~2장 많게 나올 때가 있다 (3차 N11: 751 vs 750).
        # 그 정도는 마지막 간격으로 늘려 맞추고(없는 장면이라 기록에도 없다), 더 크면 오류.
        if 0 < frames - len(times) <= MAX_COUNT_GAP and len(times) >= 2:
            gap = times[-1] - times[-2]
            times = times + [round(times[-1] + gap * (k + 1), 3) for k in range(frames - len(times))]
        else:
            raise ValueError(f"{video.name}: 실제 장면 {len(times)} ≠ 기록 장면 수 {frames}")
    return times


MAX_COUNT_GAP = 2
"""기록 장면 수가 실제보다 이만큼까지 많으면 허용 (OpenCV 장면 수 어림)."""


def load_sensor(path: Path, offset_ms: float, lanes: tuple[str, ...] | None = None):
    """3차 실제 센서 기록(labels/store01_shoot3/sensor/<짝>.csv, angle0 시각) → 판정 입력(SensorEvent, 공통 = angle1 시각).

    - 영상 안(in_video=1) 기록만 — 영상 끝 뒤의 채워 넣기(ADD)가 마지막 손님에게 '놓음'으로 붙지 않게.
    - REMOVE·ADD 만 (RESYNC 는 개수 변화 0).
    - delta_g 에 개수 변화를 그대로 넣는다 (칸 무게 sensor_unit_g 가 없는 칸은 기록 하나 = 1개로 셈).
    - lanes 를 주면 그 레인만 (예: L3 센서는 못 믿어서 L1·L2 만).
    """
    import csv

    from src.fusion.shelf_events import SensorEvent

    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if str(r.get("in_video", "1")).strip() != "1" or r["event"] not in ("REMOVE", "ADD"):
                continue
            if lanes and r["lane"] not in lanes:
                continue
            out.append(SensorEvent(float(r["time_sec"]) * 1000.0 + offset_ms, r["lane"], float(r["delta"]), 1.0))
    return out
