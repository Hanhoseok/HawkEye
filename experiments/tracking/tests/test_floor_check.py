"""바닥 통일 검증(scripts/floor_check.py) — 줄눈 어긋남 재기와 발목 바닥 좌표. 영상·기록 없이 만든 그림으로 시험한다."""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import floor_check as F  # noqa: E402
from src.zones.floor import FloorMap  # noqa: E402


def tiled_floor(shift_px=(0, 0), size=(720, 1040)) -> np.ndarray:
    """회색 타일 바닥을 위에서 본 그림: 60cm(240px) 간격의 어두운 줄눈. shift_px 만큼 민다."""
    img = np.full((size[1], size[0], 3), 170, np.uint8)
    rng = np.random.default_rng(0)
    img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)   # 타일 무늬 조금
    for x in range(40 + shift_px[0], size[0], 240):
        cv2.line(img, (x, 0), (x, size[1]), (95, 95, 95), 3)
    for y in range(70 + shift_px[1], size[1], 240):
        cv2.line(img, (0, y), (size[0], y), (95, 95, 95), 3)
    return img


def test_tile_offsets_measure_how_far_the_grout_lines_are_apart():
    same = [r for r in F.tile_offsets(tiled_floor(), tiled_floor()) if r["match"] >= F.MIN_MATCH]
    assert same and max(abs(r["dx"]) + abs(r["dy"]) for r in same) == 0.0
    # 전체 카메라 쪽 줄눈이 오른쪽 8px(2cm)·아래 12px(3cm) 에 있다
    moved = [r for r in F.tile_offsets(tiled_floor(), tiled_floor((8, 12))) if r["match"] >= F.MIN_MATCH]
    assert moved
    # 세로 줄만 지나는 창은 세로 어긋남을, 가로 줄만 지나는 창은 가로 어긋남을 알 수 없다 — 두 줄이 만나는 창이 정답을 낸다
    assert any(abs(r["dx"] - 2.0) <= 0.25 and abs(r["dy"] - 3.0) <= 0.25 for r in moved)
    assert all(abs(r["dx"]) <= 2.25 and abs(r["dy"]) <= 3.25 for r in moved)


def test_floor_mask_drops_furniture_and_notes():
    img = tiled_floor()
    img[100:300, 100:300] = (60, 140, 200)     # 나무 가구 (주황)
    img[400:440, 400:440] = (80, 230, 240)     # 노란 메모지
    mask = F.floor_mask(img)
    assert not mask[200, 200] and not mask[420, 420] and mask[600, 500]


class Obs:
    def __init__(self, bbox, keypoints=None):
        self.bbox, self.keypoints = bbox, keypoints


def test_ankle_floor_only_when_ankles_are_visible():
    fm = FloorMap("angle0", [(0.2, 0.5), (0.8, 0.5), (0.2, 1.0), (0.8, 1.0)], [(0, 0), (100, 0), (0, 100), (100, 100)])
    ankles = [[0.0, 0.0, 0.0]] * 15 + [[940.0, 800.0, 0.9], [980.0, 820.0, 0.9]]
    point = F.ankle_floor(fm, Obs((900.0, 300.0, 1020.0, 830.0), ankles), 1920, 1080)
    assert point == fm.to_floor(960 / 1920, 810 / 1080)
    assert F.ankle_floor(fm, Obs((900.0, 300.0, 1020.0, 830.0)), 1920, 1080) is None            # 관절 없음
    hidden = [[0.0, 0.0, 0.0]] * 15 + [[940.0, 800.0, 0.1], [980.0, 820.0, 0.05]]
    assert F.ankle_floor(fm, Obs((900.0, 300.0, 1020.0, 830.0), hidden), 1920, 1080) is None    # 발목이 가려 신뢰도 낮음
