"""실시간 상황판 — 정답 구간 읽기와 화면 붙이기."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.types import Frame  # noqa: E402
from src.viz.live_panel import WIDTH, LivePanel, parse_truth  # noqa: E402
from test_risk import at, engine, leave, person, take, AISLE  # noqa: E402


def test_parse_truth_reads_seconds():
    assert parse_truth("24-76; 80.5-90") == [(24.0, 76.0), (80.5, 90.0)]
    assert parse_truth(None) == []


def test_panel_is_attached_and_small_video_is_enlarged():
    """320x240 영상은 480 높이로 키우고 오른쪽에 상황판을 붙인다."""
    eng = engine()
    eng.update(at(1.0), [person(1, AISLE)], takes=[take(1)])
    panel = LivePanel("test.mp4", parse_truth("0-5"))
    panel.add_events(leave(eng, 1, 2.0))
    out = panel.render(np.zeros((240, 320, 3), np.uint8), Frame(0, "", 3000.0, None), eng)
    assert out.shape == (480, 640 + WIDTH, 3)
    assert len(panel.log) == 2   # WARNING, HIGH_RISK
