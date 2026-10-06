"""경보 전송 연결 테스트 — 위험 판정이 경보 서버 형식(app-pos/docs/design.md §5)대로 나가는가."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.types import RiskEvent, RiskLevel  # noqa: E402
from src.sinks.alert_sink import AlertSink, snapshot_jpeg  # noqa: E402

REQUIRED = ("person_id", "level", "zone", "frame", "timestamp", "time_sec", "bbox",
            "taken", "paid", "unpaid", "take_frames", "reason", "identity_check")


def event(level=RiskLevel.HIGH_RISK, **kw) -> RiskEvent:
    base = dict(person_id=3, level=level, zone="door", frame=120, timestamp="2026-10-06T10:00:04",
                time_sec=4.0, bbox=(40.0, 30.0, 120.0, 200.0), taken=2, paid=0, unpaid=2,
                take_frames=((10, 30), (50, 70)), reason="집기 행동 2번, 결제 없음 — 문 밖으로 나감",
                identity_check=0.86)
    base.update(kw)
    return RiskEvent(**base)


class FakePublisher:
    def __init__(self):
        self.published = []
        self.closed = False
        self.sent = self.failed = self.dropped = 0
        self.run_id = "test-run"

    def publish(self, event, jpeg=None):
        self.published.append((event, jpeg))
        self.sent += 1

    def close(self):
        self.closed = True


def test_event_has_every_field_the_server_requires():
    payload = event().to_dict()
    assert all(k in payload for k in REQUIRED)
    assert payload["level"] == "HIGH_RISK"
    json.dumps(payload, ensure_ascii=False)          # 그대로 JSON 으로 보낼 수 있어야 한다


def test_send_attaches_snapshot_with_box():
    pub = FakePublisher()
    sink = AlertSink("http://x", publisher=pub)
    image = np.zeros((240, 320, 3), np.uint8)
    sink.send(event(), image)
    payload, jpeg = pub.published[0]
    assert payload["person_id"] == 3 and payload["level"] == "HIGH_RISK"
    decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (240, 320, 3)
    assert decoded[30, 80].sum() > 0, "손님 상자가 그려져 있어야 한다"


def test_snapshot_scales_box_for_reduced_image():
    """줄여 둔 장면이면 상자 좌표도 같은 비율로 줄인다."""
    image = np.zeros((120, 160, 3), np.uint8)
    jpeg = snapshot_jpeg(image, event(bbox=(40.0, 30.0, 120.0, 200.0)), scale=0.5)
    decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert decoded[15, 40].sum() > 0 and decoded[100, 150].sum() == 0


def test_no_server_means_no_sink():
    assert AlertSink.create(None, None, "../app-pos/client", env={}) is None


def test_close_reports_counts():
    pub = FakePublisher()
    sink = AlertSink("http://127.0.0.1:8000", publisher=pub)
    sink.send(event(RiskLevel.WARNING))
    line = sink.close()
    assert pub.closed and "1건" in line


def test_real_client_module_is_found():
    """app-pos 전송 모듈을 실제로 불러올 수 있다 (폴더 위치 설정이 맞다)."""
    client = ROOT.parent / "app-pos" / "client"
    if not client.exists():
        pytest.skip("app-pos 폴더가 없다")
    sink = AlertSink("http://127.0.0.1:9", client_path=client)     # 9번 포트: 아무도 안 듣는다
    assert sink.run_id
    sink.publisher.max_attempts = 1
    sink.close()


def test_payload_passes_server_validation():
    """서버의 입력 모델(EventBody)을 그대로 통과한다 — 서버 의존성(pydantic)이 있을 때만."""
    pytest.importorskip("pydantic")
    server = ROOT.parent / "app-pos" / "server"
    sys.path.insert(0, str(server))
    try:
        from hawkeye_server.app import EventBody
    except ImportError:
        pytest.skip("서버 의존성(fastapi)이 없다")
    body = {"run_id": "r1", "camera_id": "cam1", "event": event().to_dict()}
    parsed = EventBody.model_validate(json.loads(json.dumps(body)))
    assert parsed.event.level == "HIGH_RISK" and parsed.event.take_frames == [[10, 30], [50, 70]]
