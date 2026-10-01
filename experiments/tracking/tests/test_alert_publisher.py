"""경보 전송기 — 위험 판정을 관리자 앱의 경보 서버로 보낸다.

설계: experiments/app-pos/docs/design.md §5·§6
핵심 요구: 서버가 느리거나 꺼져 있어도 영상 처리를 멈추지 않는다.
"""

from __future__ import annotations

import base64
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import AppConfig  # noqa: E402
from src.core.types import Detection, Frame, RiskEvent, RiskLevel  # noqa: E402
from src.pipeline import TrackingPipeline  # noqa: E402
from src.sinks.alert_publisher import AlertPublisher, http_post  # noqa: E402
from src.trackers.bytetrack_tracker import ByteTrackTracker  # noqa: E402


def make_event(level=RiskLevel.WARNING, person_id=3, frame=10) -> RiskEvent:
    return RiskEvent(
        person_id=person_id, level=level, zone="EXIT", frame=frame,
        timestamp="00:00:01.000", time_sec=1.0, bbox=(10.0, 20.0, 60.0, 120.0),
        taken=2, paid=0, unpaid=2, take_frames=((5, 8),), reason="집기 2, 결제 0",
    )


class Transport:
    """서버 대신 요청을 기록한다. status 목록을 차례로 돌려준다(끝나면 마지막 값 반복)."""

    def __init__(self, statuses=(201,), delay=0.0):
        self.statuses = list(statuses)
        self.delay = delay
        self.calls: list[tuple[str, dict, dict]] = []

    def __call__(self, url, body, headers, timeout):
        time.sleep(self.delay)
        self.calls.append((url, json.loads(body), headers))
        status = self.statuses[min(len(self.calls), len(self.statuses)) - 1]
        if isinstance(status, Exception):
            raise status
        return status


def publisher(transport, **kw) -> AlertPublisher:
    kw.setdefault("retry_wait", 0.0)
    return AlertPublisher("http://alerts:8000", camera_id="cam1", run_id="run-1", transport=transport, **kw)


def test_sends_event_with_run_and_camera():
    t = Transport()
    p = publisher(t)
    p.publish(make_event())
    p.close()

    url, payload, _ = t.calls[0]
    assert url == "http://alerts:8000/api/events"
    assert (payload["run_id"], payload["camera_id"]) == ("run-1", "cam1")
    assert (payload["event"]["person_id"], payload["event"]["level"]) == (3, "WARNING")
    assert "snapshot_jpeg_b64" not in payload
    assert p.sent == 1


def test_attaches_jpeg_snapshot_when_image_given():
    t = Transport()
    p = publisher(t)
    p.publish(make_event(), np.zeros((240, 320, 3), dtype=np.uint8))
    p.close()

    jpeg = base64.b64decode(t.calls[0][1]["snapshot_jpeg_b64"])
    assert jpeg[:2] == b"\xff\xd8"
    assert cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape == (240, 320, 3)


def test_publish_does_not_wait_for_a_slow_server():
    p = publisher(Transport(delay=0.5))
    started = time.perf_counter()
    p.publish(make_event())
    assert time.perf_counter() - started < 0.1
    p.close()


def test_retries_until_server_accepts():
    t = Transport(statuses=[ConnectionError("down"), 503, 201])
    p = publisher(t)
    p.publish(make_event())
    p.close()
    assert len(t.calls) == 3
    assert (p.sent, p.failed) == (1, 0)


def test_client_error_is_not_retried():
    t = Transport(statuses=[422])
    p = publisher(t)
    p.publish(make_event())
    p.close()
    assert len(t.calls) == 1
    assert (p.sent, p.failed) == (0, 1)


def test_gives_up_after_max_attempts():
    t = Transport(statuses=[ConnectionError("down")])
    p = publisher(t, max_attempts=3)
    p.publish(make_event())
    p.close()
    assert len(t.calls) == 3
    assert p.failed == 1


def test_full_queue_drops_oldest_event():
    gate = threading.Event()

    def blocked(url, body, headers, timeout):
        gate.wait(5)
        blocked.sent.append(json.loads(body)["event"]["person_id"])
        return 201

    blocked.sent = []
    p = AlertPublisher("http://alerts", camera_id="cam1", run_id="r", transport=blocked, max_queue=2, retry_wait=0.0)
    for pid in range(1, 6):         # 1 은 전송 중, 2~5 중 마지막 2개만 대기열에 남는다
        p.publish(make_event(person_id=pid))
        time.sleep(0.05)
    gate.set()
    p.close()
    assert blocked.sent == [1, 4, 5]
    assert p.dropped == 2


def test_close_sends_everything_still_queued():
    t = Transport(delay=0.05)
    p = publisher(t)
    for pid in range(5):
        p.publish(make_event(person_id=pid))
    p.close()
    assert [c[1]["event"]["person_id"] for c in t.calls] == [0, 1, 2, 3, 4]


def test_run_id_is_generated_when_not_given():
    a = AlertPublisher("http://x", transport=Transport())
    b = AlertPublisher("http://x", transport=Transport())
    assert a.run_id and a.run_id != b.run_id
    a.close(); b.close()


def test_http_post_sends_json_and_api_key_to_real_server():
    received = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received["path"] = self.path
            received["key"] = self.headers.get("X-API-Key")
            received["type"] = self.headers.get("Content-Type")
            received["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(201); self.end_headers()

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.handle_request, daemon=True).start()
    p = AlertPublisher(f"http://127.0.0.1:{server.server_port}", camera_id="cam1", run_id="r",
                       api_key="secret", transport=http_post)
    p.publish(make_event())
    p.close()
    server.server_close()

    assert received["path"] == "/api/events"
    assert received["key"] == "secret"
    assert received["type"] == "application/json"
    assert received["body"]["event"]["level"] == "WARNING"
    assert p.sent == 1


# --- 파이프라인 연결 ---------------------------------------------------------------

class OnePerson:
    def detect(self, frame: Frame) -> list[Detection]:
        return [Detection(bbox=(40.0, 40.0, 100.0, 200.0), score=0.9, class_id=0, class_name="person")]


class FakeRegistry:
    merges: list = []

    def reset(self):
        pass

    def assign(self, frame, observations):
        return []


class FakeRisk:
    """프레임 3 에서 WARNING 하나를 낸다."""

    def reset(self):
        self.events = []

    def update(self, frame, identities, takes, merges):
        if frame.index == 3:
            e = make_event(frame=3)
            self.events.append(e)
            return [e], []
        return [], []

    def add_late_takes(self, takes, now_ms):
        pass

    def summary(self):
        return ""


class Recorder:
    def __init__(self):
        self.published, self.closed = [], False

    def publish(self, event, image=None):
        self.published.append((event, image))

    def close(self):
        self.closed = True


def test_pipeline_publishes_risk_events_with_frame_and_closes(tmp_path):
    video = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for _ in range(8):
        w.write(np.zeros((240, 320, 3), dtype=np.uint8))
    w.release()

    config = AppConfig()
    config.video.source = str(video)
    config.output.dir = str(tmp_path / "out")
    config.output.write_video = False
    alerts = Recorder()

    TrackingPipeline(config, OnePerson(), ByteTrackTracker(config.tracker),
                     registry=FakeRegistry(), risk=FakeRisk(), alerts=alerts).run()

    assert [(e.level, e.frame) for e, _ in alerts.published] == [(RiskLevel.WARNING, 3)]
    assert alerts.published[0][1].shape == (240, 320, 3)
    assert alerts.closed


# --- 실행 옵션 -------------------------------------------------------------------

def test_from_options_is_off_without_server():
    assert AlertPublisher.from_options(None, "cam1", {}) is None


def test_from_options_reads_api_key_from_environment_only():
    p = AlertPublisher.from_options("http://h:8000", "cam2", {"HAWKEYE_API_KEY": "k"})
    try:
        assert (p.url, p.camera_id, p.api_key) == ("http://h:8000/api/events", "cam2", "k")
    finally:
        p.close()
