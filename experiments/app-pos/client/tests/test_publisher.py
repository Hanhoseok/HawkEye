"""경보 전송 클라이언트 — 어떤 탐지 방식이든 경보 서버로 경보를 보낼 수 있게 한다.

설계: ../docs/design.md §5·§6
핵심 요구: 서버가 느리거나 꺼져 있어도 호출한 쪽(영상 처리)을 멈추지 않는다.
"""

from __future__ import annotations

import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from hawkeye_client import AlertPublisher, build_event, http_post

JPEG = b"\xff\xd8\xff\xe0fake-jpeg\xff\xd9"


class Transport:
    """서버 대신 요청을 기록한다. statuses 를 차례로 돌려준다(끝나면 마지막 값 반복)."""

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


# --- 경보 만들기 ----------------------------------------------------------------

def test_build_event_fills_contract_defaults():
    e = build_event(person_id=3, level="WARNING", taken=2, paid=1)
    assert (e["person_id"], e["level"], e["taken"], e["paid"], e["unpaid"]) == (3, "WARNING", 2, 1, 1)
    for key in ("zone", "frame", "timestamp", "time_sec", "bbox", "take_frames", "reason"):
        assert key in e


def test_build_event_keeps_extra_fields_like_items():
    items = [{"name": "과자", "taken": 1, "paid": 0}]
    assert build_event(person_id=1, level="HIGH_RISK", taken=1, paid=0, items=items)["items"] == items


def test_build_event_rejects_unknown_level():
    with pytest.raises(ValueError):
        build_event(person_id=1, level="PANIC", taken=0, paid=0)


# --- 보내기 ---------------------------------------------------------------------

def test_sends_event_with_run_and_camera():
    t = Transport()
    p = publisher(t)
    p.publish(build_event(person_id=3, level="WARNING", taken=2, paid=0))
    p.close()

    url, payload, headers = t.calls[0]
    assert url == "http://alerts:8000/api/events"
    assert (payload["run_id"], payload["camera_id"]) == ("run-1", "cam1")
    assert (payload["event"]["person_id"], payload["event"]["level"]) == (3, "WARNING")
    assert "snapshot_jpeg_b64" not in payload
    assert headers["Content-Type"] == "application/json"
    assert p.sent == 1


def test_attaches_jpeg_snapshot_bytes():
    t = Transport()
    p = publisher(t)
    p.publish(build_event(person_id=3, level="WARNING", taken=1, paid=0), jpeg=JPEG)
    p.close()
    assert base64.b64decode(t.calls[0][1]["snapshot_jpeg_b64"]) == JPEG


def test_publish_does_not_wait_for_a_slow_server():
    p = publisher(Transport(delay=0.5))
    started = time.perf_counter()
    p.publish(build_event(person_id=1, level="WARNING", taken=1, paid=0))
    assert time.perf_counter() - started < 0.1
    p.close()


def test_retries_until_server_accepts():
    t = Transport(statuses=[ConnectionError("down"), 503, 201])
    p = publisher(t)
    p.publish(build_event(person_id=1, level="WARNING", taken=1, paid=0))
    p.close()
    assert len(t.calls) == 3
    assert (p.sent, p.failed) == (1, 0)


def test_client_error_is_not_retried():
    t = Transport(statuses=[422])
    p = publisher(t)
    p.publish(build_event(person_id=1, level="WARNING", taken=1, paid=0))
    p.close()
    assert len(t.calls) == 1
    assert (p.sent, p.failed) == (0, 1)


def test_gives_up_after_max_attempts():
    t = Transport(statuses=[ConnectionError("down")])
    p = publisher(t, max_attempts=3)
    p.publish(build_event(person_id=1, level="WARNING", taken=1, paid=0))
    p.close()
    assert len(t.calls) == 3
    assert p.failed == 1


def test_full_queue_drops_oldest_event():
    gate = threading.Event()
    sent = []

    def blocked(url, body, headers, timeout):
        gate.wait(5)
        sent.append(json.loads(body)["event"]["person_id"])
        return 201

    p = AlertPublisher("http://alerts", camera_id="cam1", run_id="r", transport=blocked, max_queue=2, retry_wait=0.0)
    for pid in range(1, 6):          # 1 은 전송 중, 2~5 중 마지막 2개만 대기열에 남는다
        p.publish(build_event(person_id=pid, level="WARNING", taken=1, paid=0))
        time.sleep(0.05)
    gate.set()
    p.close()
    assert sent == [1, 4, 5]
    assert p.dropped == 2


def test_close_sends_everything_still_queued():
    t = Transport(delay=0.05)
    p = publisher(t)
    for pid in range(5):
        p.publish(build_event(person_id=pid, level="WARNING", taken=1, paid=0))
    p.close()
    assert [c[1]["event"]["person_id"] for c in t.calls] == [0, 1, 2, 3, 4]


def test_run_id_is_generated_when_not_given():
    a = AlertPublisher("http://x", transport=Transport())
    b = AlertPublisher("http://x", transport=Transport())
    assert a.run_id and a.run_id != b.run_id
    a.close(); b.close()


def test_from_env_is_off_without_server_and_reads_key_from_environment():
    assert AlertPublisher.from_env({}) is None
    p = AlertPublisher.from_env({"HAWKEYE_SERVER": "http://h:8000", "HAWKEYE_CAMERA": "cam2", "HAWKEYE_API_KEY": "k"})
    try:
        assert (p.url, p.camera_id, p.api_key) == ("http://h:8000/api/events", "cam2", "k")
    finally:
        p.close()


def test_http_post_sends_json_and_api_key_to_real_server():
    received = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received["path"] = self.path
            received["key"] = self.headers.get("X-API-Key")
            received["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(201); self.end_headers()

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.handle_request, daemon=True).start()
    p = AlertPublisher(f"http://127.0.0.1:{server.server_port}", camera_id="cam1", run_id="r",
                       api_key="secret", transport=http_post)
    p.publish(build_event(person_id=1, level="WARNING", taken=1, paid=0))
    p.close()
    server.server_close()

    assert (received["path"], received["key"]) == ("/api/events", "secret")
    assert received["body"]["event"]["level"] == "WARNING"
    assert p.sent == 1
