"""시뮬레이터 — 파이프라인 없이 시나리오 경보를 보내 앱·서버를 시험·시연한다.

각 시나리오를 실제 서버 코드(../server)에 보내서 사건이 의도한 상태가 되는지 확인한다.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hawkeye_client import AlertPublisher
from hawkeye_client.simulate import SCENARIOS, run_scenario
from hawkeye_server.app import Settings, create_app


@pytest.fixture
def server(tmp_path: Path):
    return TestClient(create_app(Settings(db_path=tmp_path / "x.db", snapshot_dir=tmp_path / "snap")))


def via(client: TestClient):
    """AlertPublisher 의 전송을 실제 HTTP 대신 서버 앱으로 바로 넘긴다."""

    def transport(url, body, headers, timeout):
        return client.post("/api/events", content=body, headers=headers).status_code

    return transport


def case_of(server: TestClient, person_id: int) -> dict:
    return next(c for c in server.get("/api/cases").json() if c["person_id"] == person_id)


@pytest.mark.parametrize(
    "name, level, state",
    [
        ("theft", "HIGH_RISK", "active"),        # 출구 접근 → 미결제 퇴장
        ("partial", "REVIEW", "active"),          # 일부만 결제 → 확인 필요
        ("paid", "CLEAR", "pass"),                # 결제하고 통과 → 알림 없음
        ("came_back", "CLEAR", "auto_cleared"),   # 출구 접근했다 되돌아가 결제 → 자동 해제
    ],
)
def test_scenario_ends_in_expected_case_state(server, name, level, state):
    p = AlertPublisher("http://test", camera_id="cam1", run_id="sim", transport=via(server), retry_wait=0.0)
    pid = run_scenario(p, name, person_id=7, delay=0.0)
    p.close()
    c = case_of(server, pid)
    assert (c["level"], c["state"]) == (level, state)


def test_every_scenario_is_documented():
    for name, s in SCENARIOS.items():
        assert s.description, name
        assert s.steps, name


def test_snapshot_is_attached_to_alarm_steps(server, tmp_path):
    jpeg = b"\xff\xd8sim\xff\xd9"
    p = AlertPublisher("http://test", camera_id="cam1", run_id="sim", transport=via(server), retry_wait=0.0)
    pid = run_scenario(p, "theft", person_id=9, delay=0.0, jpeg=jpeg)
    p.close()
    c = case_of(server, pid)
    assert c["has_snapshot"] is True
    assert server.get(f"/api/events/{c['last_event_id']}/snapshot.jpg").content == jpeg


def test_cli_runs_on_korean_windows_console(monkeypatch):
    """한국어 Windows 터미널(cp949)은 '—' 같은 문자를 못 찍는다. 그래도 죽지 않아야 한다."""
    import io
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from hawkeye_client import simulate

    class Ok(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(201); self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Ok)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    out = io.TextIOWrapper(io.BytesIO(), encoding="cp949")
    monkeypatch.setattr("sys.stdout", out)
    try:
        code = simulate.main(["--server", f"http://127.0.0.1:{srv.server_port}", "--scenario", "all", "--delay", "0"])
    finally:
        srv.shutdown(); srv.server_close()
    assert code == 0
