"""PC 웹 대시보드 — 경보 서버가 같이 띄운다 (docs/design.md §8)."""

from fastapi.testclient import TestClient

from hawkeye_server.__main__ import settings_from_env
from hawkeye_server.app import Settings, create_app


def client(tmp_path, **kw):
    return TestClient(create_app(Settings(db_path=tmp_path / "x.db", snapshot_dir=tmp_path / "s", **kw)))


def test_root_serves_dashboard_page(tmp_path):
    r = client(tmp_path).get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "HawkEye" in r.text and "app.js" in r.text


def test_dashboard_script_is_served(tmp_path):
    r = client(tmp_path).get("/dashboard/app.js")
    assert r.status_code == 200
    assert "javascript" in r.headers["content-type"]


def test_dashboard_page_is_public_but_api_still_needs_key(tmp_path):
    c = client(tmp_path, api_key="secret")
    assert c.get("/").status_code == 200
    assert c.get("/dashboard/app.js").status_code == 200
    assert c.get("/api/cases").status_code == 401


def test_config_includes_browser_stream_url(tmp_path):
    c = client(tmp_path, streams=[{"camera_id": "cam1", "url": "rtsp://h:9554/cam1",
                                   "web_url": "http://127.0.0.1:8889/cam1"}])
    assert c.get("/api/config").json()["streams"][0]["web_url"] == "http://127.0.0.1:8889/cam1"


def test_web_streams_env_adds_browser_urls_to_cameras():
    s = settings_from_env({
        "HAWKEYE_STREAMS": "cam1=rtsp://h:9554/cam1,cam2=rtsp://h:9554/cam2",
        "HAWKEYE_WEB_STREAMS": "cam1=http://127.0.0.1:8889/cam1",
    })
    assert s.streams == [
        {"camera_id": "cam1", "url": "rtsp://h:9554/cam1", "web_url": "http://127.0.0.1:8889/cam1"},
        {"camera_id": "cam2", "url": "rtsp://h:9554/cam2"},
    ]


def test_web_stream_without_rtsp_camera_is_still_listed():
    s = settings_from_env({"HAWKEYE_WEB_STREAMS": "cam1=http://127.0.0.1:8889/cam1"})
    assert s.streams == [{"camera_id": "cam1", "web_url": "http://127.0.0.1:8889/cam1"}]
