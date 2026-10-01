"""경보 서버 API (docs/design.md §5)."""

import base64

import pytest
from fastapi.testclient import TestClient

from hawkeye_server.app import Settings, create_app

JPEG = b"\xff\xd8\xff\xe0fake-jpeg-bytes\xff\xd9"


def risk_event(level="WARNING", person_id=3, **over):
    e = {
        "person_id": person_id, "level": level, "zone": "EXIT", "frame": 1043,
        "timestamp": "00:00:34.766", "time_sec": 34.766, "bbox": [610, 280, 780, 700],
        "taken": 2, "paid": 0, "unpaid": 2, "take_frames": [[812, 840]],
        "reason": "집기 2회, 결제 0", "identity_check": 0.82,
    }
    e.update(over)
    return e


def body(level="WARNING", person_id=3, run_id="run-a", snapshot=None, **over):
    b = {"run_id": run_id, "camera_id": "cam1", "event": risk_event(level, person_id, **over)}
    if snapshot is not None:
        b["snapshot_jpeg_b64"] = base64.b64encode(snapshot).decode()
    return b


@pytest.fixture
def make_client(tmp_path):
    def _make(**settings):
        s = Settings(db_path=tmp_path / "hawkeye.db", snapshot_dir=tmp_path / "snapshots", **settings)
        return TestClient(create_app(s))
    return _make


@pytest.fixture
def client(make_client):
    return make_client()


def test_event_creates_case_that_appears_in_list(client):
    r = client.post("/api/events", json=body("WARNING"))
    assert r.status_code == 201
    assert r.json()["notify"] is True

    cases = client.get("/api/cases").json()
    assert len(cases) == 1
    c = cases[0]
    assert (c["person_id"], c["level"], c["state"]) == (3, "WARNING", "active")
    assert (c["taken"], c["paid"], c["unpaid"]) == (2, 0, 2)


def test_escalation_updates_the_same_case_and_keeps_history(client):
    first = client.post("/api/events", json=body("WARNING")).json()
    second = client.post("/api/events", json=body("HIGH_RISK")).json()
    assert first["case_id"] == second["case_id"]
    assert second["notify"] is True

    detail = client.get(f"/api/cases/{first['case_id']}").json()
    assert detail["level"] == "HIGH_RISK"
    assert [e["level"] for e in detail["events"]] == ["WARNING", "HIGH_RISK"]


def test_same_person_in_another_run_is_a_different_case(client):
    a = client.post("/api/events", json=body(run_id="run-a")).json()
    b = client.post("/api/events", json=body(run_id="run-b")).json()
    assert a["case_id"] != b["case_id"]


def test_snapshot_is_stored_and_served(client):
    r = client.post("/api/events", json=body(snapshot=JPEG)).json()
    detail = client.get(f"/api/cases/{r['case_id']}").json()
    event_id = detail["events"][0]["id"]
    assert detail["events"][0]["has_snapshot"] is True

    img = client.get(f"/api/events/{event_id}/snapshot.jpg")
    assert img.status_code == 200
    assert img.headers["content-type"] == "image/jpeg"
    assert img.content == JPEG


def test_event_without_snapshot_has_no_image(client):
    r = client.post("/api/events", json=body()).json()
    event_id = client.get(f"/api/cases/{r['case_id']}").json()["events"][0]["id"]
    assert client.get(f"/api/events/{event_id}/snapshot.jpg").status_code == 404


def test_invalid_level_is_rejected_and_not_stored(client):
    assert client.post("/api/events", json=body("PANIC")).status_code == 422
    assert client.get("/api/cases").json() == []


def test_missing_event_fields_are_rejected(client):
    b = body()
    del b["event"]["person_id"]
    assert client.post("/api/events", json=b).status_code == 422


def test_items_are_kept_for_item_level_display(client):
    items = [{"name": "과자", "taken": 2, "paid": 1}]
    r = client.post("/api/events", json=body(items=items)).json()
    assert client.get(f"/api/cases/{r['case_id']}").json()["items"] == items


def test_resolution_closes_case_and_is_recorded(client):
    cid = client.post("/api/events", json=body()).json()["case_id"]
    r = client.post(f"/api/cases/{cid}/resolution", json={"resolution": "false_alarm", "note": "지갑 꺼냄"})
    assert r.status_code == 200

    detail = client.get(f"/api/cases/{cid}").json()
    assert detail["state"] == "resolved"
    assert detail["resolution"] == "false_alarm"
    assert [h["resolution"] for h in detail["resolutions"]] == ["false_alarm"]


def test_invalid_resolution_is_rejected(client):
    cid = client.post("/api/events", json=body()).json()["case_id"]
    assert client.post(f"/api/cases/{cid}/resolution", json={"resolution": "maybe"}).status_code == 422


def test_resolution_of_unknown_case_is_404(client):
    assert client.post("/api/cases/999/resolution", json={"resolution": "false_alarm"}).status_code == 404


def test_escalation_after_resolution_reopens_case(client):
    cid = client.post("/api/events", json=body("WARNING")).json()["case_id"]
    client.post(f"/api/cases/{cid}/resolution", json={"resolution": "paid_confirmed"})
    r = client.post("/api/events", json=body("HIGH_RISK")).json()
    assert r["notify"] is True
    detail = client.get(f"/api/cases/{cid}").json()
    assert (detail["state"], detail["resolution"]) == ("active", None)
    assert [h["resolution"] for h in detail["resolutions"]] == ["paid_confirmed"]


def test_list_can_filter_by_state(client):
    client.post("/api/events", json=body("WARNING", person_id=1))
    client.post("/api/events", json=body("CLEAR", person_id=2))
    active = client.get("/api/cases", params={"state": "active"}).json()
    assert [c["person_id"] for c in active] == [1]


def test_list_can_return_only_cases_updated_since(client):
    client.post("/api/events", json=body(person_id=1))
    since = client.get("/api/cases").json()[0]["updated_at"]
    client.post("/api/events", json=body(person_id=2))
    newer = client.get("/api/cases", params={"updated_since": since}).json()
    assert [c["person_id"] for c in newer] == [2]


def test_stats_count_states_and_resolutions(client):
    a = client.post("/api/events", json=body("WARNING", person_id=1)).json()["case_id"]
    client.post("/api/events", json=body("CLEAR", person_id=2))
    client.post(f"/api/cases/{a}/resolution", json={"resolution": "false_alarm"})
    s = client.get("/api/stats").json()
    assert s["states"] == {"resolved": 1, "pass": 1}
    assert s["resolutions"] == {"false_alarm": 1}


def test_config_returns_stream_urls(make_client):
    c = make_client(streams=[{"camera_id": "cam1", "url": "rtsp://10.0.0.2:8554/cam1"}])
    assert c.get("/api/config").json() == {"streams": [{"camera_id": "cam1", "url": "rtsp://10.0.0.2:8554/cam1"}]}


def test_api_key_is_required_when_configured(make_client):
    c = make_client(api_key="secret")
    assert c.post("/api/events", json=body()).status_code == 401
    assert c.get("/api/cases").status_code == 401
    assert c.post("/api/events", json=body(), headers={"X-API-Key": "secret"}).status_code == 201
    assert c.get("/healthz").status_code == 200


def test_websocket_pushes_case_updates_with_notify_flag(client):
    with client.websocket_connect("/ws") as ws:
        client.post("/api/events", json=body("WARNING"))
        m1 = ws.receive_json()
        client.post("/api/events", json=body("WARNING"))
        m2 = ws.receive_json()
    assert (m1["type"], m1["notify"], m1["case"]["level"]) == ("case.updated", True, "WARNING")
    assert m2["notify"] is False


def test_websocket_rejects_wrong_key(make_client):
    c = make_client(api_key="secret")
    with pytest.raises(Exception):
        with c.websocket_connect("/ws?key=wrong") as ws:
            ws.receive_json()
    with c.websocket_connect("/ws?key=secret") as ws:
        c.post("/api/events", json=body(), headers={"X-API-Key": "secret"})
        assert ws.receive_json()["type"] == "case.updated"
