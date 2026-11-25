"""녹화 — 클립 기준 시각·구간, 파일 캐시, 녹화 API (docs/design.md §10). MediaMTX 는 가짜 재생 API 로 대신한다."""

import os
import threading
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from fastapi.testclient import TestClient

from hawkeye_server.app import Settings, create_app
from hawkeye_server.recordings import FileCache, Span, clip_window, covered_seconds, parse_abs_time, rfc3339

UTC = timezone.utc
T0 = datetime(2026, 11, 24, 3, 0, 0, tzinfo=UTC)


# --- 순수 함수 -----------------------------------------------------------------

def test_parse_abs_time_accepts_offsets_and_rejects_relative():
    assert parse_abs_time("2026-11-23T14:36:56.902510Z") == datetime(2026, 11, 23, 14, 36, 56, 902510, tzinfo=UTC)
    assert parse_abs_time("2026-11-24T12:00:00+09:00") == datetime(2026, 11, 24, 3, 0, 0, tzinfo=UTC)
    assert parse_abs_time("00:00:34.766") is None          # 녹화 파일·시뮬레이터 경보의 상대 시각
    assert parse_abs_time("2026-11-24T03:00:00") is None   # 오프셋 없음
    assert parse_abs_time("") is None and parse_abs_time(None) is None


def test_clip_window_uses_frame_time_of_live_event():
    w = clip_window("2026-11-24T12:00:00+09:00", "2026-11-24T03:00:03.500000+00:00")
    assert (w.anchor, w.anchor_time, w.delay_sec) == ("event", T0, 3.5)
    assert (w.start, w.end) == (T0 - timedelta(seconds=20), T0 + timedelta(seconds=10))


def test_clip_window_falls_back_to_received_time_for_relative_timestamps():
    w = clip_window("00:00:34.766", "2026-11-24T03:00:00+00:00", before_sec=5, after_sec=5)
    assert (w.anchor, w.anchor_time, w.delay_sec) == ("received", T0, None)
    assert (w.start, w.end) == (T0 - timedelta(seconds=5), T0 + timedelta(seconds=5))


def test_late_alert_still_points_at_the_original_scene():
    """전송 재시도로 40초 늦게 도착한 경보도 클립은 경보 장면(프레임 시각)을 가리킨다."""
    w = clip_window(rfc3339(T0), rfc3339(T0 + timedelta(seconds=40)))
    assert w.start == T0 - timedelta(seconds=20) and w.delay_sec == 40.0


def test_covered_seconds_counts_only_recorded_parts():
    spans = [Span(T0, 5.0), Span(T0 + timedelta(seconds=8), 10.0)]   # 5~8초 사이 빈 구간
    assert covered_seconds(spans, T0, T0 + timedelta(seconds=10)) == 7.0
    assert covered_seconds([], T0, T0 + timedelta(seconds=10)) == 0.0


# --- 파일 캐시 -----------------------------------------------------------------

def writer(meta=None):
    def make(p):
        p.write_bytes(b"x" * 10)
        return meta if meta is not None else {"start": rfc3339(T0), "covered": 1.0, "wanted": 1.0}
    return make


def test_cache_makes_a_file_once_and_keeps_its_info(tmp_path):
    cache = FileCache(tmp_path, max_bytes=10_000, max_age_sec=3600)
    calls = []

    def make(p):
        calls.append(p)
        return writer({"k": 1})(p)

    first = cache.get_or_make("a.mp4", make)
    second = cache.get_or_make("a.mp4", make)
    assert first == second == (tmp_path / "a.mp4", {"k": 1}) and len(calls) == 1
    assert cache.get("a.mp4") == (tmp_path / "a.mp4", {"k": 1})
    assert not list(tmp_path.glob("*.part"))


def test_cache_returns_none_and_leaves_nothing_when_there_is_nothing_to_make(tmp_path):
    cache = FileCache(tmp_path, max_bytes=10_000, max_age_sec=3600)
    assert cache.get_or_make("a.mp4", lambda p: None) is None
    assert list(tmp_path.iterdir()) == [] and cache.get("a.mp4") is None


def test_cache_removes_leftover_partial_files_on_start(tmp_path):
    (tmp_path / "a.mp4.123.part").write_bytes(b"half")
    FileCache(tmp_path, max_bytes=10_000, max_age_sec=3600)
    assert not (tmp_path / "a.mp4.123.part").exists()


def test_cache_prunes_oldest_over_size_and_expired_files(tmp_path):
    now = time.time()
    cache = FileCache(tmp_path, max_bytes=25, max_age_sec=100, clock=lambda: now, protect_sec=0)
    for i, age in enumerate([500, 50, 40, 30]):     # 0번은 나이 초과
        p = tmp_path / f"{i}.mp4"
        p.write_bytes(b"x" * 10)
        (tmp_path / f"{i}.mp4.json").write_text("{}")
        os.utime(p, (now - age, now - age))
    (tmp_path / "orphan.mp4.json").write_text("{}")   # 영상 없이 남은 정보 파일
    cache.prune()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["2.mp4", "2.mp4.json", "3.mp4", "3.mp4.json"]


def test_cache_keeps_just_used_files_even_over_size(tmp_path):
    now = time.time()
    cache = FileCache(tmp_path, max_bytes=5, max_age_sec=3600, clock=lambda: now, protect_sec=120)
    p = tmp_path / "a.mp4"
    p.write_bytes(b"x" * 10)
    os.utime(p, (now - 10, now - 10))                # 10초 전에 씀 → 보호
    cache.prune()
    assert p.exists()


def test_cache_prune_survives_files_vanishing_meanwhile(tmp_path):
    cache = FileCache(tmp_path, max_bytes=1, max_age_sec=3600, protect_sec=0)
    real = cache._files
    cache._files = lambda: real() + [tmp_path / "ghost.mp4"]   # 열거한 뒤 다른 요청이 지운 파일
    cache.get_or_make("a.mp4", writer())
    cache.prune()


def test_cache_concurrent_requests_for_different_files_do_not_fail(tmp_path):
    cache = FileCache(tmp_path, max_bytes=50, max_age_sec=3600, protect_sec=0)
    errors, got = [], []

    def worker(n):
        try:
            for i in range(20):
                r = cache.get_or_make(f"w{n}_{i}.mp4", writer())
                got.append(r is not None)
        except Exception as e:            # 정리가 겹쳐 FileNotFoundError 등
            errors.append(repr(e))

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and all(got) and len(got) == 160


# --- 가짜 MediaMTX 재생 API -------------------------------------------------------

class FakeMediaMTX:
    def __init__(self):
        self.spans: dict[str, list[Span]] = {}
        self.gets: list[dict] = []
        self.down = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("refused", request=request)
        q = {k: v[0] for k, v in parse_qs(urlparse(str(request.url)).query).items()}
        spans = self.spans.get(q["path"], [])
        start = parse_abs_time(q.get("start")) or datetime.min.replace(tzinfo=UTC)
        if request.url.path == "/list":
            end = parse_abs_time(q.get("end")) or datetime.max.replace(tzinfo=UTC)
            hit = [s for s in spans if s.end > start and s.start < end]
            if not hit:
                return httpx.Response(404, json={"error": "no recording segments found"})
            return httpx.Response(200, json=[{"start": rfc3339(s.start), "duration": s.duration} for s in hit])
        if request.url.path == "/get":
            self.gets.append(q)
            end = start + timedelta(seconds=float(q["duration"]))
            # 실제 MediaMTX 처럼: 시작 시각 앞에 이전 녹화가 있으면 그 녹화가 시작 시각을 덮어야 한다 (2단계 실기 시험)
            before = [s for s in spans if s.start <= start]
            if before and max(before, key=lambda s: s.start).end <= start:
                return httpx.Response(404, json={"error": "no recording segments found"})
            if covered_seconds(spans, start, end) <= 0:
                return httpx.Response(404, json={"error": "no recording segments found"})
            return httpx.Response(200, content=b"\x00\x00\x00\x18ftypmp42" + b"v" * 2000)
        return httpx.Response(404)


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def mtx():
    return FakeMediaMTX()


@pytest.fixture
def clock():
    return Clock(T0 + timedelta(minutes=5))


@pytest.fixture
def make(tmp_path, mtx, clock):
    def _make(playback=True, free_gb=500, **over):
        s = Settings(db_path=tmp_path / "hawkeye.db", snapshot_dir=tmp_path / "snapshots",
                     clips_dir=tmp_path / "clips",
                     playback_url="http://mediamtx:9996" if playback else None,
                     streams=[{"camera_id": "cam1", "url": "rtsp://h/cam1", "ai_url": "rtsp://h/cam1_ai"},
                              {"camera_id": "cam2", "url": "rtsp://h/cam2"}], **over)
        return TestClient(create_app(s, playback_transport=httpx.MockTransport(mtx.handler), now=clock,
                                     disk_free=lambda p: int(free_gb * 1_000_000_000)))
    return _make


def event_body(timestamp, person_id=3, level="HIGH_RISK"):
    return {"run_id": "run-a", "camera_id": "cam1", "event": {
        "person_id": person_id, "level": level, "zone": "EXIT", "frame": 1043, "timestamp": timestamp,
        "time_sec": 34.766, "bbox": [1, 2, 3, 4], "taken": 1, "paid": 0, "unpaid": 1}}


# --- 설정 · 상태 -----------------------------------------------------------------

def test_recordings_off_without_playback_url(make):
    c = make(playback=False)
    assert c.get("/api/config").json()["recordings"] is False
    assert c.get("/api/recordings", params={"camera_id": "cam1"}).status_code == 503
    c.post("/api/events", json=event_body(rfc3339(T0)))
    assert c.get("/api/cases/1/clips").status_code == 503


def test_config_gives_ai_url_and_streaming_status(make, mtx, clock):
    now = clock()
    # 10분마다 나뉜 녹화 파일 두 개가 이어져 있고, 마지막 것이 1초 전까지 쓰임
    mtx.spans["cam1"] = [Span(now - timedelta(seconds=900), 600.0), Span(now - timedelta(seconds=300), 299.0)]
    cfg = make().get("/api/config").json()
    assert cfg["recordings"] is True
    cam1, cam2 = cfg["streams"]
    assert cam1["ai_url"] == "rtsp://h/cam1_ai" and "ai_url" not in cam2
    assert cam1["live"] == {"streaming": True, "since": rfc3339(now - timedelta(seconds=900))}
    assert cam2["live"] == {"streaming": False, "since": None}


def test_config_still_works_when_mediamtx_is_down(make, mtx):
    mtx.down = True
    cfg = make().get("/api/config").json()
    assert [s["live"] for s in cfg["streams"]] == [None, None]


def test_healthz_reports_low_disk(make):
    assert make(free_gb=500).get("/healthz").json() == {"status": "ok", "disk_free_gb": 500.0}
    low = make(free_gb=60).get("/healthz").json()
    assert (low["status"], low["disk_free_gb"]) == ("degraded", 60.0)


# --- 지난 영상 ---------------------------------------------------------------------

def test_list_recordings(make, mtx):
    mtx.spans["cam1"] = [Span(T0, 30.0)]
    r = make().get("/api/recordings", params={"camera_id": "cam1", "start": rfc3339(T0 - timedelta(minutes=1)),
                                               "end": rfc3339(T0 + timedelta(minutes=1))})
    assert r.status_code == 200
    assert r.json() == {"camera_id": "cam1", "spans": [{"start": rfc3339(T0), "duration": 30.0}]}


def test_list_recordings_rejects_unknown_camera_and_bad_times(make):
    c = make()
    assert c.get("/api/recordings", params={"camera_id": "cam9"}).status_code == 404
    assert c.get("/api/recordings", params={"camera_id": "../etc"}).status_code == 404
    assert c.get("/api/recordings", params={"camera_id": "cam1", "start": "yesterday"}).status_code == 422


def test_recordings_unavailable_is_503(make, mtx):
    mtx.down = True
    assert make().get("/api/recordings", params={"camera_id": "cam1"}).status_code == 503


def test_video_chunk_is_served_once_then_from_cache_with_range_support(make, mtx):
    mtx.spans["cam1"] = [Span(T0, 120.0)]
    c = make()
    params = {"start": rfc3339(T0), "duration": 60}
    r = c.get("/api/recordings/cam1/video", params=params)
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
    assert r.headers["x-covered-seconds"] == "60.000"
    again = c.get("/api/recordings/cam1/video", params=params, headers={"Range": "bytes=0-99"})
    assert again.status_code == 206 and len(again.content) == 100     # 앱이 앞뒤로 넘길 때 (Range)
    assert len(mtx.gets) == 1                                          # 두 번째는 캐시
    assert mtx.gets[0]["format"] == "mp4"


def test_video_chunk_limits_and_missing_recording(make, mtx, clock):
    mtx.spans["cam1"] = [Span(T0, 10.0)]
    c = make()
    assert c.get("/api/recordings/cam1/video", params={"start": rfc3339(T0), "duration": 121}).status_code == 422
    assert c.get("/api/recordings/cam1/video", params={"start": "00:01:00", "duration": 10}).status_code == 422
    gone = c.get("/api/recordings/cam1/video", params={"start": rfc3339(T0 - timedelta(hours=1)), "duration": 60})
    assert gone.status_code == 404
    future = c.get("/api/recordings/cam1/video", params={"start": rfc3339(clock() - timedelta(seconds=30)),
                                                          "duration": 60})
    assert future.status_code == 202 and future.headers["retry-after"] == "32"   # 끝(30초 뒤) + 2초


# --- 사건 클립 ---------------------------------------------------------------------

def test_case_clip_waits_until_the_window_is_recorded_then_serves_it(make, mtx, clock):
    clock.t = T0 + timedelta(seconds=3)                 # 경보 3초 뒤 (구간 끝 +10초 아직 안 옴)
    mtx.spans["cam1"] = [Span(T0 - timedelta(seconds=60), 63.0)]
    mtx.spans["cam2"] = [Span(T0 - timedelta(seconds=60), 63.0)]
    c = make()
    c.post("/api/events", json=event_body(rfc3339(T0)))

    info = c.get("/api/cases/1/clips").json()
    assert (info["anchor"], info["anchor_time"]) == ("event", rfc3339(T0))
    assert info["start"] == rfc3339(T0 - timedelta(seconds=20)) and info["end"] == rfc3339(T0 + timedelta(seconds=10))
    assert [(x["camera_id"], x["is_case_camera"], x["retry_after"]) for x in info["clips"]] == [
        ("cam1", True, 9), ("cam2", False, 9)]

    wait = c.get("/api/cases/1/clips/cam1.mp4")
    assert wait.status_code == 202 and wait.headers["retry-after"] == "9"

    clock.t = T0 + timedelta(seconds=13)
    mtx.spans["cam1"] = [Span(T0 - timedelta(seconds=60), 73.0)]
    ok = c.get("/api/cases/1/clips/cam1.mp4")
    assert ok.status_code == 200 and ok.headers["content-type"] == "video/mp4"
    assert parse_abs_time(mtx.gets[-1]["start"]) == T0 - timedelta(seconds=20)
    assert float(mtx.gets[-1]["duration"]) == 30.0


def test_late_alert_clip_asks_mediamtx_for_the_original_scene(make, mtx, clock):
    mtx.spans["cam1"] = [Span(T0 - timedelta(minutes=5), 600.0)]
    c = make()
    c.post("/api/events", json=event_body(rfc3339(T0)))   # 경보 장면은 T0, 지금은 T0+5분 — 늦게 도착한 경보
    info = c.get("/api/cases/1/clips").json()
    assert info["anchor"] == "event" and info["anchor_time"] == rfc3339(T0)
    assert c.get("/api/cases/1/clips/cam1.mp4").status_code == 200
    assert parse_abs_time(mtx.gets[-1]["start"]) == T0 - timedelta(seconds=20)


def test_simulator_alert_uses_received_time(make, mtx, clock):
    c = make()
    c.post("/api/events", json=event_body("00:00:34.766"))
    info = c.get("/api/cases/1/clips").json()
    assert info["anchor"] == "received" and info["delay_sec"] is None
    received = parse_abs_time(info["anchor_time"])
    assert abs((received - datetime.now(UTC)).total_seconds()) < 60   # 서버 저장 시각(실제 시계)


def test_case_clip_partial_and_missing_recordings(make, mtx, clock):
    mtx.spans["cam1"] = [Span(T0 - timedelta(seconds=5), 15.0)]   # 구간 30초 중 15초만
    c = make()
    c.post("/api/events", json=event_body(rfc3339(T0)))
    cam1, cam2 = c.get("/api/cases/1/clips").json()["clips"]
    assert (cam1["status"], cam1["covered_sec"], cam1["partial"], cam1["has_recording"]) == ("recorded", 15.0, True, True)
    assert (cam2["status"], cam2["covered_sec"], cam2["has_recording"]) == ("no_recording", 0.0, False)
    assert c.get("/api/cases/1/clips/cam2.mp4").status_code == 404
    assert c.get("/api/cases/99/clips").status_code == 404


def test_recording_endpoints_need_the_api_key(make, mtx):
    c = make(api_key="secret")
    assert c.get("/api/recordings", params={"camera_id": "cam1"}).status_code == 401
    assert c.get("/api/recordings", params={"camera_id": "cam1"}, headers={"X-API-Key": "secret"}).status_code == 200


def test_clip_starts_where_recording_starts_when_an_older_recording_exists(make, mtx, clock):
    """앞에 이전 녹화(어제)가 있고 구간 앞부분이 녹화 전이면, 실제 녹화가 시작된 시각부터 요청해야 받는다 (실기 시험에서 찾음)."""
    rec_start = T0 - timedelta(seconds=5)
    mtx.spans["cam1"] = [Span(T0 - timedelta(days=1), 6.0), Span(rec_start, 60.0)]
    c = make()
    c.post("/api/events", json=event_body(rfc3339(T0)))            # 구간: T0-20초 ~ T0+10초
    info = c.get("/api/cases/1/clips").json()
    assert info["clips"][0]["video_start"] == rfc3339(rec_start)
    r = c.get("/api/cases/1/clips/cam1.mp4")
    assert r.status_code == 200 and r.headers["x-video-start"] == rfc3339(rec_start)
    assert parse_abs_time(mtx.gets[-1]["start"]) == rec_start and float(mtx.gets[-1]["duration"]) == 15.0

    chunk = c.get("/api/recordings/cam1/video", params={"start": rfc3339(T0 - timedelta(seconds=30)), "duration": 40})
    assert chunk.status_code == 200 and parse_abs_time(mtx.gets[-1]["start"]) == rec_start


def test_cached_clip_is_served_after_recording_is_gone_and_mediamtx_is_down(make, mtx, clock):
    mtx.spans["cam1"] = [Span(T0 - timedelta(minutes=1), 120.0)]
    c = make()
    c.post("/api/events", json=event_body(rfc3339(T0)))
    first = c.get("/api/cases/1/clips/cam1.mp4")
    assert first.status_code == 200
    mtx.spans.clear()                     # 원본 녹화를 지움
    mtx.down = True                       # MediaMTX 도 꺼짐
    again = c.get("/api/cases/1/clips/cam1.mp4")
    assert again.status_code == 200 and again.content == first.content
    assert again.headers["x-video-start"] == first.headers["x-video-start"]
    assert len(mtx.gets) == 1
    # 목록도 캐시를 먼저 본다: cam1 은 재생 가능, cam2 는 MediaMTX 에 닿지 않음 (목록 전체가 실패하지 않음)
    listing = c.get("/api/cases/1/clips")
    assert listing.status_code == 200
    cam1, cam2 = listing.json()["clips"]
    assert (cam1["status"], cam1["has_recording"], cam1["retry_after"]) == ("cached", True, 0)
    assert cam1["video_start"] == first.headers["x-video-start"] and cam1["covered_sec"] == 30.0
    assert (cam2["status"], cam2["has_recording"]) == ("unavailable", None)


def test_list_recordings_rejects_bad_end_without_start(make):
    assert make().get("/api/recordings", params={"camera_id": "cam1", "end": "yesterday"}).status_code == 422
