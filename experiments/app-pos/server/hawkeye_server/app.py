"""경보 서버 — 파이프라인 경보를 받아 사건으로 묶고, 관리자 앱에 실시간으로 전달한다.

API 표는 docs/design.md §5.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from .library import Library, LibraryError
from .recordings import (CAMERA_ID, ClipResult, FileCache, Playback, Recordings, RecordingsUnavailable,
                         clip_window, covered_seconds, first_recorded, parse_abs_time, rfc3339)
from .store import Store

GB = 1_000_000_000

Level = Literal["WARNING", "HIGH_RISK", "REVIEW", "CLEAR"]


@dataclass
class Settings:
    db_path: Path
    snapshot_dir: Path
    api_key: str | None = None
    """설정하면 모든 요청에 X-API-Key 헤더(WebSocket 은 ?key=)가 필요하다."""
    streams: list[dict[str, str]] = field(default_factory=list)
    """카메라별 라이브 주소. url = RTSP(앱), ai_url = AI 화면 RTSP(앱), web_url = WebRTC(PC 대시보드).
    [{"camera_id": "cam1", "url": "rtsp://...", "ai_url": "rtsp://...", "web_url": "http://..."}]"""
    playback_url: str | None = None
    """MediaMTX 재생 API (예: http://127.0.0.1:9996). 없으면 녹화 기능을 끈다."""
    clips_dir: Path = Path("data/clips")
    disk_path: Path | None = None
    """여유 공간을 볼 폴더 (녹화 폴더). 없으면 DB 폴더."""
    min_free_gb: float = 100.0
    clip_before_sec: float = 20.0
    clip_after_sec: float = 10.0
    flush_delay_sec: float = 2.0
    clip_cache_gb: float = 20.0
    clip_cache_days: float = 7.0
    chunk_cache_gb: float = 5.0
    chunk_cache_hours: float = 1.0
    max_chunk_sec: float = 120.0
    library_dir: Path | None = None
    """보관함(촬영 영상 폴더) 위치. 없으면 보관함 기능을 끈다 (docs/design.md §11)."""


class Item(BaseModel):
    name: str
    taken: int
    paid: int


class EventIn(BaseModel):
    """파이프라인 RiskEvent.to_dict() + 선택 항목 items."""

    model_config = ConfigDict(extra="allow")

    person_id: int
    level: Level
    zone: str
    frame: int
    timestamp: str
    time_sec: float
    bbox: list[float]
    taken: int
    paid: int
    unpaid: int
    take_frames: list[list[int]] = []
    reason: str = ""
    identity_check: float | None = None
    items: list[Item] | None = None


class EventBody(BaseModel):
    run_id: str
    camera_id: str
    event: EventIn
    snapshot_jpeg_b64: str | None = None


class ResolutionIn(BaseModel):
    resolution: Literal["paid_confirmed", "false_alarm"]
    note: str | None = None


class NameIn(BaseModel):
    name: str


class MoveIn(BaseModel):
    scene_ids: list[int]
    folder_id: int


class SceneIdsIn(BaseModel):
    scene_ids: list[int]


class Hub:
    """연결된 앱들에 메시지를 뿌린다. 끊긴 연결은 조용히 뺀다."""

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()

    async def broadcast(self, message: dict[str, Any]) -> None:
        for ws in list(self.clients):
            try:
                await ws.send_json(message)
            except Exception:
                self.clients.discard(ws)


def create_app(settings: Settings, *, playback_transport: httpx.BaseTransport | None = None,
               now: Callable[[], datetime] | None = None,
               disk_free: Callable[[Path], int] | None = None) -> FastAPI:
    """playback_transport · now · disk_free 는 시험에서 가짜를 넣기 위한 것."""
    store = Store(settings.db_path, settings.snapshot_dir)
    hub = Hub()
    app = FastAPI(title="HawkEye 경보 서버")
    free_bytes = disk_free or (lambda p: shutil.disk_usage(p).free)
    disk_path = settings.disk_path or settings.db_path.parent

    recordings: Recordings | None = None
    if settings.playback_url:
        recordings = Recordings(
            Playback(settings.playback_url, transport=playback_transport),
            FileCache(settings.clips_dir, int(settings.clip_cache_gb * GB), settings.clip_cache_days * 86400),
            FileCache(settings.clips_dir / "chunks", int(settings.chunk_cache_gb * GB),
                      settings.chunk_cache_hours * 3600),
            settings.flush_delay_sec, now)

    library = Library(settings.library_dir) if settings.library_dir else None

    def need_library() -> Library:
        if library is None:
            raise HTTPException(status_code=503, detail="library_unavailable: 보관함이 꺼져 있습니다 (HAWKEYE_LIBRARY)")
        return library

    def need_recordings() -> Recordings:
        if recordings is None:
            raise HTTPException(status_code=503, detail="recordings_unavailable: 녹화 재생이 꺼져 있습니다 (HAWKEYE_PLAYBACK_URL)")
        return recordings

    def cameras() -> list[str]:
        return [s["camera_id"] for s in settings.streams if CAMERA_ID.match(s["camera_id"])]

    def check_camera(camera_id: str) -> str:
        if not CAMERA_ID.match(camera_id) or (cameras() and camera_id not in cameras()):
            raise HTTPException(status_code=404, detail=f"모르는 카메라입니다: {camera_id}")
        return camera_id

    def video_response(result: ClipResult) -> Any:
        if result.status == "not_ready":
            return JSONResponse({"detail": "not_ready: 아직 녹화 중인 구간입니다", "retry_after": result.retry_after},
                                status_code=202, headers={"Retry-After": str(result.retry_after)})
        if result.status == "no_recording":
            raise HTTPException(status_code=404, detail="no_recording: 이 구간에 녹화가 없습니다")
        return FileResponse(result.path, media_type="video/mp4", headers={
            "X-Covered-Seconds": f"{result.covered:.3f}", "X-Wanted-Seconds": f"{result.wanted:.3f}",
            "X-Video-Start": rfc3339(result.start)})   # 앱이 영상 위치 ↔ 실제 시각을 맞출 때

    def case_window(case_id: int):
        c = store.get_case(case_id)
        if c is None:
            raise HTTPException(status_code=404, detail="사건이 없습니다")
        events = c["events"]
        alerts = [e for e in events if e["level"] != "CLEAR"]
        e = (alerts or events)[-1]                # 마지막 경보 (경보가 없던 통과 사건은 마지막 기록)
        return c, clip_window(e["timestamp"], e["received_at"], settings.clip_before_sec, settings.clip_after_sec)

    def key_ok(given: str | None) -> bool:
        return settings.api_key is None or (given is not None and hmac.compare_digest(given, settings.api_key))

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        if not key_ok(x_api_key):
            raise HTTPException(status_code=401, detail="API 키가 필요합니다")

    auth = [Depends(require_key)]

    # PC 웹 대시보드: 페이지 자체는 공개, 데이터(API)는 키로 보호한다.
    dashboard = Path(__file__).parent / "dashboard"
    app.mount("/dashboard", StaticFiles(directory=dashboard), name="dashboard")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(dashboard / "index.html", media_type="text/html")

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        try:
            free_gb = round(free_bytes(disk_path) / GB, 1)
        except OSError:
            return {"status": "ok", "disk_free_gb": None}
        if free_gb < settings.min_free_gb:
            return {"status": "degraded", "disk_free_gb": free_gb,
                    "detail": f"디스크 여유 {free_gb}GB — {settings.min_free_gb:g}GB 미만. 녹화 폴더를 정리하세요"}
        return {"status": "ok", "disk_free_gb": free_gb}

    @app.post("/api/events", status_code=201, dependencies=auth)
    async def post_event(b: EventBody) -> dict[str, Any]:
        snapshot = None
        if b.snapshot_jpeg_b64:
            try:
                snapshot = base64.b64decode(b.snapshot_jpeg_b64, validate=True)
            except (binascii.Error, ValueError):
                raise HTTPException(status_code=422, detail="snapshot_jpeg_b64 가 올바른 base64 가 아닙니다")
        case, notify = store.add_event(b.run_id, b.camera_id, b.event.model_dump(exclude_none=False), snapshot)
        await hub.broadcast({"type": "case.updated", "notify": notify, "case": case})
        return {"case_id": case["id"], "notify": notify}

    @app.get("/api/cases", dependencies=auth)
    def list_cases(state: str | None = None, updated_since: str | None = None,
                   limit: int = Query(default=100, ge=1, le=1000)) -> list[dict[str, Any]]:
        return store.list_cases(state, updated_since, limit)

    @app.get("/api/cases/{case_id}", dependencies=auth)
    def get_case(case_id: int) -> dict[str, Any]:
        c = store.get_case(case_id)
        if c is None:
            raise HTTPException(status_code=404, detail="사건이 없습니다")
        return c

    @app.post("/api/cases/{case_id}/resolution", dependencies=auth)
    async def resolve(case_id: int, r: ResolutionIn) -> dict[str, Any]:
        case = store.resolve(case_id, r.resolution, r.note)
        if case is None:
            raise HTTPException(status_code=404, detail="사건이 없습니다")
        await hub.broadcast({"type": "case.updated", "notify": False, "case": case})
        return case

    @app.get("/api/events/{event_id}/snapshot.jpg", dependencies=auth)
    def snapshot(event_id: int) -> FileResponse:
        if not store.has_snapshot(event_id):
            raise HTTPException(status_code=404, detail="사진이 없습니다")
        return FileResponse(store.snapshot_path(event_id), media_type="image/jpeg")

    @app.get("/api/stats", dependencies=auth)
    def stats() -> dict[str, dict[str, int]]:
        return store.stats()

    @app.get("/api/config", dependencies=auth)
    def config() -> dict[str, Any]:
        streams = [dict(s) for s in settings.streams]
        if recordings is not None:
            for s in streams:            # 송출 중인가 · 언제부터 (켜 둔 채 잊은 폰 알아채기). 실패해도 설정은 준다
                if CAMERA_ID.match(s["camera_id"]):
                    try:
                        s["live"] = recordings.live_status(s["camera_id"])
                    except RecordingsUnavailable:
                        s["live"] = None
        return {"streams": streams, "recordings": recordings is not None}

    # --- 녹화 (docs/design.md §10) ---

    @app.get("/api/recordings", dependencies=auth)
    def list_recordings(camera_id: str, start: str | None = None, end: str | None = None) -> dict[str, Any]:
        rec = need_recordings()
        check_camera(camera_id)
        bad = HTTPException(status_code=422, detail="start·end 는 오프셋이 있는 ISO8601 이고 start < end")
        end_dt = parse_abs_time(end) if end else rec.now()
        if end_dt is None:
            raise bad
        start_dt = parse_abs_time(start) if start else end_dt - timedelta(hours=24)
        if start_dt is None or start_dt >= end_dt:
            raise bad
        try:
            spans = rec.playback.list(camera_id, start_dt, end_dt)
        except RecordingsUnavailable as e:
            raise HTTPException(status_code=503, detail=f"recordings_unavailable: {e}")
        return {"camera_id": camera_id,
                "spans": [{"start": rfc3339(s.start), "duration": s.duration} for s in spans]}

    @app.get("/api/recordings/{camera_id}/video", dependencies=auth)
    def recording_video(camera_id: str, start: str, duration: float = Query(default=60.0)) -> Any:
        rec = need_recordings()
        check_camera(camera_id)
        start_dt = parse_abs_time(start)
        if start_dt is None or not (1.0 <= duration <= settings.max_chunk_sec):
            raise HTTPException(status_code=422,
                                detail=f"start 는 오프셋이 있는 ISO8601, duration 은 1~{settings.max_chunk_sec:g}초")
        try:
            return video_response(rec.chunk(camera_id, start_dt, duration))
        except RecordingsUnavailable as e:
            raise HTTPException(status_code=503, detail=f"recordings_unavailable: {e}")

    @app.get("/api/cases/{case_id}/clips", dependencies=auth)
    def case_clips(case_id: int) -> dict[str, Any]:
        rec = need_recordings()
        c, w = case_window(case_id)
        cams = cameras() or [c["camera_id"]]
        wanted = (w.end - w.start).total_seconds()
        clips = []
        for cam in cams:
            item: dict[str, Any] = {"camera_id": cam, "url": f"/api/cases/{case_id}/clips/{cam}.mp4",
                                    "is_case_camera": cam == c["camera_id"]}
            # status: cached(이미 만든 클립 — 원본·MediaMTX 없이도 재생) / recorded(녹화 있음) /
            #         no_recording / unavailable(MediaMTX 에 닿지 않음, 다른 카메라는 그대로 알려 줌)
            cached = rec.cached_clip(case_id, cam, w)
            if cached is not None:
                covered, first, status, wait = cached.covered, cached.start, "cached", 0
            else:
                try:
                    spans = rec.playback.list(cam, w.start - timedelta(seconds=1), w.end + timedelta(seconds=1))
                except RecordingsUnavailable:
                    clips.append({**item, "status": "unavailable", "retry_after": None, "covered_sec": None,
                                  "partial": None, "has_recording": None, "video_start": None})
                    continue
                covered = covered_seconds(spans, w.start, w.end)
                first = first_recorded(spans, w.start, w.end)
                status = "recorded" if covered > 0 else "no_recording"
                wait = rec.retry_after(w.end)
            clips.append({**item, "status": status, "retry_after": wait, "covered_sec": covered,
                          "partial": 0 < covered < wanted - 0.5, "has_recording": covered > 0,
                          "video_start": rfc3339(first) if first else None})
        return {"case_id": case_id, "anchor": w.anchor, "anchor_time": rfc3339(w.anchor_time),
                "start": rfc3339(w.start), "end": rfc3339(w.end), "delay_sec": w.delay_sec, "clips": clips}

    @app.get("/api/cases/{case_id}/clips/{camera_id}.mp4", dependencies=auth)
    def case_clip(case_id: int, camera_id: str) -> Any:
        rec = need_recordings()
        _, w = case_window(case_id)
        check_camera(camera_id)
        try:
            return video_response(rec.clip(case_id, camera_id, w))
        except RecordingsUnavailable as e:
            raise HTTPException(status_code=503, detail=f"recordings_unavailable: {e}")

    # --- 보관함 (docs/design.md §11) ---

    @app.exception_handler(LibraryError)
    async def library_error(_request: Any, e: LibraryError) -> JSONResponse:
        return JSONResponse({"detail": f"{e.code}: {e.message}", **e.extra}, status_code=e.status)

    @app.get("/api/library", dependencies=auth)
    def library_folders() -> dict[str, Any]:
        return {"folders": need_library().folders()}

    @app.post("/api/library/folders", status_code=201, dependencies=auth)
    def library_create_folder(b: NameIn) -> dict[str, Any]:
        return need_library().create_folder(b.name)

    @app.get("/api/library/folders/{folder_id}", dependencies=auth)
    def library_folder(folder_id: int) -> dict[str, Any]:
        return need_library().folder(folder_id)

    @app.patch("/api/library/folders/{folder_id}", dependencies=auth)
    def library_rename_folder(folder_id: int, b: NameIn) -> dict[str, Any]:
        return need_library().rename_folder(folder_id, b.name)

    @app.delete("/api/library/folders/{folder_id}", dependencies=auth)
    def library_delete_folder(folder_id: int, with_scenes: bool = False) -> dict[str, int]:
        return need_library().delete_folder(folder_id, with_scenes)

    @app.patch("/api/library/scenes/{scene_id}", dependencies=auth)
    def library_rename_scene(scene_id: int, b: NameIn) -> dict[str, Any]:
        return need_library().rename_scene(scene_id, b.name)

    @app.post("/api/library/scenes/move", dependencies=auth)
    def library_move(b: MoveIn) -> dict[str, int]:
        return need_library().move_scenes(b.scene_ids, b.folder_id)

    @app.post("/api/library/scenes/delete", dependencies=auth)
    def library_delete_scenes(b: SceneIdsIn) -> dict[str, int]:
        return need_library().delete_scenes(b.scene_ids)

    @app.get("/api/library/videos/{video_id}.mp4", dependencies=auth)
    def library_video(video_id: int) -> FileResponse:
        path = need_library().video_file(video_id)
        if path is None:
            raise HTTPException(status_code=404, detail="not_found: 영상 파일이 없습니다")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/api/library/videos/{video_id}/ai.mp4", dependencies=auth)
    def library_ai_video(video_id: int) -> FileResponse:
        path = need_library().ai_file(video_id)
        if path is None:
            raise HTTPException(status_code=404, detail="not_found: AI 화면 영상이 없습니다")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/api/library/scenes/{scene_id}/thumb.jpg", dependencies=auth)
    def library_thumb(scene_id: int) -> FileResponse:
        path = need_library().thumb_file(scene_id)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="not_found: 썸네일이 없습니다")
        return FileResponse(path, media_type="image/jpeg")

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket, key: str | None = None) -> None:
        if not key_ok(key):
            await ws.close(code=1008)
            return
        await ws.accept()
        hub.clients.add(ws)
        try:
            while True:
                await ws.receive_text()   # 앱은 보내지 않지만, 연결 유지·끊김 감지용
        except WebSocketDisconnect:
            pass
        finally:
            hub.clients.discard(ws)

    return app
