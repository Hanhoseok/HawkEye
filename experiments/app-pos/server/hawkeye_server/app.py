"""경보 서버 — 파이프라인 경보를 받아 사건으로 묶고, 관리자 앱에 실시간으로 전달한다.

API 표는 docs/design.md §5.
"""

from __future__ import annotations

import base64
import binascii
import hmac
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from .store import Store

Level = Literal["WARNING", "HIGH_RISK", "REVIEW", "CLEAR"]


@dataclass
class Settings:
    db_path: Path
    snapshot_dir: Path
    api_key: str | None = None
    """설정하면 모든 요청에 X-API-Key 헤더(WebSocket 은 ?key=)가 필요하다."""
    streams: list[dict[str, str]] = field(default_factory=list)
    """카메라별 라이브 주소. url = RTSP(앱), web_url = WebRTC(PC 대시보드).
    [{"camera_id": "cam1", "url": "rtsp://...", "web_url": "http://..."}]"""


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


def create_app(settings: Settings) -> FastAPI:
    store = Store(settings.db_path, settings.snapshot_dir)
    hub = Hub()
    app = FastAPI(title="HawkEye 경보 서버")

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
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

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
        return {"streams": settings.streams}

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
