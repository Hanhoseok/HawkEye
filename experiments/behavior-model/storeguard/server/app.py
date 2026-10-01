"""FastAPI 서버 + 대시보드.

영상 전달 방식 선택
  브라우저는 RTSP 를 직접 재생하지 못한다. 후보는 셋이다.
    1) MJPEG (multipart/x-mixed-replace)  — 구현 비용 최소, 브라우저 기본 지원, 지연 0.3~1초.
       대신 프레임마다 JPEG 라 대역폭이 크다(640px/8fps ≈ 1~3 Mbps).
    2) HLS (ffmpeg 로 세그먼트 생성)      — 확장성 좋지만 지연 5~15초. 알림 데모에 부적합.
    3) WebRTC                              — 지연 0.2초 미만이지만 시그널링/TURN 등 구현 비용이 크다.
  캡스톤 데모 목적에는 (1) MJPEG 이 맞다. 대시보드는 미리보기용 저해상도 스트림만 받는다.
  이 스트림은 '원본 재생'이 아니라 상태 확인용이며, 카메라가 끊기면 즉시 끊긴 것으로 표시된다.

알림 전달
  Server-Sent Events(/api/stream). WebSocket 보다 단방향 알림에 단순하고 재접속이 자동이다.

카메라 설정
  환경변수로 받는다(.env.example 참고). URL 의 계정/비밀번호는 응답과 로그에서 마스킹된다.
    CAMERAS=cam1,cam2
    CAM1_NAME=1번 카메라
    CAM1_SOURCE=rtsp://user:pass@192.168.0.10:554/stream1     # 또는 MP4 경로

PowerShell 예:
    $env:CAM1_SOURCE="D:\\computervision\\data\\raw\\val\\videos\\어떤파일.mp4"
    D:\\computervision\\.venv\\Scripts\\python.exe -m storeguard.server.app --run baseline_r2p1d
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from queue import Empty, Queue

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from storeguard.config import Config, env_report
from storeguard.infer.alerting import AlertEvent
from storeguard.infer.engine import ClipClassifier
from storeguard.infer.runner import CameraRunner
from storeguard.server.db import EventStore
from storeguard.server.playlist import Playlist
from storeguard.utils import get_logger, mask_url

log = get_logger("server")
STATIC = Path(__file__).resolve().parent / "static"


class AppState:
    def __init__(self):
        self.cfg: Config | None = None
        self.clf: ClipClassifier | None = None
        self.store: EventStore | None = None
        self.runners: dict[str, CameraRunner] = {}
        self.subscribers: list[Queue] = []
        self.started_at = time.time()
        self.run_dir: Path | None = None
        self.playlist: Playlist | None = None
        # 카메라별 재생목록 위치와 클래스 필터
        self.cursor: dict[str, int | None] = {}
        self.filter: dict[str, str | None] = {}

    def broadcast(self, payload: dict) -> None:
        dead = []
        for q in self.subscribers:
            try:
                q.put_nowait(payload)
            except Exception:
                dead.append(q)
        for q in dead:
            if q in self.subscribers:
                self.subscribers.remove(q)


state = AppState()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    # 종료 시 카메라 스레드와 녹화 작업을 정리한다.
    for r in state.runners.values():
        r.stop()


app = FastAPI(title="storeguard — 무인매장 이상행동 탐지", version="0.1.0",
              lifespan=lifespan)


def _on_event(kind: str, ev: AlertEvent) -> None:
    cfg = state.cfg
    d = ev.to_dict()
    if kind == "started":
        if state.store:
            state.store.insert(d)
        tpl = cfg.get(f"classes.alert_text.{ev.action}",
                      "{camera}에서 이상행동이 감지되었습니다.")
        name = (ev.extra or {}).get("camera_name", ev.camera_id)
        payload = {
            "type": "alert",
            "event_id": ev.event_id,
            "camera_id": ev.camera_id,
            "camera_name": name,
            "action": ev.action,
            "action_ko": cfg.get(f"classes.ko.{ev.action}", ev.action),
            "message": tpl.format(camera=name),
            "score": round(ev.score, 4),
            "started_at": ev.started_at,
            "started_media_ts": ev.started_media_ts,
            "has_snapshot": bool(ev.snapshot_path),
            "점수_주의": "보정되지 않은 모델 점수",
        }
    else:
        if state.store:
            state.store.close_event(ev.event_id, ev.ended_at or time.time(),
                                    ev.ended_media_ts, ev.peak_score)
        payload = {"type": "event_end", "event_id": ev.event_id,
                   "camera_id": ev.camera_id, "action": ev.action,
                   "peak_score": round(ev.peak_score, 4), "ended_at": ev.ended_at}
    state.broadcast(payload)


def build_cameras(cfg: Config, loop_file: bool) -> dict[str, CameraRunner]:
    ids = [c.strip() for c in os.environ.get("CAMERAS", "cam1").split(",") if c.strip()]
    runners: dict[str, CameraRunner] = {}
    for cam_id in ids:
        key = cam_id.upper()
        src = os.environ.get(f"{key}_SOURCE")
        if not src:
            log.warning("%s_SOURCE 환경변수가 없어 %s 를 건너뛴다.", key, cam_id)
            continue
        name = os.environ.get(f"{key}_NAME", f"{cam_id} 카메라")
        runners[cam_id] = CameraRunner(cam_id, src, cfg, state.clf, display_name=name,
                                       on_event=_on_event, loop_file=loop_file)
    return runners


# ---------------- API ----------------

@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    p = STATIC / "index.html"
    if not p.exists():
        return HTMLResponse("<h1>대시보드 파일이 없습니다</h1>", status_code=500)
    return HTMLResponse(p.read_text(encoding="utf-8"))


@app.get("/api/status")
def api_status() -> JSONResponse:
    cfg = state.cfg
    cams = {cid: r.status() for cid, r in state.runners.items()}
    model = state.clf.status() if state.clf else {"state": "MODEL_NOT_LOADED"}
    model_classes = set(model.get("classes") or [])
    not_covered = [c for c in cfg.class_names
                   if c != "background_unlabeled" and c not in model_classes] \
        if model_classes else []
    return JSONResponse({
        "server": {
            "uptime_sec": round(time.time() - state.started_at, 1),
            "run_dir": str(state.run_dir) if state.run_dir else None,
            "env": env_report(),
        },
        "model": model,
        "model_ready": model.get("state") == "READY",
        "cameras": cams,
        "classes": {
            "names": cfg.class_names,
            "ko": cfg.get("classes.ko", {}),
            # 로드된 체크포인트가 **실제로 예측하는** 클래스만 내보낸다.
            # 설정에만 있고 모델에는 없는 클래스를 화면에 띄우면
            # 지원하지 않는 행동을 탐지하는 것처럼 보인다.
            "actions": [c for c in cfg.action_classes if c in model_classes],
            "normals": [c for c in cfg.normal_classes if c in model_classes],
            "model_classes": sorted(model_classes),
            "not_covered": not_covered,
        },
        "model_coverage_note": (
            f"로드된 모델이 다루지 않는 클래스 {len(not_covered)}개: "
            f"{', '.join(not_covered)}. 해당 데이터로 학습해야 탐지된다."
        ) if not_covered else None,
        "events": state.store.counts() if state.store else {},
        "주의": "모델 점수는 보정되지 않았다. 절도는 '의심'으로만 표시한다.",
    })


@app.get("/api/cameras")
def api_cameras() -> JSONResponse:
    return JSONResponse({cid: r.status() for cid, r in state.runners.items()})


@app.get("/api/events")
def api_events(limit: int = Query(50, ge=1, le=500),
               camera_id: str | None = None,
               action: str | None = None) -> JSONResponse:
    if not state.store:
        raise HTTPException(503, "이벤트 저장소 없음")
    rows = state.store.recent(limit, camera_id, action)
    ko = state.cfg.get("classes.ko", {})
    for r in rows:
        r["action_ko"] = ko.get(r["action"], r["action"])
        r.pop("snapshot_path", None)
        r.pop("video_path", None)
    return JSONResponse(rows)


@app.get("/api/events/{event_id}")
def api_event(event_id: str) -> JSONResponse:
    row = state.store.get(event_id) if state.store else None
    if not row:
        raise HTTPException(404, "없는 이벤트")
    row.pop("snapshot_path", None)
    row.pop("video_path", None)
    return JSONResponse(row)


@app.get("/api/events/{event_id}/snapshot")
def api_snapshot(event_id: str):
    row = state.store.get(event_id) if state.store else None
    if not row or not row.get("snapshot_path"):
        raise HTTPException(404, "대표 이미지 없음")
    p = Path(row["snapshot_path"])
    if not p.exists():
        raise HTTPException(404, "파일 없음")
    return FileResponse(p, media_type="image/jpeg")


@app.get("/api/events/{event_id}/clip")
def api_clip(event_id: str):
    row = state.store.get(event_id) if state.store else None
    if not row or not row.get("video_path"):
        raise HTTPException(404, "영상 없음")
    p = Path(row["video_path"])
    if not p.exists():
        raise HTTPException(404, "아직 저장 중이거나 파일이 없음")
    return FileResponse(p, media_type="video/mp4")


# ---------------- 영상 소스 전환 ----------------

def _apply_source(camera_id: str, spec: str, label: str,
                  loop: bool | None, cursor: int | None) -> dict:
    r = state.runners.get(camera_id)
    if not r:
        raise HTTPException(404, "없는 카메라")
    res = r.switch_source(spec, loop_file=loop, label=label)
    state.cursor[camera_id] = cursor
    state.broadcast({"type": "source_changed", "camera_id": camera_id,
                     "label": label, "source": res["source"]})
    return res


@app.get("/api/sources")
def api_sources(camera_id: str | None = None) -> JSONResponse:
    """재생목록과 각 카메라의 현재 위치."""
    pl = state.playlist
    if pl is None:
        raise HTTPException(503, "재생목록 없음")
    ko = state.cfg.get("classes.ko", {}) or {}
    cams = {}
    for cid, r in state.runners.items():
        cur = state.cursor.get(cid)
        item = pl.get(cur) if cur is not None else None
        cams[cid] = {
            "name": r.display_name,
            "kind": r.source.kind,
            "source": r.source.status.url_masked,
            "label": r.source_label if r.source.kind == "file" else r.source.status.url_masked,
            "cursor": cur,
            "filter": state.filter.get(cid) or "all",
            "current": item.public(cur) if item else None,
        }
    items = [it.public(i) for i, it in enumerate(pl.items)]
    if camera_id:
        f = state.filter.get(camera_id)
        if f and f != "all":
            items = [x for x in items if x["class_name"] == f]
    return JSONResponse({
        "count": len(pl.items),
        "split": pl.split,
        "error": pl.error,
        "classes": [{"name": c, "ko": ko.get(c, c),
                     "count": len(pl.filtered(c))} for c in pl.classes()],
        "items": items[:500],
        "cameras": cams,
    })


@app.post("/api/cameras/{camera_id}/next")
def api_next(camera_id: str, delta: int = 1) -> JSONResponse:
    """재생목록에서 다음(또는 이전) 영상으로 전환한다."""
    pl = state.playlist
    if pl is None or not pl.items:
        raise HTTPException(503, "재생목록이 비어 있다")
    if camera_id not in state.runners:
        raise HTTPException(404, "없는 카메라")
    nxt = pl.step(state.cursor.get(camera_id), delta, state.filter.get(camera_id))
    if nxt is None:
        raise HTTPException(404, "조건에 맞는 영상이 없다")
    item = pl.get(nxt)
    res = _apply_source(camera_id, item.path, item.stem, loop=True, cursor=nxt)
    return JSONResponse({**res, "item": item.public(nxt)})


@app.post("/api/cameras/{camera_id}/play/{index}")
def api_play(camera_id: str, index: int) -> JSONResponse:
    pl = state.playlist
    item = pl.get(index) if pl else None
    if item is None:
        raise HTTPException(404, "없는 항목")
    res = _apply_source(camera_id, item.path, item.stem, loop=True, cursor=index)
    return JSONResponse({**res, "item": item.public(index)})


@app.post("/api/cameras/{camera_id}/filter")
def api_filter(camera_id: str, class_name: str = "all") -> JSONResponse:
    if camera_id not in state.runners:
        raise HTTPException(404, "없는 카메라")
    state.filter[camera_id] = None if class_name == "all" else class_name
    return JSONResponse({"ok": True, "camera_id": camera_id, "filter": class_name})


@app.post("/api/cameras/{camera_id}/source")
def api_set_source(camera_id: str, url: str = "", loop: bool = True) -> JSONResponse:
    """임의의 소스로 전환한다. RTSP 주소 또는 인덱스에 있는 MP4 경로만 허용한다."""
    if camera_id not in state.runners:
        raise HTTPException(404, "없는 카메라")
    url = (url or "").strip()
    if not url:
        raise HTTPException(400, "url 이 비어 있다")

    if "://" in url:
        scheme = url.split("://", 1)[0].lower()
        if scheme not in ("rtsp", "rtsps", "http", "https"):
            raise HTTPException(400, f"지원하지 않는 프로토콜: {scheme}")
        label = mask_url(url)
        cursor = None
    else:
        pl = state.playlist
        if pl is None or not pl.is_allowed_file(url):
            # 로컬 파일을 아무거나 열어 주지 않는다.
            raise HTTPException(
                403, "허용되지 않은 경로다. 인덱스에 있는 영상이나 data/raw 안의 파일만 재생할 수 있다.")
        if not Path(url).exists():
            raise HTTPException(404, "파일이 없다")
        label = Path(url).stem
        cursor = pl.index_of_path(url)

    res = _apply_source(camera_id, url, label, loop=loop, cursor=cursor)
    return JSONResponse(res)


@app.get("/api/thresholds")
def api_get_thresholds() -> JSONResponse:
    return JSONResponse({cid: r.alerts.status() for cid, r in state.runners.items()})


@app.post("/api/thresholds/{camera_id}/{action}")
def api_set_threshold(camera_id: str, action: str,
                      threshold: float | None = None, min_hits: int | None = None,
                      window: int | None = None, cooldown_sec: float | None = None,
                      end_below_threshold: int | None = None) -> JSONResponse:
    r = state.runners.get(camera_id)
    if not r or action not in r.alerts.states:
        raise HTTPException(404, "없는 카메라 또는 클래스")
    kw = {k: v for k, v in dict(threshold=threshold, min_hits=min_hits, window=window,
                                cooldown_sec=cooldown_sec,
                                end_below_threshold=end_below_threshold).items() if v is not None}
    r.alerts.update_params(action, **kw)
    return JSONResponse(r.alerts.status())


@app.get("/api/stream")
async def api_stream():
    """SSE 알림 스트림."""
    q: Queue = Queue(maxsize=100)
    state.subscribers.append(q)

    async def gen():
        try:
            yield f"data: {json.dumps({'type': 'hello', 'ts': time.time()}, ensure_ascii=False)}\n\n"
            while True:
                try:
                    item = await asyncio.get_event_loop().run_in_executor(None, q.get, True, 15.0)
                    yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                except Empty:
                    yield ": keepalive\n\n"
        finally:
            if q in state.subscribers:
                state.subscribers.remove(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/cameras/{camera_id}/snapshot.jpg")
def api_camera_snapshot(camera_id: str):
    """현재 프레임 1장. MJPEG 를 못 쓰는 환경의 대체 경로."""
    r = state.runners.get(camera_id)
    if not r:
        raise HTTPException(404, "없는 카메라")
    jpg = r.preview_jpeg()
    if jpg is None:
        raise HTTPException(503, "라이브 프레임 없음")
    from fastapi import Response
    return Response(jpg, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/cameras/{camera_id}/stream.mjpg")
async def api_mjpeg(camera_id: str):
    r = state.runners.get(camera_id)
    if not r:
        raise HTTPException(404, "없는 카메라")
    boundary = "frame"
    fps = max(1.0, float(state.cfg.get("runtime.mjpeg_fps", 8)))

    # 비동기 제너레이터여야 한다. 동기 제너레이터로 두면 스트림 하나가 스레드풀 스레드를
    # 계속 붙잡고, 스트림이 몇 개만 쌓여도 다른 API 응답까지 멈춘다(실제로 재현됨).
    async def gen():
        deadline = time.time() + 3.0          # 첫 프레임 대기
        while r.preview_jpeg() is None and time.time() < deadline:
            await asyncio.sleep(0.1)
        while True:
            jpg = r.preview_jpeg()
            if jpg is None:
                # 연결이 끊겼으면 마지막 화면을 계속 보내지 않는다. 스트림을 끊는다.
                break
            yield (b"--" + boundary.encode() + b"\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            await asyncio.sleep(1.0 / fps)

    return StreamingResponse(gen(),
                             media_type=f"multipart/x-mixed-replace; boundary={boundary}")


@app.post("/api/model/reload")
def api_reload() -> JSONResponse:
    if not state.clf:
        raise HTTPException(503, "분류기 없음")
    ok = state.clf.load()
    return JSONResponse({"ok": ok, **state.clf.status()})


def create_app(config: Path | None = None, overrides: list[str] | None = None,
               run: str | None = None, runs_dir: Path | None = None,
               loop_file: bool = False) -> FastAPI:
    cfg = Config.load(config, overrides or [])
    state.cfg = cfg
    runs_dir = runs_dir or cfg.path("paths.runs")
    state.run_dir = (runs_dir / run) if run else None
    state.clf = ClipClassifier(state.run_dir, cfg=cfg)
    if state.clf.state != "READY":
        log.warning("모델 미로딩: %s — 대시보드는 MODEL_NOT_LOADED 로 표시된다.", state.clf.error)
    state.store = EventStore(cfg["server.db_path"])
    state.playlist = Playlist(cfg, split=str(cfg.get("server.playlist_split", "test")))
    if state.playlist.error:
        log.warning("재생목록을 만들지 못했다: %s", state.playlist.error)
    else:
        log.info("재생목록 %d개 (%s 분할)", len(state.playlist.items), state.playlist.split)
    state.runners = build_cameras(cfg, loop_file)
    for cid, r in state.runners.items():
        r.start()
        # .env 로 지정된 파일이 재생목록에 있으면 현재 위치를 맞춰 둔다
        if r.source.kind == "file" and state.playlist:
            state.cursor[cid] = state.playlist.index_of_path(r.source.url)
    if not state.runners:
        log.warning("등록된 카메라가 없다. CAMERAS / CAM1_SOURCE 환경변수를 확인할 것.")
    return app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--run", default=None)
    ap.add_argument("--runs-dir", type=Path, default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--loop", action="store_true", help="파일 입력 반복 재생(데모용)")
    a = ap.parse_args(argv)

    try:
        from dotenv import load_dotenv
        load_dotenv(Path(r"D:\computervision\.env"))
    except Exception:
        pass

    application = create_app(a.config, a.overrides, a.run, a.runs_dir, a.loop)
    import uvicorn
    cfg = state.cfg
    uvicorn.run(application,
                host=a.host or str(cfg["server.host"]),
                port=a.port or int(cfg["server.port"]), log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
