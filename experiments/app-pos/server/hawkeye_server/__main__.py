"""경보 서버 실행.

    python -m hawkeye_server --host 0.0.0.0 --port 8000

설정은 환경변수로 받는다 (값은 .env 등에 두고 커밋하지 않는다).

    HAWKEYE_DB          SQLite 파일 (기본 data/hawkeye.db)
    HAWKEYE_SNAPSHOTS   장면 사진 폴더 (기본 data/snapshots)
    HAWKEYE_API_KEY     설정하면 모든 요청에 X-API-Key 필요
    HAWKEYE_STREAMS     앱에 내려줄 RTSP 주소. 예) cam1=rtsp://192.168.0.10:9554/cam1,cam2=...
    HAWKEYE_WEB_STREAMS PC 대시보드(브라우저)용 WebRTC 주소. 예) cam1=http://127.0.0.1:8889/cam1
    HAWKEYE_AI_STREAMS  앱에 내려줄 AI 화면(파이프라인이 그린 영상) RTSP 주소. 예) cam1=rtsp://.../cam1_ai
    HAWKEYE_PLAYBACK_URL MediaMTX 재생 API. 예) http://127.0.0.1:9996 (없으면 녹화 기능 끔)
    HAWKEYE_CLIPS       사건 클립·지난 영상 캐시 폴더 (기본 data/clips)
    HAWKEYE_RECORDINGS  디스크 여유를 볼 녹화 폴더 (기본: DB 폴더)
    HAWKEYE_MIN_FREE_GB 이보다 여유가 적으면 /healthz 가 degraded (기본 100)
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from pathlib import Path

from .app import Settings, create_app


def parse_streams(value: str) -> list[dict[str, str]]:
    streams = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        camera_id, url = part.split("=", 1)
        streams.append({"camera_id": camera_id.strip(), "url": url.strip()})
    return streams


def merge_streams(rtsp: list[dict[str, str]], web: list[dict[str, str]],
                  ai: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    """카메라별로 RTSP 주소(앱), 브라우저 주소(PC 대시보드), AI 화면 주소(앱)를 합친다."""
    streams = [dict(s) for s in rtsp]
    by_id = {s["camera_id"]: s for s in streams}
    for key, extra in (("web_url", web), ("ai_url", ai or [])):
        for w in extra:
            cam = by_id.get(w["camera_id"])
            if cam is None:
                cam = {"camera_id": w["camera_id"]}
                streams.append(cam)
                by_id[w["camera_id"]] = cam
            cam[key] = w["url"]
    return streams


def settings_from_env(env: Mapping[str, str]) -> Settings:
    return Settings(
        db_path=Path(env.get("HAWKEYE_DB", "data/hawkeye.db")),
        snapshot_dir=Path(env.get("HAWKEYE_SNAPSHOTS", "data/snapshots")),
        api_key=env.get("HAWKEYE_API_KEY") or None,
        streams=merge_streams(parse_streams(env.get("HAWKEYE_STREAMS", "")),
                              parse_streams(env.get("HAWKEYE_WEB_STREAMS", "")),
                              parse_streams(env.get("HAWKEYE_AI_STREAMS", ""))),
        playback_url=env.get("HAWKEYE_PLAYBACK_URL") or None,
        clips_dir=Path(env.get("HAWKEYE_CLIPS", "data/clips")),
        disk_path=Path(env["HAWKEYE_RECORDINGS"]) if env.get("HAWKEYE_RECORDINGS") else None,
        min_free_gb=float(env.get("HAWKEYE_MIN_FREE_GB", "100")),
    )


def main() -> None:
    import uvicorn

    p = argparse.ArgumentParser(description="HawkEye 경보 서버")
    p.add_argument("--host", default="127.0.0.1",
                   help="폰에서 접속할 주소 (렌탈 서버: Tailscale IP — 공인 IP 서버에서 0.0.0.0 은 인터넷 노출)")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    uvicorn.run(create_app(settings_from_env(os.environ)), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
