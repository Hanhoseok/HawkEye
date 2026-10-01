"""경보 서버 실행.

    python -m hawkeye_server --host 0.0.0.0 --port 8000

설정은 환경변수로 받는다 (값은 .env 등에 두고 커밋하지 않는다).

    HAWKEYE_DB          SQLite 파일 (기본 data/hawkeye.db)
    HAWKEYE_SNAPSHOTS   장면 사진 폴더 (기본 data/snapshots)
    HAWKEYE_API_KEY     설정하면 모든 요청에 X-API-Key 필요
    HAWKEYE_STREAMS     앱에 내려줄 RTSP 주소. 예) cam1=rtsp://192.168.0.10:8554/cam1,cam2=...
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


def settings_from_env(env: Mapping[str, str]) -> Settings:
    return Settings(
        db_path=Path(env.get("HAWKEYE_DB", "data/hawkeye.db")),
        snapshot_dir=Path(env.get("HAWKEYE_SNAPSHOTS", "data/snapshots")),
        api_key=env.get("HAWKEYE_API_KEY") or None,
        streams=parse_streams(env.get("HAWKEYE_STREAMS", "")),
    )


def main() -> None:
    import uvicorn

    p = argparse.ArgumentParser(description="HawkEye 경보 서버")
    p.add_argument("--host", default="127.0.0.1", help="폰에서 접속하려면 0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    uvicorn.run(create_app(settings_from_env(os.environ)), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
