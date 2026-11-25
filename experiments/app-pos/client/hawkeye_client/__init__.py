"""HawkEye 클라이언트. 어떤 탐지 방식이든 경보 서버로 경보를 보내고(AlertPublisher), 그린 화면을 관리자 앱 AI 화면으로
송출(FrameStreamer)할 수 있게 한다 (표준 라이브러리만 사용)."""

from .events import LEVELS, build_event
from .publisher import AlertPublisher, http_post
from .stream import FrameStreamer

__all__ = ["AlertPublisher", "FrameStreamer", "LEVELS", "build_event", "http_post"]
