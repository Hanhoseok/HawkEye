"""경보 전송 — 위험 판정(RiskEvent)을 관리자 앱·대시보드의 경보 서버로 보낸다.

경보 서버와 전송 모듈은 app-pos(한호석)가 만든 것을 그대로 쓴다. 이 폴더는 고치지 않는다.
    서버:      experiments/app-pos/server   (POST /api/events, 형식은 app-pos/docs/design.md §5)
    전송 모듈:  experiments/app-pos/client/hawkeye_client   (표준 라이브러리만 사용, 재시도·대기열·비차단)

서버가 받는 경보 형식은 RiskEvent.to_dict() 와 같다. 실행 번호(run_id)·카메라 이름은 전송 모듈이 붙이고,
여기서는 경보 순간의 장면 사진(JPEG)만 만들어 함께 보낸다.

서버 주소가 없으면 만들지 않는다(None) — 경보 서버 없이도 파이프라인은 그대로 돈다.
API 키는 명령줄에 쓰지 않고 환경변수 HAWKEYE_API_KEY 로만 받는다(터미널 기록에 남지 않게).
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..core.types import RiskEvent, RiskLevel

log = logging.getLogger(__name__)

# 단계별 상자 색 (BGR)
_COLORS = {
    RiskLevel.HIGH_RISK: (0, 0, 255),
    RiskLevel.REVIEW: (0, 165, 255),
    RiskLevel.WARNING: (0, 215, 255),
    RiskLevel.CLEAR: (0, 200, 0),
}


def snapshot_jpeg(image: np.ndarray, event: RiskEvent, scale: float = 1.0, quality: int = 85) -> bytes | None:
    """경보 장면 사진: 그 손님에게 단계 색의 상자와 문구를 그린 JPEG.

    scale: image 가 원본에서 줄인 사진이면 그 비율 (상자 좌표를 맞추기 위해).
    """
    if image is None:
        return None
    canvas = image.copy()
    color = _COLORS.get(event.level, (255, 255, 255))
    x1, y1, x2, y2 = (int(v * scale) for v in event.bbox)
    if x2 > x1 and y2 > y1:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
        label = f"{event.level.value} P{event.person_id} unpaid {event.unpaid}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        # 글자가 사진 밖으로 나가지 않게 (작은 영상에서 오른쪽 끝이 잘렸다)
        org = (max(2, min(x1, canvas.shape[1] - tw - 2)), max(th + 4, y1 - 8))
        cv2.putText(canvas, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


class AlertSink:
    """RiskEvent 를 경보 서버로 보낸다. 실제 전송은 app-pos 의 AlertPublisher 가 별도 스레드에서 한다."""

    def __init__(self, server: str, camera_id: str = "cam1", client_path: str | Path = "../app-pos/client",
                 api_key: str | None = None, publisher: Any = None) -> None:
        if publisher is None:
            path = str(Path(client_path).resolve())
            if path not in sys.path:
                sys.path.insert(0, path)
            try:
                from hawkeye_client import AlertPublisher  # app-pos 의 전송 모듈
            except ImportError as e:
                raise ImportError(
                    f"경보 전송 모듈을 찾지 못했습니다: {path}/hawkeye_client "
                    "(app-pos 폴더가 있는지, alerts.client_path 설정을 확인하세요)"
                ) from e
            publisher = AlertPublisher(server, camera_id=camera_id, api_key=api_key)
        self.publisher = publisher
        self.server = server
        self.camera_id = camera_id

    @classmethod
    def create(cls, server: str | None, camera_id: str | None, client_path: str | Path,
               env: Mapping[str, str] = os.environ) -> "AlertSink | None":
        """서버 주소가 설정(인자 또는 HAWKEYE_SERVER)돼 있을 때만 만든다."""
        server = server or env.get("HAWKEYE_SERVER")
        if not server:
            return None
        return cls(server, camera_id or env.get("HAWKEYE_CAMERA") or "cam1", client_path,
                   api_key=env.get("HAWKEYE_API_KEY") or None)

    @property
    def run_id(self) -> str:
        return getattr(self.publisher, "run_id", "")

    def send(self, event: RiskEvent, image: np.ndarray | None = None, scale: float = 1.0) -> None:
        """대기열에 넣고 바로 돌아온다 (영상 처리를 멈추지 않는다)."""
        self.publisher.publish(event.to_dict(), jpeg=snapshot_jpeg(image, event, scale) if image is not None else None)

    def close(self) -> str:
        """남은 경보를 모두 보내고 결과 요약을 돌려준다."""
        self.publisher.close()
        sent = getattr(self.publisher, "sent", 0)
        failed = getattr(self.publisher, "failed", 0)
        dropped = getattr(self.publisher, "dropped", 0)
        line = f"경보 전송       : {self.server} 로 {sent}건 보냄"
        if failed or dropped:
            line += f" (실패 {failed}건, 대기열 넘쳐 버림 {dropped}건)"
        return line
