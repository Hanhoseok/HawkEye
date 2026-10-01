"""경보 전송기 — 위험 판정(RiskEvent)을 관리자 앱의 경보 서버로 보낸다.

서버 쪽 설계·API: experiments/app-pos/docs/design.md
켜는 법: run_tracking.py --alert-server http://<서버>:8000 (켜지 않으면 아무 일도 하지 않는다)

영상 처리를 멈추지 않는 것이 가장 중요하다.
- publish() 는 대기열에 넣고 바로 돌아온다. 사진 압축과 전송은 별도 스레드가 한다.
- 서버가 꺼져 있으면 간격을 늘려 가며 재시도하고, 그래도 안 되면 버린다(실패 수만 센다).
  경보는 risk.jsonl 에도 그대로 남으므로 잃어버리지 않는다.
- 대기열이 가득 차면 가장 오래된 경보부터 버린다.
- 4xx(형식 오류·인증 실패)는 다시 보내도 같으므로 재시도하지 않는다.
"""

from __future__ import annotations

import base64
import collections
import json
import logging
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from collections.abc import Mapping
from typing import Callable

import cv2
import numpy as np

from ..core.types import RiskEvent

log = logging.getLogger(__name__)

Transport = Callable[[str, bytes, dict, float], int]
"""(url, body, headers, timeout) -> HTTP 상태 코드. 연결 실패는 예외로 알린다."""


def http_post(url: str, body: bytes, headers: dict, timeout: float) -> int:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code


class AlertPublisher:
    def __init__(
        self,
        server_url: str,
        camera_id: str = "cam1",
        run_id: str | None = None,
        api_key: str | None = None,
        *,
        transport: Transport = http_post,
        max_queue: int = 100,
        max_attempts: int = 5,
        retry_wait: float = 1.0,
        timeout: float = 5.0,
        jpeg_quality: int = 80,
    ) -> None:
        self.url = server_url.rstrip("/") + "/api/events"
        self.camera_id = camera_id
        self.run_id = run_id or f"{datetime.now():%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
        """파이프라인을 다시 켜면 손님 번호가 1부터 다시 시작하므로, 실행마다 다른 값으로 구분한다."""
        self.api_key = api_key
        self.transport = transport
        self.max_queue = max_queue
        self.max_attempts = max_attempts
        self.retry_wait = retry_wait
        self.timeout = timeout
        self.jpeg_quality = jpeg_quality

        self.sent = 0
        self.failed = 0
        self.dropped = 0

        self._queue: collections.deque = collections.deque()
        self._cv = threading.Condition()
        self._closing = False
        self._thread = threading.Thread(target=self._run, name="alert-publisher", daemon=True)
        self._thread.start()

    @classmethod
    def from_options(cls, server_url: str | None, camera_id: str, env: Mapping[str, str]) -> "AlertPublisher | None":
        """run_tracking.py 옵션으로 만든다. 서버가 없으면 None(전송 안 함).

        API 키는 명령줄이 아니라 환경변수 HAWKEYE_API_KEY 로만 받는다 (터미널 기록에 남지 않게).
        """
        if not server_url:
            return None
        return cls(server_url, camera_id=camera_id, api_key=env.get("HAWKEYE_API_KEY") or None)

    def publish(self, event: RiskEvent, image: np.ndarray | None = None) -> None:
        """대기열에 넣고 바로 돌아온다. image 는 경보 순간의 프레임(BGR)."""
        item = (event, None if image is None else image.copy())
        with self._cv:
            if len(self._queue) >= self.max_queue:
                old, _ = self._queue.popleft()
                self.dropped += 1
                log.warning("경보 대기열이 가득 차 오래된 경보를 버림: 손님 %s %s", old.person_id, old.level.value)
            self._queue.append(item)
            self._cv.notify()

    def close(self, timeout: float | None = 30.0) -> None:
        """남은 경보를 모두 보낸 뒤 끝낸다."""
        with self._cv:
            self._closing = True
            self._cv.notify()
        self._thread.join(timeout)

    # --- 전송 스레드 ---
    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._queue and not self._closing:
                    self._cv.wait()
                if not self._queue:
                    return
                event, image = self._queue.popleft()
            try:
                self._send(event, image)
            except Exception:
                self.failed += 1
                log.exception("경보 전송 중 예상치 못한 오류")

    def _payload(self, event: RiskEvent, image: np.ndarray | None) -> bytes:
        data = {"run_id": self.run_id, "camera_id": self.camera_id, "event": event.to_dict()}
        if image is not None:
            x1, y1, x2, y2 = (int(v) for v in event.bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if ok:
                data["snapshot_jpeg_b64"] = base64.b64encode(jpeg.tobytes()).decode("ascii")
        return json.dumps(data, ensure_ascii=False).encode("utf-8")

    def _send(self, event: RiskEvent, image: np.ndarray | None) -> None:
        body = self._payload(event, image)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        for attempt in range(1, self.max_attempts + 1):
            try:
                status = self.transport(self.url, body, headers, self.timeout)
            except Exception as e:  # 연결 실패 → 재시도
                status, reason = None, repr(e)
            else:
                reason = f"HTTP {status}"
                if 200 <= status < 300:
                    self.sent += 1
                    return
                if 400 <= status < 500:
                    self.failed += 1
                    log.warning("경보 서버가 거절함(%s): 손님 %s", reason, event.person_id)
                    return
            if attempt < self.max_attempts:
                time.sleep(self.retry_wait * 2 ** (attempt - 1))

        self.failed += 1
        log.warning("경보 전송 포기(%d회 실패, 마지막 %s): 손님 %s", self.max_attempts, reason, event.person_id)
