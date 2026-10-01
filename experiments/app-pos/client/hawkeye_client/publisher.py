"""경보 전송 클라이언트 — 경보를 관리자 앱의 경보 서버로 보낸다. 표준 라이브러리만 쓴다.

    from hawkeye_client import AlertPublisher, build_event

    alerts = AlertPublisher("http://127.0.0.1:8000", camera_id="cam1")
    alerts.publish(build_event(person_id=3, level="HIGH_RISK", taken=2, paid=1), jpeg=jpeg_bytes)
    ...
    alerts.close()   # 끝날 때 남은 경보를 모두 보낸다

호출한 쪽(영상 처리)을 멈추지 않는 것이 가장 중요하다.
- publish() 는 대기열에 넣고 바로 돌아온다. 전송은 별도 스레드가 한다.
- 서버가 꺼져 있거나 5xx 면 간격을 늘려 재시도하고, 그래도 안 되면 버린다(실패 수만 센다).
- 4xx(형식 오류·인증 실패)는 다시 보내도 같으므로 재시도하지 않는다.
- 대기열이 가득 차면 가장 오래된 경보부터 버린다.
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
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Callable

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
    ) -> None:
        self.url = server_url.rstrip("/") + "/api/events"
        self.camera_id = camera_id
        self.run_id = run_id or f"{datetime.now():%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
        """실행마다 다른 값. 다시 켜면 손님 번호가 1부터 다시 시작해도 서버에서 섞이지 않는다."""
        self.api_key = api_key
        self.transport = transport
        self.max_queue = max_queue
        self.max_attempts = max_attempts
        self.retry_wait = retry_wait
        self.timeout = timeout

        self.sent = 0
        self.failed = 0
        self.dropped = 0

        self._queue: collections.deque = collections.deque()
        self._cv = threading.Condition()
        self._closing = False
        self._thread = threading.Thread(target=self._run, name="hawkeye-alerts", daemon=True)
        self._thread.start()

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "AlertPublisher | None":
        """환경변수로 만든다. HAWKEYE_SERVER 가 없으면 None(전송 안 함).

        HAWKEYE_SERVER   경보 서버 주소 (예: http://127.0.0.1:8000)
        HAWKEYE_CAMERA   카메라 이름 (기본 cam1)
        HAWKEYE_API_KEY  서버에 키를 설정한 경우 (명령줄에 쓰지 않아 터미널 기록에 남지 않는다)
        """
        server = env.get("HAWKEYE_SERVER")
        if not server:
            return None
        return cls(server, camera_id=env.get("HAWKEYE_CAMERA") or "cam1", api_key=env.get("HAWKEYE_API_KEY") or None)

    def publish(self, event: dict[str, Any], jpeg: bytes | None = None) -> None:
        """대기열에 넣고 바로 돌아온다. jpeg 는 경보 순간의 장면 사진."""
        with self._cv:
            if len(self._queue) >= self.max_queue:
                old, _ = self._queue.popleft()
                self.dropped += 1
                log.warning("경보 대기열이 가득 차 오래된 경보를 버림: 손님 %s %s", old.get("person_id"), old.get("level"))
            self._queue.append((dict(event), jpeg))
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
                event, jpeg = self._queue.popleft()
            try:
                self._send(event, jpeg)
            except Exception:
                self.failed += 1
                log.exception("경보 전송 중 예상치 못한 오류")

    def _send(self, event: dict[str, Any], jpeg: bytes | None) -> None:
        data: dict[str, Any] = {"run_id": self.run_id, "camera_id": self.camera_id, "event": event}
        if jpeg is not None:
            data["snapshot_jpeg_b64"] = base64.b64encode(jpeg).decode("ascii")
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        reason = ""
        for attempt in range(1, self.max_attempts + 1):
            try:
                status = self.transport(self.url, body, headers, self.timeout)
            except Exception as e:  # 연결 실패 → 재시도
                reason = repr(e)
            else:
                reason = f"HTTP {status}"
                if 200 <= status < 300:
                    self.sent += 1
                    return
                if 400 <= status < 500:
                    self.failed += 1
                    log.warning("경보 서버가 거절함(%s): 손님 %s", reason, event.get("person_id"))
                    return
            if attempt < self.max_attempts:
                time.sleep(self.retry_wait * 2 ** (attempt - 1))

        self.failed += 1
        log.warning("경보 전송 포기(%d회 실패, 마지막 %s): 손님 %s", self.max_attempts, reason, event.get("person_id"))
