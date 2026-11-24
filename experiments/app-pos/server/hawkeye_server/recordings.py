"""녹화 영상 — MediaMTX 재생 API 를 감싸 사건 클립·지난 영상을 앱에 준다 (docs/design.md §10).

MediaMTX 재생 API (127.0.0.1 에서만 열림, 서버만 쓴다):
    GET /list?path=cam1[&start=&end=]        → [{"start": RFC3339, "duration": 초, "url": ...}]
    GET /get?path=cam1&start=&duration=&format=mp4 → MP4 (녹화된 만큼만)
0단계 시험: 끝이 1초 이상 지난 구간은 온전히 받을 수 있다 → 클립은 구간 끝 + 2초 뒤에 준비된다.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import httpx

CAMERA_ID = re.compile(r"^cam[0-9]+$")

Anchor = Literal["event", "received"]


class RecordingsUnavailable(Exception):
    """재생 API 에 닿지 않음 (MediaMTX 꺼짐 등)."""


# --- 시각 ------------------------------------------------------------------

def parse_abs_time(value: str | None) -> datetime | None:
    """오프셋이 있는 ISO8601 이면 UTC datetime, 아니면 None ('00:00:34.766' 같은 상대 시각 포함)."""
    if not value or "T" not in value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return None
    return dt.astimezone(timezone.utc)


def rfc3339(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class ClipWindow:
    anchor: Anchor
    """event = 경보에 담긴 프레임 시각(실시간 입력), received = 서버가 받은 시각(대체값)"""
    anchor_time: datetime
    start: datetime
    end: datetime
    delay_sec: float | None
    """서버가 받은 시각 − 경보 시각 (전송 지연). anchor 가 received 면 None"""


def clip_window(event_timestamp: str | None, received_at: str, before_sec: float = 20.0,
                after_sec: float = 10.0) -> ClipWindow:
    """경보 하나의 클립 구간. 실시간 경보는 프레임 시각을, 녹화 파일·시뮬레이터 경보는 받은 시각을 기준으로 한다."""
    received = parse_abs_time(received_at)
    if received is None:
        raise ValueError(f"received_at 이 절대 시각이 아닙니다: {received_at}")
    event = parse_abs_time(event_timestamp)
    if event is not None:
        anchor, at, delay = "event", event, round((received - event).total_seconds(), 3)
    else:
        anchor, at, delay = "received", received, None
    return ClipWindow(anchor, at, at - timedelta(seconds=before_sec), at + timedelta(seconds=after_sec), delay)


@dataclass(frozen=True)
class Span:
    start: datetime
    duration: float

    @property
    def end(self) -> datetime:
        return self.start + timedelta(seconds=self.duration)


def covered_seconds(spans: list[Span], start: datetime, end: datetime) -> float:
    """[start, end) 중 녹화가 있는 시간(초)."""
    total = 0.0
    for s in spans:
        lo, hi = max(s.start, start), min(s.end, end)
        if hi > lo:
            total += (hi - lo).total_seconds()
    return round(total, 3)


# --- 재생 API 클라이언트 ------------------------------------------------------

class Playback:
    def __init__(self, base_url: str, transport: httpx.BaseTransport | None = None, timeout: float = 15.0) -> None:
        self._http = httpx.Client(base_url=base_url.rstrip("/"), transport=transport, timeout=timeout)

    def list(self, camera_id: str, start: datetime | None = None, end: datetime | None = None) -> list[Span]:
        params = {"path": camera_id}
        if start is not None:
            params["start"] = rfc3339(start)
        if end is not None:
            params["end"] = rfc3339(end)
        try:
            r = self._http.get("/list", params=params)
        except httpx.HTTPError as e:
            raise RecordingsUnavailable(str(e)) from e
        if r.status_code == 404:          # 그 경로에 녹화가 없음
            return []
        if r.status_code != 200:
            raise RecordingsUnavailable(f"/list HTTP {r.status_code}")
        spans = []
        for item in r.json():
            st = parse_abs_time(item.get("start"))
            if st is not None:
                spans.append(Span(st, float(item.get("duration", 0.0))))
        return spans

    def download(self, camera_id: str, start: datetime, duration: float, dest: Path) -> bool:
        """[start, start+duration) 을 MP4 로 dest 에 쓴다. 녹화가 없으면 False."""
        params = {"path": camera_id, "start": rfc3339(start), "duration": f"{duration:.3f}", "format": "mp4"}
        try:
            with self._http.stream("GET", "/get", params=params) as r:
                if r.status_code == 404:
                    return False
                if r.status_code != 200:
                    raise RecordingsUnavailable(f"/get HTTP {r.status_code}")
                with open(dest, "wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
        except httpx.HTTPError as e:
            raise RecordingsUnavailable(str(e)) from e
        return dest.stat().st_size > 0


# --- 파일 캐시 ---------------------------------------------------------------

class FileCache:
    """MP4 파일 캐시 (+ 파일마다 정보 한 줄 `<이름>.json`). 크기·나이 상한을 넘으면 오래된 것부터 지운다.

    - 같은 이름을 동시에 만들면 한쪽만 만든다 (이름별 잠금). 임시 파일(.part)에 쓰고 이름을 바꿔 반쯤 쓴 파일을 내주지 않는다.
    - 조회·이름 바꾸기·정리는 캐시 전체 잠금 하나로 순서를 맞춘다 (정리끼리 겹치면 같은 파일을 두 번 지우려다 실패).
    - 방금 쓴(조회한) 파일은 `protect_sec` 동안 지우지 않는다 — 응답을 보내는 중에 다른 요청의 정리가 지우지 않게.
    """

    def __init__(self, directory: Path, max_bytes: int, max_age_sec: float,
                 clock: Callable[[], float] = time.time, protect_sec: float = 120.0) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes, self.max_age, self.protect = max_bytes, max_age_sec, protect_sec
        self._clock = clock
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        self._state = threading.RLock()
        for leftover in self.dir.glob("*.part"):      # 지난 실행에서 남은 임시 파일
            leftover.unlink(missing_ok=True)

    def _lock(self, name: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(name, threading.Lock())

    def path(self, name: str) -> Path:
        return self.dir / name

    @staticmethod
    def _meta_path(final: Path) -> Path:
        return final.with_name(final.name + ".json")

    def get(self, name: str) -> tuple[Path, dict] | None:
        """있으면 (파일, 정보). 최근 쓴 것으로 표시한다 (정리 때 보호)."""
        final = self.path(name)
        with self._state:
            try:
                meta = json.loads(self._meta_path(final).read_text(encoding="utf-8"))
                now = self._clock()
                os.utime(final, (now, now))
            except (FileNotFoundError, ValueError):
                return None
            return final, meta

    def get_or_make(self, name: str, make: Callable[[Path], dict | None]) -> tuple[Path, dict] | None:
        """있으면 (파일, 정보), 없으면 make(임시 경로) 로 만든다. make 는 정보(dict) 또는 None(만들 것이 없음)."""
        final = self.path(name)
        with self._lock(name):
            hit = self.get(name)
            if hit is not None:
                return hit
            part = final.with_name(final.name + f".{threading.get_ident()}.part")
            try:
                meta = make(part)
                if meta is None:
                    return None
                with self._state:
                    os.replace(part, final)
                    self._meta_path(final).write_text(json.dumps(meta), encoding="utf-8")
                    now = self._clock()
                    os.utime(final, (now, now))
            finally:
                part.unlink(missing_ok=True)
        self.prune()
        return final, meta

    def _files(self) -> list[Path]:
        return [p for p in self.dir.iterdir() if p.suffix == ".mp4"]

    def _remove(self, p: Path) -> None:
        p.unlink(missing_ok=True)
        self._meta_path(p).unlink(missing_ok=True)

    def prune(self) -> None:
        with self._state:
            now = self._clock()
            for meta in self.dir.glob("*.mp4.json"):    # 영상 없이 남은 정보 파일
                if not meta.with_suffix("").exists():
                    meta.unlink(missing_ok=True)
            keep = []
            for p in self._files():
                try:
                    st = p.stat()
                except FileNotFoundError:          # 열거한 뒤 사라짐
                    continue
                if now - st.st_mtime > self.max_age:
                    self._remove(p)
                else:
                    keep.append((st.st_mtime, st.st_size, p))
            total = sum(size for _, size, _ in keep)
            for mtime, size, p in sorted(keep):        # 오래된 것부터
                if total <= self.max_bytes:
                    break
                if now - mtime < self.protect:         # 방금 쓴 것은 남긴다 (잠깐 상한을 넘을 수 있음)
                    continue
                self._remove(p)
                total -= size


# --- 서비스 ------------------------------------------------------------------

def first_recorded(spans: list[Span], start: datetime, end: datetime) -> datetime | None:
    """[start, end) 안에서 녹화가 처음 시작되는 시각.

    MediaMTX /get 은 요청 시작 시각이 녹화 구간 안에 있어야 한다 (2단계 실기 시험: 그 앞에 이전 녹화가 있으면
    그 녹화를 기준으로 찾다가 'no recording segments found' 로 404). 그래서 실제 녹화가 시작된 시각부터 요청한다.
    """
    starts = [max(s.start, start) for s in spans if s.end > start and s.start < end]
    return min(starts) if starts else None


@dataclass
class ClipResult:
    status: Literal["ready", "not_ready", "no_recording"]
    path: Path | None = None
    retry_after: int = 0
    covered: float = 0.0
    wanted: float = 0.0
    start: datetime | None = None
    """실제 영상이 시작되는 시각 (구간 앞부분에 녹화가 없으면 요청보다 늦다)"""


class Recordings:
    """경보 서버가 쓰는 녹화 기능 묶음."""

    def __init__(self, playback: Playback, clips: FileCache, chunks: FileCache,
                 flush_delay_sec: float = 2.0, now: Callable[[], datetime] | None = None) -> None:
        self.playback, self.clips, self.chunks = playback, clips, chunks
        self.flush_delay = flush_delay_sec
        self._now = now or (lambda: datetime.now(timezone.utc))

    def now(self) -> datetime:
        return self._now()

    def retry_after(self, end: datetime) -> int:
        """end 까지 녹화가 파일로 다 쓰이려면 몇 초 더 기다려야 하나 (0 이면 지금 받을 수 있음)."""
        wait = (end + timedelta(seconds=self.flush_delay) - self.now()).total_seconds()
        return max(0, int(wait + 0.999))

    @staticmethod
    def _ready(path: Path, meta: dict) -> ClipResult:
        return ClipResult("ready", path=path, covered=meta["covered"], wanted=meta["wanted"],
                          start=parse_abs_time(meta["start"]))

    def _fetch(self, cache: FileCache, name: str, camera_id: str, start: datetime, end: datetime) -> ClipResult:
        wanted = round((end - start).total_seconds(), 3)
        hit = cache.get(name)                 # 먼저 캐시 — 원본 녹화를 지웠거나 MediaMTX 가 꺼져도 이미 만든 영상은 준다
        if hit is not None:
            return self._ready(*hit)
        wait = self.retry_after(end)
        if wait > 0:
            return ClipResult("not_ready", retry_after=wait, wanted=wanted)
        spans = self.playback.list(camera_id, start - timedelta(seconds=1), end + timedelta(seconds=1))
        covered = covered_seconds(spans, start, end)
        real_start = first_recorded(spans, start, end)
        if covered <= 0 or real_start is None:
            return ClipResult("no_recording", wanted=wanted)
        length = round((end - real_start).total_seconds(), 3)

        def make(tmp: Path) -> dict | None:
            if not self.playback.download(camera_id, real_start, length, tmp):
                return None
            return {"start": rfc3339(real_start), "covered": covered, "wanted": wanted}

        made = cache.get_or_make(name, make)
        if made is None:
            return ClipResult("no_recording", wanted=wanted)
        return self._ready(*made)

    @staticmethod
    def _clip_name(case_id: int, camera_id: str, window: ClipWindow) -> str:
        return f"case{case_id}_{camera_id}_{int(window.start.timestamp() * 1000)}.mp4"

    def cached_clip(self, case_id: int, camera_id: str, window: ClipWindow) -> ClipResult | None:
        hit = self.clips.get(self._clip_name(case_id, camera_id, window))
        return self._ready(*hit) if hit is not None else None

    def clip(self, case_id: int, camera_id: str, window: ClipWindow) -> ClipResult:
        return self._fetch(self.clips, self._clip_name(case_id, camera_id, window), camera_id, window.start, window.end)

    def chunk(self, camera_id: str, start: datetime, duration: float) -> ClipResult:
        end = start + timedelta(seconds=duration)
        name = f"{camera_id}_{int(start.timestamp() * 1000)}_{int(duration * 1000)}.mp4"
        return self._fetch(self.chunks, name, camera_id, start, end)

    def live_status(self, camera_id: str, within_sec: float = 5.0) -> dict[str, object]:
        """송출 중인가 (마지막 녹화 구간의 끝이 지금에 가까우면), 언제부터 이어서 송출 중인가."""
        now = self.now()
        spans = self.playback.list(camera_id, now - timedelta(hours=24), None)
        if not spans:
            return {"streaming": False, "since": None}
        spans = sorted(spans, key=lambda s: s.start)
        last = spans[-1]
        if (now - last.end).total_seconds() > within_sec:
            return {"streaming": False, "since": None}
        since = last.start
        for prev in reversed(spans[:-1]):          # 녹화 파일은 10분마다 나뉘므로 이어진 구간을 거슬러 합친다
            if (since - prev.end).total_seconds() > 2.0:
                break
            since = prev.start
        return {"streaming": True, "since": rfc3339(since)}
