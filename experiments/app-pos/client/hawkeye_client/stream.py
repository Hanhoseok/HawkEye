"""AI 화면 송출 — 파이프라인이 그린 이미지(사람 상자·번호·구역·경보)를 MediaMTX `<카메라>_ai` 경로로 보낸다.

    from hawkeye_client import FrameStreamer

    ai = FrameStreamer()                 # 송출 주소: HAWKEYE_AI_RTSP → tailscale ip -4 → 127.0.0.1 (포트 8554)
    ai.push("cam1", annotated_bgr)       # 바로 돌아온다 (numpy BGR 배열, (높이, 폭, 3) uint8)
    ...
    ai.close()

관리자 앱은 `/api/config` 의 `ai_url` (rtsp://.../cam1_ai) 로 이 화면을 본다. 설계: ../../docs/design.md §10.

호출한 쪽(영상 처리)을 멈추지 않는 것이 가장 중요하다.
- push() 는 카메라마다 '최신 장면 1장' 칸만 갈아 끼우고 돌아온다. 대기열이 없어 밀리면 자연히 버려진다(버린 수를 센다).
- 카메라마다 송출 스레드가 고정 속도(fps)로 최신 장면을 ffmpeg 표준 입력에 쓴다. 새 장면이 없으면 직전 장면을
  다시 쓴다 — 처리 간격이 들쭉날쭉해도 영상 시간축이 일정하다.
- 안드로이드 호환: 두 인코더 모두 yuv420p · H.264 Main(libx264 ultrafast 는 Constrained Baseline) · B-프레임 없음 ·
  1초마다 키프레임.
- 인코더: 처음 송출 전에 h264_nvenc 로 한 장 시험 인코딩해 보고, 안 되면 libx264. 환경변수 HAWKEYE_AI_ENCODER 로 고정 가능.
- ffmpeg 가 죽으면 간격을 늘려 다시 띄운다. 장면 크기가 바뀌면 다시 띄운다.
- 새 장면이 idle_stop_sec 동안 없으면 송출을 멈춘다 — 파이프라인이 멈췄는데 마지막 장면을 '라이브'처럼 계속 보여 주지
  않게. 새 장면이 오면 다시 띄운다.

표준 라이브러리만 쓴다. 이미지는 .shape · .tobytes() 만 쓰므로 numpy 를 import 하지 않는다.
"""

from __future__ import annotations

import collections
import logging
import os
import re
import subprocess
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable

log = logging.getLogger(__name__)

ENCODERS = ("h264_nvenc", "libx264")
CAMERA_ID = re.compile(r"^cam[0-9]+$")
"""MediaMTX 규칙: 서버 안에서는 계정 없이 `cam<번호>_ai` 에만 송출할 수 있다 (relay/mediamtx.server.yml)."""

ENCODER_ERROR = re.compile(
    r"(?i)OpenEncodeSession|Cannot load \S*nv|No (?:NVENC )?capable devices|CUDA_ERROR|cuInit|nvenc API"
    r"|opening encoder|Error initializing output stream"
)
"""실행 중 nvenc ffmpeg 가 곧바로 죽었을 때 인코더 탓인지 가리는 stderr 문구 (MediaMTX 가 꺼진 것과 구별)."""

Spawn = Callable[[list[str]], Any]
"""ffmpeg 인자 -> Popen 같은 객체 (stdin, stderr, poll, wait, kill)."""


def default_rtsp_base() -> str:
    """송출할 MediaMTX 주소. MediaMTX 는 Tailscale IP 에만 열려 있다 (relay/run-server.sh)."""
    env = os.environ.get("HAWKEYE_AI_RTSP", "").strip()
    if env:
        return env.rstrip("/")
    ip = ""
    try:
        out = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=3)
        words = out.stdout.split() if out.returncode == 0 else []
        ip = words[0] if words else ""
    except (OSError, subprocess.SubprocessError):
        pass
    return f"rtsp://{ip or '127.0.0.1'}:8554"


def safe_url(url: str) -> str:
    """로그에 남길 주소: 계정(user:pass@)을 지운다."""
    return re.sub(r"//[^/@]+@", "//", url)


def encoder_args(encoder: str, fps: float, bitrate: str) -> list[str]:
    """두 인코더 공통: Main 프로파일 · B-프레임 없음 · 1초마다 키프레임 (앱이 중간에 붙어도 바로 화면)."""
    gop = max(1, round(fps))
    common = ["-profile:v", "main", "-g", str(gop), "-bf", "0", "-b:v", bitrate]
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ll", *common]
    if encoder == "libx264":
        return ["-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", *common]
    raise ValueError(f"모르는 인코더: {encoder} ({', '.join(ENCODERS)} 중 하나)")


def ffmpeg_args(
    ffmpeg: Sequence[str],
    encoder: str,
    width: int,
    height: int,
    fps: float,
    url: str,
    *,
    out_width: int = 1280,
    bitrate: str = "2M",
) -> list[str]:
    """원본 크기 BGR 장면을 표준 입력으로 받아 폭 out_width 이하로 줄이고 yuv420p 로 바꿔 RTSP 로 보낸다.

    입력이 bgr24 라 출력 형식을 정하지 않으면 libx264 가 yuv444p(High 4:4:4)를 만든다 — 안드로이드는 못 튼다.
    폭은 줄이기만 하고(작은 입력은 그대로) 폭·높이 모두 짝수로 맞춘다 (yuv420p 조건).
    """
    scale = f"scale='trunc(min({out_width},iw)/2)*2':-2,format=yuv420p"
    return [
        *ffmpeg, "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-",
        "-vf", scale,
        *encoder_args(encoder, fps, bitrate),
        "-an", "-f", "rtsp", "-rtsp_transport", "tcp", url,
    ]


def probe_args(ffmpeg: Sequence[str], encoder: str) -> list[str]:
    """인코더가 이 서버에서 되는지 한 장만 인코딩해 본다 (어디에도 보내지 않음)."""
    return [
        *ffmpeg, "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=black:s=256x144:r=1", "-frames:v", "1",
        "-vf", "format=yuv420p", *encoder_args(encoder, 1, "1M"), "-f", "null", "-",
    ]


def popen(args: list[str]) -> subprocess.Popen:
    return subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


@dataclass(frozen=True)
class _Frame:
    width: int
    height: int
    data: bytes


class FrameStreamer:
    def __init__(
        self,
        rtsp_base: str | None = None,
        fps: float = 10.0,
        out_width: int = 1280,
        bitrate: str = "2M",
        *,
        encoder: str | None = None,
        ffmpeg: str | Sequence[str] = "ffmpeg",
        spawn: Spawn = popen,
        restart_wait: float = 1.0,
        max_restart_wait: float = 30.0,
        healthy_sec: float = 10.0,
        idle_stop_sec: float = 10.0,
        write_timeout: float = 5.0,
        close_timeout: float = 5.0,
        stats_every: float = 60.0,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps 는 0보다 커야 합니다")
        self.rtsp_base = (rtsp_base or default_rtsp_base()).rstrip("/")
        self.fps = float(fps)
        self.out_width = int(out_width)
        self.bitrate = bitrate
        self.ffmpeg = [ffmpeg] if isinstance(ffmpeg, str) else list(ffmpeg)
        self.spawn = spawn
        self.restart_wait = restart_wait
        self.max_restart_wait = max_restart_wait
        self.healthy_sec = healthy_sec
        """이만큼 버틴 ffmpeg 는 '정상'으로 보고 재시작 간격을 처음으로 되돌린다."""
        self.idle_stop_sec = idle_stop_sec
        self.write_timeout = write_timeout
        """장면 한 장 쓰기가 이보다 오래 막히면(ffmpeg 가 살아 있지만 입력을 안 받음) 감시 스레드가 ffmpeg 를 끝낸다."""
        self.close_timeout = close_timeout
        self.stats_every = stats_every

        forced = encoder or os.environ.get("HAWKEYE_AI_ENCODER", "").strip() or None
        if forced is not None and forced not in ENCODERS:
            raise ValueError(f"HAWKEYE_AI_ENCODER={forced!r} — {', '.join(ENCODERS)} 중 하나")
        self._encoder: str | None = forced
        self._encoder_lock = threading.Lock()
        self._probe: subprocess.Popen | None = None

        self._channels: dict[str, _Channel] = {}
        self._channels_lock = threading.Lock()
        self._watchdog: threading.Thread | None = None
        self._stop = threading.Event()

    # ---- 파이프라인 쪽 ----

    def push(self, camera_id: str, image: Any) -> None:
        """그린 장면을 넘긴다. 바로 돌아온다. 이미지는 복사하므로 호출 뒤 같은 배열을 다시 써도 된다.

        잘못된 카메라 이름·이미지(BGR (높이, 폭, 3) uint8 이 아님)는 ValueError — 설정 실수라 처음 장면에서 드러난다.
        close() 뒤에는 무시한다.
        """
        if self._stop.is_set():
            return
        if not isinstance(camera_id, str) or not CAMERA_ID.match(camera_id):
            raise ValueError(f"카메라 이름은 cam1, cam2 처럼 써야 합니다: {camera_id!r}")
        shape = getattr(image, "shape", None)
        if shape is None or len(shape) != 3 or int(shape[2]) != 3:
            raise ValueError(f"BGR 이미지 (높이, 폭, 3) 가 아닙니다: shape={shape!r}")
        dtype = getattr(image, "dtype", None)
        if dtype is not None and str(dtype) != "uint8":
            raise ValueError(f"이미지 형식은 uint8 이어야 합니다: {dtype}")
        height, width = int(shape[0]), int(shape[1])
        if width <= 0 or height <= 0:
            raise ValueError(f"빈 이미지입니다: shape={shape!r}")
        data = image.tobytes()
        if len(data) != width * height * 3:
            raise ValueError(f"이미지 바이트 수가 맞지 않습니다: {len(data)} != {width}x{height}x3")
        self._channel(camera_id).offer(_Frame(width, height, data))

    def stats(self) -> dict[str, dict[str, int]]:
        """카메라별 누적 수: 보낸 장면(sent)·버린 장면(dropped)·ffmpeg 재시작(restarts)."""
        with self._channels_lock:
            channels = list(self._channels.values())
        return {c.camera_id: {"sent": c.sent, "dropped": c.dropped, "restarts": c.restarts} for c in channels}

    @property
    def encoder(self) -> str | None:
        """지금 쓰는 인코더. 첫 송출 전에는 None 일 수 있다."""
        return self._encoder

    def close(self, timeout: float | None = None) -> None:
        """송출을 끝낸다. ffmpeg 표준 입력을 닫고 기다리다 제한 시간이 지나면 강제 종료한다."""
        timeout = self.close_timeout if timeout is None else timeout
        deadline = time.monotonic() + max(0.0, timeout)
        self._deadline = deadline
        self._stop.set()
        with self._channels_lock:
            channels = list(self._channels.values())
        for c in channels:
            c.wake.set()
        for c in channels:
            c.join(max(0.0, deadline - time.monotonic()))
        if any(c.is_alive() for c in channels):
            self._kill_probe()               # 인코더 시험이 늦어지는 경우
            for c in channels:               # ffmpeg 가 입력을 안 받아 쓰기에서 막힌 경우
                c.kill_process()
            for c in channels:
                c.join(2.0)
        for c in channels:
            if c.is_alive():
                log.warning("%s: 송출 스레드가 끝나지 않았습니다", c.path)

    def __enter__(self) -> "FrameStreamer":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- 내부 ----

    _deadline: float = 0.0

    def _remaining(self) -> float:
        return max(0.0, self._deadline - time.monotonic())

    def _channel(self, camera_id: str) -> "_Channel":
        ch = self._channels.get(camera_id)
        if ch is not None:
            return ch
        with self._channels_lock:
            ch = self._channels.get(camera_id)
            if ch is None:
                ch = _Channel(self, camera_id)
                self._channels[camera_id] = ch
                ch.start()
                if self._watchdog is None:
                    self._watchdog = threading.Thread(target=self._watch, name="hawkeye-ai-watchdog", daemon=True)
                    self._watchdog.start()
        return ch

    def _watch(self) -> None:
        """송출 스레드가 ffmpeg 쓰기에서 막히면 그 스레드는 스스로 빠져나올 수 없다 — 밖에서 ffmpeg 를 끝내 준다."""
        interval = max(0.05, min(0.5, self.write_timeout / 2))
        while not self._stop.wait(interval):
            with self._channels_lock:
                channels = list(self._channels.values())
            now = time.monotonic()
            for c in channels:
                c.check_stall(now)

    def _kill_probe(self) -> None:
        proc = self._probe
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass

    def _resolve_encoder(self) -> str:
        """첫 송출 때 한 번: h264_nvenc 로 한 장 인코딩해 보고 안 되면 libx264. close() 가 끊을 수 있게 Popen 으로 둔다."""
        with self._encoder_lock:
            if self._encoder is None:
                if self._stop.is_set():
                    return "libx264"             # 다른 카메라 시험을 기다리는 사이 close() — 새 시험을 띄우지 않는다
                try:
                    self._probe = subprocess.Popen(probe_args(self.ffmpeg, "h264_nvenc"), stdin=subprocess.DEVNULL,
                                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    try:
                        _, err = self._probe.communicate(timeout=20)
                    except subprocess.TimeoutExpired:
                        self._probe.kill()
                        _, err = self._probe.communicate()
                        err = (err or b"") + "시험 인코딩이 20초 안에 끝나지 않음".encode()
                    ok = self._probe.returncode == 0
                    detail = err.decode("utf-8", "replace").strip().splitlines()[-1:]
                except OSError as e:
                    ok, detail = False, [str(e)]
                finally:
                    self._probe = None
                if self._stop.is_set():
                    return "libx264"             # 끝내는 중에 끊긴 시험 — 결과로 삼지 않는다
                if ok:
                    self._encoder = "h264_nvenc"
                    log.info("AI 화면 인코더: h264_nvenc (GPU)")
                else:
                    self._encoder = "libx264"
                    log.warning("h264_nvenc 를 쓸 수 없어 libx264 로 보냅니다: %s", " ".join(detail) or "이유 모름")
            return self._encoder

    def _fall_back(self, reason: str) -> None:
        with self._encoder_lock:
            if self._encoder == "h264_nvenc":
                self._encoder = "libx264"
                log.warning("h264_nvenc 가 실패해 libx264 로 바꿉니다: %s", reason)


class _Channel(threading.Thread):
    """카메라 하나의 송출: 최신 장면 칸 + 고정 속도로 ffmpeg 에 쓰는 스레드."""

    def __init__(self, owner: FrameStreamer, camera_id: str) -> None:
        super().__init__(name=f"hawkeye-ai-{camera_id}", daemon=True)
        self.owner = owner
        self.camera_id = camera_id
        self.path = f"{camera_id}_ai"
        self.url = f"{owner.rtsp_base}/{self.path}"
        self.wake = threading.Event()

        self._lock = threading.Lock()
        self._latest: _Frame | None = None
        self._fresh = False
        self._last_new = 0.0

        self.sent = 0
        self.dropped = 0
        self.restarts = 0
        self._starts = 0

        self._proc: Any = None
        self._proc_size: tuple[int, int] | None = None
        self._proc_encoder = ""
        self._proc_started = 0.0
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=20)
        self._stderr_thread: threading.Thread | None = None
        self._backoff = owner.restart_wait
        self._next_start = 0.0
        self._write_since = 0.0              # 지금 쓰는 장면을 쓰기 시작한 시각 (0 = 쓰는 중 아님)
        self._stalled = False

        self._summary_at = time.monotonic()
        self._summary_sent = 0
        self._summary_dropped = 0

    # push() 가 부른다 (호출한 쪽 스레드)
    def offer(self, frame: _Frame) -> None:
        with self._lock:
            if self._fresh:
                self.dropped += 1            # 아직 보내지 못한 장면을 새 장면으로 덮음
            self._latest = frame
            self._fresh = True
            self._last_new = time.monotonic()
        self.wake.set()

    def check_stall(self, now: float) -> None:
        """감시 스레드가 부른다: 장면 한 장 쓰기가 write_timeout 넘게 막혀 있으면 ffmpeg 를 끝낸다 → 쓰기가 풀려 재시작."""
        since = self._write_since
        if since and not self._stalled and now - since > self.owner.write_timeout:
            self._stalled = True
            log.warning("%s: ffmpeg 가 %.0f초 넘게 장면을 받지 않아 끝냅니다", self.path, self.owner.write_timeout)
            self.kill_process()

    def kill_process(self) -> None:
        proc = self._proc
        if proc is not None:
            try:
                proc.kill()
            except OSError:
                pass

    def run(self) -> None:
        stop = self.owner._stop
        period = 1.0 / self.owner.fps
        next_tick = time.monotonic()
        try:
            while not stop.is_set():
                delay = next_tick - time.monotonic()
                if delay > 0 and stop.wait(delay):
                    break
                now = time.monotonic()
                next_tick += period
                if now - next_tick > period:     # 한참 밀렸으면(쓰기가 막혔다 풀림) 시간축을 지금으로 다시 맞춤
                    next_tick = now + period
                self._summary(now)

                with self._lock:
                    frame, last_new = self._latest, self._last_new
                    self._fresh = False
                if frame is None or now - last_new > self.owner.idle_stop_sec:
                    if self._proc is not None:
                        log.info("%s: 새 장면이 %.0f초 없어 송출을 멈춥니다 (새 장면이 오면 다시 보냄)",
                                 self.path, self.owner.idle_stop_sec)
                        self._stop_process(self.owner.close_timeout)
                    self._wait_for_frame()
                    next_tick = time.monotonic()
                    continue
                self._send(frame, now)
        finally:
            self._summary(time.monotonic(), force=True)
            self._stop_process(self.owner._remaining() if stop.is_set() else self.owner.close_timeout)

    def _wait_for_frame(self) -> None:
        stop = self.owner._stop
        self.wake.clear()
        with self._lock:
            if self._latest is not None and time.monotonic() - self._last_new <= self.owner.idle_stop_sec:
                return                           # clear 직전에 새 장면이 들어옴
        while not stop.is_set() and not self.wake.wait(0.5):
            pass

    def _send(self, frame: _Frame, now: float) -> None:
        if self._proc is not None and self._proc.poll() is not None:
            self._on_exit("ffmpeg 가 끝났습니다")
        size = (frame.width, frame.height)
        if self._proc is not None and self._proc_size != size:
            log.info("%s: 장면 크기가 %sx%s 로 바뀌어 ffmpeg 를 다시 띄웁니다", self.path, *size)
            self._stop_process(self.owner.close_timeout)
            self._next_start = 0.0
        if self._proc is None:
            if now < self._next_start or not self._start(size):
                return
        self._write_since = time.monotonic()
        try:
            self._proc.stdin.write(frame.data)
            self._proc.stdin.flush()
            ok = True
        except (OSError, ValueError):
            ok = False
        finally:
            self._write_since = 0.0
        if ok:
            self.sent += 1
        else:
            self._on_exit("멈춘 ffmpeg 를 끝냈습니다" if self._stalled else "ffmpeg 에 장면을 쓰지 못했습니다")

    def _start(self, size: tuple[int, int]) -> bool:
        owner = self.owner
        encoder = owner._resolve_encoder()
        if owner._stop.is_set():
            return False
        args = ffmpeg_args(owner.ffmpeg, encoder, size[0], size[1], owner.fps, self.url,
                           out_width=owner.out_width, bitrate=owner.bitrate)
        try:
            proc = owner.spawn(args)
        except OSError as e:
            log.error("%s: ffmpeg 를 실행하지 못했습니다: %s", self.path, e)
            self._schedule_restart(ran=0.0)
            return False
        self._proc, self._proc_size, self._proc_encoder = proc, size, encoder
        self._proc_started = time.monotonic()
        self._stalled = False
        self._stderr_tail.clear()
        self._stderr_thread = None
        if getattr(proc, "stderr", None) is not None:
            self._stderr_thread = threading.Thread(target=self._read_stderr, args=(proc.stderr,),
                                                   name=f"hawkeye-ai-{self.camera_id}-stderr", daemon=True)
            self._stderr_thread.start()
        if self._starts:
            self.restarts += 1
        self._starts += 1
        log.info("%s 송출 시작 (%s, %sx%s → 폭 최대 %d, %g장/초): %s",
                 self.path, encoder, size[0], size[1], owner.out_width, owner.fps, safe_url(self.url))
        return True

    def _read_stderr(self, stream: Any) -> None:
        try:
            for raw in iter(stream.readline, b""):
                line = raw.decode("utf-8", "replace").strip() if isinstance(raw, bytes) else str(raw).strip()
                if line:
                    self._stderr_tail.append(line)
                    log.warning("[%s ffmpeg] %s", self.path, safe_url(line))
        except (OSError, ValueError):
            pass

    def _on_exit(self, what: str) -> None:
        """ffmpeg 가 죽었거나 쓰기에 실패함: 정리하고, 인코더 탓이면 libx264 로, 간격을 두고 다시 띄운다."""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        code = self._reap(proc, 2.0)
        ran = time.monotonic() - self._proc_started
        tail = " | ".join(self._stderr_tail)
        if self._proc_encoder == "h264_nvenc" and ran < self.owner.healthy_sec and ENCODER_ERROR.search(tail):
            self.owner._fall_back(tail[-300:])
        wait = self._schedule_restart(ran)
        log.warning("%s: %s (코드 %s, %.1f초 실행) — %.1f초 뒤 다시 띄움", self.path, what, code, ran, wait)

    def _schedule_restart(self, ran: float) -> float:
        if ran >= self.owner.healthy_sec:
            self._backoff = self.owner.restart_wait
        wait = self._backoff
        self._backoff = min(self._backoff * 2, self.owner.max_restart_wait)
        self._next_start = time.monotonic() + wait
        return wait

    def _stop_process(self, wait: float) -> None:
        """정상 종료: 표준 입력을 닫으면 ffmpeg 가 남은 것을 보내고 끝난다. 제한 시간이 지나면 강제 종료."""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.stdin.close()
        except (OSError, ValueError):
            pass
        self._reap(proc, wait)

    def _reap(self, proc: Any, wait: float) -> int | None:
        try:
            return proc.wait(timeout=max(0.0, wait))
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except OSError:
                pass
            try:
                return proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                return None
        finally:
            t = self._stderr_thread
            if t is not None:
                t.join(1.0)

    def _summary(self, now: float, force: bool = False) -> None:
        if not force and now - self._summary_at < self.owner.stats_every:
            return
        sent, dropped = self.sent - self._summary_sent, self.dropped - self._summary_dropped
        if sent or dropped:
            log.info("%s: %.0f초 동안 보냄 %d장, 버림 %d장 (재시작 누적 %d)",
                     self.path, now - self._summary_at, sent, dropped, self.restarts)
        self._summary_at, self._summary_sent, self._summary_dropped = now, self.sent, self.dropped
