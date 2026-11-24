"""AI 화면 시험 송출 — 탐지 파이프라인 없이 합성 장면(시계·움직이는 상자·번호)을 `cam1_ai` 로 보낸다.

    python -m hawkeye_client.stream_demo                       # cam1_ai, 1280x720, 처리 초당 5장 → 송출 초당 10장
    python -m hawkeye_client.stream_demo --camera cam1 --camera cam2 --seconds 60
    HAWKEYE_AI_ENCODER=libx264 python -m hawkeye_client.stream_demo   # GPU 인코더 없이

관리자 앱 라이브에서 'AI 탐지 화면 보기' 를 켜면 이 화면이 보인다. 화면 아래 시계(서버 시각, 0.1초 단위)를
앱 화면과 함께 찍으면 AI 화면 지연을 잴 수 있다. 송출 주소는 FrameStreamer 와 같다 (HAWKEYE_AI_RTSP → tailscale ip).

표준 라이브러리만 쓴다 (numpy·OpenCV 없이 bytearray 에 직접 그림).
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime

from .stream import CAMERA_ID, FrameStreamer, safe_url

# 7-세그먼트: a 위, b 오른쪽 위, c 오른쪽 아래, d 아래, e 왼쪽 아래, f 왼쪽 위, g 가운데
SEGMENTS = {
    "0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc",
    "5": "afgcd", "6": "afgedc", "7": "abc", "8": "abcdefg", "9": "abcdfg", "-": "g",
}

BG = (40, 40, 40)            # BGR
GREEN = (60, 200, 60)
RED = (40, 40, 230)
WHITE = (235, 235, 235)
YELLOW = (40, 220, 240)


class Frame:
    """FrameStreamer 가 받는 이미지: .shape (높이, 폭, 3) 와 .tobytes() 만 있으면 된다."""

    def __init__(self, width: int, height: int, color=BG):
        self.width, self.height = width, height
        self.shape = (height, width, 3)
        self.buf = bytearray(bytes(color) * (width * height))

    def tobytes(self) -> bytes:
        return bytes(self.buf)

    def fill(self, x0: int, y0: int, x1: int, y1: int, color) -> None:
        x0, x1 = max(0, min(x0, self.width)), max(0, min(x1, self.width))
        y0, y1 = max(0, min(y0, self.height)), max(0, min(y1, self.height))
        if x1 <= x0 or y1 <= y0:
            return
        row = bytes(color) * (x1 - x0)
        stride = self.width * 3
        for y in range(y0, y1):
            off = y * stride + x0 * 3
            self.buf[off:off + len(row)] = row

    def box(self, x0: int, y0: int, x1: int, y1: int, color, t: int = 4) -> None:
        self.fill(x0, y0, x1, y0 + t, color)
        self.fill(x0, y1 - t, x1, y1, color)
        self.fill(x0, y0, x0 + t, y1, color)
        self.fill(x1 - t, y0, x1, y1, color)

    def text(self, x: int, y: int, s: str, color, size: int = 6) -> int:
        """숫자·':'·'.'·'-' 만 그린다. 칸 하나는 폭 5s·높이 9s. 다 그린 오른쪽 끝 x 를 돌려준다."""
        for ch in s:
            if ch == ":":
                self.fill(x + size, y + 2 * size, x + 2 * size, y + 3 * size, color)
                self.fill(x + size, y + 6 * size, x + 2 * size, y + 7 * size, color)
                x += 3 * size
                continue
            if ch == ".":
                self.fill(x + size, y + 8 * size, x + 2 * size, y + 9 * size, color)
                x += 3 * size
                continue
            if ch == " ":
                x += 3 * size
                continue
            s_ = size
            seg = {
                "a": (x + s_, y, x + 4 * s_, y + s_),
                "b": (x + 4 * s_, y + s_, x + 5 * s_, y + 4 * s_),
                "c": (x + 4 * s_, y + 5 * s_, x + 5 * s_, y + 8 * s_),
                "d": (x + s_, y + 8 * s_, x + 4 * s_, y + 9 * s_),
                "e": (x, y + 5 * s_, x + s_, y + 8 * s_),
                "f": (x, y + s_, x + s_, y + 4 * s_),
                "g": (x + s_, y + 4 * s_, x + 4 * s_, y + 5 * s_),
            }
            for name in SEGMENTS.get(ch, ""):
                self.fill(*seg[name], color)
            x += 6 * size
        return x


def render(camera_index: int, n: int, width: int, height: int, now: datetime) -> Frame:
    """처리한 장면 n 번째: 구역(초록 테두리), 좌우로 움직이는 사람 상자(빨강)와 번호, 아래에 시계."""
    f = Frame(width, height)
    u = max(1, height // 120)                                 # 화면 크기에 맞춘 단위
    f.box(width * 3 // 5, height // 6, width - 4 * u, height * 3 // 4, GREEN, t=max(2, u))   # 출구 구역
    f.text(4 * u, 4 * u, str(camera_index), YELLOW, size=3 * u)                             # 카메라 번호

    bw, bh = width // 8, height // 2
    span = max(1, width - bw - 8 * u)
    step = (n * 6 * u) % (2 * span)
    x = 4 * u + (step if step < span else 2 * span - step)   # 끝에 닿으면 되돌아옴
    y = height // 6
    f.box(x, y, x + bw, y + bh, RED, t=max(2, u))
    f.text(x, y - 11 * u, str(7 + camera_index), RED, size=max(1, u))                      # 사람 번호

    clock = now.strftime("%H:%M:%S.") + str(now.microsecond // 100_000)
    f.text(4 * u, height - 22 * u, clock, WHITE, size=2 * u)
    f.text(width - 40 * u, height - 22 * u, f"{n % 100000:05d}", WHITE, size=u)             # 처리 장면 번호
    return f


def parse_size(s: str) -> tuple[int, int]:
    try:
        w, h = (int(v) for v in s.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError("크기는 1280x720 처럼 씁니다") from None
    if w < 16 or h < 16:
        raise argparse.ArgumentTypeError("너무 작습니다")
    return w, h


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m hawkeye_client.stream_demo", description=__doc__.split("\n")[0])
    p.add_argument("--camera", action="append", help="보낼 카메라 (여러 번 가능, 기본 cam1)")
    p.add_argument("--rtsp", help="MediaMTX 주소 (기본: HAWKEYE_AI_RTSP → tailscale ip -4 → 127.0.0.1, 포트 8554)")
    p.add_argument("--size", type=parse_size, default=(1280, 720), help="그릴 장면 크기 (기본 1280x720)")
    p.add_argument("--fps", type=float, default=10.0, help="송출 초당 장면 수 (기본 10)")
    p.add_argument("--process-fps", type=float, default=5.0,
                   help="파이프라인이 장면을 넘기는 속도 흉내 (기본 5 = 지금 live_process_fps)")
    p.add_argument("--seconds", type=float, default=0.0, help="이만큼 보내고 끝냄 (기본 0 = Ctrl+C 까지)")
    a = p.parse_args(argv)
    if a.fps <= 0 or a.process_fps <= 0:
        p.error("--fps · --process-fps 는 0보다 커야 합니다")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cameras = a.camera or ["cam1"]
    bad = [c for c in cameras if not CAMERA_ID.match(c)]
    if bad:
        p.error(f"카메라 이름은 cam1, cam2 처럼 씁니다: {', '.join(bad)}")
    try:
        streamer = FrameStreamer(a.rtsp, fps=a.fps)
    except ValueError as e:
        print(f"[stream_demo] {e}", file=sys.stderr)
        return 2

    stop = {"now": False}

    def on_signal(*_):
        stop["now"] = True

    signal.signal(signal.SIGINT, on_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, on_signal)

    w, h = a.size
    print(f"[stream_demo] {', '.join(c + '_ai' for c in cameras)} → {safe_url(streamer.rtsp_base)} "
          f"({w}x{h}, 처리 {a.process_fps:g}장/초 → 송출 {a.fps:g}장/초). Ctrl+C 로 끝냄")
    period = 1.0 / a.process_fps
    started = time.monotonic()
    n = 0
    try:
        while not stop["now"]:
            now = datetime.now()
            for i, c in enumerate(cameras, start=1):
                try:
                    streamer.push(c, render(i, n, w, h, now))
                except ValueError as e:
                    print(f"[stream_demo] {e}", file=sys.stderr)
                    return 2
            n += 1
            if a.seconds and time.monotonic() - started >= a.seconds:
                break
            time.sleep(max(0.0, started + n * period - time.monotonic()))
    finally:
        streamer.close()
        for c, s in streamer.stats().items():
            print(f"[stream_demo] {c}_ai: 보냄 {s['sent']}장, 버림 {s['dropped']}장, 재시작 {s['restarts']}번 "
                  f"(인코더 {streamer.encoder or '없음'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
