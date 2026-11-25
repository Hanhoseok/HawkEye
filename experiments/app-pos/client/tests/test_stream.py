"""AI 화면 송출 (FrameStreamer) — 가짜 ffmpeg(tests/fake_ffmpeg.py)로 시험한다.

설계: ../../.ai-review 계획 §5-7 / docs/design.md §10
핵심 요구: push() 가 파이프라인을 멈추지 않는다. 송출은 고정 속도, 밀리면 최신 장면만 남긴다.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from hawkeye_client import FrameStreamer
from hawkeye_client.stream import default_rtsp_base, encoder_args, ffmpeg_args, probe_args, safe_url

FAKE = str(Path(__file__).with_name("fake_ffmpeg.py"))


class Image:
    """numpy 배열 대신: .shape · .dtype · .tobytes() 만 있으면 된다."""

    def __init__(self, width: int, height: int, value: int = 0, dtype: str = "uint8", channels: int = 3):
        self.shape = (height, width, channels)
        self.dtype = dtype
        self._data = bytes([value % 256]) * (width * height * channels)

    def tobytes(self) -> bytes:
        return self._data


@pytest.fixture
def logdir(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_FFMPEG_LOG", str(tmp_path))
    for name in ("FAKE_PROBE_EXIT", "FAKE_EXIT_AFTER", "FAKE_NVENC_RUN_FAIL", "FAKE_HANG", "FAKE_NO_READ",
                 "FAKE_NO_READ_ONCE", "FAKE_PROBE_HANG", "HAWKEYE_AI_ENCODER", "HAWKEYE_AI_RTSP"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def streamer(**kw) -> FrameStreamer:
    kw.setdefault("rtsp_base", "rtsp://127.0.0.1:1")
    kw.setdefault("fps", 20)
    kw.setdefault("restart_wait", 0.05)
    kw.setdefault("max_restart_wait", 0.2)
    kw.setdefault("close_timeout", 3.0)
    return FrameStreamer(ffmpeg=[sys.executable, FAKE], **kw)


def records(folder: Path, kind: str = "run") -> list[dict]:
    out = [json.loads(p.read_text(encoding="utf-8")) for p in folder.glob(f"{kind}-*.json")]
    return sorted(out, key=lambda r: r.get("started", 0))


def wait_for(cond, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


def arg(args: list[str], name: str) -> str:
    return args[args.index(name) + 1]


# ---- ffmpeg 인자 (안드로이드 호환) ----

@pytest.mark.parametrize("encoder", ["h264_nvenc", "libx264"])
def test_both_encoders_make_android_compatible_h264(encoder):
    args = ffmpeg_args(["ffmpeg"], encoder, 1920, 1080, 10, "rtsp://10.0.0.1:8554/cam1_ai")
    assert arg(args, "-f") == "rawvideo" and arg(args, "-pix_fmt") == "bgr24"
    assert arg(args, "-s") == "1920x1080" and arg(args, "-r") == "10" and arg(args, "-i") == "-"
    assert "format=yuv420p" in arg(args, "-vf")              # bgr24 그대로면 libx264 가 yuv444p 를 만든다
    assert "min(1280,iw)" in arg(args, "-vf")                # 폭 1280 이하로만 줄임
    assert arg(args, "-c:v") == encoder
    assert arg(args, "-profile:v") == "main"
    assert arg(args, "-bf") == "0"
    assert arg(args, "-g") == "10"                           # 1초마다 키프레임
    assert arg(args, "-rtsp_transport") == "tcp" and "-an" in args
    assert args[-1] == "rtsp://10.0.0.1:8554/cam1_ai"


def test_gop_follows_fps_and_unknown_encoder_rejected():
    assert arg(encoder_args("libx264", 15, "2M"), "-g") == "15"
    assert arg(encoder_args("h264_nvenc", 0.5, "2M"), "-g") == "1"
    with pytest.raises(ValueError):
        encoder_args("h265", 10, "2M")


def test_probe_encodes_one_frame_nowhere():
    args = probe_args(["ffmpeg"], "h264_nvenc")
    assert arg(args, "-frames:v") == "1" and args[-2:] == ["null", "-"]
    assert arg(args, "-c:v") == "h264_nvenc" and arg(args, "-profile:v") == "main"


def test_default_rtsp_base_from_env(monkeypatch):
    monkeypatch.setenv("HAWKEYE_AI_RTSP", "rtsp://100.1.2.3:8554/ ")
    assert default_rtsp_base() == "rtsp://100.1.2.3:8554"


def test_safe_url_hides_account():
    assert safe_url("rtsp://user:pw@10.0.0.1:8554/cam1_ai") == "rtsp://10.0.0.1:8554/cam1_ai"


# ---- push ----

def test_push_rejects_bad_camera_and_image(logdir):
    s = streamer(encoder="libx264")
    try:
        for bad in ("cam1_ai", "../cam1", "Cam1", "", None):
            with pytest.raises(ValueError):
                s.push(bad, Image(4, 2))
        with pytest.raises(ValueError):
            s.push("cam1", Image(4, 2, channels=4))
        with pytest.raises(ValueError):
            s.push("cam1", Image(4, 2, dtype="float32"))
        with pytest.raises(ValueError):
            s.push("cam1", Image(0, 2))
        with pytest.raises(ValueError):
            s.push("cam1", object())
    finally:
        s.close()
    assert records(logdir) == []


def test_push_returns_immediately_even_when_ffmpeg_does_not_read(logdir, monkeypatch):
    monkeypatch.setenv("FAKE_NO_READ", "1")                  # 쓰기가 막힘 (ffmpeg 가 멈춘 상황)
    s = streamer(encoder="libx264")
    frame = Image(640, 360)
    t0 = time.monotonic()
    for _ in range(200):
        s.push("cam1", frame)
    took = time.monotonic() - t0
    assert wait_for(lambda: records(logdir))                 # ffmpeg 는 떴다
    time.sleep(0.3)                                          # 송출 스레드가 쓰기에서 막히도록
    for _ in range(50):
        s.push("cam1", frame)
    t1 = time.monotonic()
    s.close(timeout=0.5)                                     # 막힌 쓰기도 강제 종료로 풀린다
    assert took < 1.0, f"push 200번에 {took:.2f}초"
    assert time.monotonic() - t1 < 5.0
    assert s.stats()["cam1"]["dropped"] > 200


def test_repeats_last_frame_at_fixed_rate(logdir):
    s = streamer(fps=20, encoder="libx264")
    s.push("cam1", Image(4, 2, value=7))
    time.sleep(1.0)
    s.close()
    runs = records(logdir)
    assert len(runs) == 1 and runs[0]["reason"] == "eof" and runs[0]["partial"] == 0   # 장면이 잘리지 않음
    assert 10 <= runs[0]["frames"] <= 30, runs[0]["frames"]  # 새 장면 1장이지만 초당 20장으로 계속 보냄
    assert runs[0]["last"] == 7
    assert s.stats()["cam1"]["sent"] == runs[0]["frames"]


def test_keeps_only_latest_frame(logdir):
    s = streamer(fps=2, encoder="libx264")                   # 0.5초마다 한 장만 보냄
    for i in range(50):
        s.push("cam1", Image(4, 2, value=i))
    time.sleep(0.8)                                          # 두 번째 틱(0.5초)에 최신 장면이 나감
    s.close()
    runs = records(logdir)
    assert runs and runs[-1]["last"] == 49                   # 마지막에 넘긴 장면이 나감
    assert s.stats()["cam1"]["dropped"] >= 45


def test_cameras_stream_to_their_own_ai_path(logdir):
    s = streamer(encoder="libx264")
    s.push("cam1", Image(4, 2, value=1))
    s.push("cam2", Image(6, 4, value=2))
    time.sleep(0.3)
    s.close()
    urls = {r["args"][-1]: r for r in records(logdir)}
    assert set(urls) == {"rtsp://127.0.0.1:1/cam1_ai", "rtsp://127.0.0.1:1/cam2_ai"}
    assert arg(urls["rtsp://127.0.0.1:1/cam2_ai"]["args"], "-s") == "6x4"
    assert urls["rtsp://127.0.0.1:1/cam2_ai"]["last"] == 2


# ---- 재시작 ----

def test_restarts_after_ffmpeg_dies(logdir, monkeypatch):
    monkeypatch.setenv("FAKE_EXIT_AFTER", "3")
    s = streamer(fps=50, encoder="libx264")
    s.push("cam1", Image(4, 2))
    assert wait_for(lambda: len(records(logdir)) >= 3)
    s.close()
    runs = records(logdir)
    assert all(r["reason"] == "crash" for r in runs[:3])
    assert s.stats()["cam1"]["restarts"] >= 2


def test_restarts_ffmpeg_that_stops_reading_without_close(logdir, monkeypatch):
    """ffmpeg 가 살아 있지만 입력을 안 받음 → 쓰기가 막힘 → 감시 스레드가 끝내고 다시 띄워 최신 장면을 보낸다."""
    monkeypatch.setenv("FAKE_NO_READ_ONCE", str(logdir / "stalled-once"))
    s = streamer(encoder="libx264", write_timeout=0.4)
    stop = time.monotonic() + 6
    i = 0

    def resumed() -> bool:
        runs = [r for r in records(logdir) if r["reason"] != "no_read"]
        return bool(runs)

    while time.monotonic() < stop:
        s.push("cam1", Image(640, 360, value=i))             # 파이프 버퍼보다 커서 첫 장면부터 쓰기가 막힘
        i += 1
        if s.stats()["cam1"]["restarts"] >= 1 and i > 40:
            break
        time.sleep(0.05)
    last = i - 1
    time.sleep(0.3)
    s.close()
    runs = records(logdir)
    assert runs[0]["reason"] == "no_read"
    assert s.stats()["cam1"]["restarts"] >= 1
    assert resumed() and runs[-1]["reason"] == "eof" and runs[-1]["frames"] > 0
    assert runs[-1]["last"] == last % 256                    # 다시 띄운 뒤 최신 장면을 보냄


def test_restarts_when_frame_size_changes(logdir):
    s = streamer(encoder="libx264")
    s.push("cam1", Image(4, 2, value=1))
    time.sleep(0.3)
    s.push("cam1", Image(6, 4, value=2))
    time.sleep(0.3)
    s.close()
    runs = records(logdir)
    assert [arg(r["args"], "-s") for r in runs] == ["4x2", "6x4"]
    assert [r["reason"] for r in runs] == ["eof", "eof"] and runs[0]["partial"] == 0
    assert runs[1]["last"] == 2


def test_stops_when_no_new_frames_and_resumes(logdir):
    s = streamer(encoder="libx264", idle_stop_sec=0.3)
    s.push("cam1", Image(4, 2, value=1))
    assert wait_for(lambda: len(records(logdir)) == 1, timeout=3)   # 0.3초 뒤 스스로 닫음
    assert records(logdir)[0]["reason"] == "eof"
    s.push("cam1", Image(4, 2, value=2))
    time.sleep(0.2)
    s.close()
    runs = records(logdir)
    assert len(runs) == 2 and runs[1]["last"] == 2


# ---- 인코더 ----

def test_uses_nvenc_when_probe_succeeds(logdir):
    s = streamer()
    s.push("cam1", Image(4, 2))
    time.sleep(0.3)
    s.close()
    assert [arg(p["args"], "-c:v") for p in records(logdir, "probe")] == ["h264_nvenc"]
    assert arg(records(logdir)[0]["args"], "-c:v") == "h264_nvenc"
    assert s.encoder == "h264_nvenc"


def test_falls_back_to_libx264_when_probe_fails(logdir, monkeypatch):
    monkeypatch.setenv("FAKE_PROBE_EXIT", "1")
    s = streamer()
    s.push("cam1", Image(4, 2))
    s.push("cam2", Image(4, 2))
    time.sleep(0.3)
    s.close()
    assert len(records(logdir, "probe")) == 1                # 카메라가 둘이어도 한 번만 확인
    assert {arg(r["args"], "-c:v") for r in records(logdir)} == {"libx264"}
    assert s.encoder == "libx264"


def test_falls_back_when_nvenc_fails_while_running(logdir, monkeypatch):
    monkeypatch.setenv("FAKE_NVENC_RUN_FAIL", "1")           # 시험 인코딩은 됐지만 실제 송출에서 인코더 오류
    s = streamer()
    s.push("cam1", Image(4, 2))
    assert wait_for(lambda: len(records(logdir)) >= 1 and s.encoder == "libx264")
    time.sleep(0.4)
    s.close()
    runs = records(logdir)
    assert runs[0]["reason"] == "nvenc" and arg(runs[0]["args"], "-c:v") == "h264_nvenc"
    assert arg(runs[1]["args"], "-c:v") == "libx264" and runs[1]["reason"] == "eof"


def test_crash_without_encoder_error_keeps_nvenc(logdir, monkeypatch):
    monkeypatch.setenv("FAKE_EXIT_AFTER", "2")               # MediaMTX 가 꺼진 것처럼 그냥 죽음
    s = streamer(fps=50)
    s.push("cam1", Image(4, 2))
    assert wait_for(lambda: len(records(logdir)) >= 2)
    s.close()
    assert {arg(r["args"], "-c:v") for r in records(logdir)} == {"h264_nvenc"}


def test_encoder_env_skips_probe(logdir, monkeypatch):
    monkeypatch.setenv("HAWKEYE_AI_ENCODER", "libx264")
    s = streamer()
    s.push("cam1", Image(4, 2))
    time.sleep(0.3)
    s.close()
    assert records(logdir, "probe") == []
    assert arg(records(logdir)[0]["args"], "-c:v") == "libx264"


def test_bad_encoder_env_rejected(logdir, monkeypatch):
    monkeypatch.setenv("HAWKEYE_AI_ENCODER", "hevc")
    with pytest.raises(ValueError):
        streamer()


# ---- 끝내기 ----

def test_close_kills_ffmpeg_that_ignores_end_of_input(logdir, monkeypatch):
    monkeypatch.setenv("FAKE_HANG", "1")
    s = streamer(encoder="libx264")
    s.push("cam1", Image(4, 2))
    time.sleep(0.3)
    t0 = time.monotonic()
    s.close(timeout=0.5)
    assert time.monotonic() - t0 < 4.0
    assert records(logdir)[0]["reason"] == "eof"             # 입력은 정상으로 닫혔다


def test_close_cuts_slow_encoder_probe(logdir, monkeypatch):
    monkeypatch.setenv("FAKE_PROBE_HANG", "1")               # 인코더 시험이 끝나지 않음
    s = streamer()
    s.push("cam1", Image(4, 2))
    s.push("cam2", Image(4, 2))                              # cam2 는 cam1 시험이 끝나기를 기다림
    assert wait_for(lambda: records(logdir, "probe"))
    time.sleep(0.2)
    t0 = time.monotonic()
    s.close(timeout=0.3)
    assert time.monotonic() - t0 < 3.0
    assert not s._channels["cam1"].is_alive() and not s._channels["cam2"].is_alive()
    time.sleep(0.3)
    assert len(records(logdir, "probe")) == 1                # 끝낸 뒤 cam2 가 새 시험을 띄우지 않음
    assert records(logdir) == []                             # 끊긴 시험으로 송출을 시작하지 않음


def test_push_after_close_is_ignored(logdir):
    s = streamer(encoder="libx264")
    s.close()
    s.push("cam1", Image(4, 2))
    time.sleep(0.2)
    assert records(logdir) == [] and s.stats() == {}


def test_context_manager_closes(logdir):
    with streamer(encoder="libx264") as s:
        s.push("cam1", Image(4, 2, value=5))
        time.sleep(0.2)
    runs = records(logdir)
    assert len(runs) == 1 and runs[0]["reason"] == "eof" and runs[0]["last"] == 5


# ---- 시험 송출 프로그램 ----

def test_demo_render_makes_full_bgr_frame():
    from datetime import datetime

    from hawkeye_client.stream_demo import BG, RED, render

    f = render(1, 3, 320, 180, datetime(2026, 11, 24, 14, 30, 5, 300000))
    assert f.shape == (180, 320, 3)
    data = f.tobytes()
    assert len(data) == 320 * 180 * 3
    pixels = {data[i:i + 3] for i in range(0, len(data), 3)}
    assert bytes(BG) in pixels and bytes(RED) in pixels       # 배경과 사람 상자가 그려짐


def test_demo_rejects_bad_camera_name(capsys):
    from hawkeye_client.stream_demo import main

    with pytest.raises(SystemExit):
        main(["--camera", "cam1_ai"])
