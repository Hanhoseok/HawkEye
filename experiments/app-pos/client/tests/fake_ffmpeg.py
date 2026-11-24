"""FrameStreamer 시험용 가짜 ffmpeg. 표준 입력으로 받은 장면 수·마지막 장면을 기록한다.

환경변수로 동작을 바꾼다 (시험이 monkeypatch.setenv 로 정함):
    FAKE_FFMPEG_LOG     기록 폴더 (필수). 실행마다 run-*.json, 시험 인코딩마다 probe-*.json
    FAKE_PROBE_EXIT     시험 인코딩(-f null)에서 h264_nvenc 일 때 돌려줄 종료 코드 (기본 0)
    FAKE_EXIT_AFTER     장면을 이만큼 받으면 코드 1 로 죽음 (ffmpeg 가 죽는 상황)
    FAKE_NVENC_RUN_FAIL h264_nvenc 로 실행되면 첫 장면을 받자마자 인코더 오류를 찍고 죽음
    FAKE_HANG           입력이 끝나도 끝나지 않음 (close 제한 시간 시험)
    FAKE_NO_READ        입력을 읽지 않음 (쓰기가 막히는 상황)
    FAKE_NO_READ_ONCE   이 파일이 없으면 만들고 FAKE_NO_READ 처럼 동작 — 첫 실행만 멈추고 다음 실행은 정상
    FAKE_PROBE_HANG     시험 인코딩이 끝나지 않음
"""

import json
import os
import sys
import time


def record(kind: str, data: dict) -> None:
    folder = os.environ["FAKE_FFMPEG_LOG"]
    name = os.path.join(folder, f"{kind}-{time.time_ns()}-{os.getpid()}.json")
    with open(name + ".tmp", "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(name + ".tmp", name)


def main() -> int:
    args = sys.argv[1:]
    if "null" in args:                                   # 인코더 시험 (-f null -)
        code = int(os.environ.get("FAKE_PROBE_EXIT", "0")) if "h264_nvenc" in args else 0
        record("probe", {"args": args, "code": code})
        if os.environ.get("FAKE_PROBE_HANG"):
            time.sleep(60)
        if code:
            print("[h264_nvenc @ 0x1] Cannot load libnvidia-encode.so.1", file=sys.stderr)
        return code

    w, h = map(int, args[args.index("-s") + 1].split("x"))
    size = w * h * 3
    started = time.time()
    once = os.environ.get("FAKE_NO_READ_ONCE")
    first = False
    if once:
        try:
            with open(once, "x"):
                first = True
        except FileExistsError:
            pass
    if os.environ.get("FAKE_NO_READ") or first:
        record("run", {"args": args, "frames": 0, "last": None, "reason": "no_read", "started": started})
        time.sleep(60)
        return 0

    exit_after = int(os.environ.get("FAKE_EXIT_AFTER", "0"))
    frames, last, buf = 0, None, b""
    stdin = sys.stdin.buffer
    while True:
        chunk = stdin.read(size - len(buf))
        if not chunk:
            break
        buf += chunk
        if len(buf) < size:
            continue
        frames, last, buf = frames + 1, buf[0], b""
        if os.environ.get("FAKE_NVENC_RUN_FAIL") and "h264_nvenc" in args:
            print("[h264_nvenc @ 0x1] OpenEncodeSessionEx failed: out of memory (10)", file=sys.stderr, flush=True)
            print("[vost#0:0/h264_nvenc @ 0x2] Error while opening encoder", file=sys.stderr, flush=True)
            record("run", {"args": args, "frames": frames, "last": last, "reason": "nvenc", "started": started})
            return 1
        if exit_after and frames >= exit_after:
            record("run", {"args": args, "frames": frames, "last": last, "reason": "crash", "started": started})
            return 1
    record("run", {"args": args, "frames": frames, "last": last, "reason": "eof", "partial": len(buf),
                   "started": started})
    if os.environ.get("FAKE_HANG"):
        time.sleep(60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
