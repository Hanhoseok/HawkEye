"""오판 장면 보기 페이지용 로컬 서버 — 내 컴퓨터(127.0.0.1)에서만 열린다.

영상 구간 이동(seek)에는 HTTP Range 응답이 필요해 표준 http.server 를 조금 고쳤다.
저장소 맨 위 폴더를 기준으로 열어, 페이지(experiments/tracking/outputs/review/)와 원본 영상(source/)을 함께 준다.
영상·기록은 저장소에 올리지 않는다 (outputs/·source/ 는 커밋 금지).

    python scripts/review_export.py      # 기록 → 보기 페이지 데이터 (처음 한 번, 기록이 바뀌면 다시)
    python scripts/review_serve.py       # http://127.0.0.1:8765/experiments/tracking/outputs/review/
"""
from __future__ import annotations

import argparse
import os
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PAGE = "/experiments/tracking/outputs/review/"


class RangeHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not rng or not os.path.isfile(path):
            return super().send_head()
        m = re.match(r"bytes=(\d*)-(\d*)", rng)
        size = os.path.getsize(path)
        start = int(m.group(1)) if m and m.group(1) else 0
        end = int(m.group(2)) if m and m.group(2) else size - 1
        end = min(end, size - 1)
        if start > end:
            self.send_error(416)
            return None
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self._remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        left = getattr(self, "_remaining", None)
        if left is None:
            return super().copyfile(source, outputfile)
        try:
            while left > 0:
                chunk = source.read(min(256 * 1024, left))
                if not chunk:
                    break
                outputfile.write(chunk)
                left -= len(chunk)
        except (ConnectionResetError, BrokenPipeError):
            pass  # 브라우저가 구간을 바꾸며 연결을 끊는 것은 정상
        self._remaining = None

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(RangeHandler, directory=str(REPO)))
    print(f"열기: http://127.0.0.1:{args.port}{PAGE}  (끄기: Ctrl+C)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
