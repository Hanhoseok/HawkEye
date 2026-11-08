"""오판 장면 보기 페이지용 로컬 서버 — 내 컴퓨터(127.0.0.1)에서만 열린다.

영상 구간 이동(seek)에는 HTTP Range 응답이 필요해 표준 http.server 를 조금 고쳤다.
저장소 맨 위 폴더를 기준으로 열어, 페이지(experiments/tracking/outputs/review/)와 원본 영상(source/)을 함께 준다.
영상·기록은 저장소에 올리지 않는다 (outputs/·source/ 는 커밋 금지).

    python scripts/review_export.py      # 기록 → 보기 페이지 데이터 (처음 한 번, 기록이 바뀌면 다시)
    python scripts/review_serve.py       # http://127.0.0.1:8765/experiments/tracking/outputs/review/

구역 편집 페이지도 같은 서버로 연다 (저장·판정 돌리기는 /api/ 요청 → scripts/zone_editor.py):
    http://127.0.0.1:8765/experiments/tracking/scripts/zone_editor.html
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PAGE = "/experiments/tracking/outputs/review/"
EDITOR = "/experiments/tracking/scripts/zone_editor.html"
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _zone_api():
    import zone_editor  # 처음 /api/ 요청 때만 (판정 코드를 읽어 오는 데 몇 초 걸림)
    return zone_editor


class RangeHandler(SimpleHTTPRequestHandler):
    def _same_origin(self) -> bool:
        """다른 웹사이트가 내 컴퓨터의 이 서버로 저장 요청을 보내지 못하게 — 주소·출처가 이 서버일 때만."""
        host = self.headers.get("Host", "")
        port = self.server.server_address[1]
        if host not in (f"127.0.0.1:{port}", f"localhost:{port}"):
            return False
        origin = self.headers.get("Origin")
        return origin is None or origin in (f"http://127.0.0.1:{port}", f"http://localhost:{port}")

    def _api(self, method: str) -> None:
        if not self._same_origin():
            self.send_error(403)
            return
        body = None
        if method == "POST":
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            except json.JSONDecodeError:
                self.send_error(400)
                return
        try:
            code, data = _zone_api().handle(method, self.path.split("?")[0], body)
        except Exception as e:  # noqa: BLE001 — 페이지에 그대로 보여 준다
            code, data = 500, {"ok": False, "errors": [repr(e)]}
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path.startswith("/api/"):
            return self._api("GET")
        return super().do_GET()

    def do_POST(self):
        if self.path.startswith("/api/"):
            return self._api("POST")
        self.send_error(405)

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
    print(f"구역 편집: http://127.0.0.1:{args.port}{EDITOR}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
