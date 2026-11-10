"""분석 도구 서버 — 내 컴퓨터(127.0.0.1)에서만 열린다.

    python workbench/server.py            # http://127.0.0.1:8770/
    python workbench/server.py --port 9000

저장소 맨 위 폴더를 기준으로 열어 화면(workbench/static/), 만들어 둔 데이터(outputs/workbench/), 원본 영상(source/)을 함께 준다.
영상 구간 이동과 '같은 컴퓨터에서 온 요청만 받기'는 오판 장면 보기 서버(scripts/review_serve.py)의 것을 그대로 쓴다.
구역 편집 페이지의 /api/zones* 요청도 그대로 받으므로 예전 페이지도 이 서버에서 열린다.

/api/wb/index   갤러리 목록 + 만드는 중인지. 바뀐 짝이 있으면 뒤에서 다시 만들기 시작한다.
/api/wb/status  만드는 중 상태만.
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from review_serve import RangeHandler  # noqa: E402
from workbench.data import OUT, Builder  # noqa: E402

STATIC = "/" + (ROOT / "workbench" / "static").relative_to(REPO).as_posix() + "/"
DATA = "/" + OUT.relative_to(REPO).as_posix() + "/"
BUILDER = Builder()


def index_payload() -> dict:
    started = BUILDER.build_async()
    index = BUILDER.read_index()
    return {"static": STATIC, "data": DATA, "build": dict(BUILDER.state), "started": started,
            "stale": [] if BUILDER.state["running"] else BUILDER.stale(), **index}


class Handler(RangeHandler):
    def _json(self, code: int, data: dict) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _api(self, method: str) -> None:
        path = self.path.split("?")[0]
        if not path.startswith("/api/wb/"):
            return super()._api(method)   # 구역 편집 페이지
        if not self._same_origin():
            self.send_error(403)
            return
        try:
            if path == "/api/wb/index":
                return self._json(200, index_payload())
            if path == "/api/wb/status":
                return self._json(200, {"build": dict(BUILDER.state)})
            self.send_error(404)
        except Exception as e:  # noqa: BLE001 — 화면에 그대로 보여 준다
            self._json(500, {"error": repr(e)})

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", STATIC)
            self.end_headers()
            return
        return super().do_GET()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(REPO)))
    print(f"분석 도구: http://127.0.0.1:{args.port}/  (끄기: Ctrl+C)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
