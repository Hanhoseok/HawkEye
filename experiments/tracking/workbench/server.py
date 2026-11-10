"""분석 도구 서버 — 내 컴퓨터(127.0.0.1)에서만 열린다.

    python workbench/server.py            # http://127.0.0.1:8770/
    python workbench/server.py --port 9000

저장소 맨 위 폴더를 기준으로 열어 화면(workbench/static/), 만들어 둔 데이터(outputs/workbench/), 원본 영상(source/)을 함께 준다.
영상 구간 이동과 '같은 컴퓨터에서 온 요청만 받기'는 오판 장면 보기 서버(scripts/review_serve.py)의 것을 그대로 쓴다.
구역 편집 페이지의 /api/zones* 요청도 그대로 받으므로 예전 페이지도 이 서버에서 열린다.

/api/wb/index   갤러리 목록 + 만드는 중인지. 바뀐 짝이 있으면 뒤에서 다시 만들기 시작한다.
/api/wb/status  만드는 중 상태만.
/api/wb/label   정답 적기 (workbench/labels.py): POST {pair, labels, payments} → 저장하고 그 짝을 다시 채점해 요약을 돌려준다.
/api/wb/tune    수치 조정 (workbench/tune.py): GET 칸 목록·기록, POST run(초안 판정) / result(전후 비교) / apply(적용).
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
from workbench import labels as label_store  # noqa: E402
from workbench.data import OUT, Builder  # noqa: E402
from workbench.tune import Tuner  # noqa: E402

STATIC = "/" + (ROOT / "workbench" / "static").relative_to(REPO).as_posix() + "/"
DATA = "/" + OUT.relative_to(REPO).as_posix() + "/"
BUILDER = Builder()
TUNER = Tuner(BUILDER)


def save_labels(body: dict) -> dict:
    """정답을 저장하고 그 짝만 다시 채점한다 → 새 요약. 영상을 다시 돌리지 않으므로 1초 안팎."""
    pair = str(body.get("pair") or "")
    if pair not in BUILDER.pairs():
        raise ValueError(f"모르는 영상: {pair}")
    if BUILDER.state["running"] or TUNER.job["running"]:
        raise ValueError("다른 판정이 돌고 있습니다. 잠시 뒤 다시 저장됩니다")
    label_store.save(pair, body.get("labels") or {})
    if body.get("payments") is not None:
        label_store.save_payments(pair, body["payments"])
    state = BUILDER.build([pair], force=True)
    if state["errors"]:
        raise RuntimeError("; ".join(state["errors"]))
    data = json.loads((OUT / "pairs" / f"{pair}.json").read_text(encoding="utf-8"))
    return {"summary": data["summary"], "labels": data["labels"], "payment_labels": data["payment_labels"]}


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
        body = {}
        if method == "POST":
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            except json.JSONDecodeError:
                return self._json(400, {"error": "요청을 읽지 못했습니다"})
        try:
            if method == "GET" and path == "/api/wb/index":
                return self._json(200, index_payload())
            if method == "GET" and path == "/api/wb/status":
                return self._json(200, {"build": dict(BUILDER.state), "job": dict(TUNER.job)})
            if method == "GET" and path == "/api/wb/tune":
                return self._json(200, TUNER.page())
            if method == "POST" and path == "/api/wb/label":
                return self._json(200, save_labels(body))
            if method == "POST" and path == "/api/wb/tune/run":
                return self._json(200, TUNER.start(body.get("values") or {}, body.get("scope") or "normal"))
            if method == "POST" and path == "/api/wb/tune/result":
                return self._json(200, TUNER.result(body.get("values") or {}))
            if method == "POST" and path == "/api/wb/tune/apply":
                return self._json(200, TUNER.apply(body.get("values") or {}))
            self.send_error(404)
        except ValueError as e:   # 화면에서 고칠 수 있는 것 (바뀐 값 없음, 범위 밖 …)
            self._json(400, {"error": str(e)})
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
