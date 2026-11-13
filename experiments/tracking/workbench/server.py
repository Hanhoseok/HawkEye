"""분석 도구 서버 — 내 컴퓨터(127.0.0.1)에서만 열린다.

    python workbench/server.py            # http://127.0.0.1:8770/
    python workbench/server.py --port 9000

저장소 맨 위 폴더를 기준으로 열어 화면(workbench/static/), 만들어 둔 데이터(outputs/workbench/), 원본 영상(source/)을 함께 준다.
영상 구간 이동과 '같은 컴퓨터에서 온 요청만 받기'는 오판 장면 보기 서버(scripts/review_serve.py)의 것을 그대로 쓴다.
구역 편집 페이지의 /api/zones* 요청도 그대로 받으므로 예전 페이지도 이 서버에서 열린다.

촬영 회차(2차·3차)마다 영상·기록·정답표·적용한 값이 따로다 — 요청마다 shoot 을 받는다 (GET 은 ?shoot=3, POST 는 본문의 shoot).
안 주면 /api/wb/index 는 기록이 있는 가장 최근 회차, 나머지는 2차.

/api/wb/index   갤러리 목록 + 만드는 중인지. 바뀐 짝이 있으면 뒤에서 다시 만들기 시작한다.
/api/wb/status  만드는 중 상태만.
/api/wb/label   정답 적기 (workbench/labels.py): POST {pair, labels, payments} → 저장하고 그 짝을 다시 채점해 요약을 돌려준다.
/api/wb/upload  새 영상 올리기 (workbench/intake.py): POST 본문이 영상 파일 그대로, ?name=&cam=&ext=
/api/wb/add     올린 두 영상을 등록하고 처리 대기열에 넣는다. /api/wb/add/retry 는 다시 처리.
/api/wb/sync    두 영상의 시차를 사람이 맞춘 값으로 적는다.
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
from urllib.parse import parse_qs, urlsplit  # noqa: E402

from workbench import intake  # noqa: E402
from workbench import labels as label_store  # noqa: E402
from workbench.data import PROFILES, Builder  # noqa: E402
from workbench.tune import Tuner  # noqa: E402

STATIC = "/" + (ROOT / "workbench" / "static").relative_to(REPO).as_posix() + "/"
BUILDERS = {shoot: Builder(shoot) for shoot in PROFILES}
TUNERS = {shoot: Tuner(builder) for shoot, builder in BUILDERS.items()}
RUNNER = intake.Runner(BUILDERS[2])
"""새 영상 추가는 2차 촬영 자리에서 찍은 영상만 받는다 (PROFILES[..].can_add)."""


def shoot_of(value, default: int | None = 2) -> int:
    """요청의 촬영 회차. 안 왔으면 default — None 이면 기록이 있는 가장 최근 회차."""
    if value in (None, ""):
        if default is not None:
            return default
        have = [s for s, b in BUILDERS.items() if b.pairs()]
        return max(have) if have else 2
    try:
        shoot = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"모르는 촬영 회차: {value!r}") from None
    if shoot not in BUILDERS:
        raise ValueError(f"모르는 촬영 회차: {shoot}")
    return shoot


def busy() -> bool:
    """어느 회차든 판정이 돌고 있나 — 판정 코드는 한 번에 한 회차만 돈다."""
    return any(b.state["running"] for b in BUILDERS.values()) or any(t.job["running"] for t in TUNERS.values())


def data_url(builder: Builder) -> str:
    return "/" + builder.out.relative_to(REPO).as_posix() + "/"


def save_labels(body: dict) -> dict:
    """정답을 저장하고 그 짝만 다시 채점한다 → 새 요약. 영상을 다시 돌리지 않으므로 1초 안팎."""
    builder = BUILDERS[shoot_of(body.get("shoot"))]
    pair = str(body.get("pair") or "")
    if pair not in builder.pairs():
        raise ValueError(f"모르는 영상: {pair}")
    if busy():
        raise ValueError("다른 판정이 돌고 있습니다. 잠시 뒤 다시 저장됩니다")
    label_store.save(pair, body.get("labels") or {}, builder.events_dir())
    if body.get("payments") is not None:
        label_store.save_payments(pair, body["payments"], builder.payments_file())
    state = builder.build([pair], force=True)
    if state["errors"]:
        raise RuntimeError("; ".join(state["errors"]))
    data = json.loads((builder.out / "pairs" / f"{pair}.json").read_text(encoding="utf-8"))
    return {"summary": data["summary"], "labels": data["labels"], "payment_labels": data["payment_labels"]}


def index_payload(shoot: int) -> dict:
    builder = BUILDERS[shoot]
    started = builder.build_async()
    index = builder.read_index()
    p = builder.profile
    add = {"pending": RUNNER.pending(), "suggest": intake.suggest_names()} if p.can_add else {"pending": [], "suggest": {}}
    return {"static": STATIC, "data": data_url(builder), "build": dict(builder.state), "started": started,
            "stale": [] if builder.state["running"] else builder.stale(),
            "shoot": shoot, "shoot_name": p.name, "can_add": p.can_add,
            "labels_dir": p.labels.relative_to(ROOT).as_posix(),
            "shoots": [{"shoot": s, "name": b.profile.name, "n": len(b.pairs())} for s, b in BUILDERS.items()],
            **add, **index}


def add_video(body: dict) -> dict:
    out = intake.register(body)
    RUNNER.notes[out["pair"]] = out["notes"]
    RUNNER.enqueue(out["pair"])
    return out


def save_sync(body: dict) -> dict:
    builder = BUILDERS[shoot_of(body.get("shoot"))]
    pair = str(body.get("pair") or "")
    if pair not in builder.pairs():
        raise ValueError(f"모르는 영상: {pair}")
    if busy():
        raise ValueError("다른 판정이 돌고 있습니다. 잠시 뒤 다시 눌러 주세요")
    try:
        offset = float(body.get("offset_ms"))
    except (TypeError, ValueError):
        raise ValueError("시차가 숫자가 아닙니다") from None
    intake.set_offset(builder, pair, offset, str(body.get("checked_by") or ""))
    state = builder.build([pair], force=True)   # 그 짝만 다시 판정한다 (1초 안팎)
    if state["errors"]:
        raise RuntimeError("; ".join(state["errors"]))
    return {"pair": pair, "offset_ms": offset}


def tune_guard(tuner: Tuner) -> None:
    """수치 조정은 그 회차의 Tuner 가 제 판정만 살핀다 — 다른 회차가 도는 중이면 여기서 막는다."""
    if any(b.state["running"] for b in BUILDERS.values() if b is not tuner.builder) or \
            any(t.job["running"] for t in TUNERS.values() if t is not tuner):
        raise ValueError("다른 촬영의 판정이 돌고 있습니다. 끝난 뒤 다시 눌러 주세요")


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
        if method == "POST" and path == "/api/wb/upload":   # 본문이 영상 파일 그대로 — JSON 으로 읽지 않는다
            q = {k: v[0] for k, v in parse_qs(urlsplit(self.path).query).items()}
            try:
                return self._json(200, intake.save_upload(q.get("name", ""), q.get("cam", ""), q.get("ext", ""), self.rfile,
                                                          int(self.headers.get("Content-Length") or 0)))
            except ValueError as e:
                return self._json(400, {"error": str(e)})
        body = {}
        if method == "POST":
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            except json.JSONDecodeError:
                return self._json(400, {"error": "요청을 읽지 못했습니다"})
            if not isinstance(body, dict):
                return self._json(400, {"error": "요청을 읽지 못했습니다"})
        query = {k: v[0] for k, v in parse_qs(urlsplit(self.path).query).items()}
        try:
            if method == "GET" and path == "/api/wb/index":
                return self._json(200, index_payload(shoot_of(query.get("shoot"), None)))
            shoot = shoot_of(query.get("shoot") if method == "GET" else body.get("shoot"))
            builder, tuner = BUILDERS[shoot], TUNERS[shoot]
            if method == "GET" and path == "/api/wb/status":
                return self._json(200, {"build": dict(builder.state), "job": dict(tuner.job)})
            if method == "GET" and path == "/api/wb/tune":
                return self._json(200, tuner.page())
            if path.startswith("/api/wb/add") and not builder.profile.can_add:
                raise ValueError(f"{builder.profile.name}은 여기서 영상을 추가하지 못합니다 (영상마다 카메라 맞추기가 필요)")
            if method == "POST" and path == "/api/wb/add":
                return self._json(200, add_video(body))
            if method == "POST" and path == "/api/wb/add/retry":
                RUNNER.enqueue(str(body.get("pair") or ""))
                return self._json(200, {"ok": True})
            if method == "POST" and path == "/api/wb/sync":
                return self._json(200, save_sync(body))
            if method == "POST" and path == "/api/wb/label":
                return self._json(200, save_labels(body))
            if method == "POST" and path == "/api/wb/tune/run":
                tune_guard(tuner)
                return self._json(200, tuner.start(body.get("values") or {}, body.get("scope") or "normal"))
            if method == "POST" and path == "/api/wb/tune/result":
                return self._json(200, tuner.result(body.get("values") or {}))
            if method == "POST" and path == "/api/wb/tune/apply":
                tune_guard(tuner)
                return self._json(200, tuner.apply(body.get("values") or {}))
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
