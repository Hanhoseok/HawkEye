"""기록 → 분석 도구 데이터 (outputs/workbench/).

저장된 기록(기본 outputs/dual_eval_pose3/<짝>/)을 지금 설정으로 다시 판정하며 보기 데이터를 만들고,
질문별 요약(analyze.py)을 붙여 짝마다 한 파일로 둔다. 영상을 다시 돌리지 않는다 — 짝당 1초 안팎.

    python workbench/data.py              # 바뀐 짝만 다시 만들기
    python workbench/data.py --force      # 전부 다시

만들어 둔 것은 구역·시차·정답표·설정 파일이나 기록이 바뀌면 그 짝만 다시 만든다(서명으로 확인).
결과와 영상은 저장소에 올리지 않는다 (outputs/·source/).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from workbench.analyze import analyze, card  # noqa: E402

OUT = ROOT / "outputs" / "workbench"
VERSION = 1
"""보기 데이터의 모양이 바뀌면 올린다 — 예전에 만들어 둔 것을 전부 다시 만들게 된다."""
RUN_KEY = "current"


def natural(pair: str) -> tuple:
    """N1, N2, …, N10 순서 (글자 순이면 N10 이 N2 앞에 온다). 세트는 정상 → 절도 → 복합."""
    digits = "".join(c if c.isdigit() else " " for c in pair).split()
    return ("NTC".find(pair[0]) % 9, [int(d) for d in digits])


class Builder:
    """짝별 데이터를 만들고 상태를 알려 준다. 서버는 build_async 로 뒤에서 돌리고 status 를 화면에 보인다."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.state = {"running": False, "done": 0, "total": 0, "current": "", "errors": [], "seconds": 0.0}
        self._ctx = None

    # -- 판정에 쓰는 것들 (한 번 읽어 둔다)
    def ctx(self):
        if self._ctx is None:
            import csv

            import dual_eval
            import review_export
            import sync_pair
            from src.core.config import AppConfig
            from src.inputs.sync import SyncTable
            self._ctx = {"dual_eval": dual_eval, "review_export": review_export, "sync_pair": sync_pair,
                         "AppConfig": AppConfig, "SyncTable": SyncTable, "csv": csv}
        return self._ctx

    def records_dir(self) -> Path:
        return ROOT / "outputs" / "dual_eval_pose3"

    def pairs(self) -> list[str]:
        names = [p.name for p in self.records_dir().iterdir() if (p / ".done").exists()] if self.records_dir().exists() else []
        return sorted(names, key=natural)

    def signature(self, pair: str) -> str:
        """이 짝의 결과를 바꿀 수 있는 파일들의 크기·수정 시각."""
        files = [ROOT / "config.yaml", ROOT / "zones" / "store01_dual" / "floor.yaml",
                 ROOT / "zones" / "store01_dual" / "angle0.yaml", ROOT / "labels" / "store01_dual" / "sync.csv",
                 ROOT / "labels" / "store01_dual" / "truth.csv", self.records_dir() / pair / ".done",
                 self.records_dir() / pair / "meta.json"]
        h = hashlib.sha1(f"v{VERSION}".encode())
        for f in files:
            st = f.stat() if f.exists() else None
            h.update(f"{f.name}:{st.st_size if st else -1}:{int(st.st_mtime) if st else 0};".encode())
        return h.hexdigest()[:16]

    def stale(self) -> list[str]:
        index = self.read_index()
        have = {c["pair"]: c.get("signature") for c in index.get("cards", [])}
        return [p for p in self.pairs()
                if have.get(p) != self.signature(p) or not (OUT / "pairs" / f"{p}.json").exists()]

    def read_index(self) -> dict:
        path = OUT / "index.json"
        if not path.exists():
            return {"cards": []}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"cards": []}

    # -- 만들기
    def build(self, pairs: list[str] | None = None, force: bool = False) -> dict:
        with self.lock:
            started = time.time()
            todo = self.pairs() if force else self.stale()
            if pairs is not None:
                todo = [p for p in todo if p in pairs]
            self.state.update(running=True, done=0, total=len(todo), current="", errors=[], seconds=0.0)
            try:
                if todo:
                    self._build(todo)
                self._write_index()
            finally:
                self.state.update(running=False, current="", seconds=round(time.time() - started, 1))
            return dict(self.state)

    def build_async(self) -> bool:
        if self.state["running"] or not self.stale():
            return False
        self.state.update(running=True, done=0, total=len(self.stale()), current="준비 중")
        threading.Thread(target=self.build, daemon=True).start()
        return True

    def _build(self, todo: list[str]) -> None:
        c = self.ctx()
        de, rx = c["dual_eval"], c["review_export"]
        base = c["AppConfig"].load(ROOT / "config.yaml")
        table = c["SyncTable"].load(de.LABELS / "sync.csv")
        band = de.load_band_map()
        with open(de.LABELS / "truth.csv", encoding="utf-8", newline="") as f:
            truth = {r["pair"]: {"people": int(r["people"]), "expected": r["expected"], "notes": r["notes"], "set": r["set"]}
                     for r in c["csv"].DictReader(f)}
        run_dir = self.records_dir().name
        (OUT / "pairs").mkdir(parents=True, exist_ok=True)
        (OUT / "cards").mkdir(parents=True, exist_ok=True)
        for pair in todo:
            self.state["current"] = pair
            try:
                built = rx.pair_data(RUN_KEY, run_dir, {}, pair, base, table, band, truth)
                if built is None:
                    continue
                data, result = built
                dwell = base.risk.dwell_pay_seconds
                summary = analyze(data, result, dwell)
                data["summary"] = summary
                (OUT / "pairs" / f"{pair}.json").write_text(
                    json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
                item = card(summary)
                item["signature"] = self.signature(pair)
                item["thumb"] = self._thumb(pair, data)
                item["score"] = {k: result[k] for k in ("correct", "correct_last", "no_verdict", "people_ok",
                                                         "expected_source")}
                (OUT / "cards" / f"{pair}.json").write_text(json.dumps(item, ensure_ascii=False), encoding="utf-8")
            except Exception as e:  # noqa: BLE001 — 한 짝이 실패해도 나머지는 만든다. 화면에 그대로 보인다
                self.state["errors"].append(f"{pair}: {e!r}")
            self.state["done"] += 1
            self._write_index()   # 만드는 동안에도 다 된 짝부터 갤러리에 보인다

    def _thumb(self, pair: str, data: dict) -> str | None:
        """전체 카메라 영상의 한 장면을 작은 그림으로. 손님이 가장 크게 보이는 때를 고른다."""
        path = OUT / "thumbs" / f"{pair}.jpg"
        try:
            import cv2
            track = data["tracks"]["angle1"]
            best, best_area = None, 0
            for key, rows in track["frames"].items():
                area = sum((r[5] - r[3]) * (r[6] - r[4]) for r in rows if r[1] > 0)
                if area > best_area:
                    best, best_area = int(key), area
            video = ROOT.parents[1] / data["video"]["angle1"].lstrip("/")
            cap = cv2.VideoCapture(str(video))
            target = best if best is not None else track["frames_total"] // 2
            image = None
            # 이 영상(H.265)은 위치 이동이 어긋나므로 처음부터 차례로 읽는다
            for i in range(target + 1):
                ok = cap.grab()
                if not ok:
                    break
                if i == target:
                    ok, image = cap.retrieve()
            cap.release()
            if image is None:
                return None
            h, w = image.shape[:2]
            image = cv2.resize(image, (480, int(480 * h / w)))
            path.parent.mkdir(parents=True, exist_ok=True)
            ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])   # 한글 경로라 imwrite 대신
            if ok:
                path.write_bytes(buf.tobytes())
            return f"thumbs/{pair}.jpg" if ok else None
        except Exception:  # noqa: BLE001 — 그림이 없어도 카드는 보인다
            return None

    def _write_index(self) -> None:
        cards = []
        for pair in self.pairs():
            path = OUT / "cards" / f"{pair}.json"
            if path.exists():
                cards.append(json.loads(path.read_text(encoding="utf-8")))
        n = len(cards)
        right = [c for c in cards if c["correct"]]
        counted = [c for c in cards if c["people_ok"] is not None]
        totals = {
            "n": n, "correct": len(right),
            "correct_decided": sum(not c["no_verdict"] for c in right),
            "no_verdict": sum(c["no_verdict"] for c in cards),
            "correct_last": sum(c["correct_last"] for c in cards),
            "people_ok": sum(c["people_ok"] for c in counted), "people_n": len(counted),
            "by_set": {s: {"n": sum(c["set"] == s for c in cards), "correct": sum(c["set"] == s and c["correct"] for c in cards)}
                       for s in ("normal", "theft", "complex")},
        }
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "index.json").write_text(json.dumps(
            {"version": VERSION, "built": time.strftime("%Y-%m-%d %H:%M"), "records": self.records_dir().name,
             "payment": "계산대 체류 = 결제", "cards": cards, "totals": totals}, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="바뀌지 않은 짝도 다시 만든다")
    ap.add_argument("--only", help="짝 이름 (쉼표)")
    args = ap.parse_args()
    b = Builder()
    state = b.build(args.only.split(",") if args.only else None, args.force)
    print(f"만든 짝 {state['done']}/{state['total']}, {state['seconds']}초 → {OUT}")
    for e in state["errors"]:
        print("  실패:", e)
    return 1 if state["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
