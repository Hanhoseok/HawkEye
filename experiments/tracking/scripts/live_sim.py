"""실시간 흉내 — 녹화본을 카메라 속도로 흘려 실시간 입력(LiveSource)으로 카메라 2대 파이프라인을 돌리고 채점한다.

실제 rtsp 카메라가 없어서, 영상 파일을 원래 속도대로만 내보내는 가짜 카메라 두 대를 만들고
파이프라인을 실시간 모드(최신 장면만 처리, 밀린 장면은 버림, 판정은 파이프라인 안에서 바로)로 돌린다.
보는 것: 이 컴퓨터로 실시간 시연이 되는가 — 처리 속도(장면/초)·버린 장면, 판정이 녹화본 judge 단계 점수와 얼마나 다른가.

- 두 카메라는 같은 시각 0점(LiveSource.start_s)에서 시작한다. 짝의 시차(sync.csv)는 기다리지 않고 앞부분을 건너뛰어 맞춘다
  (offset>0: 전체 카메라 앞 offset ms, offset<0: 선반 카메라 앞 -offset ms) — 첫 장면을 늦추면 두 입력을 섞는 쪽이 멈춰 장면이 버려진다.
- 3차 전체 카메라는 장면을 빠뜨리는 영상(VFR) → 실제 장면 시각표대로 내보낸다.
- 결과는 실행마다 조금씩 다르다 (벽시계, 장면 버림). 결제 라벨·센서는 쓰지 않는다 (비교 기준: judge 단계 '영상만').

    python scripts/live_sim.py --shoot 3            # 3차 전부
    python scripts/live_sim.py --shoot 2 "C1*"      # 2차 일부
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import dual_eval  # noqa: E402
import sync_pair  # noqa: E402
from src.core.config import AppConfig  # noqa: E402
from src.core.types import RiskLevel  # noqa: E402
from src.inputs.live_source import LiveSource  # noqa: E402
from src.inputs.sync import SyncTable  # noqa: E402
import src.multicam as multicam  # noqa: E402


class PacedCapture:
    """영상 파일을 원래 속도로만 내보내는 가짜 카메라. 끝나면 읽기 실패(=끊김).

    start_s: 공통 시각 0점(time.time 기준) — 이 순간에 (skip_ms 뒤) 첫 장면이 나온다.
    times: 장면 번호 → 영상 시각(ms). 없으면 번호 ÷ fps.
    """

    def __init__(self, path, start_s: float, skip_ms: float = 0.0, times: list[float] | None = None):
        self.cap = cv2.VideoCapture(str(path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.times = times
        self.start_s, self.skip_ms = start_s, max(0.0, skip_ms)
        self.n = 0
        while self.skip_ms > 0 and self._t(self.n) < self.skip_ms:   # 앞부분 건너뛰기 (기다리지 않음)
            if not self.cap.grab():
                break
            self.n += 1

    def _t(self, n: int) -> float:
        if self.times:
            return self.times[min(n, len(self.times) - 1)] if n < len(self.times) else float("inf")
        return n * 1000.0 / self.fps

    def isOpened(self):
        return self.cap.isOpened()

    def get(self, prop):
        return self.cap.get(prop)

    def read(self):
        t = self._t(self.n)
        if t == float("inf"):
            return False, None
        wait = self.start_s + (t - self.skip_ms) / 1000.0 - time.time()
        if wait > 0:
            time.sleep(wait)
        self.n += 1
        return self.cap.read()

    def release(self):
        self.cap.release()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("only", nargs="?", help="짝 이름 패턴 (dual_eval 과 같음)")
    ap.add_argument("--shoot", type=int, choices=[2, 3], default=2)
    ap.add_argument("--out", help="outputs/ 아래 폴더 (기본 live_sim_shoot<N>)")
    args = ap.parse_args()
    dual_eval.set_profile(args.shoot)
    out_root = ROOT / "outputs" / (args.out or f"live_sim_shoot{args.shoot}")
    base = AppConfig.load(ROOT / "config.yaml")
    table = SyncTable.load(dual_eval.LABELS / "sync.csv")
    truth = dual_eval.load_truth()
    from src.detectors.yolo_detector import build_detector, build_item_detector
    from src.identity.embedder import build_embedder
    from src.identity.registry import IdentityRegistry
    from src.interaction.detector import InteractionDetector
    from src.risk.engine import RiskEngine
    from src.risk.payments import PaymentFeed
    from src.shelf.watcher import ShelfWatcher
    from src.trackers import build_tracker

    multicam.is_live_source = lambda source: True   # 영상 파일을 실시간 카메라로 다룬다 (시간축·처리 속도)
    rows = []
    for pair in dual_eval.pairs(args.only):
        try:
            cfg = dual_eval.dual_config(base, pair, 0.0)   # 시차 0 — 실시간은 시계 하나, 시차는 건너뛰기로
        except dual_eval.PAIR_ERRORS as e:
            print(f"{pair}: 건너뜀 {type(e).__name__}: {e}", flush=True)
            continue
        offset = table.offset_ms(pair) or 0.0
        skip = {"angle1": max(0.0, offset), "angle0": max(0.0, -offset)}
        times = {}
        if args.shoot == 3:
            import shoot3_eval
            video, ref = sync_pair.pair_files(pair, 3)
            for cam, path in (("angle0", video), ("angle1", ref)):
                times[cam] = shoot3_eval.cached_times(path, dual_eval.OUT / pair / f"times_{cam}.json")
        live: dict[str, LiveSource] = {}

        def live_source(self, cam, start_s=None):
            opener = lambda src, cam=cam: PacedCapture(src, start_s, skip[cam.name], times.get(cam.name))
            live[cam.name] = LiveSource(cam.source, cam.name, max_reconnects=0, opener=opener,
                                        idle_timeout_seconds=3.0, start_s=start_s)
            return live[cam.name]

        multicam.MultiCameraPipeline._source = live_source
        pipe = multicam.build_multicam(cfg, out_root / pair, build_detector=build_detector, build_tracker=build_tracker,
                                       build_embedder=build_embedder, build_item_detector=build_item_detector,
                                       IdentityRegistry=IdentityRegistry, InteractionDetector=InteractionDetector,
                                       ShelfWatcher=ShelfWatcher, RiskEngine=RiskEngine, PaymentFeed=PaymentFeed)
        if args.shoot == 3:
            pipe.judge.detail_point = dual_eval.detail_point_for(pair)   # judge 단계와 같게 (카메라 움직임 되돌림)
        lag: dict[str, list[float]] = {}

        def timed(worker, name):
            step = worker.step

            def run_step(frame):
                out = step(frame)
                if frame.wall_ms is not None:   # 카메라에서 받은 순간 → 이 장면 처리가 끝난 순간 (ms)
                    lag.setdefault(name, []).append(time.time() * 1000.0 - frame.wall_ms)
                return out
            worker.step = run_step

        timed(pipe.overview, pipe.overview_cam.name)
        if pipe.detail is not None:
            timed(pipe.detail, pipe.detail_cam.name)
        started = time.perf_counter()
        stats = pipe.run()
        wall = time.perf_counter() - started
        resolve = pipe.overview.registry.resolve
        finals = [e for e in stats.events if e.level != RiskLevel.WARNING]
        people = sum(1 for c in pipe.judge.risk.book.customers.values() if c.seen_inside)
        scored = dual_eval.score(pair, [(resolve(e.person_id), e.level.value) for e in finals], people, truth)
        rate = {cam: round(n / max(wall, 1e-6), 2) for cam, n in stats.frames.items()}
        row = {"pair": pair, "kind": dual_eval.kind(pair), "verdicts": [e.level.value for e in finals], **scored,
               "people": people, "reasons": [f"P{e.person_id} {e.level.value}: {e.reason}" for e in finals],
               "frames": dict(stats.frames), "dropped": {k: v.dropped for k, v in live.items()},
               "rate": rate, "wall_sec": round(wall, 1),
               "lag_ms": {k: {"median": round(sorted(v)[len(v) // 2]), "p90": round(sorted(v)[int(len(v) * 0.9)]),
                              "max": round(max(v))} for k, v in lag.items() if v}}
        rows.append(row)
        mark = "OK" if row["correct"] else "XX"
        print(f"{mark} {pair}: 장면/초 {rate} 버림 {row['dropped']} 지연 {row['lag_ms']} {wall:.1f}s "
              f"판정 {row['verdicts']} 손님 {people}", flush=True)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "results_live.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    dual_eval.print_summary(dual_eval.summarize(rows), f"실시간 {args.shoot}차")
    if rows:
        tot = {cam: sum(r["frames"].get(cam, 0) for r in rows) / sum(r["wall_sec"] for r in rows) for cam in rows[0]["frames"]}
        print("평균 처리 장면/초: " + ", ".join(f"{k} {v:.2f}" for k, v in tot.items()))
        for cam in rows[0]["frames"]:
            meds = sorted(r["lag_ms"][cam]["median"] for r in rows if cam in r.get("lag_ms", {}))
            if meds:
                print(f"처리 지연 {cam}: 짝별 중앙값의 중앙값 {meds[len(meds) // 2]}ms, 가장 큰 짝 {meds[-1]}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
