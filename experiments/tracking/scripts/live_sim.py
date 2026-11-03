"""실시간 흉내 — 녹화본을 카메라 속도로 흘려 실시간 입력(LiveSource)을 시험한다 (설계 1-7, limitations F-6).

실제 rtsp 카메라가 없어서, 영상 파일을 원래 초당 장면 수대로만 내보내는 가짜 카메라를 두 대 만들고
카메라 2대 파이프라인을 실시간 모드(최신 장면만 처리, 밀린 장면은 버림)로 돌린다.
보는 것: 처리가 카메라보다 느려도 지연이 쌓이지 않는가(버린 장면 수, 처리한 장면의 지연), 판정이 녹화본과 같은가.

    python scripts/live_sim.py N1 T1
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import dual_eval  # noqa: E402
from src.core.config import AppConfig  # noqa: E402
from src.core.types import RiskLevel  # noqa: E402
from src.inputs.live_source import LiveSource  # noqa: E402
import src.multicam as multicam  # noqa: E402


class PacedCapture:
    """영상 파일을 원래 속도로만 내보내는 가짜 카메라. 끝나면 읽기 실패(=끊김)."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.start = time.perf_counter()
        self.n = 0

    def isOpened(self):
        return self.cap.isOpened()

    def get(self, prop):
        return self.cap.get(prop)

    def read(self):
        due = self.start + self.n / self.fps
        wait = due - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        self.n += 1
        return self.cap.read()

    def release(self):
        self.cap.release()


def main(pairs: list[str]) -> int:
    base = AppConfig.load(ROOT / "config.yaml")
    lag = {}

    def live_source(self, cam):
        src = LiveSource(cam.source, cam.name, max_reconnects=0, opener=PacedCapture, idle_timeout_seconds=3.0)
        lag[cam.name] = src
        return src

    multicam.MultiCameraPipeline._source = live_source
    from src.detectors.yolo_detector import build_detector, build_item_detector
    from src.identity.embedder import build_embedder
    from src.identity.registry import IdentityRegistry
    from src.interaction.detector import InteractionDetector
    from src.risk.engine import RiskEngine
    from src.risk.payments import PaymentFeed
    from src.shelf.watcher import ShelfWatcher
    from src.trackers import build_tracker

    for pair in pairs:
        cfg = dual_eval.dual_config(base, pair, 0.0)
        out = ROOT / "outputs" / "live_sim" / pair
        pipe = multicam.build_multicam(cfg, out, build_detector=build_detector, build_tracker=build_tracker,
                                       build_embedder=build_embedder, build_item_detector=build_item_detector,
                                       IdentityRegistry=IdentityRegistry, InteractionDetector=InteractionDetector,
                                       ShelfWatcher=ShelfWatcher, RiskEngine=RiskEngine, PaymentFeed=PaymentFeed)
        started = time.perf_counter()
        stats = pipe.run()
        wall = time.perf_counter() - started
        levels = [e.level.value for e in stats.events if e.level != RiskLevel.WARNING]
        drops = {k: v.dropped for k, v in lag.items()}
        print(f"{pair}: 처리 {stats.frames} / 버린 장면 {drops} / 걸린 시간 {wall:.1f}s (영상 길이와 비교) / 판정 {levels}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["N1", "T1"]))
