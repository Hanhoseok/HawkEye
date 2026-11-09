"""테스트용: 모의 거리 시퀀스를 추정기/모니터에 넣는 도구."""
from __future__ import annotations

import json
import random

from shelf_monitor.config import LaneConfig
from shelf_monitor.estimator import LaneEstimator, Sample


def lane_cfg(lane_id: int = 1, **kw) -> LaneConfig:
    base = dict(lane_id=lane_id, enabled=True, product_id=f"P{lane_id}", product_name="콜라",
                thickness_mm=66, empty_distance_mm=430, max_count=5, tca_channel=lane_id - 1)
    base.update(kw)
    return LaneConfig(**base)


class Feeder:
    """장치 시각 t(ms)를 진행시키며 LaneEstimator 에 샘플을 넣고 이벤트를 모은다."""

    def __init__(self, est: LaneEstimator, period_ms: int = 100, seed: int = 1, start_ms: int = 1000):
        self.est = est
        self.period = period_ms
        self.t = start_ms
        self.rng = random.Random(seed)
        self.events = []

    def _push(self, dist, valid=True, status="OK"):
        s = Sample(lane_id=self.est.cfg.lane_id, t_ms=self.t, distance_mm=dist, valid=valid,
                   sensor_status=status, pc_mono=self.t / 1000, meta={"uptime_ms": self.t})
        self.events.extend(self.est.on_sample(s))
        self.t += self.period

    def hold(self, dist: float, ms: int, noise: float = 2.5, spike_prob: float = 0.0):
        for _ in range(ms // self.period):
            d = dist + self.rng.gauss(0, noise)
            if self.rng.random() < spike_prob:
                d += self.rng.choice((-1, 1)) * 40
            self._push(round(d))

    def values(self, seq):
        for d in seq:
            self._push(d)

    def invalid(self, ms: int, status: str = "TIMEOUT"):
        for _ in range(ms // self.period):
            self._push(None, valid=False, status=status)

    def spring(self, start: float, target: float, ms: int, freq_hz: float = 3.0, zeta: float = 0.25):
        """스프링 반동: 감쇠 진동하며 target 으로 수렴."""
        import math
        w = 2 * math.pi * freq_hz
        x, v, dt = start, 0.0, 0.001
        for _ in range(ms // self.period):
            for _ in range(int(self.period / 1000 / dt)):
                a = -w * w * (x - target) - 2 * zeta * w * v
                v += a * dt
                x += v * dt
            self._push(round(x + self.rng.gauss(0, 2)))

    def names(self):
        return [e.event for e in self.events]

    def take(self):
        ev, self.events = self.events, []
        return ev


def line(seq: int, uptime: int, mtype: str = "MEASUREMENT", boot: str = "b1", **fields) -> bytes:
    msg = {"message_type": mtype, "device_id": "dev", "boot_id": boot, "sequence": seq, "uptime_ms": uptime}
    msg.update(fields)
    return json.dumps(msg).encode()


class ListSink:
    def __init__(self):
        self.events = []
        self.status = []

    def on_event(self, rec):
        self.events.append(rec)

    def on_status(self, recs):
        self.status.extend(recs)

    def on_device_log(self, msg):
        pass

    def lane_events(self, lane_id=None):
        return [(r["event"], r.get("previous_count"), r.get("current_count"))
                for r in self.events
                if r["message_type"] == "EVENT" and (lane_id is None or r["lane_id"] == lane_id)]
