"""짧은 보정: 안정 구간 측정, 설정 계산, config.toml 부분 수정, 적용 후 RESYNC."""
import random
import shutil
import tempfile
import tomllib
import unittest
from pathlib import Path

from shelf_monitor.app import Monitor
from shelf_monitor.calibration import (DONE, MEASURING, WAIT_STABLE, CalibrationError, LevelCapture,
                                       compute_settings)
from shelf_monitor.config import AppConfig, load_config, update_lane_in_toml
from shelf_monitor.estimator import LaneEstimator, Sample
from tests.helpers import Feeder, ListSink, lane_cfg

ROOT = Path(__file__).resolve().parents[1]


def capture(level, seconds=10, noise=3.0, seed=1, start_ms=1000, moving_ms=0):
    rng = random.Random(seed)
    c = LevelCapture()
    t = start_ms
    for i in range(int(seconds * 10)):
        d = level + rng.gauss(0, noise)
        if t - start_ms < moving_ms:                       # 처음엔 밀판이 출렁임
            d += 40 * ((i % 4) - 1.5)
        c.feed(t, d)
        t += 100
    return c


class Capture(unittest.TestCase):
    def test_measures_stable_level(self):
        c = capture(345)
        self.assertEqual(c.state, DONE)
        self.assertAlmostEqual(c.result.median, 345, delta=2)
        self.assertLess(c.result.lo, c.result.median)
        self.assertGreater(c.result.hi, c.result.median)

    def test_waits_while_moving(self):
        c = capture(345, seconds=4, moving_ms=3000)
        self.assertEqual(c.state, WAIT_STABLE)            # 흔들리는 동안은 측정 시작 안 함
        c2 = capture(345, seconds=12, moving_ms=3000)
        self.assertEqual(c2.state, DONE)
        self.assertAlmostEqual(c2.result.median, 345, delta=2)

    def test_restarts_when_disturbed_during_measure(self):
        c = LevelCapture()
        t = 0
        for _ in range(40):
            c.feed(t, 345); t += 100
        self.assertEqual(c.state, MEASURING)
        for _ in range(10):
            c.feed(t, 300); t += 100                       # 손이 닿아 크게 바뀜
        self.assertEqual(c.state, WAIT_STABLE)


class Settings(unittest.TestCase):
    def test_lane1_like_rail(self):
        e, o, t = capture(345).result, capture(328, seed=2).result, capture(263, seed=3).result
        r = compute_settings(e, o, t)
        self.assertAlmostEqual(r.thickness_mm, 65, delta=2)
        self.assertAlmostEqual(r.gap01_mm, 17, delta=3)
        self.assertGreaterEqual(r.max_count_geometry, 4)
        cfg = lane_cfg(thickness_mm=r.thickness_mm, empty_distance_mm=r.empty_distance_mm, max_count=4,
                       count_distance_mm=r.count_distance_mm[:5], count_tolerance_mm=r.count_tolerance_mm[:5],
                       hysteresis_mm=r.hysteresis_mm, median_window=r.median_window)
        self.assertEqual(cfg.validate(), [])
        est = LaneEstimator(cfg)
        f = Feeder(est)
        for level in (345, 328, 345, 263, 198):
            f.hold(level, 3000, noise=3.5)
        self.assertEqual([(e.event, e.current_count) for e in f.take()],
                         [("INIT", 0), ("ADD", 1), ("REMOVE", 0), ("ADD", 2), ("ADD", 3)])

    def test_first_item_does_not_move_pusher(self):
        e, o, t = capture(345).result, capture(344, seed=2).result, capture(279, seed=3).result
        with self.assertRaises(CalibrationError):
            compute_settings(e, o, t)

    def test_second_item_missing(self):
        e, o, t = capture(345).result, capture(328, seed=2).result, capture(326, seed=3).result
        with self.assertRaises(CalibrationError):
            compute_settings(e, o, t)

    def test_narrow_gap_warns(self):
        e, o, t = capture(339, noise=4).result, capture(332, noise=4, seed=2).result, capture(266, seed=3).result
        r = compute_settings(e, o, t)
        self.assertTrue(r.warnings)


class TomlUpdate(unittest.TestCase):
    def test_updates_only_target_lane(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            path = tmp / "config.toml"
            shutil.copy(ROOT / "config.toml", path)
            before = tomllib.loads(path.read_text(encoding="utf-8"))
            vals = {"product_name": '새 "과자" 이름', "product_id": "SNACK-NEW", "thickness_mm": 44.5,
                    "empty_distance_mm": 341.0, "max_count": 4,
                    "count_distance_mm": [343.5, 330.0, 285.5, 241.0, 196.5],
                    "count_tolerance_mm": [5.0, 5.0, 15.0, 15.0, 15.0], "hysteresis_mm": 2.0, "median_window": 5}
            backup = update_lane_in_toml(path, 3, vals, note="자동 보정 테스트")
            self.assertTrue(backup.exists())
            after = tomllib.loads(path.read_text(encoding="utf-8"))
            lane3 = next(l for l in after["lanes"] if l["lane_id"] == 3)
            for k, v in vals.items():
                self.assertEqual(lane3[k], v)
            for i, l in enumerate(after["lanes"]):
                if l["lane_id"] != 3:
                    self.assertEqual(l, before["lanes"][i])          # 다른 레인은 그대로
            self.assertEqual(after["serial"], before["serial"])
            self.assertIn("# 자동 보정 테스트", path.read_text(encoding="utf-8"))
            self.assertEqual(load_config(path).lane(3).product_name, '새 "과자" 이름')
        finally:
            shutil.rmtree(tmp)

    def test_invalid_result_is_rolled_back(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            path = tmp / "config.toml"
            shutil.copy(ROOT / "config.toml", path)
            original = path.read_text(encoding="utf-8")
            with self.assertRaises(Exception):
                update_lane_in_toml(path, 3, {"max_count": 4, "count_distance_mm": [300, 310]})
            self.assertEqual(path.read_text(encoding="utf-8"), original)
        finally:
            shutil.rmtree(tmp)


class Rename(unittest.TestCase):
    def test_rename_keeps_calibration_and_comments(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            path = tmp / "config.toml"
            shutil.copy(ROOT / "config.toml", path)
            before_text = path.read_text(encoding="utf-8")
            before = next(l for l in tomllib.loads(before_text)["lanes"] if l["lane_id"] == 1)
            update_lane_in_toml(path, 1, {"product_name": "하이트 제로", "product_id": "CAN-HITE-ZERO"},
                                drop_comments=False)
            text = path.read_text(encoding="utf-8")
            after = next(l for l in tomllib.loads(text)["lanes"] if l["lane_id"] == 1)
            self.assertEqual(after["product_name"], "하이트 제로")
            for k in ("count_distance_mm", "count_tolerance_mm", "empty_distance_mm", "thickness_mm"):
                self.assertEqual(after[k], before[k])
            self.assertEqual(text.count("#"), before_text.count("#"))      # 보정 설명 주석 유지
        finally:
            shutil.rmtree(tmp)

    def test_monitor_rename_keeps_count(self):
        sink = ListSink()
        mon = Monitor(AppConfig(lanes=[lane_cfg(1)]), [sink])
        f = Feeder(mon.lanes[1])
        f.hold(232, 3000)
        mon.rename_product(1, "새 이름", "NEW-ID", 1_800_000_000.0)
        self.assertEqual(mon.cfg.lane(1).product_name, "새 이름")
        self.assertEqual(mon.lanes[1].cfg.product_id, "NEW-ID")
        self.assertEqual(mon.lanes[1].confirmed, 3)
        self.assertIn("PRODUCT_RENAMED", [r["event"] for r in sink.events])


class Apply(unittest.TestCase):
    def test_recalibrate_keeps_count_and_resyncs(self):
        sink = ListSink()
        mon = Monitor(AppConfig(lanes=[lane_cfg(1)]), [sink])
        lane = mon.lanes[1]
        f = Feeder(lane)
        f.hold(232, 3000)                                    # 기존 설정으로 3개
        mon.set_restock(1, True, 0, 1_800_000_000.0)
        new = lane_cfg(1, product_name="새 상품", count_distance_mm=[345, 328, 263, 198, 133, 68],
                       count_tolerance_mm=[7, 6, 15, 15, 15, 15], hysteresis_mm=2)
        mon.recalibrate_lane(new, 0, 1_800_000_000.0)
        events = [r["event"] for r in sink.events]
        self.assertIn("RESTOCK_END", events)
        self.assertIn("LANE_CALIBRATED", events)
        est = mon.lanes[1]
        self.assertIsNot(est, lane)
        self.assertEqual(mon.cfg.lane(1).product_name, "새 상품")
        f2 = Feeder(est, start_ms=f.t)
        f2.hold(263, 3000)                                   # 보정 뒤 2개 남음
        self.assertEqual([(e.event, e.previous_count, e.current_count) for e in f2.take()], [("RESYNC", 3, 2)])


if __name__ == "__main__":
    unittest.main()
