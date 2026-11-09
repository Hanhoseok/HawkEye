"""모의 거리 시퀀스로 수량 추정 로직 검증 (하드웨어 불필요).

실행: python -m unittest discover -s tests -v   (python 폴더에서)
기준거리 (thickness 66, empty 430): 0개 430, 1개 364, 2개 298, 3개 232, 4개 166, 5개 100
"""
import unittest

from shelf_monitor.config import ConfigError, LaneConfig, load_config
from shelf_monitor.estimator import (CHANGING, OUT_OF_RANGE, SENSOR_ERROR, STABLE, UNCERTAIN,
                                     LaneEstimator)
from tests.helpers import Feeder, lane_cfg

R = {n: 430 - 66 * n for n in range(6)}


def started(count=3, **kw):
    est = LaneEstimator(lane_cfg(**kw))
    f = Feeder(est)
    f.hold(R[count], 3000)
    return est, f


class BootAndNoise(unittest.TestCase):
    def test_boot_with_items_is_init_not_add(self):
        est, f = started(3)
        evs = f.take()
        self.assertEqual([e.event for e in evs], ["INIT"])
        self.assertEqual(evs[0].current_count, 3)
        self.assertIsNone(evs[0].quantity)
        self.assertEqual(est.state, STABLE)

    def test_init_waits_for_stability(self):
        est = LaneEstimator(lane_cfg())
        f = Feeder(est)
        f.hold(R[3], 1000)          # stable_time 1200ms 미만
        self.assertEqual(f.names(), [])
        self.assertIsNone(est.confirmed)

    def test_noise_and_spikes_no_event(self):
        est, f = started(3)
        f.take()
        f.hold(R[3], 60000, noise=4, spike_prob=0.03)
        self.assertEqual(f.names(), [])
        self.assertEqual(est.confirmed, 3)


class Removal(unittest.TestCase):
    def test_remove_one(self):
        est, f = started(3)
        f.take()
        f.hold(R[2], 3000)
        evs = f.take()
        self.assertEqual([(e.event, e.previous_count, e.current_count, e.quantity) for e in evs],
                         [("REMOVE", 3, 2, 1)])

    def test_remove_many_at_once_with_fast_jump(self):
        # 중간 위치를 하나도 측정하지 못하고 바로 0개 위치로 점프
        est, f = started(4)
        f.take()
        f.hold(R[0], 3000)
        evs = f.take()
        self.assertEqual([(e.event, e.quantity, e.current_count) for e in evs], [("REMOVE", 4, 0)])

    def test_spring_rebound_single_event(self):
        est, f = started(3)
        f.take()
        f.spring(R[3], R[1], 4000)      # 2개 제거 후 반동하며 정착
        evs = f.take()
        self.assertEqual([(e.event, e.quantity) for e in evs], [("REMOVE", 2)])

    def test_large_change_is_not_discarded(self):
        est, f = started(5)
        f.take()
        f.hold(R[0], 3000)
        self.assertEqual(f.names(), ["REMOVE"])
        self.assertEqual(est.confirmed, 0)


class Addition(unittest.TestCase):
    def test_add_one_and_many(self):
        est, f = started(1)
        f.take()
        f.hold(R[2], 3000)
        f.hold(R[5], 3000)
        evs = f.take()
        self.assertEqual([(e.event, e.previous_count, e.current_count, e.quantity) for e in evs],
                         [("ADD", 1, 2, 1), ("ADD", 2, 5, 3)])

    def test_same_count_no_event(self):
        est, f = started(3)
        f.take()
        f.hold(R[3] + 8, 3000)
        f.hold(R[3] - 8, 3000)
        self.assertEqual(f.names(), [])


class Transients(unittest.TestCase):
    def test_brief_hand_cover(self):
        est, f = started(3)
        f.take()
        f.hold(70, 500, noise=3)        # 손이 센서 앞 70mm
        f.hold(R[3], 3000)
        self.assertEqual(f.names(), [])
        self.assertEqual(est.confirmed, 3)

    def test_hand_held_long_is_reported_out_of_range_not_clipped(self):
        est, f = started(3)
        f.take()
        f.hold(70, 3000, noise=2)       # max_count(5개)=100mm 보다 가까움
        evs = f.take()
        self.assertEqual([e.event for e in evs], ["OUT_OF_RANGE"])
        self.assertEqual(est.confirmed, 3)          # max_count 로 잘라 확정하지 않음
        self.assertEqual(est.state, OUT_OF_RANGE)
        f.hold(R[3], 3000)
        self.assertEqual(f.names(), [])

    def test_beyond_empty_is_out_of_range(self):
        est, f = started(0)
        f.take()
        f.hold(R[0] + 60, 3000)
        self.assertEqual(f.names(), ["OUT_OF_RANGE"])
        self.assertEqual(est.confirmed, 0)

    def test_boundary_oscillation_hysteresis(self):
        est, f = started(3)
        f.take()
        # 3개 유지 구간(232±25) 가장자리에서 반복 흔들림
        f.values([R[3] - 14, R[3] - 22] * 200)
        self.assertEqual(f.names(), [])
        # 3과 4 사이(두 허용 구간 밖)에서 흔들림 → 불확실, 이벤트 없음
        f.values([R[3] - 30, R[3] - 36] * 100)
        self.assertEqual(f.names(), [])
        self.assertEqual(est.state, UNCERTAIN)
        self.assertEqual(est.confirmed, 3)

    def test_isolated_invalid_breaks_stability(self):
        est, f = started(3)
        f.take()
        f.hold(R[2], 800)
        f.invalid(100)                   # 무효 1회 → 안정 구간 다시 시작
        f.hold(R[2], 1000)               # 무효 이후 1000ms 로는 부족
        self.assertEqual(f.names(), [])
        f.hold(R[2], 800)
        self.assertEqual(f.names(), ["REMOVE"])

    def test_long_gap_not_continuous(self):
        est, f = started(3)
        f.take()
        f.hold(R[2], 600)
        f.t += 2000                      # 측정 공백 (max_sample_gap 600ms 초과)
        f.hold(R[2], 700)
        self.assertEqual(f.names(), [])
        f.hold(R[2], 1000)
        self.assertEqual(f.names(), ["REMOVE"])


class SensorErrors(unittest.TestCase):
    def test_disconnect_and_recover_gives_resync(self):
        est, f = started(3)
        f.take()
        f.invalid(3000, "NOT_CONNECTED")
        self.assertEqual(f.names(), ["SENSOR_ERROR"])
        self.assertEqual(est.state, SENSOR_ERROR)
        self.assertEqual(est.confirmed, 3)          # 0 으로 바꾸지 않음
        f.take()
        f.hold(R[2], 3000)                          # 오류 중 1개 빠짐
        evs = f.take()
        self.assertEqual([e.event for e in evs], ["RECOVERED", "RESYNC"])
        rs = evs[1]
        self.assertEqual((rs.previous_count, rs.current_count, rs.detail["net_change"]), (3, 2, -1))
        self.assertFalse(rs.detail["observed"])

    def test_timeouts_need_threshold(self):
        est, f = started(3)
        f.take()
        f.invalid(300, "TIMEOUT")                   # 3회 < error_invalid_count(5)
        f.hold(R[3], 2000)
        self.assertEqual(f.names(), [])

    def test_error_before_first_count(self):
        est = LaneEstimator(lane_cfg())
        f = Feeder(est)
        f.invalid(1000, "INIT_FAILED")
        self.assertEqual(f.names(), ["SENSOR_ERROR"])
        self.assertIsNone(est.confirmed)
        f.take()
        f.hold(R[4], 3000)
        self.assertEqual(f.names(), ["RECOVERED", "INIT"])

    def test_out_of_valid_range_is_invalid(self):
        est, f = started(3)
        f.take()
        f.hold(5000, 2000, noise=0)                 # valid_max_mm 초과
        self.assertEqual(f.names(), ["SENSOR_ERROR"])


class Restock(unittest.TestCase):
    def test_restock_pull_and_release(self):
        est, f = started(2)
        f.take()
        est.set_restock(True)
        f.hold(R[5] - 5, 2500)          # 밀판을 끝까지 당겨 잡고 있음 (안정되어도 이벤트 억제)
        f.hold(R[4], 3000)              # 상품 넣고 놓음
        self.assertEqual(f.names(), [])
        self.assertEqual(est.restock_observed, 4)
        evs = est.set_restock(False)
        self.assertEqual([e.event for e in evs], ["RESTOCK_END"])
        f.hold(R[4], 3000)
        evs = f.take()
        self.assertEqual([(e.event, e.previous_count, e.current_count, e.detail["net_change"]) for e in evs],
                         [("RESYNC", 2, 4, 2)])


class ConfigValidation(unittest.TestCase):
    def test_overlapping_tolerance_rejected(self):
        c = lane_cfg(tolerance_mm=30, hysteresis_mm=10)   # 2*30+10 >= 66
        self.assertTrue(any("겹침" in e for e in c.validate()))

    def test_table_length_and_order(self):
        self.assertTrue(lane_cfg(count_distance_mm=[430, 360]).validate())
        self.assertTrue(lane_cfg(count_distance_mm=[430, 360, 370, 230, 160, 100]).validate())
        self.assertEqual(lane_cfg(count_distance_mm=[431, 362, 300, 229, 168, 99]).validate(), [])

    def test_table_overrides_thickness(self):
        table = [440, 380, 300, 240, 170, 110]
        est = LaneEstimator(lane_cfg(count_distance_mm=table))
        f = Feeder(est)
        f.hold(300, 3000)
        self.assertEqual(est.confirmed, 2)          # 두께식이면 298 → 2 이기도 하지만
        f.take()
        f.hold(380, 3000)                           # 두께식(364)로는 1개 범위 밖, 표로는 1개
        self.assertEqual([(e.event, e.current_count) for e in f.take()], [("REMOVE", 1)])

    def test_per_count_tolerance_separates_close_first_item(self):
        # 밀판 앞 고정 블록 때문에 첫 상품만 18mm 움직이는 레일 (실측: 0개 336, 1개 318)
        kw = dict(count_distance_mm=[340, 318, 253, 188, 123, 58], count_tolerance_mm=[10, 6, 15, 15, 15, 15],
                  hysteresis_mm=4)
        self.assertEqual(lane_cfg(**kw).validate(), [])
        est = LaneEstimator(lane_cfg(**kw))
        f = Feeder(est)
        f.hold(336, 3000, noise=4)
        f.hold(318, 3000, noise=3.5)
        f.hold(336, 3000, noise=4)
        f.hold(253, 3000, noise=4)
        self.assertEqual([(e.event, e.current_count) for e in f.take()],
                         [("INIT", 0), ("ADD", 1), ("REMOVE", 0), ("ADD", 2)])
        f.hold(253, 30000, noise=4, spike_prob=0.03)
        self.assertEqual(f.names(), [])

    def test_per_count_tolerance_validation(self):
        bad = lane_cfg(count_distance_mm=[340, 318, 253, 188, 123, 58], count_tolerance_mm=[15, 15, 15, 15, 15, 15])
        self.assertTrue(any("0개와 1개" in e for e in bad.validate()))
        short = lane_cfg(count_tolerance_mm=[10, 10])
        self.assertTrue(any("count_tolerance_mm 길이" in e for e in short.validate()))

    def test_sample_config_file_loads(self):
        from pathlib import Path
        cfg = load_config(Path(__file__).resolve().parents[1] / "config.toml", enable_all=True)
        self.assertEqual(len(cfg.lanes), 5)
        self.assertGreater(len({l.thickness_mm for l in cfg.lanes}), 1)   # 레인마다 따로 읽힘


if __name__ == "__main__":
    unittest.main()
