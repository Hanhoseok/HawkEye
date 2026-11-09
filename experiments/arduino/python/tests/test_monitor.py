"""JSON Lines 수신부터 이벤트까지 전체 경로 검증 (시리얼 없이 줄을 직접 넣음)."""
import unittest

from shelf_monitor.config import AppConfig
from shelf_monitor.app import Monitor
from shelf_monitor.link import LinkTracker
from tests.helpers import ListSink, lane_cfg, line

R = {n: 430 - 66 * n for n in range(6)}
W = 1_800_000_000.0   # PC 벽시계 (테스트 고정값)


class Rig:
    """5개 레인 순차 측정(레인당 약 200ms 재방문)을 흉내 내는 장치 + PC 시계."""

    def __init__(self, lanes=(1,), boot="b1"):
        self.cfg = AppConfig(lanes=[lane_cfg(i) for i in lanes])
        self.sink = ListSink()
        self.mon = Monitor(self.cfg, [self.sink])
        self.lanes = list(lanes)
        self.seq = 0
        self.up = 5000
        self.pc = 100.0
        self.boot = boot
        self.mon.handle_link_up("TEST", self.pc, W)

    def send(self, raw: bytes, advance_ms: int = 40):
        self.mon.handle_line(raw, self.pc, W)
        self.up += advance_ms
        self.pc += advance_ms / 1000
        self.mon.tick(self.pc, W)

    def run(self, dist_by_lane: dict, ms: int, status_by_lane: dict = None):
        status_by_lane = status_by_lane or {}
        end = self.up + ms
        while self.up < end:
            for ln in self.lanes:
                st = status_by_lane.get(ln, "OK")
                d = dist_by_lane.get(ln)
                if st == "OK":
                    raw = line(self.seq, self.up, boot=self.boot, lane_id=ln, distance_mm=d, valid=True,
                               sensor_status="OK")
                else:
                    raw = line(self.seq, self.up, boot=self.boot, lane_id=ln, distance_mm=None, valid=False,
                               sensor_status=st)
                self.seq += 1
                self.send(raw)


class EndToEnd(unittest.TestCase):
    def test_startup_init_then_remove(self):
        rig = Rig()
        rig.run({1: R[3]}, 3000)
        rig.run({1: R[1]}, 3000)
        self.assertEqual(rig.sink.lane_events(), [("INIT", None, 3), ("REMOVE", 3, 1)])
        rec = [r for r in rig.sink.events if r["event"] == "REMOVE"][0]
        for k in ("message_type", "device_id", "boot_id", "sequence", "uptime_ms", "lane_id", "product_id",
                  "event", "previous_count", "current_count", "quantity", "distance_mm", "status"):
            self.assertIn(k, rec)
        self.assertEqual(rec["quantity"], 2)

    def test_status_does_not_create_events(self):
        rig = Rig()
        rig.cfg.status_interval_s = 0.1
        rig.run({1: R[3]}, 10000)
        self.assertEqual(rig.sink.lane_events(), [("INIT", None, 3)])
        self.assertGreater(len(rig.sink.status), 10)

    def test_one_of_five_lanes_fails(self):
        rig = Rig(lanes=(1, 2, 3, 4, 5))
        base = {i: R[3] for i in range(1, 6)}
        rig.run(base, 3000)
        rig.run({**base, 1: R[2], 5: R[4]}, 3000, status_by_lane={3: "NOT_CONNECTED"})
        ev = rig.sink.lane_events
        self.assertEqual(ev(1), [("INIT", None, 3), ("REMOVE", 3, 2)])
        self.assertEqual(ev(5), [("INIT", None, 3), ("ADD", 3, 4)])
        self.assertEqual(ev(3), [("INIT", None, 3), ("SENSOR_ERROR", 3, 3)])
        self.assertEqual(ev(2), [("INIT", None, 3)])
        rig.run({**base, 1: R[2], 5: R[4], 3: R[1]}, 3000)
        self.assertEqual(ev(3)[2:], [("RECOVERED", 3, 3), ("RESYNC", 3, 1)])

    def test_device_reboot_resync(self):
        rig = Rig()
        rig.run({1: R[3]}, 3000)
        rig.boot, rig.seq, rig.up = "b2", 0, 300       # 재부팅: 새 boot_id, sequence/uptime 처음부터
        rig.run({1: R[2]}, 3000)
        self.assertEqual(rig.sink.lane_events(), [("INIT", None, 3), ("RESYNC", 3, 2)])
        self.assertIn("DEVICE_REBOOT", [r["event"] for r in rig.sink.events])

    def test_usb_disconnect_resync(self):
        rig = Rig()
        rig.run({1: R[3]}, 3000)
        rig.mon.handle_link_down("unplugged", rig.pc, W)
        rig.pc += 10
        rig.up += 10000
        rig.seq += 250
        rig.mon.handle_link_up("TEST", rig.pc, W)
        rig.run({1: R[3]}, 3000)
        self.assertEqual(rig.sink.lane_events(), [("INIT", None, 3), ("RESYNC", 3, 3)])

    def test_no_data_marks_stale_not_zero(self):
        rig = Rig()
        rig.run({1: R[3]}, 3000)
        rig.pc += 5
        rig.mon.tick(rig.pc, W)
        self.assertIn("DEVICE_STALE", [r["event"] for r in rig.sink.events])
        snap = rig.mon.snapshots(rig.pc)[0]
        self.assertTrue(snap["stale"])
        self.assertEqual(snap["count"], 3)

    def test_duplicates_and_garbage_ignored(self):
        rig = Rig()
        rig.mon.handle_line(b"ets Jun  8 2016 00:22:57 rst:0x1", rig.pc, W)
        rig.mon.handle_line(b'{"message_type": "MEAS', rig.pc, W)
        raw = line(0, rig.up, lane_id=1, distance_mm=R[3], valid=True, sensor_status="OK")
        rig.send(raw)
        rig.send(raw)        # 같은 sequence 재수신
        self.assertEqual(rig.mon.parse_errors, 2)
        self.assertEqual(rig.mon.stats()["duplicates"], 1)

    def test_backlog_burst_not_treated_as_realtime(self):
        rig = Rig()
        rig.run({1: R[3]}, 3000)
        # PC 가 2초 멈춘 사이 장치는 계속 측정 → 2초 분량이 한꺼번에 도착
        rig.pc += 2.0
        for _ in range(50):
            rig.mon.handle_line(line(rig.seq, rig.up, lane_id=1, distance_mm=R[1], valid=True,
                                     sensor_status="OK"), rig.pc, W)
            rig.seq += 1
            rig.up += 40
        self.assertIn("BACKLOG_START", [r["event"] for r in rig.sink.events])
        self.assertEqual(rig.sink.lane_events(), [("INIT", None, 3)])   # 몰려온 데이터로 확정하지 않음
        rig.run({1: R[1]}, 3000)                                         # 실시간 수신 재개 후 확정
        self.assertEqual(rig.sink.lane_events()[-1], ("REMOVE", 3, 1))


class Link(unittest.TestCase):
    def msg(self, seq, up, boot="b"):
        return {"message_type": "HEARTBEAT", "boot_id": boot, "sequence": seq, "uptime_ms": up}

    def test_uptime_wrap(self):
        lt = LinkTracker()
        r1 = lt.process(self.msg(1, 0xFFFFFF00), 0.0)
        r2 = lt.process(self.msg(2, 0x00000010), 0.272)
        self.assertEqual(r2.dev_ms - r1.dev_ms, 0x110)
        self.assertFalse(r2.late)

    def test_sequence_gap_and_wrap(self):
        lt = LinkTracker()
        lt.process(self.msg(0xFFFFFFFE, 0), 0)
        r = lt.process(self.msg(0x00000001, 100), 0.1)
        self.assertEqual(r.gap, 2)
        self.assertTrue(lt.process(self.msg(0, 50), 0.1).duplicate)   # 과거 sequence


if __name__ == "__main__":
    unittest.main()
