"""레인마다 ESP32 가 1대씩 있는 구성: 보드별 추적이 서로 섞이지 않는지 검증."""
import unittest

from shelf_monitor.app import Monitor
from shelf_monitor.config import AppConfig
from tests.helpers import ListSink, lane_cfg, line

R = {n: 430 - 66 * n for n in range(6)}
W = 1_800_000_000.0


class Board:
    def __init__(self, device_id: str, lane_id: int, boot: str):
        self.device_id, self.lane_id, self.boot = device_id, lane_id, boot
        self.seq, self.up = 0, 3000

    def reboot(self, boot: str):
        self.boot, self.seq, self.up = boot, 0, 500

    def measurement(self, dist):
        raw = line(self.seq, self.up, boot=self.boot, device_id=self.device_id, lane_id=self.lane_id,
                   distance_mm=dist, valid=True, sensor_status="OK")
        self.seq += 1
        self.up += 100
        return raw


class Rig:
    def __init__(self, lanes=(1, 2)):
        self.sink = ListSink()
        self.mon = Monitor(AppConfig(lanes=[lane_cfg(i) for i in lanes]), [self.sink])
        self.pc = 100.0
        self.mon.handle_link_up("Wi-Fi UDP:4210", self.pc, W)

    def run(self, boards_and_dist, ms: int):
        """[(board, distance or None=silent), ...] 를 100ms 마다 번갈아 보낸다."""
        for _ in range(ms // 100):
            for board, dist in boards_and_dist:
                if dist is not None:
                    self.mon.handle_line(board.measurement(dist), self.pc, W)
            self.pc += 0.1
            self.mon.tick(self.pc, W)

    def events(self, lane_id):
        return self.sink.lane_events(lane_id)

    def system(self):
        return [r["event"] for r in self.sink.events if r["message_type"] == "SYSTEM"]


class MultiBoard(unittest.TestCase):
    def test_reboot_affects_only_its_lane(self):
        rig = Rig()
        a, b = Board("shelf-lane-1", 1, "a1"), Board("shelf-lane-2", 2, "b1")
        rig.run([(a, R[3]), (b, R[4])], 3000)
        b.reboot("b2")
        rig.run([(a, R[2]), (b, R[1])], 3000)
        self.assertEqual(rig.events(1), [("INIT", None, 3), ("REMOVE", 3, 2)])   # 정상 관찰 → REMOVE
        self.assertEqual(rig.events(2), [("INIT", None, 4), ("RESYNC", 4, 1)])   # 재부팅 → RESYNC
        self.assertEqual(rig.system().count("DEVICE_REBOOT"), 1)
        self.assertEqual(rig.mon.stats()["devices"], 2)

    def test_silent_board_marks_only_its_lane_stale(self):
        rig = Rig()
        a, b = Board("shelf-lane-1", 1, "a1"), Board("shelf-lane-2", 2, "b1")
        rig.run([(a, R[3]), (b, R[4])], 3000)
        rig.run([(a, R[3]), (b, None)], 5000)                                    # 2번 보드 전원 꺼짐
        snaps = {s["lane_id"]: s for s in rig.mon.snapshots(rig.pc)}
        self.assertTrue(snaps[1]["link_ok"])
        self.assertFalse(snaps[2]["link_ok"])
        self.assertTrue(snaps[2]["stale"])
        self.assertEqual(snaps[2]["count"], 4)                                  # 0 으로 바꾸지 않음
        self.assertIn("DEVICE_STALE", rig.system())
        rig.run([(a, R[3]), (b, R[4])], 3000)                                    # 다시 켜짐(같은 부팅)
        self.assertIn("DEVICE_RESTORED", rig.system())
        self.assertEqual(rig.events(2), [("INIT", None, 4), ("RESYNC", 4, 4)])
        self.assertEqual(rig.events(1), [("INIT", None, 3)])

    def test_two_boards_same_lane_number_conflict(self):
        rig = Rig()
        a, a2 = Board("shelf-lane-1", 1, "a1"), Board("shelf-lane-1b", 1, "c1")  # 레인 번호를 같게 잘못 설정
        rig.run([(a, R[3]), (a2, R[0])], 4000)
        self.assertEqual(rig.events(1), [("INIT", None, 3)])                    # 두 번째 보드 값은 섞이지 않음
        self.assertEqual(rig.system().count("LANE_CONFLICT"), 1)

    def test_replacement_board_takes_over_after_old_goes_silent(self):
        rig = Rig()
        old, new = Board("shelf-lane-1", 1, "a1"), Board("shelf-lane-1-new", 1, "n1")
        rig.run([(old, R[3])], 3000)
        rig.run([(old, None)], 5000)                                             # 옛 보드 고장
        rig.run([(new, R[2])], 3000)                                             # 새 보드로 교체
        self.assertIn("LANE_OWNER_CHANGED", rig.system())
        self.assertEqual(rig.events(1), [("INIT", None, 3), ("RESYNC", 3, 2)])


if __name__ == "__main__":
    unittest.main()
