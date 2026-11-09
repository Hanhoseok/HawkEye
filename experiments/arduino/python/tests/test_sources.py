"""Wi-Fi(UDP) 수신 경로 검증: 로컬 루프백으로 데이터그램을 보내 Monitor 까지 전달되는지 확인."""
import json
import queue
import socket
import threading
import time
import unittest

from shelf_monitor.app import Monitor
from shelf_monitor.config import AppConfig
from shelf_monitor.sources import UdpSource
from tests.helpers import ListSink, lane_cfg


def free_udp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class UdpReceive(unittest.TestCase):
    def test_datagrams_reach_monitor(self):
        port = free_udp_port()
        q: queue.Queue = queue.Queue()
        stop = threading.Event()
        UdpSource(port, q, stop, host="127.0.0.1").start()
        kind, payload, _, _ = q.get(timeout=3)
        self.assertEqual((kind, payload), ("UP", f"Wi-Fi UDP:{port}"))

        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        seq, up = 0, 1000
        for _ in range(40):                       # 4초 분량, 232mm = 3개 기준
            msg = {"message_type": "MEASUREMENT", "device_id": "dev", "boot_id": "b1", "sequence": seq,
                   "uptime_ms": up, "lane_id": 1, "distance_mm": 232, "valid": True, "sensor_status": "OK"}
            tx.sendto((json.dumps(msg) + "\n").encode(), ("127.0.0.1", port))
            seq, up = seq + 1, up + 100
            if seq == 10:
                seq += 3                          # UDP 손실 흉내: 3줄 누락
        tx.close()

        sink = ListSink()
        mon = Monitor(AppConfig(lanes=[lane_cfg(1)]), [sink])
        mon.handle_link_up("UDP", time.monotonic(), time.time())
        lines = 0
        end = time.monotonic() + 3
        pc = 100.0
        while time.monotonic() < end and lines < 40:
            try:
                kind, payload, _, wall = q.get(timeout=0.5)
            except queue.Empty:
                continue
            if kind == "LINE":
                lines += 1
                pc += 0.1                          # 실시간 간격으로 처리한 것처럼
                mon.handle_line(payload, pc, wall)
        stop.set()
        self.assertEqual(lines, 40)
        self.assertEqual(mon.stats()["missing"], 3)
        self.assertNotIn("SEQUENCE_GAP", [r["event"] for r in sink.events])   # 작은 누락은 구간을 끊지 않음
        self.assertEqual(sink.lane_events(), [("INIT", None, 3)])


if __name__ == "__main__":
    unittest.main()
