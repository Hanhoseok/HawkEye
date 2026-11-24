"""서버 중계(노트북 RelaySender → 서버 relay_server.RelayReceiver) 테스트."""
import importlib.util
import json
import shutil
import socket
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shelf_monitor import relay as relay_mod
from shelf_monitor.relay import RelaySender, is_tailscale_ip, parse_target

_spec = importlib.util.spec_from_file_location("relay_server", ROOT / "tools" / "relay_server.py")
relay_server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(relay_server)


class _FakeSock:
    def __init__(self):
        self.out = []

    def sendto(self, data, addr):
        self.out.append((data, addr))

    def close(self):
        pass


class Parsing(unittest.TestCase):
    def test_parse_target(self):
        self.assertEqual(parse_target("100.118.214.101:4220"), ("100.118.214.101", 4220))
        self.assertEqual(parse_target(" da-perfect57:4220 "), ("da-perfect57", 4220))
        for bad in ("100.1.1.1", ":4220", "host:", "host:abc", "host:70000"):
            with self.assertRaises(ValueError):
                parse_target(bad)

    def test_tailscale_range(self):
        self.assertTrue(is_tailscale_ip("100.118.214.101"))
        self.assertTrue(is_tailscale_ip("100.64.0.1"))
        self.assertFalse(is_tailscale_ip("100.128.0.1"))
        self.assertFalse(is_tailscale_ip("192.168.43.10"))


class RouteGuard(unittest.TestCase):
    """Tailscale 주소로 보낼 때, 경로가 Tailscale 이 아니면(꺼져 있으면) 보내지 않는다."""

    def setUp(self):
        self._orig = relay_mod.local_ip_for

    def tearDown(self):
        relay_mod.local_ip_for = self._orig

    def _sender(self, local_ip):
        relay_mod.local_ip_for = lambda addr: local_ip
        s = RelaySender("100.118.214.101:4220", log=lambda m: None)
        s.sock.close()
        s.sock = _FakeSock()
        return s

    def test_blocks_when_tailscale_off(self):
        s = self._sender("192.168.43.10")
        s.send_raw(b'{"message_type":"HEARTBEAT"}')
        self.assertEqual(s.sock.out, [])
        self.assertEqual(s.failed, 1)

    def test_sends_when_tailscale_route(self):
        s = self._sender("100.80.1.2")
        s.send_raw(b'{"message_type":"HEARTBEAT"}')
        self.assertEqual(len(s.sock.out), 1)
        self.assertEqual(s.sock.out[0][1], ("100.118.214.101", 4220))


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.rx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx_sock.bind(("127.0.0.1", 0))
        self.rx_sock.settimeout(1.0)
        self.port = self.rx_sock.getsockname()[1]
        self.echoed = []
        self.rx = relay_server.RelayReceiver(self.tmp, echo=self.echoed.append)

    def tearDown(self):
        self.rx.close()
        self.rx_sock.close()
        shutil.rmtree(self.tmp)

    def _drain(self, n):
        for _ in range(n):
            data, (ip, _p) = self.rx_sock.recvfrom(65535)
            self.rx.handle(data, ip)

    def test_roundtrip_with_dedup_and_latest(self):
        tx = RelaySender(f"127.0.0.1:{self.port}", log=lambda m: None)
        raw = (b'{"message_type":"MEASUREMENT","device_id":"shelf-lane-2","boot_id":"ab","sequence":7,'
               b'"uptime_ms":1000,"lane_id":2,"distance_mm":314,"valid":true,"sensor_status":"OK"}')
        tx.send_raw(raw)
        tx.on_event({"message_type": "EVENT", "event": "REMOVE", "lane_id": 2, "product_name": "동과루",
                     "quantity": 1, "current_count": 3})
        tx.on_status([{"message_type": "STATUS", "lane_id": 1, "count": 4, "state": "STABLE",
                       "product_name": "칸타타", "distance_mm": 185.0},
                      {"message_type": "STATUS", "lane_id": 2, "count": 3, "state": "STABLE",
                       "product_name": "동과루", "distance_mm": 248.0}])
        self._drain(1 + 2 + 2)          # 원본 1, 이벤트 2벌, 상태 2
        tx.close()

        day = next(self.tmp.iterdir())
        read = lambda k: [json.loads(l) for l in (day / f"{k}.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(read("raw")[0]["data"]["distance_mm"], 314)
        self.assertEqual(len(read("events")), 1)          # 두 벌 중 하나는 중복으로 버림
        self.assertEqual(self.rx.dups, 1)
        self.assertEqual(len(read("status")), 2)
        latest = json.loads((self.tmp / "latest.json").read_text(encoding="utf-8"))
        self.assertEqual(latest["lanes"]["2"]["count"], 3)
        self.assertEqual(latest["lanes"]["1"]["count"], 4)
        self.assertIn("제거", self.echoed[0])
        self.assertEqual(self.rx.runs[tx.run_id].loss_ratio(), 0.0)

    def test_rejects_non_tailscale_source_and_garbage(self):
        env = json.dumps({"relay_v": 1, "run_id": "x", "rseq": 1, "kind": "raw", "sent": 0, "data": "{}"}).encode()
        self.assertIsNone(self.rx.handle(env, "203.0.113.5"))
        self.assertEqual(self.rx.rejected, 1)
        self.assertIsNone(self.rx.handle(b"not json", "100.70.0.1"))
        self.assertEqual(self.rx.bad, 1)
        self.assertEqual(self.rx.handle(env, "100.70.0.1"), "raw")

    def test_loss_counted_from_sequence_gap(self):
        st = relay_server.RunStats()
        for r in (1, 2, 4, 5):
            st.add(r)
        self.assertAlmostEqual(st.loss_ratio(), 0.2)


if __name__ == "__main__":
    unittest.main()
