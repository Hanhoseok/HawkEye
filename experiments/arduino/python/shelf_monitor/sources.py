"""데이터 소스: 실제 시리얼 포트, 또는 하드웨어 없는 모의 장치.

둘 다 별도 스레드에서 돌며 큐에 다음 튜플을 넣는다 (수신 시각은 스레드에서 바로 기록):
  ("UP",   port,  pc_mono, pc_wall)
  ("DOWN", 사유,   pc_mono, pc_wall)
  ("LINE", bytes, pc_mono, pc_wall)
"""
from __future__ import annotations

import json
import math
import queue
import random
import threading
import time
from typing import Optional

from .config import AppConfig, LaneConfig

KNOWN_USB_VIDS = {0x10C4: "CP210x", 0x1A86: "CH340", 0x0403: "FTDI", 0x303A: "Espressif"}


def list_ports() -> list[str]:
    from serial.tools import list_ports as lp
    out = []
    for p in lp.comports():
        tag = KNOWN_USB_VIDS.get(p.vid or -1, "")
        out.append(f"{p.device}\t{tag or '-'}\t{p.description}")
    return out


def find_esp32_port() -> Optional[str]:
    from serial.tools import list_ports as lp
    for p in lp.comports():
        if p.vid in KNOWN_USB_VIDS:
            return p.device
    return None


class SerialSource(threading.Thread):
    def __init__(self, port: str, baud: int, q: queue.Queue, stop: threading.Event):
        super().__init__(daemon=True)
        self.port_cfg, self.baud, self.q, self.stop = port, baud, q, stop

    def _put(self, kind: str, payload) -> None:
        self.q.put((kind, payload, time.monotonic(), time.time()))

    def run(self) -> None:
        import serial
        last_down = None
        while not self.stop.is_set():
            port = find_esp32_port() if self.port_cfg == "auto" else self.port_cfg
            if not port:
                reason = "ESP32 USB 포트를 찾지 못함 (--list-ports 로 확인 후 --port 지정)"
                if reason != last_down:
                    self._put("DOWN", reason)
                    last_down = reason
                self.stop.wait(2)
                continue
            ser = serial.Serial()
            ser.port, ser.baudrate, ser.timeout = port, self.baud, 0.2
            # 포트를 열 때 DTR/RTS 로 ESP32 가 리셋되지 않도록 (보드에 따라 효과가 다를 수 있음)
            ser.dtr = False
            ser.rts = False
            try:
                ser.open()
                ser.reset_input_buffer()          # 열기 전 OS 버퍼에 쌓인 오래된 데이터 폐기
            except (serial.SerialException, OSError) as e:
                msg = str(e)
                if "PermissionError" in msg or "Access is denied" in msg or "액세스" in msg:
                    reason = f"{port} 사용 중 (Arduino 시리얼 모니터 등 다른 프로그램을 닫으세요)"
                else:
                    reason = f"{port} 열기 실패: {msg}"
                if reason != last_down:
                    self._put("DOWN", reason)
                    last_down = reason
                self.stop.wait(2)
                continue
            last_down = None
            self._put("UP", port)
            buf = b""
            try:
                while not self.stop.is_set():
                    chunk = ser.readline()
                    if not chunk:
                        continue
                    buf += chunk
                    if buf.endswith(b"\n"):
                        line, buf = buf.strip(), b""
                        if line:
                            self._put("LINE", line)
                    elif len(buf) > 4096:
                        buf = b""                  # 줄바꿈 없는 쓰레기 데이터
            except (serial.SerialException, OSError) as e:
                last_down = f"{port} 연결 끊김: {e}"
                self._put("DOWN", last_down)
            finally:
                try:
                    ser.close()
                except Exception:
                    pass
            self.stop.wait(1)


class UdpSource(threading.Thread):
    """Wi-Fi: ESP32 가 보내는 UDP 데이터그램(한 개 = JSON 한 줄)을 받는다.

    UDP 는 연결 개념이 없으므로 포트를 열면 UP 으로 보고, 데이터가 끊기면
    Monitor 의 link_stale 판정(데이터 없음)이 처리한다.
    """

    def __init__(self, port: int, q: queue.Queue, stop: threading.Event, host: str = ""):
        super().__init__(daemon=True)
        self.port, self.host, self.q, self.stop = port, host, q, stop

    def _put(self, kind: str, payload) -> None:
        self.q.put((kind, payload, time.monotonic(), time.time()))

    def run(self) -> None:
        import socket
        last_down = None
        while not self.stop.is_set():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((self.host, self.port))
            except OSError as e:
                sock.close()
                reason = f"UDP {self.port} 포트 열기 실패: {e}"
                if reason != last_down:
                    self._put("DOWN", reason)
                    last_down = reason
                self.stop.wait(2)
                continue
            sock.settimeout(0.2)
            self._put("UP", f"Wi-Fi UDP:{self.port}")
            try:
                while not self.stop.is_set():
                    try:
                        data, _addr = sock.recvfrom(4096)
                    except socket.timeout:
                        continue
                    except ConnectionResetError:
                        continue           # Windows: 이전 송신에 대한 ICMP 오류, 수신과 무관
                    for line in data.split(b"\n"):
                        line = line.strip()
                        if line:
                            self._put("LINE", line)
            except OSError as e:
                self._put("DOWN", f"UDP 수신 오류: {e}")
            finally:
                sock.close()
            self.stop.wait(1)


# ---------------------------------------------------------------------------
# 모의 장치: 펌웨어와 같은 JSON Lines 를 만든다 (스프링 반동, 잡음, 손 가림, 센서 오류 포함)
# ---------------------------------------------------------------------------
class _SimLane:
    def __init__(self, cfg: LaneConfig, rng: random.Random):
        self.cfg = cfg
        self.rng = rng
        self.count = rng.randint(max(1, cfg.max_count // 2), cfg.max_count)
        self.calib_offset = rng.uniform(-3, 3)          # 실제 설치와 설정값의 작은 차이
        self.x = self.target()
        self.v = 0.0
        self.hold_until = 0.0                            # 사람이 밀판을 잡고 있음
        self.hold_x = 0.0
        self.cover_until = 0.0                           # 손이 센서 앞을 가림
        self.fault_until = 0.0
        self.next_action = rng.uniform(4, 10)

    def target(self) -> float:
        return self.cfg.reference_mm(self.count) + self.calib_offset

    def step(self, t: float, dt: float) -> None:
        if t < self.hold_until:
            self.x, self.v = self.hold_x, 0.0
            return
        w = 2 * math.pi * 3.0                            # 스프링 고유진동 약 3 Hz, 감쇠비 0.25
        a = -w * w * (self.x - self.target()) - 2 * 0.25 * w * self.v
        self.v += a * dt
        self.x += self.v * dt

    def act(self, t: float, lane_index: int) -> None:
        if t < self.next_action:
            return
        r = self.rng.random()
        c = self.cfg
        if r < 0.35 and self.count > 0:
            self.count -= 1                                       # 1개 제거
        elif r < 0.50 and self.count > 1:
            self.count -= self.rng.randint(2, min(3, self.count))  # 여러 개 제거
        elif r < 0.70 and self.count < c.max_count:
            self.count += self.rng.randint(1, min(2, c.max_count - self.count))  # 추가
        elif r < 0.80:
            self.cover_until = t + 0.4                            # 순간 손 가림
        elif r < 0.90 and self.count < c.max_count:
            # 재입고 동작: 밀판을 뒤로 끝까지 당겨 0.8초 잡고 있다가 상품을 넣고 놓음
            self.hold_x = c.reference_mm(c.max_count) - 5
            self.hold_until = t + 0.8
            self.count = c.max_count
        elif r < 0.95 and lane_index == 2:
            self.fault_until = t + 6                              # 3번째 레인 센서 끊김
            if self.count > 0:
                self.count -= 1                                   # 오류 중 실제 변화 (관찰 불가)
        self.next_action = t + self.rng.uniform(6, 14)

    def measure(self, t: float) -> tuple[Optional[int], bool, str]:
        if t < self.fault_until:
            return None, False, "I2C_ERROR" if self.fault_until - t > 5 else "NOT_CONNECTED"
        if t < self.cover_until:
            return int(self.rng.gauss(70, 3)), True, "OK"
        if self.rng.random() < 0.005:
            return None, False, "RANGE_INVALID"
        d = self.x + self.rng.gauss(0, 2.5)
        if self.rng.random() < 0.01:
            d += self.rng.choice((-1, 1)) * self.rng.uniform(20, 60)  # 순간 튐
        return int(round(d)), True, "OK"


class MockSource(threading.Thread):
    def __init__(self, cfg: AppConfig, q: queue.Queue, stop: threading.Event, seed: Optional[int] = None):
        super().__init__(daemon=True)
        self.q, self.stop = q, stop
        self.rng = random.Random(seed)
        self.lanes = [_SimLane(l, self.rng) for l in cfg.lanes if l.enabled]
        self.boot_id = f"{self.rng.getrandbits(32):08x}"
        self.seq = 0
        self.uptime0 = 12345

    def _line(self, t: float, mtype: str, **fields) -> None:
        self._stamp = max(getattr(self, "_stamp", 0.0), t)    # 실제 장치처럼 uptime 은 줄지 않음
        msg = {"message_type": mtype, "device_id": "mock-esp32", "boot_id": self.boot_id,
               "sequence": self.seq, "uptime_ms": self.uptime0 + int(self._stamp * 1000), **fields}
        self.seq += 1
        self.q.put(("LINE", json.dumps(msg).encode(), time.monotonic(), time.time()))

    def run(self) -> None:
        self.q.put(("UP", "MOCK", time.monotonic(), time.time()))
        start = time.monotonic()
        last = 0.0
        next_hb = 0.0
        slot = max(0.1, 0.04 * len(self.lanes)) / max(1, len(self.lanes))   # 펌웨어와 비슷한 순차 측정
        next_meas = 0.0
        idx = 0
        self._line(0.0, "HELLO", fw_version="mock", bus_mode="TCA9548A",
                   enabled_lanes=[l.cfg.lane_id for l in self.lanes], config_ok=True)
        while not self.stop.is_set():
            t = time.monotonic() - start
            while last < t:                                  # 물리 적분 (5ms)
                last += 0.005
                for i, ln in enumerate(self.lanes):
                    ln.act(last, i)
                    ln.step(last, 0.005)
            if t >= next_hb:
                next_hb += 1.0
                self._line(t, "HEARTBEAT", config_ok=True, enabled_lanes=[l.cfg.lane_id for l in self.lanes])
            while self.lanes and t >= next_meas:
                ln = self.lanes[idx]
                d, valid, st = ln.measure(next_meas)
                self._line(next_meas, "MEASUREMENT", lane_id=ln.cfg.lane_id, distance_mm=d,
                           valid=valid, sensor_status=st)
                idx = (idx + 1) % len(self.lanes)
                next_meas += slot
            time.sleep(0.01)
