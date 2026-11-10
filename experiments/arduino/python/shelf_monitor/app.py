"""모니터 본체: 수신 줄 → 보드별 링크 추적 → 레인 추정기 → 이벤트 기록(sink).

입출력(시리얼, UDP, 모의 장치, GUI)은 이 클래스 밖에 있다. handle_line / handle_link_up /
handle_link_down / tick 만 호출하면 되므로 다른 전송 수단에도 그대로 쓸 수 있다.

여러 ESP32 보드(레인마다 1대)가 같은 Wi-Fi 로 보내는 경우를 위해, 재부팅·sequence·수신 지연·
데이터 끊김은 device_id 별로 따로 추적하고, 그 보드가 맡은 레인에만 영향을 준다.
"""
from __future__ import annotations

import json
from collections import deque
from datetime import datetime
from typing import Optional, Protocol

from .config import AppConfig
from .estimator import LaneEstimator, LaneEvent, Sample
from .link import LinkTracker


SMALL_GAP = 5   # 이 개수 이하의 sequence 누락은 기록만 하고 안정 구간을 끊지 않는다


class Sink(Protocol):
    def on_event(self, rec: dict) -> None: ...
    def on_status(self, recs: list[dict]) -> None: ...
    def on_device_log(self, msg: dict) -> None: ...


def _iso(pc_wall: float) -> str:
    return datetime.fromtimestamp(pc_wall).astimezone().isoformat(timespec="milliseconds")


class DeviceState:
    """보드 1대의 수신 상태."""

    def __init__(self, device_id: str, backlog_limit_ms: int, pc_mono: float):
        self.device_id = device_id
        self.link = LinkTracker("", backlog_limit_ms)
        self.last_msg_pc = pc_mono
        self.stale = False
        self.backlog = False
        self.lanes: set[int] = set()
        self.info: dict = {}          # 마지막 HELLO/HEARTBEAT (wifi, ip, rssi ...)
        self.config_error: Optional[str] = None


class Monitor:
    def __init__(self, cfg: AppConfig, sinks: list):
        self.cfg = cfg
        self.sinks = sinks
        self.lanes = {l.lane_id: LaneEstimator(l, cfg.resync_gap_ms) for l in cfg.lanes if l.enabled}
        self.devices: dict[str, DeviceState] = {}
        self.lane_owner: dict[int, str] = {}
        self.connected = False
        self.port: Optional[str] = None
        self.link_note = "연결 대기"
        self.parse_errors = 0
        self.foreign = 0
        self.event_seq = 0
        self.recent = deque(maxlen=40)
        self.restock_since: dict[int, float] = {}
        self.last_status_pc: Optional[float] = None
        self._warned: set[str] = set()

    # ------------------------------------------------------------ 출력
    def _publish(self, rec: dict) -> None:
        self.event_seq += 1
        rec["pc_event_seq"] = self.event_seq
        self.recent.appendleft(rec)
        for s in self.sinks:
            s.on_event(rec)

    def _lane_record(self, ev: LaneEvent, pc_wall: float) -> dict:
        lane = self.lanes[ev.lane_id].cfg
        owner = self.devices.get(self.lane_owner.get(ev.lane_id, ""))
        rec = {
            "message_type": "EVENT",
            "event": ev.event,
            "device_id": ev.meta.get("device_id", owner.device_id if owner else None),
            "boot_id": ev.meta.get("boot_id", owner.link.boot_id if owner else None),
            "sequence": ev.meta.get("sequence"),
            "uptime_ms": ev.meta.get("uptime_ms"),      # ESP32 부팅 후 경과 ms (날짜·시각 아님)
            "lane_id": ev.lane_id,
            "product_id": lane.product_id,
            "product_name": lane.product_name,
            "previous_count": ev.previous_count,
            "current_count": ev.current_count,
            "quantity": ev.quantity,
            "distance_mm": None if ev.distance_mm is None else round(ev.distance_mm, 1),
            "status": ev.status,
        }
        rec.update(ev.detail)
        rec["pc_time"] = _iso(pc_wall)                    # PC 기록 시각 (실제 날짜·시각)
        return rec

    def _publish_lane(self, events: list[LaneEvent], pc_wall: float) -> None:
        for ev in events:
            self._publish(self._lane_record(ev, pc_wall))

    def _system(self, event: str, pc_wall: float, dev: Optional[DeviceState] = None, **extra) -> None:
        rec = {"message_type": "SYSTEM", "event": event,
               "device_id": dev.device_id if dev else None,
               "boot_id": dev.link.boot_id if dev else None,
               "lane_id": None, **extra, "pc_time": _iso(pc_wall)}
        self._publish(rec)

    def _lanes_link_lost(self, reason: str, lane_ids=None) -> None:
        for lane_id, lane in self.lanes.items():
            if lane_ids is None or lane_id in lane_ids:
                lane.on_link_lost(reason)

    # ------------------------------------------------------------ 입력
    def handle_link_up(self, port: str, pc_mono: float, pc_wall: float) -> None:
        self.connected = True
        self.port = port
        self.link_note = f"{port} 연결됨"
        for dev in self.devices.values():
            dev.link.reset_connection()
            dev.last_msg_pc = pc_mono
        self._system("LINK_UP", pc_wall, port=port)

    def handle_link_down(self, reason: str, pc_mono: float, pc_wall: float) -> None:
        self.link_note = reason
        if not self.connected:
            return
        self.connected = False
        self._lanes_link_lost("USB_DISCONNECTED")
        self._system("LINK_LOST", pc_wall, reason=reason)

    def handle_line(self, raw: bytes, pc_mono: float, pc_wall: float) -> None:
        try:
            msg = json.loads(raw.decode("utf-8", errors="strict"))
            if (not isinstance(msg, dict) or not isinstance(msg.get("message_type"), str)
                    or not isinstance(msg.get("device_id"), str) or not isinstance(msg.get("boot_id"), str)
                    or not isinstance(msg.get("sequence"), int) or not isinstance(msg.get("uptime_ms"), int)):
                raise ValueError("필수 필드 없음")
        except (ValueError, UnicodeDecodeError):
            # ESP32 리셋 직후 ROM 부트로그, 포트를 연 순간 잘린 첫 줄 등은 여기서 버려진다.
            self.parse_errors += 1
            return
        if self.cfg.device_id and msg["device_id"] != self.cfg.device_id:
            self.foreign += 1
            return

        dev = self.devices.get(msg["device_id"])
        if dev is None:
            dev = DeviceState(msg["device_id"], self.cfg.backlog_limit_ms, pc_mono)
            self.devices[dev.device_id] = dev
            self._system("DEVICE_SEEN", pc_wall, dev)
        dev.last_msg_pc = pc_mono
        if dev.stale:
            # 끊긴 구간은 이미 "관찰 못함"(RESYNC 대기)으로 처리했다. 수신 간격 기준도 새로 시작해,
            # 공백 시간이 수신 지연(백로그)으로 계산되지 않게 한다.
            dev.stale = False
            dev.link.reset_connection()
            self._system("DEVICE_RESTORED", pc_wall, dev)

        r = dev.link.process(msg, pc_mono)
        if not r.accept:
            return
        if r.reboot:
            self._lanes_link_lost("DEVICE_REBOOT", dev.lanes)
            self._system("DEVICE_REBOOT", pc_wall, dev)
        if r.gap > SMALL_GAP:
            # 몇 줄 누락(Wi-Fi UDP 손실 등)은 레인별 샘플 간격 검사가 처리한다. 크게 빠졌을 때만 구간을 끊는다.
            for lane_id in dev.lanes:
                if lane_id in self.lanes:
                    self.lanes[lane_id].on_discontinuity()
            self._system("SEQUENCE_GAP", pc_wall, dev, missing=r.gap, sequence=msg["sequence"])
        if r.late != dev.backlog:
            dev.backlog = r.late
            self._system("BACKLOG_START" if r.late else "BACKLOG_END", pc_wall, dev,
                         lateness_ms=round(dev.link.lateness_ms))

        mt = msg["message_type"]
        if mt == "MEASUREMENT":
            self._on_measurement(dev, msg, r.dev_ms, r.late, pc_mono, pc_wall)
        elif mt in ("HELLO", "HEARTBEAT", "WIFI_STATUS"):
            dev.info.update({k: v for k, v in msg.items() if k in ("fw_version", "bus_mode", "wifi", "ip", "rssi")})
            if mt != "WIFI_STATUS" and msg.get("config_ok", True):
                dev.config_error = None
        elif mt == "CONFIG_ERROR":
            text = str(msg.get("text", ""))
            if text != dev.config_error:
                dev.config_error = text
                self._system("DEVICE_CONFIG_ERROR", pc_wall, dev, text=text)
        elif mt == "LOG":
            for s in self.sinks:
                s.on_device_log(msg)

    def _warn_once(self, key: str, event: str, pc_wall: float, dev: DeviceState, **extra) -> None:
        if key not in self._warned:
            self._warned.add(key)
            self._system(event, pc_wall, dev, **extra)

    def _on_measurement(self, dev: DeviceState, msg: dict, dev_ms: int, late: bool,
                        pc_mono: float, pc_wall: float) -> None:
        lane_id = msg.get("lane_id")
        lane = self.lanes.get(lane_id) if isinstance(lane_id, int) else None
        if lane is None:
            self._warn_once(f"lane:{lane_id}:{dev.device_id}", "UNKNOWN_LANE", pc_wall, dev, reported_lane_id=lane_id)
            return
        owner_id = self.lane_owner.get(lane_id)
        if owner_id != dev.device_id:
            owner = self.devices.get(owner_id) if owner_id else None
            if owner is not None and not owner.stale:
                # 다른 보드가 이미 이 레인을 보고 중: 두 보드의 레인 번호가 같게 설정된 것
                self._warn_once(f"conflict:{lane_id}:{dev.device_id}", "LANE_CONFLICT", pc_wall, dev,
                                reported_lane_id=lane_id, owner_device_id=owner_id)
                return
            if owner is not None:
                owner.lanes.discard(lane_id)
                lane.on_link_lost("DEVICE_CHANGED")
                self._system("LANE_OWNER_CHANGED", pc_wall, dev, reported_lane_id=lane_id, previous_device_id=owner_id)
            self.lane_owner[lane_id] = dev.device_id
            dev.lanes.add(lane_id)
        dist = msg.get("distance_mm")
        valid = msg.get("valid") is True and isinstance(dist, (int, float))
        sample = Sample(
            lane_id=lane_id, t_ms=dev_ms, distance_mm=dist if valid else None, valid=valid,
            sensor_status=str(msg.get("sensor_status", "UNKNOWN")), pc_mono=pc_mono, late=late,
            meta={k: msg.get(k) for k in ("device_id", "boot_id", "sequence", "uptime_ms")},
        )
        self._publish_lane(lane.on_sample(sample), pc_wall)

    # ------------------------------------------------------------ 상태
    @property
    def link_ok(self) -> bool:
        """수신 경로가 열려 있고 데이터를 보내는 보드가 하나 이상 있다."""
        return self.connected and any(not d.stale for d in self.devices.values())

    def lane_link_ok(self, lane_id: int) -> bool:
        dev = self.devices.get(self.lane_owner.get(lane_id, ""))
        return self.connected and dev is not None and not dev.stale

    @property
    def backlog(self) -> bool:
        return any(d.backlog for d in self.devices.values())

    @property
    def device_config_error(self) -> Optional[str]:
        for d in self.devices.values():
            if d.config_error:
                return f"{d.device_id}: {d.config_error}"
        return None

    def stats(self) -> dict:
        total = {"accepted": 0, "duplicates": 0, "missing": 0, "reboots": 0}
        for d in self.devices.values():
            for k in total:
                total[k] += d.link.stats[k]
        total["devices"] = len(self.devices)
        total["foreign"] = self.foreign
        return total

    def tick(self, pc_mono: float, pc_wall: float) -> None:
        if self.connected:
            for dev in self.devices.values():
                if not dev.stale and pc_mono - dev.last_msg_pc > self.cfg.link_stale_s:
                    dev.stale = True
                    self._lanes_link_lost("NO_DATA", dev.lanes)
                    self._system("DEVICE_STALE", pc_wall, dev, silent_s=round(pc_mono - dev.last_msg_pc, 1))
        if self.cfg.restock_timeout_s > 0:
            for lane_id, since in list(self.restock_since.items()):
                if pc_mono - since > self.cfg.restock_timeout_s:
                    self.set_restock(lane_id, False, pc_mono, pc_wall, reason="TIMEOUT")
        if self.last_status_pc is None or pc_mono - self.last_status_pc >= self.cfg.status_interval_s:
            self.last_status_pc = pc_mono
            recs = []
            for snap in self.snapshots(pc_mono):
                owner = self.devices.get(snap["device_id"] or "")
                recs.append({"message_type": "STATUS", "boot_id": owner.link.boot_id if owner else None,
                             **snap, "pc_time": _iso(pc_wall)})
            for s in self.sinks:
                s.on_status(recs)

    def snapshots(self, pc_mono: float) -> list[dict]:
        out = []
        for lane_id, lane in self.lanes.items():
            ok = self.lane_link_ok(lane_id)
            snap = lane.snapshot(pc_mono, self.cfg.lane_stale_s, ok)
            snap["device_id"] = self.lane_owner.get(lane_id)
            snap["link_ok"] = ok
            out.append(snap)
        return out

    def set_restock(self, lane_id: int, on: bool, pc_mono: float, pc_wall: float, reason: str = "USER") -> None:
        lane = self.lanes.get(lane_id)
        if lane is None:
            return
        events = lane.set_restock(on)
        if on and events:
            self.restock_since[lane_id] = pc_mono
        if not on:
            self.restock_since.pop(lane_id, None)
        for ev in events:
            ev.detail["reason"] = reason
        self._publish_lane(events, pc_wall)
