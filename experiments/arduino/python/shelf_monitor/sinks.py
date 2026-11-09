"""이벤트 출력: 콘솔(한국어 요약), JSONL 파일."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

STATE_KO = {
    "INITIALIZING": "초기화 중",
    "STABLE": "안정",
    "CHANGING": "변화 확인 중",
    "UNCERTAIN": "불확실",
    "OUT_OF_RANGE": "범위 밖",
    "SENSOR_ERROR": "센서 오류",
}


def _cnt(v) -> str:
    return "?" if v is None else f"{v}개"


def describe(rec: dict) -> str:
    """이벤트 레코드 → 사람이 읽는 한 줄."""
    ev = rec["event"]
    if rec["message_type"] == "SYSTEM":
        extra = {k: v for k, v in rec.items()
                 if k not in ("message_type", "event", "device_id", "boot_id", "lane_id", "pc_time", "pc_event_seq")}
        names = {
            "LINK_UP": "수신 시작", "LINK_LOST": "USB 연결 끊김", "LINK_STALE": "데이터 수신 없음(stale)",
            "DEVICE_SEEN": "보드 발견", "DEVICE_STALE": "보드 데이터 끊김(stale)", "DEVICE_RESTORED": "보드 데이터 재개",
            "LANE_CONFLICT": "두 보드가 같은 레인 번호 사용 — 레인 번호 다시 설정 필요",
            "LANE_OWNER_CHANGED": "레인을 보내는 보드가 바뀜",
            "LINK_RESTORED": "데이터 수신 재개", "DEVICE_REBOOT": "ESP32 재부팅 감지",
            "SEQUENCE_GAP": "메시지 누락", "BACKLOG_START": "수신 지연(버퍼 적체) - 해당 샘플은 판정 제외",
            "BACKLOG_END": "수신 지연 해소", "DEVICE_CONFIG_ERROR": "펌웨어 설정 오류",
            "LANE_CONFIG_MISMATCH": "펌웨어/파이썬 레인 설정 불일치", "UNKNOWN_LANE": "설정에 없는 레인 데이터",
        }
        tail = " ".join(f"{k}={v}" for k, v in extra.items())
        return f"[시스템] {names.get(ev, ev)} {tail}".rstrip()

    head = f"{rec['lane_id']}번 레인 / {rec['product_name']}"
    cur = rec.get("current_count")
    if ev == "INIT":
        return f"{head} / 초기 수량 {_cnt(cur)} (INIT)"
    if ev == "ADD":
        return f"{head} / {rec['quantity']}개 추가 / 현재 {_cnt(cur)}"
    if ev == "REMOVE":
        return f"{head} / {rec['quantity']}개 제거 / 현재 {_cnt(cur)}"
    if ev == "RESYNC":
        net = rec.get("net_change")
        return (f"{head} / 재동기화 {_cnt(rec.get('previous_count'))} → {_cnt(cur)} "
                f"(순변화 {net:+d}, 관찰 못한 구간: {rec.get('reason')}) — 입출고로 단정하지 않음")
    if ev == "SENSOR_ERROR":
        return f"{head} / 센서 오류 {rec.get('error_kind')} / 마지막 확정 {_cnt(cur)} 유지(stale)"
    if ev == "RECOVERED":
        return f"{head} / 센서 복구 (오류 {rec.get('error_kind')}, {rec.get('outage_ms')}ms) / 안정화 후 RESYNC"
    if ev == "OUT_OF_RANGE":
        return (f"{head} / 수량 범위 밖 거리 {rec.get('distance_mm')}mm ({rec.get('reason')}, "
                f"추정 {rec.get('estimated_count')}) — 설치·보정 확인 필요")
    if ev == "RESTOCK_START":
        return f"{head} / 재입고 모드 시작 (기준 {_cnt(rec.get('previous_count'))}, 중간 이벤트 억제)"
    if ev == "RESTOCK_END":
        return f"{head} / 재입고 모드 종료 ({rec.get('reason')}) — 안정되면 RESYNC"
    return f"{head} / {ev}"


class ConsoleSink:
    def __init__(self, show_status: bool = True):
        self.show_status = show_status

    def on_event(self, rec: dict) -> None:
        t = datetime.now().strftime("%H:%M:%S")
        print(f"[{t}] {describe(rec)}", flush=True)

    def on_status(self, recs: list[dict]) -> None:
        if not self.show_status:
            return
        parts = []
        for r in recs:
            st = STATE_KO.get(r["state"], r["state"])
            if r["restock"]:
                st = "재입고 중"
            flag = " stale" if r["stale"] else ""
            dist = "-" if r["distance_mm"] is None else f"{r['distance_mm']:.0f}mm"
            if not r["link_ok"]:
                flag += " 보드없음"
            parts.append(f"L{r['lane_id']} {r['product_name']} {_cnt(r['count'])} {st}{flag} {dist}")
        link = ""
        print(f"[{datetime.now():%H:%M:%S}] 상태{link} | " + " | ".join(parts), flush=True)

    def on_device_log(self, msg: dict) -> None:
        lane = f" L{msg['lane_id']}" if "lane_id" in msg else ""
        print(f"[{datetime.now():%H:%M:%S}] [ESP32 {msg.get('level', '')}{lane}] {msg.get('text', '')}", flush=True)


class JsonlSink:
    """확정 이벤트는 events 파일, 주기 STATUS 는 status 파일(선택)에 한 줄씩 추가."""

    def __init__(self, event_path: Path, status_path: Optional[Path] = None):
        event_path.parent.mkdir(parents=True, exist_ok=True)
        self._ev = open(event_path, "a", encoding="utf-8")
        self._st = None
        if status_path is not None:
            status_path.parent.mkdir(parents=True, exist_ok=True)
            self._st = open(status_path, "a", encoding="utf-8")

    @staticmethod
    def _write(f, rec: dict) -> None:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()

    def on_event(self, rec: dict) -> None:
        self._write(self._ev, rec)

    def on_status(self, recs: list[dict]) -> None:
        if self._st is not None:
            for r in recs:
                self._write(self._st, r)

    def on_device_log(self, msg: dict) -> None:
        self._write(self._ev, {**msg, "message_type": "DEVICE_LOG"})

    def close(self) -> None:
        self._ev.close()
        if self._st is not None:
            self._st.close()
