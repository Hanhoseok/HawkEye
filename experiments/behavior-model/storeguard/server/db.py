"""이벤트 저장소 (SQLite).

기록 항목: 이벤트 ID, 카메라 ID, 행동 종류, 시각, 모델 점수, 대표 이미지, 관련 영상 경로.
모델 점수는 보정되지 않았으므로 확률로 표기하지 않는다(컬럼명 score).
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id        TEXT PRIMARY KEY,
    camera_id       TEXT NOT NULL,
    camera_name     TEXT,
    action          TEXT NOT NULL,
    started_at      REAL NOT NULL,
    started_media_ts REAL,
    ended_at        REAL,
    ended_media_ts  REAL,
    score           REAL NOT NULL,
    peak_score      REAL,
    snapshot_path   TEXT,
    video_path      TEXT,
    extra_json      TEXT,
    created_at      REAL DEFAULT (strftime('%s','now'))
);
CREATE INDEX IF NOT EXISTS idx_events_started ON events(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_camera  ON events(camera_id, started_at DESC);
"""


class EventStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def insert(self, ev: dict) -> None:
        extra = ev.get("extra") or {}
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO events
                   (event_id, camera_id, camera_name, action, started_at, started_media_ts,
                    ended_at, ended_media_ts, score, peak_score, snapshot_path, video_path, extra_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (ev["event_id"], ev["camera_id"], extra.get("camera_name"), ev["action"],
                 ev["started_at"], ev.get("started_media_ts"), ev.get("ended_at"),
                 ev.get("ended_media_ts"), ev["score"], ev.get("peak_score"),
                 ev.get("snapshot_path"), ev.get("video_path"),
                 json.dumps(extra, ensure_ascii=False)))
            self._conn.commit()

    def close_event(self, event_id: str, ended_at: float, ended_media_ts: float | None,
                    peak_score: float | None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE events SET ended_at=?, ended_media_ts=?, peak_score=? WHERE event_id=?",
                (ended_at, ended_media_ts, peak_score, event_id))
            self._conn.commit()

    def recent(self, limit: int = 50, camera_id: str | None = None,
               action: str | None = None) -> list[dict]:
        q = "SELECT * FROM events"
        cond, args = [], []
        if camera_id:
            cond.append("camera_id=?")
            args.append(camera_id)
        if action:
            cond.append("action=?")
            args.append(action)
        if cond:
            q += " WHERE " + " AND ".join(cond)
        q += " ORDER BY started_at DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._conn.execute(q, args).fetchall()
        return [self._row(r) for r in rows]

    def get(self, event_id: str) -> dict | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone()
        return self._row(r) if r else None

    def counts(self) -> dict:
        with self._lock:
            rows = self._conn.execute(
                "SELECT action, COUNT(*) c FROM events GROUP BY action").fetchall()
            total = self._conn.execute("SELECT COUNT(*) c FROM events").fetchone()["c"]
        return {"total": total, "by_action": {r["action"]: r["c"] for r in rows}}

    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        d = dict(r)
        if d.get("extra_json"):
            try:
                d["extra"] = json.loads(d["extra_json"])
            except Exception:
                d["extra"] = {}
        d.pop("extra_json", None)
        # 파일 경로는 내부 정보다. 클라이언트에는 존재 여부만 준다.
        d["has_snapshot"] = bool(d.get("snapshot_path") and Path(d["snapshot_path"]).exists())
        d["has_video"] = bool(d.get("video_path") and Path(d["video_path"]).exists())
        return d
