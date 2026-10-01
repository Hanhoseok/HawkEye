"""SQLite 저장소 — 경보(events), 사건(cases), 관리자 처리 이력(resolutions)."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .cases import Current, decide

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL, camera_id TEXT NOT NULL, person_id INTEGER NOT NULL,
    level TEXT NOT NULL, state TEXT NOT NULL,
    taken INTEGER, paid INTEGER, unpaid INTEGER, reason TEXT, identity_check REAL, items TEXT,
    last_event_id INTEGER,
    resolution TEXT, resolution_note TEXT, resolved_at TEXT,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    UNIQUE (run_id, camera_id, person_id)
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER NOT NULL REFERENCES cases(id),
    level TEXT NOT NULL, time_sec REAL, frame INTEGER, timestamp TEXT,
    taken INTEGER, paid INTEGER, unpaid INTEGER, reason TEXT, identity_check REAL,
    raw TEXT NOT NULL, has_snapshot INTEGER NOT NULL DEFAULT 0, received_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS resolutions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER NOT NULL REFERENCES cases(id),
    resolution TEXT NOT NULL, note TEXT, at TEXT NOT NULL
);
"""

CASE_FIELDS = (
    "id run_id camera_id person_id level state taken paid unpaid reason identity_check "
    "last_event_id resolution resolution_note resolved_at created_at updated_at"
).split()


class Store:
    def __init__(self, db_path: str | Path, snapshot_dir: str | Path) -> None:
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()
        self._last_ts = datetime.min.replace(tzinfo=timezone.utc)

    # --- 시각: updated_since 비교를 위해 항상 증가하는 ISO8601 문자열 ---
    def _now(self) -> str:
        now = datetime.now(timezone.utc)
        if now <= self._last_ts:
            now = self._last_ts + timedelta(microseconds=1)
        self._last_ts = now
        return now.isoformat(timespec="microseconds")

    def snapshot_path(self, event_id: int) -> Path:
        return self.snapshot_dir / f"{event_id}.jpg"

    # --- 쓰기 ---
    def add_event(self, run_id: str, camera_id: str, ev: dict[str, Any],
                  snapshot: bytes | None) -> tuple[dict[str, Any], bool]:
        """경보 하나를 저장하고 사건을 갱신한다. (사건, 알림 여부)를 돌려준다."""
        with self._lock, self._db:
            now = self._now()
            row = self._db.execute(
                "SELECT * FROM cases WHERE run_id=? AND camera_id=? AND person_id=?",
                (run_id, camera_id, ev["person_id"]),
            ).fetchone()
            d = decide(Current(row["level"], row["state"]) if row else None, ev["level"])

            if row is None:
                case_id = self._db.execute(
                    "INSERT INTO cases (run_id, camera_id, person_id, level, state, created_at, updated_at)"
                    " VALUES (?,?,?,?,?,?,?)",
                    (run_id, camera_id, ev["person_id"], d.level, d.state, now, now),
                ).lastrowid
            else:
                case_id = row["id"]

            event_id = self._db.execute(
                "INSERT INTO events (case_id, level, time_sec, frame, timestamp, taken, paid, unpaid,"
                " reason, identity_check, raw, has_snapshot, received_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (case_id, ev["level"], ev.get("time_sec"), ev.get("frame"), ev.get("timestamp"),
                 ev.get("taken"), ev.get("paid"), ev.get("unpaid"), ev.get("reason"),
                 ev.get("identity_check"), json.dumps(ev, ensure_ascii=False), int(snapshot is not None), now),
            ).lastrowid
            if snapshot is not None:
                self.snapshot_path(event_id).write_bytes(snapshot)

            reopen = ", resolution=NULL, resolution_note=NULL, resolved_at=NULL" if d.reopened else ""
            self._db.execute(
                f"UPDATE cases SET level=?, state=?, taken=?, paid=?, unpaid=?, reason=?, identity_check=?,"
                f" items=?, last_event_id=?, updated_at=?{reopen} WHERE id=?",
                (d.level, d.state, ev.get("taken"), ev.get("paid"), ev.get("unpaid"), ev.get("reason"),
                 ev.get("identity_check"),
                 json.dumps(ev["items"], ensure_ascii=False) if ev.get("items") is not None else None,
                 event_id, now, case_id),
            )
        return self._case(case_id), d.notify

    def resolve(self, case_id: int, resolution: str, note: str | None) -> dict[str, Any] | None:
        with self._lock, self._db:
            if self._db.execute("SELECT 1 FROM cases WHERE id=?", (case_id,)).fetchone() is None:
                return None
            now = self._now()
            self._db.execute(
                "INSERT INTO resolutions (case_id, resolution, note, at) VALUES (?,?,?,?)",
                (case_id, resolution, note, now),
            )
            self._db.execute(
                "UPDATE cases SET state='resolved', resolution=?, resolution_note=?, resolved_at=?, updated_at=?"
                " WHERE id=?",
                (resolution, note, now, now, case_id),
            )
        return self._case(case_id)

    # --- 읽기 ---
    def _case(self, case_id: int) -> dict[str, Any] | None:
        row = self._db.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
        return self._case_dict(row) if row else None

    def _case_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        c = {k: row[k] for k in CASE_FIELDS}
        c["items"] = json.loads(row["items"]) if row["items"] else None
        last = self._db.execute("SELECT has_snapshot FROM events WHERE id=?", (row["last_event_id"],)).fetchone()
        c["has_snapshot"] = bool(last and last["has_snapshot"])
        return c

    def get_case(self, case_id: int) -> dict[str, Any] | None:
        with self._lock:
            c = self._case(case_id)
            if c is None:
                return None
            c["events"] = [
                {"id": e["id"], "level": e["level"], "time_sec": e["time_sec"], "frame": e["frame"],
                 "timestamp": e["timestamp"], "taken": e["taken"], "paid": e["paid"], "unpaid": e["unpaid"],
                 "reason": e["reason"], "identity_check": e["identity_check"],
                 "has_snapshot": bool(e["has_snapshot"]), "received_at": e["received_at"]}
                for e in self._db.execute("SELECT * FROM events WHERE case_id=? ORDER BY id", (case_id,))
            ]
            c["resolutions"] = [
                {"resolution": r["resolution"], "note": r["note"], "at": r["at"]}
                for r in self._db.execute("SELECT * FROM resolutions WHERE case_id=? ORDER BY id", (case_id,))
            ]
            return c

    def list_cases(self, state: str | None = None, updated_since: str | None = None,
                   limit: int = 100) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM cases WHERE 1=1", []
        if state:
            sql += " AND state=?"; args.append(state)
        if updated_since:
            sql += " AND updated_at > ?"; args.append(updated_since)
        sql += " ORDER BY updated_at DESC LIMIT ?"; args.append(limit)
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
            return [self._case_dict(r) for r in rows]

    def has_snapshot(self, event_id: int) -> bool:
        with self._lock:
            r = self._db.execute("SELECT has_snapshot FROM events WHERE id=?", (event_id,)).fetchone()
        return bool(r and r["has_snapshot"]) and self.snapshot_path(event_id).exists()

    def stats(self) -> dict[str, dict[str, int]]:
        with self._lock:
            states = dict(self._db.execute("SELECT state, COUNT(*) FROM cases GROUP BY state").fetchall())
            res = dict(self._db.execute(
                "SELECT resolution, COUNT(*) FROM cases WHERE resolution IS NOT NULL GROUP BY resolution"
            ).fetchall())
        return {"states": states, "resolutions": res}
