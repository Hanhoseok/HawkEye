"""보관함 — 촬영 영상을 폴더(회차) → 장면 → 카메라 영상으로 모아 두고 앱에서 보고 관리한다 (docs/design.md §11).

녹화(MediaMTX)와 별개다. 영상 파일은 한 번 넣으면 옮기지 않고, 폴더·이름·소속은 DB 만 바꾼다.

    <보관함>/library.db        폴더·장면·영상 (경보 DB 와 분리)
    <보관함>/files/<uuid>.mp4   영상
    <보관함>/thumbs/<장면 id>.jpg

가져오기 (서버에서):

    python -m hawkeye_server.library import --folder "2차 촬영" --camera cam1=<폴더> --camera cam2=<폴더> [--sync sync.csv]
    python -m hawkeye_server.library prepare <원본 폴더> <출력 폴더>   # DB 없이 변환만 (PC 에서 먼저 줄일 때)
    python -m hawkeye_server.library gc                                # DB 에 없는 파일 지우기
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import time
import re
import sqlite3
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS folders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL, name_key TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS scenes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    folder_id INTEGER NOT NULL REFERENCES folders(id),
    name TEXT NOT NULL, origin TEXT NOT NULL DEFAULT '',
    source_key TEXT UNIQUE, created_at TEXT NOT NULL, media_at TEXT
);
CREATE INDEX IF NOT EXISTS scenes_folder ON scenes(folder_id);
CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scene_id INTEGER NOT NULL REFERENCES scenes(id),
    camera_id TEXT NOT NULL, file TEXT NOT NULL,
    duration REAL, width INTEGER, height INTEGER, size INTEGER,
    offset_ms REAL NOT NULL DEFAULT 0,
    UNIQUE (scene_id, camera_id)
);
CREATE TABLE IF NOT EXISTS ai_videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL UNIQUE REFERENCES videos(id),
    file TEXT NOT NULL, duration REAL, width INTEGER, height INTEGER, size INTEGER
);
CREATE TABLE IF NOT EXISTS scene_verdicts (
    scene_id INTEGER PRIMARY KEY REFERENCES scenes(id),
    data TEXT NOT NULL, source TEXT, updated_at TEXT NOT NULL
);
"""
VERDICT_LEVELS = {"CLEAR", "WARNING", "REVIEW", "HIGH_RISK"}
MAX_VERDICTS = 200
MAX_REASON = 500
MAX_PERSON = 2_147_483_647
"""앱(Kotlin Int)이 받을 수 있는 가장 큰 손님 번호."""
MAX_COUNT = 1_000_000
AI_LENGTH_TOLERANCE_SEC = 1.0
"""AI 화면 영상은 원본과 같은 시간축이어야 한다 (원본에 그려 만든 것). 길이가 이보다 다르면 넣지 않는다."""

GC_GRACE_SEC = 600
"""gc 는 이보다 최근에 바뀐 파일은 건드리지 않는다 (가져오는 중일 수 있음)."""
MAX_NAME = 40
MAX_IDS = 500
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".m4v"}
CAMERA = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class LibraryError(Exception):
    """API 로 그대로 내보내는 오류. code 는 앱이 문구를 고르는 데 쓴다 (detail = "code: 문장")."""

    def __init__(self, status: int, code: str, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.status, self.code, self.message, self.extra = status, code, message, extra


def not_found(what: str) -> LibraryError:
    return LibraryError(404, "not_found", f"{what}이(가) 없습니다")


# --- 순수 함수 -------------------------------------------------------------------

def clean_name(name: Any) -> str:
    """앞뒤 공백을 빼고 1~40자, 제어 문자 없음. 어기면 422 invalid_name."""
    if not isinstance(name, str):
        raise LibraryError(422, "invalid_name", "이름은 글자여야 합니다")
    name = name.strip()
    if not name or len(name) > MAX_NAME or _CONTROL.search(name):
        raise LibraryError(422, "invalid_name", f"이름은 1~{MAX_NAME}자, 줄바꿈 없이 써 주세요")
    return name


def natural_key(s: str) -> list[Any]:
    """C2 < C10, '0차' < '1차' < '시연용'. re.split 은 글자·숫자가 번갈아 나와 같은 자리끼리만 비교된다."""
    parts = re.split(r"(\d+)", s.casefold())
    return [int(p) if i % 2 else p for i, p in enumerate(parts)]


_DATE_PREFIX = re.compile(r"^\d{8}_")
_SCENE = re.compile(r"^([A-Za-z]+)0*(\d+)(.*)$")


def scene_key(stem: str) -> str:
    """카메라마다 다른 파일 이름을 같은 장면 이름으로: '20261028_C1' → 'C01', 'c04-1' → 'C04-1', 'N8' → 'N08'."""
    s = _DATE_PREFIX.sub("", stem.strip())
    m = _SCENE.match(s)
    if not m:
        return s
    prefix, num, rest = m.groups()
    return f"{prefix.upper()}{int(num):02d}{rest}"


def plan_import(cameras: dict[str, list[Path]]) -> list[tuple[str, dict[str, Path]]]:
    """카메라별 파일 목록 → [(장면 이름, {카메라: 파일})] (자연 정렬). 같은 카메라에 같은 장면이 둘이면 앞의 것만."""
    scenes: dict[str, dict[str, Path]] = {}
    for cam, files in cameras.items():
        for f in sorted(files, key=lambda p: natural_key(p.name)):
            scenes.setdefault(scene_key(f.stem), {}).setdefault(cam, f)
    return sorted(scenes.items(), key=lambda kv: natural_key(kv[0]))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ids(scene_ids: Iterable[Any]) -> list[int]:
    out: list[int] = []
    for i in scene_ids:
        if not isinstance(i, int) or isinstance(i, bool):
            raise LibraryError(422, "invalid_ids", "장면 번호가 올바르지 않습니다")
        if i not in out:
            out.append(i)
    if not 1 <= len(out) <= MAX_IDS:
        raise LibraryError(422, "invalid_ids", f"장면은 1~{MAX_IDS}개를 골라 주세요")
    return out


# --- 저장소 ----------------------------------------------------------------------

class Library:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.files = self.root / "files"
        self.thumbs = self.root / "thumbs"
        self.files.mkdir(parents=True, exist_ok=True)
        self.thumbs.mkdir(parents=True, exist_ok=True)
        # 가져오기 도구(다른 프로세스)가 같은 DB 에 쓸 수 있어 잠금을 기다린다.
        self._db = sqlite3.connect(str(self.root / "library.db"), check_same_thread=False, timeout=15)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()

    def close(self) -> None:
        self._db.close()

    # --- 읽기 ---
    def _folder_rows(self, folder_id: int | None = None) -> list[dict[str, Any]]:
        sql = ("SELECT f.id, f.name, COUNT(s.id) AS scene_count, COALESCE(SUM(s.dur), 0) AS total_sec"
               " FROM folders f LEFT JOIN (SELECT sc.id, sc.folder_id,"
               "   (SELECT MAX(v.duration) FROM videos v WHERE v.scene_id = sc.id) AS dur FROM scenes sc) s"
               " ON s.folder_id = f.id")
        args: tuple[Any, ...] = ()
        if folder_id is not None:
            sql += " WHERE f.id = ?"
            args = (folder_id,)
        rows = self._db.execute(sql + " GROUP BY f.id", args).fetchall()
        return [{"id": r["id"], "name": r["name"], "scene_count": r["scene_count"],
                 "total_sec": round(float(r["total_sec"]), 1)} for r in rows]

    def folders(self) -> list[dict[str, Any]]:
        with self._lock:
            return sorted(self._folder_rows(), key=lambda f: natural_key(f["name"]))

    def _folder(self, folder_id: int) -> dict[str, Any]:
        rows = self._folder_rows(folder_id)
        if not rows:
            raise not_found("폴더")
        return rows[0]

    def _scene_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        videos = [{"id": v["id"], "camera_id": v["camera_id"], "duration": v["duration"],
                   "width": v["width"], "height": v["height"], "offset_ms": v["offset_ms"], "has_ai": bool(v["has_ai"])}
                  for v in self._db.execute(
                      "SELECT v.*, EXISTS(SELECT 1 FROM ai_videos a WHERE a.video_id = v.id) AS has_ai"
                      " FROM videos v WHERE v.scene_id=? ORDER BY v.camera_id", (row["id"],))]
        durations = [v["duration"] for v in videos if v["duration"] is not None]
        vr = self._db.execute("SELECT data, source FROM scene_verdicts WHERE scene_id=?", (row["id"],)).fetchone()
        return {"id": row["id"], "folder_id": row["folder_id"], "name": row["name"], "origin": row["origin"],
                "duration": max(durations) if durations else None, "media_at": row["media_at"], "videos": videos,
                "has_thumb": self.thumb_file(row["id"]).exists(),
                "verdicts": json.loads(vr["data"]) if vr else [], "verdict_source": vr["source"] if vr else None}

    def folder(self, folder_id: int) -> dict[str, Any]:
        with self._lock:
            folder = self._folder(folder_id)
            rows = self._db.execute("SELECT * FROM scenes WHERE folder_id=?", (folder_id,)).fetchall()
            scenes = sorted((self._scene_dict(r) for r in rows), key=lambda s: (natural_key(s["name"]),
                                                                              natural_key(s["origin"]), s["id"]))
        return {"folder": folder, "scenes": scenes}

    def scene(self, scene_id: int) -> dict[str, Any]:
        with self._lock:
            row = self._db.execute("SELECT * FROM scenes WHERE id=?", (scene_id,)).fetchone()
            if row is None:
                raise not_found("장면")
            return self._scene_dict(row)

    def video_file(self, video_id: int) -> Path | None:
        with self._lock:
            row = self._db.execute("SELECT file FROM videos WHERE id=?", (video_id,)).fetchone()
        if row is None:
            return None
        path = self.files / row["file"]
        return path if path.is_file() else None

    def ai_file(self, video_id: int) -> Path | None:
        """그 영상의 AI 화면(사람 상자·번호를 그린 것) 파일. 없으면 None."""
        with self._lock:
            row = self._db.execute("SELECT file FROM ai_videos WHERE video_id=?", (video_id,)).fetchone()
        if row is None:
            return None
        path = self.files / row["file"]
        return path if path.is_file() else None

    def thumb_file(self, scene_id: int) -> Path:
        return self.thumbs / f"{scene_id}.jpg"

    # --- 폴더 쓰기 ---
    def create_folder(self, name: Any) -> dict[str, Any]:
        name = clean_name(name)
        with self._lock:
            try:
                with self._db:
                    now = _now()
                    fid = self._db.execute(
                        "INSERT INTO folders (name, name_key, created_at, updated_at) VALUES (?,?,?,?)",
                        (name, name.casefold(), now, now)).lastrowid
            except sqlite3.IntegrityError:
                raise LibraryError(409, "duplicate_name", "같은 이름의 폴더가 있습니다")
            return self._folder(fid)

    def ensure_folder(self, name: str) -> dict[str, Any]:
        """가져오기용: 같은 이름(대소문자 무시) 폴더가 있으면 그것, 없으면 만든다."""
        name = clean_name(name)
        with self._lock:
            row = self._db.execute("SELECT id FROM folders WHERE name_key=?", (name.casefold(),)).fetchone()
        return self._folder_locked(row["id"]) if row else self.create_folder(name)

    def _folder_locked(self, folder_id: int) -> dict[str, Any]:
        with self._lock:
            return self._folder(folder_id)

    def rename_folder(self, folder_id: int, name: Any) -> dict[str, Any]:
        name = clean_name(name)
        with self._lock:
            try:
                with self._db:
                    n = self._db.execute("UPDATE folders SET name=?, name_key=?, updated_at=? WHERE id=?",
                                         (name, name.casefold(), _now(), folder_id)).rowcount
            except sqlite3.IntegrityError:
                raise LibraryError(409, "duplicate_name", "같은 이름의 폴더가 있습니다")
            if n == 0:
                raise not_found("폴더")
            return self._folder(folder_id)

    def delete_folder(self, folder_id: int, with_scenes: bool = False) -> dict[str, int]:
        with self._lock:
            with self._db:
                if self._db.execute("SELECT 1 FROM folders WHERE id=?", (folder_id,)).fetchone() is None:
                    raise not_found("폴더")
                ids = [r["id"] for r in self._db.execute("SELECT id FROM scenes WHERE folder_id=?", (folder_id,))]
                if ids and not with_scenes:
                    raise LibraryError(409, "not_empty", "폴더에 장면이 있습니다", scene_count=len(ids))
                files = self._delete_scene_rows(ids)
                self._db.execute("DELETE FROM folders WHERE id=?", (folder_id,))
        self._remove(files, ids)
        return {"deleted_scenes": len(ids)}

    # --- 장면 쓰기 ---
    def rename_scene(self, scene_id: int, name: Any) -> dict[str, Any]:
        name = clean_name(name)
        with self._lock:
            with self._db:
                n = self._db.execute("UPDATE scenes SET name=? WHERE id=?", (name, scene_id)).rowcount
            if n == 0:
                raise not_found("장면")
        return self.scene(scene_id)

    def _missing(self, ids: list[int]) -> list[int]:
        q = ",".join("?" * len(ids))
        found = {r["id"] for r in self._db.execute(f"SELECT id FROM scenes WHERE id IN ({q})", ids)}
        return [i for i in ids if i not in found]

    def move_scenes(self, scene_ids: Iterable[Any], folder_id: int) -> dict[str, int]:
        """한 트랜잭션: 폴더나 장면이 하나라도 없으면 아무것도 바꾸지 않는다."""
        ids = _ids(scene_ids)
        with self._lock:
            with self._db:
                if self._db.execute("SELECT 1 FROM folders WHERE id=?", (folder_id,)).fetchone() is None:
                    raise not_found("옮길 폴더")
                missing = self._missing(ids)
                if missing:
                    raise LibraryError(404, "not_found", "없는 장면이 있습니다", missing=missing)
                q = ",".join("?" * len(ids))
                moved = self._db.execute(f"UPDATE scenes SET folder_id=? WHERE id IN ({q}) AND folder_id<>?",
                                         [folder_id, *ids, folder_id]).rowcount
        return {"moved": moved}

    def delete_scenes(self, scene_ids: Iterable[Any]) -> dict[str, int]:
        ids = _ids(scene_ids)
        with self._lock:
            with self._db:
                missing = self._missing(ids)
                if missing:
                    raise LibraryError(404, "not_found", "없는 장면이 있습니다", missing=missing)
                files = self._delete_scene_rows(ids)
        self._remove(files, ids)
        return {"deleted": len(ids)}

    def _delete_scene_rows(self, ids: list[int]) -> list[str]:
        """트랜잭션 안에서 부른다. 지울 영상 파일 이름을 돌려준다 (파일은 커밋 뒤에 지운다)."""
        if not ids:
            return []
        q = ",".join("?" * len(ids))
        files = [r["file"] for r in self._db.execute(f"SELECT file FROM videos WHERE scene_id IN ({q})", ids)]
        sub = f"SELECT id FROM videos WHERE scene_id IN ({q})"
        files += [r["file"] for r in self._db.execute(f"SELECT file FROM ai_videos WHERE video_id IN ({sub})", ids)]
        self._db.execute(f"DELETE FROM ai_videos WHERE video_id IN ({sub})", ids)
        self._db.execute(f"DELETE FROM scene_verdicts WHERE scene_id IN ({q})", ids)
        self._db.execute(f"DELETE FROM videos WHERE scene_id IN ({q})", ids)
        self._db.execute(f"DELETE FROM scenes WHERE id IN ({q})", ids)
        return files

    def _remove(self, files: list[str], scene_ids: list[int]) -> None:
        for name in files:
            _unlink(self.files / name)
        for sid in scene_ids:
            _unlink(self.thumb_file(sid))

    # --- 가져오기 ---
    def add_scene(self, folder_id: int, name: str, origin: str, source_key: str) -> tuple[int, bool]:
        """source_key 로 찾고 없으면 만든다. (장면 id, 새로 만듦)."""
        with self._lock:
            with self._db:
                row = self._db.execute("SELECT id FROM scenes WHERE source_key=?", (source_key,)).fetchone()
                if row:
                    return row["id"], False
                sid = self._db.execute(
                    "INSERT INTO scenes (folder_id, name, origin, source_key, created_at) VALUES (?,?,?,?,?)",
                    (folder_id, clean_name(name), origin, source_key, _now())).lastrowid
        return sid, True

    def scene_id_by_source(self, source_key: str) -> int | None:
        with self._lock:
            row = self._db.execute("SELECT id FROM scenes WHERE source_key=?", (source_key,)).fetchone()
        return row["id"] if row else None

    def has_video(self, scene_id: int, camera_id: str) -> bool:
        with self._lock:
            return self._db.execute("SELECT 1 FROM videos WHERE scene_id=? AND camera_id=?",
                                    (scene_id, camera_id)).fetchone() is not None

    def video_id(self, scene_id: int, camera_id: str) -> int | None:
        with self._lock:
            row = self._db.execute("SELECT id FROM videos WHERE scene_id=? AND camera_id=?",
                                   (scene_id, camera_id)).fetchone()
        return row["id"] if row else None

    def has_ai(self, video_id: int) -> bool:
        with self._lock:
            return self._db.execute("SELECT 1 FROM ai_videos WHERE video_id=?", (video_id,)).fetchone() is not None

    def put_ai(self, video_id: int, file: str, info: VideoInfo) -> None:
        """원본 영상(video_id)에 AI 화면 영상을 붙이거나 바꾼다. 길이가 원본과 다르면 LibraryError."""
        with self._lock:
            with self._db:
                raw = self._db.execute("SELECT duration FROM videos WHERE id=?", (video_id,)).fetchone()
                if raw is None:
                    raise not_found("원본 영상")
                if raw["duration"] and info.duration and abs(raw["duration"] - info.duration) > AI_LENGTH_TOLERANCE_SEC:
                    raise LibraryError(422, "length_mismatch",
                                       f"AI 화면 길이({info.duration:.1f}s)가 원본({raw['duration']:.1f}s)과 다릅니다")
                old = self._db.execute("SELECT file FROM ai_videos WHERE video_id=?", (video_id,)).fetchone()
                self._db.execute(
                    "INSERT INTO ai_videos (video_id, file, duration, width, height, size) VALUES (?,?,?,?,?,?)"
                    " ON CONFLICT (video_id) DO UPDATE SET file=excluded.file, duration=excluded.duration,"
                    " width=excluded.width, height=excluded.height, size=excluded.size",
                    (video_id, file, info.duration, info.width, info.height, (self.files / file).stat().st_size))
        if old and old["file"] != file:
            _unlink(self.files / old["file"])

    def put_video(self, scene_id: int, camera_id: str, file: str, info: VideoInfo) -> None:
        """영상을 넣거나 바꾼다 (바뀐 옛 파일은 지운다). 원본을 바꾸면 그 AI 화면과 장면 판정은 시간축 · 내용이
        달라질 수 있어 지운다 (다시 넣어야 함)."""
        stale: list[str] = []
        with self._lock:
            with self._db:
                old = self._db.execute("SELECT id, file FROM videos WHERE scene_id=? AND camera_id=?",
                                       (scene_id, camera_id)).fetchone()
                if old and old["file"] != file:
                    stale = [r["file"] for r in self._db.execute("SELECT file FROM ai_videos WHERE video_id=?", (old["id"],))]
                    self._db.execute("DELETE FROM ai_videos WHERE video_id=?", (old["id"],))
                    # 판정도 그 영상의 시간축 · 내용 기준이라 함께 지운다 (새 판정을 다시 넣어야 함)
                    self._db.execute("DELETE FROM scene_verdicts WHERE scene_id=?", (scene_id,))
                self._db.execute(
                    "INSERT INTO videos (scene_id, camera_id, file, duration, width, height, size)"
                    " VALUES (?,?,?,?,?,?,?) ON CONFLICT (scene_id, camera_id) DO UPDATE SET"
                    " file=excluded.file, duration=excluded.duration, width=excluded.width,"
                    " height=excluded.height, size=excluded.size",
                    (scene_id, camera_id, file, info.duration, info.width, info.height,
                     (self.files / file).stat().st_size))
                self._db.execute("UPDATE scenes SET media_at=? WHERE id=?",
                                 (datetime.now(timezone.utc).isoformat(timespec="microseconds"), scene_id))
        if old and old["file"] != file:
            _unlink(self.files / old["file"])
        for name in stale:
            _unlink(self.files / name)

    def set_verdicts(self, scene_id: int, verdicts: list[dict[str, Any]], source: str | None) -> None:
        """장면의 AI 판정 목록을 바꾼다 (clean_verdicts 를 거친 것)."""
        with self._lock:
            with self._db:
                self._db.execute(
                    "INSERT INTO scene_verdicts (scene_id, data, source, updated_at) VALUES (?,?,?,?)"
                    " ON CONFLICT (scene_id) DO UPDATE SET data=excluded.data, source=excluded.source,"
                    " updated_at=excluded.updated_at",
                    (scene_id, json.dumps(verdicts, ensure_ascii=False), source, _now()))

    def set_offsets(self, scene_id: int, offsets: dict[str, float]) -> None:
        with self._lock:
            with self._db:
                for cam, ms in offsets.items():
                    self._db.execute("UPDATE videos SET offset_ms=? WHERE scene_id=? AND camera_id=?",
                                     (float(ms), scene_id, cam))

    def gc(self, grace_sec: float = GC_GRACE_SEC) -> int:
        """DB 에 없는 영상·썸네일을 지운다 (임시 파일·최근 파일은 남김 — 가져오는 중일 수 있음). 지운 개수.
        가져오기와 겹치지 않게 maintenance_lock 안에서 부른다."""
        with self._lock:
            used = {r["file"] for r in self._db.execute("SELECT file FROM videos UNION SELECT file FROM ai_videos")}
            scenes = {r["id"] for r in self._db.execute("SELECT id FROM scenes")}
        cutoff = time.time() - grace_sec
        n = 0
        checks: list[tuple[Path, Callable[[Path], bool]]] = [
            (self.files, lambda q: q.name in used),
            (self.thumbs, lambda q: q.stem.isdigit() and int(q.stem) in scenes),
        ]
        for folder, keep in checks:
            for p in folder.iterdir():
                if keep(p) or ".part." in p.name or not p.is_file():
                    continue
                try:
                    if p.stat().st_mtime > cutoff:
                        continue
                except FileNotFoundError:
                    continue
                n += _unlink(p)
        return n


class Busy(Exception):
    """다른 가져오기·정리가 돌고 있다."""


@contextlib.contextmanager
def maintenance_lock(root: Path) -> Iterator[None]:
    """가져오기·gc 를 프로세스끼리 하나씩만 (Linux flock). fcntl 이 없는 Windows 개발 PC 에서는 잠그지 않는다."""
    try:
        import fcntl
    except ImportError:
        yield
        return
    root.mkdir(parents=True, exist_ok=True)
    with open(root / ".maintenance.lock", "a+") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Busy("다른 가져오기·정리가 돌고 있습니다") from None
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _unlink(p: Path) -> int:
    try:
        p.unlink()
        return 1
    except FileNotFoundError:
        return 0
    except OSError as e:
        print(f"[library] 파일을 지우지 못했습니다: {p} ({e})", file=sys.stderr)
        return 0


# --- 변환 (ffmpeg) -----------------------------------------------------------------

@dataclass
class VideoInfo:
    codec: str
    pix_fmt: str
    width: int
    height: int
    fps: float
    duration: float | None


def probe(path: Path, ffprobe: str = "ffprobe") -> VideoInfo:
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,pix_fmt,width,height,r_frame_rate:format=duration", "-of", "default=nw=1", str(path)],
        capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe 실패: {out.stderr.strip()[:200]}")
    kv = dict(line.split("=", 1) for line in out.stdout.splitlines() if "=" in line)
    num, _, den = kv.get("r_frame_rate", "0/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    dur = kv.get("duration")
    return VideoInfo(kv.get("codec_name", ""), kv.get("pix_fmt", ""), int(kv.get("width", 0) or 0),
                     int(kv.get("height", 0) or 0), fps,
                     float(dur) if dur not in (None, "", "N/A") else None)


def playable_as_is(info: VideoInfo) -> bool:
    """다시 인코딩하지 않고 그대로 담아도 되는가 (안드로이드가 확실히 트는 H.264 8비트, 폭 1280·30fps 이하)."""
    return info.codec == "h264" and info.pix_fmt == "yuv420p" and 0 < info.width <= 1280 and info.fps <= 30.5


def encoder_args(encoder: str) -> list[str]:
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "23", "-b:v", "0"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]


def convert_args(src: Path, dst: Path, info: VideoInfo, encoder: str, ffmpeg: str = "ffmpeg") -> list[str]:
    """그대로 담거나(-c copy) 720p H.264 Main 으로. 소리는 뺀다 (시연에 필요 없고 크기·사생활)."""
    head = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src)]
    tail = ["-an", "-movflags", "+faststart", "-f", "mp4", str(dst)]
    if playable_as_is(info):
        return head + ["-map", "0:v:0", "-c:v", "copy"] + tail
    vf = "scale='min(1280,iw)':-2"
    if info.fps > 30.5:
        vf += ",fps=30"
    vf += ",format=yuv420p"
    return head + ["-map", "0:v:0", "-vf", vf] + encoder_args(encoder) + ["-profile:v", "main"] + tail


def pick_encoder(choice: str, ffmpeg: str = "ffmpeg") -> str:
    """auto: h264_nvenc 로 1장을 만들어 보고 되면 그것, 아니면 libx264."""
    if choice != "auto":
        return choice
    try:
        r = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                            "color=c=black:s=256x144:d=0.1", "-frames:v", "1", "-c:v", "h264_nvenc", "-f", "null", "-"],
                           capture_output=True, timeout=30)
        return "h264_nvenc" if r.returncode == 0 else "libx264"
    except (OSError, subprocess.TimeoutExpired):
        return "libx264"


def convert(src: Path, dst: Path, encoder: str, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> VideoInfo:
    """src → dst (임시 이름으로 쓰고 끝나면 바꿈). 만든 파일의 정보를 돌려준다."""
    info = probe(src, ffprobe)
    tmp = dst.with_name(dst.stem + ".part.mp4")
    r = subprocess.run(convert_args(src, tmp, info, encoder, ffmpeg), capture_output=True, text=True,
                       stdin=subprocess.DEVNULL)
    if r.returncode != 0 or not tmp.is_file():
        _unlink(tmp)
        raise RuntimeError(f"ffmpeg 실패: {r.stderr.strip()[-300:]}")
    os.replace(tmp, dst)
    return probe(dst, ffprobe)


def make_thumb(src: Path, dst: Path, duration: float | None, ffmpeg: str = "ffmpeg") -> bool:
    at = min(1.0, (duration or 2.0) / 2)
    tmp = dst.with_name(dst.stem + ".part.jpg")
    r = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{at:.2f}", "-i", str(src),
                        "-frames:v", "1", "-vf", "scale=320:-2", "-q:v", "5", str(tmp)],
                       capture_output=True, stdin=subprocess.DEVNULL)
    if r.returncode != 0 or not tmp.is_file():
        _unlink(tmp)
        return False
    os.replace(tmp, dst)
    return True


def load_sync(path: Path) -> list[tuple[str, str, float]]:
    """시차 표(experiments/tracking/labels/<촬영>/sync.csv) → [(영상 경로, 기준 경로, offset_ms)].
    경로는 표에 적힌 그대로 ('2(angle0)/C01.mp4'). 공통 시각 = 그 카메라 재생 시각 + offset_ms (기준 카메라는 0)."""
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append((r["video"].strip(), r["reference"].strip(), float(r["offset_ms"])))
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
    return rows


def _tail(path: str | Path, parts: int) -> tuple[str, ...]:
    """경로의 끝 parts 조각 ('a/2(angle0)/C01.mp4', 2 → ('2(angle0)', 'C01.mp4')). 윈도 구분자도 받는다."""
    return tuple(str(path).replace("\\", "/").split("/")[-parts:])


def scene_offsets(files: dict[str, Path], sync: list[tuple[str, str, float]]) -> dict[str, float] | None:
    """장면의 카메라별 원본 파일과 시차 표를 맞춘다. 짝이 표에 없으면 None.
    3·4차처럼 두 카메라 파일 이름이 같으면(C01.mp4) 폴더 이름(2(angle0))까지 본다. 파일 이름이 서로 다르면 이름만으로도 맞춘다."""
    for parts in (2, 1):
        cam_of: dict[tuple[str, ...], str] = {}
        for cam, p in files.items():
            k = _tail(p.as_posix(), parts)
            cam_of[k] = "" if k in cam_of else cam        # 이 깊이에서 겹치면 쓰지 않음
        for video, reference, ms in sync:
            v, r = cam_of.get(_tail(video, parts)), cam_of.get(_tail(reference, parts))
            if v and r and v != r:
                return {v: ms, r: 0.0}
    return None


def video_files(folder: Path) -> list[Path]:
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXT and ".part." not in p.name]


def run_import(lib: Library, folder: str, cameras: dict[str, Path], *, encoder: str = "auto",
               sync: list[tuple[str, str, float]] | None = None, replace: bool = False, dry_run: bool = False,
               ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe",
               log: Callable[[str], None] = print) -> dict[str, int]:
    """가져오기. maintenance_lock 안에서 부른다 (gc 와 겹치지 않게)."""
    plan = plan_import({cam: video_files(d) for cam, d in cameras.items()})
    counts = {"scenes": len(plan), "added": 0, "skipped": 0, "failed": 0, "no_sync": 0}
    if dry_run:
        for name, files in plan:
            off = scene_offsets(files, sync) if sync is not None else None
            log(f"  {name}: " + ", ".join(f"{c}={p.name}" for c, p in sorted(files.items()))
                + (f"  시차 {off}" if off else ""))
        return counts
    folder_id = lib.ensure_folder(folder)["id"]
    enc = pick_encoder(encoder, ffmpeg)
    log(f"[library] '{folder}' 장면 {len(plan)}개, 인코더 {enc}")
    for name, files in plan:
        sid, created = lib.add_scene(folder_id, name, folder, f"{folder}/{name}")
        replaced = False
        for cam, src in sorted(files.items()):
            if lib.has_video(sid, cam) and not replace:
                counts["skipped"] += 1
                continue
            file = f"{uuid.uuid4().hex}.mp4"
            try:
                info = convert(src, lib.files / file, enc, ffmpeg, ffprobe)
                lib.put_video(sid, cam, file, info)
                counts["added"] += 1
                replaced = True
                log(f"  {name} {cam} ← {src.name} ({info.width}x{info.height}, {info.duration or 0:.1f}s)")
            except Exception as e:  # 한 파일이 실패해도 나머지는 계속
                _unlink(lib.files / file)
                counts["failed"] += 1
                log(f"  {name} {cam} 실패: {src.name} — {e}")
        if created and not replaced:        # 이번에 만든 장면에 영상이 하나도 안 들어갔으면 빈 장면을 남기지 않는다
            lib.delete_scenes([sid])
            log(f"  {name}: 영상을 하나도 넣지 못해 장면을 만들지 않았습니다")
            continue
        if len(files) > 1 and sync is not None:     # 이미 넣은 장면도 시차는 다시 읽어 갱신
            off = scene_offsets(files, sync)
            if off is None:
                counts["no_sync"] += 1
                log(f"  {name}: 시차 표에 없음 — 카메라를 바꿀 때 같은 재생 위치로 이어 봄")
            else:
                lib.set_offsets(sid, off)
        thumb = lib.thumb_file(sid)
        if replaced or not thumb.exists():
            first = lib.scene(sid)["videos"]
            pick = next((v for v in first if v["camera_id"] == "cam1"), first[0] if first else None)
            path = lib.video_file(pick["id"]) if pick else None
            if path is not None:
                make_thumb(path, thumb, pick["duration"], ffmpeg)
    log(f"[library] 끝: 장면 {counts['scenes']}, 넣음 {counts['added']}, 건너뜀 {counts['skipped']}, "
        f"실패 {counts['failed']}" + (f", 시차 없음 {counts['no_sync']}" if sync is not None else ""))
    return counts


def run_import_ai(lib: Library, folder: str, cameras: dict[str, Path], *, encoder: str = "auto",
                  replace: bool = False, dry_run: bool = False, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe",
                  log: Callable[[str], None] = print) -> dict[str, int]:
    """AI 화면 영상을 이미 넣은 장면·카메라 영상에 붙인다 (장면을 새로 만들지 않음). 장면은 원본 가져오기와 같은
    source_key ('<폴더>/<장면>') 로 찾는다. maintenance_lock 안에서 부른다."""
    plan = plan_import({cam: video_files(d) for cam, d in cameras.items()})
    counts = {"scenes": len(plan), "added": 0, "skipped": 0, "failed": 0, "no_source": 0}
    if dry_run:
        for name, files in plan:
            log(f"  {name}: " + ", ".join(f"{c}={p.name}" for c, p in sorted(files.items())))
        return counts
    enc = pick_encoder(encoder, ffmpeg)
    log(f"[library] '{folder}' AI 화면 {len(plan)}장면, 인코더 {enc}")
    for name, files in plan:
        sid = lib.scene_id_by_source(f"{folder}/{name}")
        for cam, src in sorted(files.items()):
            vid = lib.video_id(sid, cam) if sid is not None else None
            if vid is None:
                counts["no_source"] += 1
                log(f"  {name} {cam}: 원본 영상이 보관함에 없어 건너뜀")
                continue
            if lib.has_ai(vid) and not replace:
                counts["skipped"] += 1
                continue
            file = f"{uuid.uuid4().hex}.mp4"
            try:
                info = convert(src, lib.files / file, enc, ffmpeg, ffprobe)
                lib.put_ai(vid, file, info)
                counts["added"] += 1
                log(f"  {name} {cam} AI ← {src.name}")
            except Exception as e:
                _unlink(lib.files / file)
                counts["failed"] += 1
                log(f"  {name} {cam} AI 실패: {src.name} — {getattr(e, 'message', e)}")
    log(f"[library] 끝: AI 넣음 {counts['added']}, 건너뜀 {counts['skipped']}, 원본 없음 {counts['no_source']}, "
        f"실패 {counts['failed']}")
    return counts


def _whole(value: Any, low: int, high: int) -> int:
    """불리언·소수가 아닌 정수이고 범위 안이면 그 값, 아니면 ValueError (변환하지 않는다 — 1.9 를 1 로 만들지 않게)."""
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(value)
    return value


def clean_verdicts(items: Any) -> tuple[list[dict[str, Any]], int]:
    """판정 목록 검증 → (쓸 것 시각 순, 버린 수). level 은 넷 중 하나, ms 는 0 이상 유한한 수, person 은 1 ~ 앱 Int 범위
    정수, 개수는 0 ~ 백만 정수(없으면 0), 이유 500자까지. 어긋난 항목은 버리고 센다 (가져오기 전체를 멈추지 않음)."""
    if not isinstance(items, list):
        return [], 1
    out, dropped = [], 0
    for v in items[:MAX_VERDICTS]:
        try:
            raw_ms = v["ms"]
            if isinstance(raw_ms, bool) or not isinstance(raw_ms, (int, float)):
                raise ValueError(raw_ms)
            ms, level = float(raw_ms), v["level"]
            if not isinstance(level, str) or level not in VERDICT_LEVELS or not math.isfinite(ms) or ms < 0:
                raise ValueError(level)
            out.append({"ms": round(ms, 1), "person": _whole(v["person"], 1, MAX_PERSON), "level": level,
                        "zone": str(v.get("zone") or "")[:40], "reason": str(v.get("reason") or "")[:MAX_REASON],
                        "taken": _whole(v.get("taken", 0), 0, MAX_COUNT), "paid": _whole(v.get("paid", 0), 0, MAX_COUNT),
                        "unpaid": _whole(v.get("unpaid", 0), 0, MAX_COUNT)})
        except (KeyError, TypeError, ValueError, OverflowError, AttributeError):
            dropped += 1
    dropped += max(0, len(items) - MAX_VERDICTS)
    return sorted(out, key=lambda x: (x["ms"], x["person"])), dropped


def run_import_verdicts(lib: Library, folder: str, path: Path, *, dry_run: bool = False,
                        log: Callable[[str], None] = print) -> dict[str, int]:
    """tools/export_verdicts.py 가 만든 JSON 의 판정을 이미 넣은 장면에 붙인다 (장면 이름 = source_key '<폴더>/<장면>')."""
    data = json.loads(path.read_text(encoding="utf-8"))
    scenes = data.get("scenes") if isinstance(data, dict) else None
    if not isinstance(scenes, dict):
        raise LibraryError(422, "invalid_verdicts", "판정 파일 형식이 아닙니다 ({'scenes': {...}})")
    source = str(data.get("source") or "")[:200] or None
    counts = {"scenes": len(scenes), "set": 0, "no_scene": 0, "dropped": 0}
    for name, items in sorted(scenes.items(), key=lambda kv: natural_key(kv[0])):
        sid = lib.scene_id_by_source(f"{folder}/{scene_key(str(name))}")
        if sid is None:
            counts["no_scene"] += 1
            log(f"  {name}: 보관함에 장면이 없어 건너뜀")
            continue
        good, dropped = clean_verdicts(items)
        counts["dropped"] += dropped
        if dry_run:
            log(f"  {name}: 판정 {len(good)}개 (버림 {dropped}) — 미리 보기라 넣지 않음")
        else:
            lib.set_verdicts(sid, good, source)
        counts["set"] += 1
    log(f"[library] '{folder}' 판정{' (미리 보기)' if dry_run else ''}: 넣음 {counts['set']}장면, "
        f"장면 없음 {counts['no_scene']}, 버린 항목 {counts['dropped']}")
    return counts


def run_prepare(src: Path, out: Path, *, encoder: str = "auto", ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe",
                log: Callable[[str], None] = print) -> dict[str, int]:
    """src 아래 영상을 같은 폴더 구조로 out 에 변환해 둔다 (이미 있으면 건너뜀)."""
    enc = pick_encoder(encoder, ffmpeg)
    counts = {"done": 0, "skipped": 0, "failed": 0}
    for p in sorted(src.rglob("*"), key=lambda x: natural_key(str(x))):
        if not (p.is_file() and p.suffix.lower() in VIDEO_EXT) or ".part." in p.name:
            continue
        dst = out / p.relative_to(src).with_suffix(".mp4")
        if dst.is_file():
            counts["skipped"] += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            convert(p, dst, enc, ffmpeg, ffprobe)
            counts["done"] += 1
            log(f"  {p.relative_to(src)}")
        except Exception as e:
            counts["failed"] += 1
            log(f"  실패: {p.relative_to(src)} — {e}")
    return counts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m hawkeye_server.library", description="HawkEye 보관함 가져오기")
    p.add_argument("--library", default=os.environ.get("HAWKEYE_LIBRARY", "data/library"),
                   help="보관함 폴더 (기본: HAWKEYE_LIBRARY 또는 data/library)")
    sub = p.add_subparsers(dest="cmd", required=True)
    imp = sub.add_parser("import", help="영상을 보관함 폴더에 넣는다")
    imp.add_argument("--folder", required=True, help='보관함 폴더 이름 (예: "1차 촬영")')
    imp.add_argument("--camera", action="append", default=[], metavar="CAM=DIR",
                     help="카메라와 영상 폴더 (예: cam1=source/1(angle1)). 여러 번")
    imp.add_argument("--sync", help="카메라 시차 표 (experiments/tracking/labels/<촬영>/sync.csv)")
    imp.add_argument("--encoder", default="auto", choices=["auto", "libx264", "h264_nvenc"])
    imp.add_argument("--replace", action="store_true", help="이미 있는 카메라 영상도 다시 넣는다")
    imp.add_argument("--verdicts", help="AI 판정 JSON (tools/export_verdicts.py) 을 이미 넣은 장면에 붙인다 — --camera 없이")
    imp.add_argument("--ai", action="store_true",
                     help="AI 화면 영상(tools/ai_render.py 로 그린 것)을 이미 넣은 장면에 붙인다 (장면을 새로 만들지 않음)")
    imp.add_argument("--dry-run", action="store_true", help="장면 짝 맞추기만 보여 준다")
    pre = sub.add_parser("prepare", help="DB 없이 변환만 (같은 폴더 구조)")
    pre.add_argument("src")
    pre.add_argument("out")
    pre.add_argument("--encoder", default="auto", choices=["auto", "libx264", "h264_nvenc"])
    sub.add_parser("gc", help="DB 에 없는 파일 지우기")
    a = p.parse_args(argv)

    if a.cmd == "prepare":
        c = run_prepare(Path(a.src), Path(a.out), encoder=a.encoder)
        print(f"[library] 변환 {c['done']}, 건너뜀 {c['skipped']}, 실패 {c['failed']}")
        return 1 if c["failed"] else 0

    cameras: dict[str, Path] = {}
    sync = None
    if a.cmd == "import":
        if not a.camera and not a.verdicts:
            p.error("--camera 또는 --verdicts 가 필요합니다")
        if a.verdicts and not Path(a.verdicts).is_file():
            p.error(f"판정 파일이 없습니다: {a.verdicts}")
        for spec in a.camera:
            cam, sep, folder = spec.partition("=")
            if not sep or not CAMERA.match(cam) or not Path(folder).is_dir():
                p.error(f"--camera 는 cam1=<있는 폴더> 형식입니다: {spec}")
            cameras[cam] = Path(folder)
        if a.sync:
            if not Path(a.sync).is_file():
                p.error(f"시차 표가 없습니다: {a.sync}")
            sync = load_sync(Path(a.sync))
    root = Path(a.library)
    try:
        with maintenance_lock(root):
            lib = Library(root)
            try:
                if a.cmd == "gc":
                    print(f"[library] 지운 파일 {lib.gc()}개")
                    return 0
                if a.verdicts and not a.camera:
                    run_import_verdicts(lib, a.folder, Path(a.verdicts), dry_run=a.dry_run)
                    return 0
                if a.ai:
                    c = run_import_ai(lib, a.folder, cameras, encoder=a.encoder, replace=a.replace, dry_run=a.dry_run)
                else:
                    c = run_import(lib, a.folder, cameras, encoder=a.encoder, sync=sync, replace=a.replace,
                                   dry_run=a.dry_run)
                if a.verdicts and not a.dry_run:
                    run_import_verdicts(lib, a.folder, Path(a.verdicts))
                return 1 if c["failed"] else 0
            finally:
                lib.close()
    except Busy as e:
        print(f"[library] {e}", file=sys.stderr)
        return 3
    except LibraryError as e:
        print(f"[library] {e.message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
