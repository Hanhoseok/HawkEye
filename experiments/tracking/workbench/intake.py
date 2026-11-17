"""새 영상 추가 — 두 각도 영상을 받아 등록하고, 뒤에서 처리(탐지·추적)해 갤러리에 올린다.

**같은 매장·같은 카메라 위치에서 찍은 영상만.** 구역(zones/store01_dual)·바닥 보정·띠 축을 그대로 쓰기 때문이다.
카메라가 움직였거나 다른 매장이면 구역부터 다시 잡아야 한다 — 이 도구는 그것을 해 주지 않는다.

흐름
  1. 올리기   — 화면에서 고른 두 파일을 source/added/<짝>_angle1.* · <짝>_angle0.* 로 받는다 (저장소에 올리지 않는 폴더).
  2. 등록     — 영상을 열어 보고(해상도·길이), 정답표(truth.csv)에 한 줄, 추가 기록(added.csv)에 한 줄.
  3. 처리     — scripts/dual_eval.py run 을 따로 띄운다 (CPU 로 영상 길이의 8~10배). 한 번에 하나씩.
  4. 시차     — 두 화면의 손님 위치로 자동 추정(믿기 어려우면 기존 영상들의 중앙값). **확인 필요** — 분석 화면의 정답 적기에서 고친다.
  5. 갤러리   — 보기 데이터를 만들어 카드로 올린다.

짝 이름은 세트 글자(N 정상 · T 절도 · C 복합) + 번호 — 기존 도구들이 첫 글자로 세트를 읽기 때문에 이 규칙을 지킨다.
"""

from __future__ import annotations

import csv
import io
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from workbench.data import OUT, ROOT, Builder, natural

LABELS = ROOT / "labels" / "store01_dual"
TRUTH = LABELS / "truth.csv"
ADDED = LABELS / "added.csv"
ADDED_FIELDS = ("pair", "added", "angle1_file", "angle0_file", "note")
NAME = re.compile(r"^[NTC]\d{1,3}(-[0-9A-Za-z]{1,12})?$")
SETS = {"N": "normal", "T": "theft", "C": "complex"}
SET_NAME = {"normal": "정상", "theft": "절도", "complex": "복합"}
CAMS = ("angle1", "angle0")
MAX_BYTES = 2 * 1024 ** 3
BATCHES = (("shoot2", "2차 촬영"), ("added", "추가한 영상"))


def _sync_pair():
    sys.path.insert(0, str(ROOT / "scripts"))
    import sync_pair
    return sync_pair


def known_pairs() -> set[str]:
    """이미 쓰는 이름: 2차 촬영, 등록한 추가 영상, 정답표에 있는 것.
    파일만 올라와 있고 아직 등록하지 않은 이름은 넣지 않는다 — 올린 뒤 등록할 때 제 파일 때문에 막히지 않게, 다시 올릴 수도 있게."""
    sp = _sync_pair()
    names = (set(sp.all_pairs()) - set(sp.added_pairs())) | set(added_rows())
    if TRUTH.exists():
        with open(TRUTH, encoding="utf-8", newline="") as f:
            names |= {r["pair"] for r in csv.DictReader(f)}
    return names


def check_name(name: str, taken: set[str] | None = None) -> str:
    if not NAME.match(name or ""):
        raise ValueError("이름은 N·T·C 중 한 글자와 번호입니다 (예: N13, T11, C22-1)")
    if name in (known_pairs() if taken is None else taken):
        raise ValueError(f"'{name}' 은 이미 있는 이름입니다")
    return name


def suggest_names(taken: set[str] | None = None) -> dict[str, str]:
    """세트마다 다음 번호 (정상 → N13 …)."""
    taken = known_pairs() if taken is None else taken
    out = {}
    for letter, key in SETS.items():
        nums = [int(m.group(1)) for n in taken if (m := re.match(rf"^{letter}(\d+)", n))]
        out[key] = f"{letter}{max(nums, default=0) + 1}"
    return out


def added_rows() -> dict[str, dict]:
    if not ADDED.exists():
        return {}
    with open(ADDED, encoding="utf-8", newline="") as f:
        return {r["pair"]: r for r in csv.DictReader(f)}


def batch_of(pair: str, added: dict | None = None) -> str:
    return "added" if pair in (added_rows() if added is None else added) else "shoot2"


def _append_csv(path: Path, fields: tuple, row: dict) -> None:
    """표 끝에 한 줄만 덧붙인다 — 있던 줄은 건드리지 않는다 (줄바꿈도 그 파일 것을 따른다)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.read_bytes() if path.exists() else b""
    nl = "\r\n" if b"\r\n" in raw else "\n"
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator=nl)
    if not raw:
        writer.writeheader()
    writer.writerow(row)
    with open(path, "ab") as f:
        if raw and not raw.endswith((b"\n", b"\r")):
            f.write(nl.encode())
        f.write(buf.getvalue().encode("utf-8"))


# ---------------------------------------------------------------------- 올리기

def upload_path(name: str, cam: str, ext: str) -> Path:
    sp = _sync_pair()
    if cam not in CAMS:
        raise ValueError("카메라는 angle1(전체) 또는 angle0(선반)")
    ext = (ext or "").lower()
    if ext not in sp.VIDEO_EXT:
        raise ValueError(f"영상 파일만 받습니다 ({', '.join(sp.VIDEO_EXT)})")
    return sp.ADDED_DIR / f"{name}_{cam}{ext}"


def save_upload(name: str, cam: str, ext: str, stream, length: int) -> dict:
    """요청 본문을 그대로 파일로 받는다. 다 받은 뒤에야 제 이름으로 바꾼다 (받다 끊긴 파일이 영상으로 보이지 않게)."""
    check_name(name)
    if not 0 < length <= MAX_BYTES:
        raise ValueError("파일 크기가 0 이거나 너무 큽니다 (2GB 까지)")
    path = upload_path(name, cam, ext)
    path.parent.mkdir(parents=True, exist_ok=True)
    for old in path.parent.glob(f"{name}_{cam}.*"):   # 같은 자리에 다시 올리면 바꿔 쓴다
        old.unlink()
    part = path.with_name(path.name + ".part")
    left = length
    with open(part, "wb") as f:
        while left > 0:
            chunk = stream.read(min(1024 * 1024, left))
            if not chunk:
                break
            f.write(chunk)
            left -= len(chunk)
    if left:
        part.unlink(missing_ok=True)
        raise ValueError("올리다 끊겼습니다. 다시 올려 주세요")
    os.replace(part, path)
    return {"saved": path.name, "bytes": length}


def probe(path: Path) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"영상을 열 수 없습니다: {path.name}")
        info = {"fps": cap.get(cv2.CAP_PROP_FPS) or 0.0, "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    finally:
        cap.release()
    if info["fps"] <= 0 or info["frames"] < 30:
        raise ValueError(f"영상이 너무 짧거나 읽을 수 없습니다: {path.name}")
    info["seconds"] = round(info["frames"] / info["fps"], 1)
    return info


def check_videos(a1: dict, a0: dict) -> list[str]:
    """등록해도 되는지 — 안 되면 ValueError, 되지만 알아 둘 것은 목록으로."""
    notes = []
    for name, info in (("전체 카메라", a1), ("선반 카메라", a0)):
        if abs(info["width"] / info["height"] - 16 / 9) > 0.03:
            raise ValueError(f"{name} 영상의 화면 비율이 16:9 가 아닙니다 ({info['width']}x{info['height']}) — "
                             "구역이 지금 카메라 화면에 맞춰져 있어 다른 화면은 받을 수 없습니다")
    if abs(a1["seconds"] - a0["seconds"]) > 5:
        raise ValueError(f"두 영상의 길이가 많이 다릅니다 ({a1['seconds']}초 · {a0['seconds']}초) — 같은 장면을 찍은 두 각도인지 확인해 주세요")
    if max(a1["seconds"], a0["seconds"]) > 300:
        raise ValueError("영상이 5분을 넘습니다 — 처리에 한 시간 가까이 걸립니다. 잘라서 올려 주세요")
    for name, info, want in (("전체 카메라", a1, 60), ("선반 카메라", a0, 30)):
        if abs(info["fps"] - want) > want * 0.15:
            notes.append(f"{name}가 초당 {info['fps']:.0f}장입니다 (기존 영상은 {want}장). 처리 간격이 달라져 결과가 조금 다를 수 있습니다")
    return notes


def register(body: dict) -> dict:
    """올린 두 파일을 영상으로 등록한다 → 정답표와 추가 기록에 한 줄씩."""
    sp = _sync_pair()
    name = check_name(str(body.get("name") or ""))
    files = sp.added_files(name)
    if files is None:
        raise ValueError("두 영상을 먼저 올려 주세요")
    a0_path, a1_path = files
    notes = check_videos(probe(a1_path), probe(a0_path))
    set_key = SETS[name[0]]
    people = int(body.get("people") or 1)
    if not 1 <= people <= 9:
        raise ValueError("손님 수는 1~9")
    expected = "HIGH_RISK" if (body["alarm"] if "alarm" in body else set_key == "theft") else "CLEAR"
    text = " ".join(str(body.get("notes") or "").split())[:200]
    _append_csv(TRUTH, ("pair", "set", "people", "expected", "notes"),
                {"pair": name, "set": set_key, "people": people, "expected": expected, "notes": text})
    _append_csv(ADDED, ADDED_FIELDS, {"pair": name, "added": time.strftime("%Y-%m-%d"),
                                      "angle1_file": " ".join(str(body.get("angle1_file") or "").split())[:120],
                                      "angle0_file": " ".join(str(body.get("angle0_file") or "").split())[:120], "note": text})
    return {"pair": name, "notes": notes}


# ---------------------------------------------------------------------- 시차

def estimate_offset(builder: Builder, pair: str) -> dict:
    """처리가 끝난 짝의 시차를 자동으로 추정해 sync.csv 에 적는다 (사람이 맞춘 값이 있으면 건드리지 않는다)."""
    c = builder.ctx()
    SyncTable = c["SyncTable"]
    path = builder.profile.labels / "sync.csv"
    with builder.engine() as de:
        before = SyncTable.load(path)
        if before.rows.get(pair) is not None and before.rows[pair].method == "manual":
            return {"offset_ms": before.rows[pair].offset_ms, "method": "manual"}
        de.sync(pair)
        table = SyncTable.load(path)
        row = table.rows.get(pair)
        if row is None:
            return {"offset_ms": 0.0, "method": "none"}
        video, ref = c["sync_pair"].pair_files(pair, builder.shoot)   # dual_eval.sync 는 2차 촬영 폴더 이름을 적는다 — 실제 자리로 고친다
        want = (f"{video.parent.name}/{video.name}", f"{ref.parent.name}/{ref.name}")
        if (row.video, row.reference) != want:
            row.video, row.reference = want
            table.upsert(row)
            table.save(path)
        if row.method == "trajectory-fallback":
            # 한 짝만 추정하면 dual_eval.sync 의 '믿을 만한 짝들의 중앙값'이 0 이 된다 — 기존 영상들의 값으로 바꾼다
            trusted = sorted(r.offset_ms for p, r in table.rows.items() if p != pair and r.method in ("manual", "trajectory"))
            if trusted:
                row.offset_ms = trusted[len(trusted) // 2]
                row.note = f"자동 추정을 믿기 어려워 기존 영상들의 중앙값 {row.offset_ms:+.0f}ms 사용 — 확인 필요"
                table.upsert(row)
                table.save(path)
        return {"offset_ms": row.offset_ms, "method": row.method}


def set_offset(builder: Builder, pair: str, offset_ms: float, who: str = "") -> None:
    """사람이 두 화면을 보고 맞춘 시차를 적는다 (method=manual). 예전 값은 메모에 남긴다.
    그 촬영의 시차표(<라벨 폴더>/sync.csv)에 적는다 — 이미 사람이 맞춘 값을 바꾸면 그 값도 메모에 남긴다."""
    c = builder.ctx()
    SyncTable = c["SyncTable"]
    from src.inputs.sync import SyncRow
    if not -30000 <= offset_ms <= 30000:
        raise ValueError("시차가 범위를 벗어났습니다 (±30초)")
    path = builder.profile.labels / "sync.csv"
    with builder.lock:
        table = SyncTable.load(path)
        old = table.rows.get(pair)
        video, ref = c["sync_pair"].pair_files(pair, builder.shoot)
        note = "눈으로 맞춤 (분석 도구)" + (f", 처음 {old.method} {old.offset_ms:+.0f}ms" if old and old.method != "manual" else "")
        if old and old.method == "manual" and "분석 도구" in old.note:
            note = old.note   # 이 도구로 이미 한 번 고친 값 — 처음 값은 메모에 남아 있다
        elif old and old.method == "manual":
            # 다른 사람이(다른 도구로) 맞춘 값을 덮어쓰는 일이 조용히 지나가지 않게 그 값을 남긴다
            was = f"분석 도구에서 다시 맞춤 (전 값 {old.offset_ms:+.0f}ms" + (f", {old.checked_by}" if old.checked_by else "") + ")"
            note = "; ".join(x for x in (old.note, was) if x)[:300]
        table.upsert(SyncRow(pair, old.video if old else f"{video.parent.name}/{video.name}",
                             old.reference if old else f"{ref.parent.name}/{ref.name}",
                             round(float(offset_ms), 1), "manual", " ".join(who.split())[:40], note))
        table.save(path)


# ---------------------------------------------------------------------- 처리 대기열

class Runner:
    """추가한 영상을 하나씩 처리한다. 무거운 일(탐지·추적)은 따로 띄운 프로그램이 하고, 여기서는 기다리며 상태를 알린다."""

    def __init__(self, builder: Builder) -> None:
        self.builder = builder
        self.jobs: dict[str, dict] = {}
        self.queue: queue.Queue[str] = queue.Queue()
        self.notes: dict[str, list[str]] = {}
        threading.Thread(target=self._loop, daemon=True).start()

    def done(self, pair: str) -> bool:
        return (self.builder.records_dir() / pair / ".done").exists()

    def enqueue(self, pair: str) -> None:
        if pair not in added_rows():
            raise ValueError(f"추가한 영상이 아닙니다: {pair}")
        job = self.jobs.get(pair)
        if job and job["state"] in ("queued", "running", "finishing"):
            return
        self.jobs[pair] = {"state": "queued", "started": None, "error": "", "log": ""}
        self.queue.put(pair)

    def pending(self) -> list[dict]:
        """갤러리에 아직 못 올라간 추가 영상 — 처리 대기·처리 중·실패·아직 안 돌림."""
        out, truth = [], {}
        if TRUTH.exists():
            with open(TRUTH, encoding="utf-8", newline="") as f:
                truth = {r["pair"]: r for r in csv.DictReader(f)}
        cards = {c["pair"] for c in self.builder.read_index().get("cards", [])}
        for pair in sorted(added_rows(), key=natural):
            job = self.jobs.get(pair)
            if pair in cards and not (job and job["state"] != "done"):
                continue
            state = job["state"] if job else ("finishing" if self.done(pair) else "idle")
            seconds = round(time.time() - job["started"]) if job and job["started"] and state in ("running", "finishing") else 0
            out.append({"pair": pair, "state": state, "seconds": seconds, "error": job["error"] if job else "",
                        "log": job["log"] if job else "", "set_name": SET_NAME.get(truth.get(pair, {}).get("set"), ""),
                        "notes": self.notes.get(pair, [])})
        return out

    def _loop(self) -> None:
        while True:
            pair = self.queue.get()
            job = self.jobs[pair]
            try:
                self._process(pair, job)
                job["state"] = "done"
            except Exception as e:  # noqa: BLE001 — 화면에 그대로 보인다
                job["state"], job["error"] = "failed", str(e)

    def _process(self, pair: str, job: dict) -> None:
        job.update(state="running", started=time.time())
        log = OUT / "jobs" / f"{pair}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        if not self.done(pair):
            env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
            with open(log, "w", encoding="utf-8") as f:
                code = subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / "dual_eval.py"), "run", "--only", pair],
                                      cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, env=env).returncode
            job["log"] = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:])
            if code != 0 or not self.done(pair):
                raise RuntimeError(f"영상을 처리하지 못했습니다 (자세한 내용: outputs/workbench/jobs/{pair}.log)")
        job["state"] = "finishing"
        estimate_offset(self.builder, pair)
        state = self.builder.build()
        if any(e.startswith(f"{pair}:") for e in state["errors"]):
            raise RuntimeError("; ".join(e for e in state["errors"] if e.startswith(f"{pair}:")))
