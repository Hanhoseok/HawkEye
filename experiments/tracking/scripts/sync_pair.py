"""두 각도 녹화본의 시차 맞추기 — 사람이 두 화면을 보며 확정하고 표에 남긴다 (design 1-0, M-9).

기준은 전체 카메라(angle1). 선반 카메라(angle0) 영상을 offset_ms 만큼 밀어 같은 시간축에 놓는다
(정의는 src/inputs/sync.py). 자동 추정(사람이 보이나 신호의 상관)은 약해서 시작값으로만 쓴다 (limitations B-10).

    # 한 짝: 자동 추정값으로 시작 → 화면을 보며 맞춤 → s 로 저장
    python scripts/sync_pair.py N1

    # 자동 추정만 표에 채우기 (method=suggest, 확인 전 표시)
    python scripts/sync_pair.py --suggest-all

화면 조작 (왼쪽 angle0, 오른쪽 angle1 — 같은 공통 시각의 장면):
    스페이스 재생/정지   ← → 0.1초 이동   [ ] 1초 이동
    a d   angle0 을 한 장면(1/30초)씩 밀기   A D  10 장면씩
    s 저장   q 끝
같은 순간(박수, 문 닫힘, 물건이 손을 떠나는 순간)이 양쪽에 함께 보이게 맞추고 저장한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inputs.sync import SyncRow, SyncTable, best_lag  # noqa: E402

SOURCE = ROOT.parent.parent / "source"
VIDEO_DIR = SOURCE / "1(angle0)"
REF_DIR = SOURCE / "1(angle1)"
TABLE = ROOT / "labels" / "store01_dual" / "sync.csv"
ADDED_DIR = SOURCE / "added"
"""분석 도구에서 추가한 영상: <짝>_angle1.<확장자>(전체 카메라) · <짝>_angle0.<확장자>(선반 카메라). 같은 매장·같은 카메라 위치만."""
VIDEO_EXT = (".mp4", ".mov", ".m4v", ".mkv", ".avi")
HZ = 10.0


def pair_files(pair: str) -> tuple[Path, Path]:
    """짝 이름(N1, T3, C10, N8-1) → (angle0 파일, angle1 파일). 추가한 영상이면 그 파일, 아니면 2차 촬영의 이름 규칙."""
    added = added_files(pair)
    if added is not None:
        return added
    kind, rest = pair[0], pair[1:]
    num, _, part = rest.partition("-")
    video = VIDEO_DIR / f"20261028_{kind}{int(num)}{'-' + part if part else ''}.mp4"
    ref = REF_DIR / f"{kind}{int(num):02d}{'-' + part if part else ''}.mp4"
    return video, ref


def added_files(pair: str) -> tuple[Path, Path] | None:
    """추가한 영상의 (angle0 파일, angle1 파일). 두 파일이 다 있을 때만."""
    found = []
    for cam in ("angle0", "angle1"):
        hits = [p for p in sorted(ADDED_DIR.glob(f"{pair}_{cam}.*")) if p.suffix.lower() in VIDEO_EXT] if ADDED_DIR.exists() else []
        if not hits:
            return None
        found.append(hits[0])
    return found[0], found[1]


def added_pairs() -> list[str]:
    if not ADDED_DIR.exists():
        return []
    names = {p.name.rsplit("_angle0.", 1)[0] for p in ADDED_DIR.glob("*_angle0.*") if p.suffix.lower() in VIDEO_EXT}
    return sorted(n for n in names if added_files(n) is not None)


def all_pairs() -> list[str]:
    names = []
    for p in sorted(VIDEO_DIR.glob("20261028_*.mp4")):
        name = p.stem.split("_", 1)[1]
        if pair_files(name)[1].exists():
            names.append(name)
    return names + [n for n in added_pairs() if n not in names]


class Clip:
    """영상을 한 번 다 읽어 줄인 장면을 메모리에 둔다 (H.265 는 위치 이동이 불안정해서)."""

    def __init__(self, path: Path, width: int = 640) -> None:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise SystemExit(f"영상을 열 수 없습니다: {path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frames = []
        while True:
            ok, image = cap.read()
            if not ok:
                break
            self.frames.append(cv2.resize(image, (width, int(width * image.shape[0] / image.shape[1]))))
        cap.release()
        self.duration_ms = len(self.frames) * 1000.0 / self.fps

    def at(self, ms: float):
        i = int(round(ms * self.fps / 1000.0))
        if 0 <= i < len(self.frames):
            return self.frames[i]
        return np.zeros_like(self.frames[0])


def presence_signal(path: Path, model) -> np.ndarray:
    """초당 10번, 사람이 보이면 1."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step, nxt, i, out = fps / HZ, 0.0, 0, []
    while True:
        ok, image = cap.read()
        if not ok:
            break
        if i >= nxt:
            nxt += step
            r = model.predict(image, imgsz=640, conf=0.25, classes=[0], verbose=False)[0]
            out.append(1.0 if len(r.boxes) else 0.0)
        i += 1
    cap.release()
    return np.asarray(out)


def suggest(pair: str, model) -> tuple[float, float]:
    video, ref = pair_files(pair)
    return best_lag(presence_signal(video, model), presence_signal(ref, model), HZ)


def interactive(pair: str, start_offset: float, who: str) -> SyncRow | None:
    video, ref = pair_files(pair)
    a, b = Clip(video), Clip(ref)
    offset, t, playing = start_offset, 0.0, False
    frame_ms = 1000.0 / a.fps
    print(f"{pair}: angle0 {a.fps:.0f}FPS {a.duration_ms / 1000:.1f}s / angle1 {b.fps:.0f}FPS {b.duration_ms / 1000:.1f}s, 시작 시차 {offset:+.0f}ms")
    while True:
        left, right = a.at(t - offset), b.at(t)   # 공통 시각 t: angle1 은 그대로, angle0 은 t - offset
        canvas = cv2.hconcat([left, cv2.resize(right, (left.shape[1], left.shape[0]))])
        cv2.putText(canvas, f"{pair}  t={t / 1000:.2f}s  offset={offset:+.0f}ms  {'PLAY' if playing else ''}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("sync", canvas)
        key = cv2.waitKeyEx(int(frame_ms) if playing else 30)
        if playing:
            t = min(t + frame_ms, b.duration_ms)
        # 화살표는 운영체제마다 코드가 다르다 (Windows 2424832/2555904, Linux 65361/65363)
        moves = {2424832: -100, 65361: -100, 2555904: 100, 65363: 100, ord("["): -1000, ord("]"): 1000}
        shifts = {ord("a"): -frame_ms, ord("d"): frame_ms, ord("A"): -10 * frame_ms, ord("D"): 10 * frame_ms}
        if key == ord(" "):
            playing = not playing
        elif key in moves:
            t = float(np.clip(t + moves[key], 0, b.duration_ms))
        elif key in shifts:
            offset += shifts[key]
        elif key == ord("s"):
            cv2.destroyWindow("sync")
            return SyncRow(pair, f"1(angle0)/{video.name}", f"1(angle1)/{ref.name}", round(offset, 1), "manual", who)
        elif key == ord("q"):
            cv2.destroyWindow("sync")
            return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pair", nargs="?", help="짝 이름 (예: N1, T3, C10, N8-1)")
    ap.add_argument("--suggest-all", action="store_true", help="모든 짝의 자동 추정값을 표에 채움 (확인된 짝은 건드리지 않음)")
    ap.add_argument("--no-suggest", action="store_true", help="자동 추정 없이 표의 값(없으면 0)에서 시작")
    ap.add_argument("--by", default="", help="확인한 사람")
    ap.add_argument("--table", default=str(TABLE))
    args = ap.parse_args()

    table = SyncTable.load(args.table)
    model = None
    if args.suggest_all or (args.pair and not args.no_suggest and table.offset_ms(args.pair) is None):
        from ultralytics import YOLO
        model = YOLO(str(ROOT / "yolov8n.pt"))

    if args.suggest_all:
        for pair in all_pairs():
            row = table.rows.get(pair)
            if row and row.method == "manual":
                continue
            off, corr = suggest(pair, model)
            video, ref = pair_files(pair)
            table.upsert(SyncRow(pair, f"1(angle0)/{video.name}", f"1(angle1)/{ref.name}", off, "suggest", "",
                                 f"사람 보임 신호 상관 {corr:.2f} — 확인 필요"))
            print(f"{pair}: {off:+.0f}ms (상관 {corr:.2f})", flush=True)
        table.save(args.table)
        print(f"저장: {args.table}")
        return 0

    if not args.pair:
        ap.error("짝 이름 또는 --suggest-all 이 필요합니다")
    start = table.offset_ms(args.pair)
    if start is None:
        start = 0.0 if args.no_suggest else suggest(args.pair, model)[0]
    row = interactive(args.pair, start, args.by)
    if row is not None:
        table.upsert(row)
        table.save(args.table)
        print(f"저장: {args.pair} {row.offset_ms:+.0f}ms → {args.table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
