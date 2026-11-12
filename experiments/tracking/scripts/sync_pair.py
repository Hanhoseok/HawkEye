"""두 각도 녹화본의 시차 맞추기 — 사람이 두 화면을 보며 확정하고 표에 남긴다 (design 1-0, M-9).

기준은 전체 카메라(angle1). 선반 카메라(angle0) 영상을 offset_ms 만큼 밀어 같은 시간축에 놓는다
(정의는 src/inputs/sync.py). 자동 추정(사람이 보이나 신호의 상관)은 약해서 시작값으로만 쓴다 (limitations B-10).

    # 한 짝: 자동 추정값으로 시작 → 화면을 보며 맞춤 → s 로 저장
    python scripts/sync_pair.py N1

    # 자동 추정만 표에 채우기 (method=suggest, 확인 전 표시)
    python scripts/sync_pair.py --suggest-all

    # 3차 촬영 (source/2(angle0)·2(angle1), 표 labels/store01_shoot3/sync.csv).
    # 시작값은 발 위치 자동 추정(scripts/sync_floor.py estimate)이 표에 써 둔 값 — 먼저 그것을 돌린다.
    python scripts/sync_pair.py T01 --shoot 3

    # 3차 촬영 39쌍을 차례로 (아직 manual 이 아닌 짝만, --redo 면 전부) / 진행 상황
    python scripts/sync_pair.py --all --shoot 3 --by hoseok
    python scripts/sync_pair.py --status --shoot 3

화면 조작 (왼쪽 angle0, 오른쪽 angle1 — 같은 공통 시각의 장면):
    스페이스 재생/정지   ← → 0.1초 이동   [ ] 1초 이동
    a d   angle0 을 한 장면(1/30초)씩 밀기   A D  10 장면씩
    s 저장   q 끝
    --all 일 때: n 저장 없이 다음 짝, c 후보 바꾸기 (표 note 의 '발 위치'·'소리' 시차로 옮김), q 전체 끝
같은 순간(박수, 문 닫힘, 물건이 손을 떠나는 순간)이 양쪽에 함께 보이게 맞추고 저장한다.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inputs.frame_times import end_ms, first_ms, index_at  # noqa: E402
from src.inputs.sync import SyncRow, SyncTable, best_lag  # noqa: E402

SOURCE = ROOT.parent.parent / "source"
VIDEO_DIR = SOURCE / "1(angle0)"
REF_DIR = SOURCE / "1(angle1)"
TABLE = ROOT / "labels" / "store01_dual" / "sync.csv"
SHOOT3 = {"video": SOURCE / "2(angle0)", "ref": SOURCE / "2(angle1)",
          "table": ROOT / "labels" / "store01_shoot3" / "sync.csv"}
"""3차 촬영: 두 각도 모두 <짝>.mp4 (예: T01.mp4, N08-1.mp4)."""
ADDED_DIR = SOURCE / "added"
"""분석 도구에서 추가한 영상: <짝>_angle1.<확장자>(전체 카메라) · <짝>_angle0.<확장자>(선반 카메라). 같은 매장·같은 카메라 위치만."""
VIDEO_EXT = (".mp4", ".mov", ".m4v", ".mkv", ".avi")
HZ = 10.0


def pair_files(pair: str, shoot: int = 2) -> tuple[Path, Path]:
    """짝 이름(N1, T3, C10, N8-1) → (angle0 파일, angle1 파일). 추가한 영상이면 그 파일, 아니면 2차 촬영의 이름 규칙.
    shoot=3 이면 3차 촬영 폴더의 같은 이름 파일 (T01 → 2(angle0)/T01.mp4, 2(angle1)/T01.mp4)."""
    if shoot == 3:
        return SHOOT3["video"] / f"{pair}.mp4", SHOOT3["ref"] / f"{pair}.mp4"
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


def all_pairs(shoot: int = 2) -> list[str]:
    if shoot == 3:
        return sorted(p.stem for p in SHOOT3["ref"].glob("*.mp4")
                      if p.stem != "area_setting" and (SHOOT3["video"] / p.name).exists())
    names = []
    for p in sorted(VIDEO_DIR.glob("20261028_*.mp4")):
        name = p.stem.split("_", 1)[1]
        if pair_files(name)[1].exists():
            names.append(name)
    return names + [n for n in added_pairs() if n not in names]


class Clip:
    """영상을 한 번 다 읽어 줄인 장면을 메모리에 둔다 (H.265 는 위치 이동이 불안정해서).
    장면마다 실제 시각(POS_MSEC)으로 찾는다 — 3차 angle1 은 장면을 빠뜨려 '번호 ÷ fps' 가 끝에서 최대 10초 틀린다."""

    def __init__(self, path: Path, width: int = 640) -> None:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise SystemExit(f"영상을 열 수 없습니다: {path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frames = []
        self.times: list[float] = []
        while True:
            ok, image = cap.read()
            if not ok:
                break
            self.times.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            self.frames.append(cv2.resize(image, (width, int(width * image.shape[0] / image.shape[1]))))
        cap.release()
        if len(self.times) > 1:   # 첫 장면 시각만 OpenCV 가 엉뚱하게 줄 때가 있다
            self.times[0] = first_ms(self.times[0], self.times[1], self.fps)
        self.duration_ms = end_ms(self.times, self.fps)

    def at(self, ms: float):
        """공통 시각 ms 의 장면. 영상 구간 [0, 끝) 밖이면 검은 화면 (두 영상이 겹치지 않는 곳을 장면처럼 보이지 않게)."""
        if not self.frames:
            return None
        if ms < 0 or ms >= self.duration_ms:
            return np.zeros_like(self.frames[0])
        return self.frames[index_at(self.times, ms)]


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


def rel(path: Path) -> str:
    """표에 남기는 경로: source 아래 '폴더/파일' (예: 1(angle0)/20261028_N1.mp4, 2(angle1)/T01.mp4)."""
    return f"{path.parent.name}/{path.name}"


CANDIDATE = re.compile(r"(발 위치|소리) ([+-]?\d+(?:\.\d+)?)ms")


def candidates(note: str) -> list[tuple[str, float]]:
    """표 note 의 시차 후보 (sync_floor.py 가 적음): [('발 위치', 654.0), ('소리', 1.0)]. 같은 값은 한 번만."""
    out: list[tuple[str, float]] = []
    for name, value in CANDIDATE.findall(note or ""):
        v = float(value)
        if all(abs(v - o) > 0.5 for _, o in out):
            out.append((name, v))
    return out


def next_pairs(table: SyncTable, pairs: list[str], redo: bool = False) -> list[str]:
    """--all 로 볼 짝: 아직 사람이 확인(manual)하지 않은 짝, redo 면 전부."""
    return [p for p in pairs if redo or (table.rows.get(p) is None or table.rows[p].method != "manual")]


def status_text(table: SyncTable, pairs: list[str]) -> str:
    counts: dict[str, int] = {}
    for p in pairs:
        m = table.rows[p].method if p in table.rows else "(없음)"
        counts[m] = counts.get(m, 0) + 1
    todo = next_pairs(table, pairs)
    done = len(pairs) - len(todo)
    return (f"확인(manual) {done}/{len(pairs)}  |  " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
            + (f"\n남은 짝: {', '.join(todo)}" if todo else "\n모두 확인했다"))


def interactive(pair: str, start_offset: float, who: str, shoot: int = 2, loop: bool = False,
                cands: list[tuple[str, float]] | None = None, label: str = ""):
    """두 화면을 보며 시차를 맞춘다. 돌려주는 값: SyncRow(s 저장) / None(q 끝) / "skip"(n, loop 일 때만)."""
    video, ref = pair_files(pair, shoot)
    a, b = Clip(video), Clip(ref)
    offset, t, playing = start_offset, 0.0, False
    frame_ms = 1000.0 / a.fps
    cands = cands or []
    ci = -1
    print(f"{pair}: angle0 {a.fps:.0f}FPS {a.duration_ms / 1000:.1f}s / angle1 {b.fps:.0f}FPS {b.duration_ms / 1000:.1f}s, 시작 시차 {offset:+.0f}ms")
    while True:
        left, right = a.at(t - offset), b.at(t)   # 공통 시각 t: angle1 은 그대로, angle0 은 t - offset
        canvas = cv2.hconcat([left, cv2.resize(right, (left.shape[1], left.shape[0]))])
        cv2.putText(canvas, f"{label}{pair}  t={t / 1000:.2f}s  offset={offset:+.0f}ms  {'PLAY' if playing else ''}",
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
            return SyncRow(pair, rel(video), rel(ref), round(offset, 1), "manual", who)
        elif key == ord("q"):
            cv2.destroyWindow("sync")
            return None
        elif loop and key == ord("n"):
            cv2.destroyWindow("sync")
            return "skip"
        elif loop and key == ord("c") and cands:
            ci = (ci + 1) % len(cands)
            offset = cands[ci][1]
            print(f"   후보 {cands[ci][0]} {offset:+.0f}ms")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pair", nargs="?", help="짝 이름 (예: N1, T3, C10, N8-1)")
    ap.add_argument("--suggest-all", action="store_true", help="모든 짝의 자동 추정값을 표에 채움 (확인된 짝은 건드리지 않음)")
    ap.add_argument("--no-suggest", action="store_true", help="자동 추정 없이 표의 값(없으면 0)에서 시작")
    ap.add_argument("--by", default="", help="확인한 사람")
    ap.add_argument("--table", help="시차 표 (기본: 2차 labels/store01_dual/sync.csv, 3차 labels/store01_shoot3/sync.csv)")
    ap.add_argument("--shoot", type=int, choices=[2, 3], default=2, help="촬영 회차 (기본 2)")
    ap.add_argument("--all", action="store_true", help="짝을 차례로 (아직 manual 이 아닌 짝만)")
    ap.add_argument("--redo", action="store_true", help="--all 에서 manual 짝도 다시")
    ap.add_argument("--status", action="store_true", help="확인 진행 상황만 출력")
    args = ap.parse_args()
    args.table = args.table or str(SHOOT3["table"] if args.shoot == 3 else TABLE)
    if args.shoot == 3 and args.suggest_all:
        ap.error("3차 촬영의 자동 추정은 scripts/sync_floor.py estimate 를 쓴다 (발 위치)")

    table = SyncTable.load(args.table)
    model = None
    if args.suggest_all or (args.shoot == 2 and args.pair and not args.no_suggest and table.offset_ms(args.pair) is None):
        from ultralytics import YOLO
        model = YOLO(str(ROOT / "yolov8n.pt"))

    if args.suggest_all:
        for pair in all_pairs():
            row = table.rows.get(pair)
            if row and row.method == "manual":
                continue
            off, corr = suggest(pair, model)
            video, ref = pair_files(pair)
            table.upsert(SyncRow(pair, rel(video), rel(ref), off, "suggest", "",
                                 f"사람 보임 신호 상관 {corr:.2f} — 확인 필요"))
            print(f"{pair}: {off:+.0f}ms (상관 {corr:.2f})", flush=True)
        table.save(args.table)
        print(f"저장: {args.table}")
        return 0

    pairs = all_pairs(args.shoot)
    if args.status:
        print(status_text(table, pairs))
        return 0
    if args.all:
        todo = next_pairs(table, pairs, args.redo)
        print(f"{len(todo)}쌍을 차례로 — s 저장 후 다음, n 저장 없이 다음, c 후보 바꾸기, q 끝")
        for k, pair in enumerate(todo, 1):
            row = table.rows.get(pair)
            start = row.offset_ms if row else 0.0
            print(f"\n[{k}/{len(todo)}] {pair}: {row.method if row else '(표에 없음)'} {start:+.0f}ms"
                  + (f"\n   {row.note}" if row and row.note else ""))
            if row is None and args.shoot == 2 and not args.no_suggest:
                if model is None:
                    from ultralytics import YOLO
                    model = YOLO(str(ROOT / "yolov8n.pt"))
                start = suggest(pair, model)[0]
            res = interactive(pair, start, args.by, args.shoot, loop=True,
                              cands=candidates(row.note if row else ""), label=f"[{k}/{len(todo)}] ")
            if res is None:
                break
            if res == "skip":
                continue
            if row is not None and row.note and row.method != "manual":
                res.note = f"확인 전 {row.method} {row.offset_ms:+.0f}ms: {row.note}"
            table.upsert(res)
            table.save(args.table)
            print(f"   저장: {pair} {res.offset_ms:+.0f}ms")
        print(status_text(table, pairs))
        return 0

    if not args.pair:
        ap.error("짝 이름, --all, --status 또는 --suggest-all 이 필요합니다")
    start = table.offset_ms(args.pair)
    if start is None and args.shoot == 3:
        print(f"{args.pair}: 표에 시작값 없음 — 0 에서 시작 (먼저 sync_floor.py estimate 권장)")
        start = 0.0
    if start is None:
        start = 0.0 if args.no_suggest else suggest(args.pair, model)[0]
    row = interactive(args.pair, start, args.by, args.shoot)
    if row is not None:
        table.upsert(row)
        table.save(args.table)
        print(f"저장: {args.pair} {row.offset_ms:+.0f}ms → {args.table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
