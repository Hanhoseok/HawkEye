"""2차 촬영(카메라 2대) 영상별 결제 라벨링 도구.

지금 판정은 결제 기록이 없어 "계산대에 서 있으면 결제"(payment_mode=dwell)로 본다.
그래서 키오스크 앞에 서 있기만 한 절도(T4 일부 결제·T5 수량 적게·T6 결제 미완료)를 놓친다 (limitations E-13).
이 도구로 영상마다 **결제 시각·품목 수·결과**를 찍어 두면, 판정을 결제 기록(payment_mode=feed)으로 다시 돌릴 수 있다.

    python scripts/label_payments.py              # 아직 안 찍은 짝부터 차례로
    python scripts/label_payments.py T4 T5 T6     # 고른 짝만
    python scripts/label_payments.py --summary    # 찍은 결과 요약만 출력

화면: 왼쪽 전체 카메라(angle1, 계산대·출구), 오른쪽 선반 카메라(angle0, sync.csv 시차로 맞춤).
시각은 **전체 카메라 재생 시각(초)** = 공통 시간축 (src/inputs/sync.py). 결제 기록(PaymentFeed)과 같은 축이다.

두 카메라 시차도 여기서 눈으로 맞춘다: 두 화면에 같은 순간(손이 칸에 닿음, 문 열림)을 띄워 놓고 [ ] 로 민다.
바꾼 시차는 저장할 때 labels/store01_dual/sync.csv 에 method=manual, checked_by 와 함께 기록된다.

조작키 (화면 HUD 에도 표시됨)
    SPACE   재생 / 일시정지
    ← →     1 프레임 뒤로 / 앞으로 (a d 도 됨)
    ↑ ↓     1초 뒤로 / 앞으로
    [ ]     시차 조절: 선반 카메라(오른쪽)를 1 프레임(약 33ms) 더 이른 장면 / 더 늦은 장면으로
    { }     시차 조절 10 프레임씩 (Shift+[ ])
    0       시차를 sync.csv 에 저장된 값으로 되돌림
    1~9     다음에 찍을 품목 수 (기본 1)
    p       결제 완료(paid)     — 지금 시각, 품목 수
    f       결제 실패(failed)   — 품목 수 0 으로 기록
    i       결제 미완료(incomplete) — 스캔·화면 조작만 하고 결제를 끝내지 않음
    x       결제 취소(cancelled)
    n       이 영상은 결제 시도 없음(none)
    u       이 영상의 마지막 기록 삭제
    TAB     다음 짝 (저장)       BACKSPACE   이전 짝 (저장)
    s       저장          q/ESC   저장하고 종료

결과: labels/store01_dual/payments.csv

    pair,attempt,time_sec,items,result,checked_by,note
    N1,1,11.2,1,paid,hoseok,
    T6,1,9.8,0,incomplete,hoseok,
    T1,0,,0,none,hoseok,

- 한 영상에 여러 줄(결제 시도마다 한 줄). 결제 시도가 없으면 attempt 0·result none 한 줄.
- 판정에 넣을 때는 result=paid 줄만 결제 기록으로 쓴다 (`load_feed`). 실제 POS 도 완료된 결제만 알려준다.
- 촬영 시나리오 문서(무인매장_촬영_시나리오_관리.pdf)의 짝별 결제 결과를 터미널에 '시나리오' 로 보여준다.
  실제 촬영이 대본과 다를 수 있으니 **정답은 영상에서 본 것**이다.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inputs.sync import SyncTable  # noqa: E402
from src.risk.payments import PaymentEvent, PaymentFeed  # noqa: E402

SOURCE = ROOT.parent.parent / "source"
LABELS = ROOT / "labels" / "store01_dual"
OUT = LABELS / "payments.csv"
FIELDS = ("pair", "attempt", "time_sec", "items", "result", "checked_by", "note")
RESULTS = ("paid", "failed", "incomplete", "cancelled", "none")

# 촬영 시나리오 문서 v1.1 의 대본 (영상 이름 앞부분 기준). 힌트일 뿐 정답이 아니다.
SCENARIO = {
    "N1": "기본 구매: 1개 결제",
    "N2": "가방 이용: 전체 결제",
    "N3": "주머니(개인 물건): 정상 결제",
    "N4": "다른 상품 3개 모두 결제",
    "N5": "같은 상품 수량 3개 결제",
    "N6": "스캔 -> 구매 취소 -> 반납 (결제 없음)",
    "N7": "결제 실패 -> 재결제 완료",
    "N8": "결제 -> 추가 집기 -> 추가 결제 (결제 2번)",
    "N9": "A 가 B 대신 결제",
    "N10": "2개 집고 1개 반납 -> 1개 결제",
    "N11": "비교 후 모두 반납 (결제 없음)",
    "N12": "출구 근처 -> 돌아와 결제",
    "T1": "들고 바로 퇴장 (결제 없음)",
    "T2": "가방에 넣고 퇴장 (결제 없음)",
    "T3": "주머니에 넣고 퇴장 (결제 없음)",
    "T4": "3개 중 2개만 결제",
    "T5": "같은 상품 3개 중 1개만 결제",
    "T6": "스캔·화면 조작 -> 결제 미완료",
    "T7": "결제 실패 -> 그대로 퇴장",
    "T8": "1개 결제 -> 추가 집기 -> 결제 없이 퇴장",
    "T9": "동행자에게 전달 (결제 없음)",
    "T10": "2개 집고 1개 반납 -> 1개 결제 없이 퇴장",
    "C10": "화면 이탈 -> 반납 후 퇴장 (대본 기준)",
    "C11": "화면 이탈 -> 결제 후 퇴장 (대본 기준)",
}

FONT = cv2.FONT_HERSHEY_SIMPLEX
# 방향키 코드는 OS/백엔드마다 다르다. Windows 는 waitKeyEx 로만 잡히고 값도 크다.
LEFT_KEYS = {2424832, 65361}
RIGHT_KEYS = {2555904, 65363}
UP_KEYS = {2490368, 65362}
DOWN_KEYS = {2621440, 65364}
KEY_UP, KEY_DOWN = 1001, 1002  # normalize_key 가 돌려주는 가상 코드 (0~255 와 겹치지 않게)
RESULT_COLOR = {"paid": (80, 200, 80), "failed": (60, 60, 255), "incomplete": (0, 165, 255),
                "cancelled": (200, 120, 200), "none": (160, 160, 160)}


@dataclass
class PaymentLabel:
    pair: str
    attempt: int
    time_sec: float | None
    items: int
    result: str
    checked_by: str = ""
    note: str = ""


def scenario_hint(pair: str) -> str:
    base = re.match(r"[A-Z]\d+", pair)
    return SCENARIO.get(base.group(0) if base else pair, "대본 없음 — 영상에서 판단")


# ---------------------------------------------------------------- 파일 읽고 쓰기

def load_labels(path: Path = OUT) -> dict[str, list[PaymentLabel]]:
    out: dict[str, list[PaymentLabel]] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            t = (r.get("time_sec") or "").strip()
            out.setdefault(r["pair"], []).append(PaymentLabel(
                pair=r["pair"], attempt=int(r.get("attempt") or 0), time_sec=float(t) if t else None,
                items=int(r.get("items") or 0), result=r["result"].strip(),
                checked_by=r.get("checked_by", ""), note=r.get("note", "")))
    return out


def save_labels(labels: dict[str, list[PaymentLabel]], path: Path = OUT) -> None:
    def pair_key(p: str):
        m = re.match(r"([A-Z])(\d+)(.*)", p)
        return (m.group(1), int(m.group(2)), m.group(3)) if m else (p, 0, "")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for pair in sorted(labels, key=pair_key):
            for lab in labels[pair]:
                row = asdict(lab)
                row["time_sec"] = "" if lab.time_sec is None else f"{lab.time_sec:.2f}"
                w.writerow(row)


def load_feed(pair: str, path: Path = OUT) -> PaymentFeed:
    """판정용: 그 짝의 완료된 결제만 결제 기록으로. 라벨이 없으면 빈 기록."""
    rows = load_labels(path).get(pair, [])
    return PaymentFeed([PaymentEvent(time_sec=r.time_sec, items=r.items)
                        for r in rows if r.result == "paid" and r.time_sec is not None])


def renumber(rows: list[PaymentLabel]) -> list[PaymentLabel]:
    """시도는 시각 순 1,2,3… / 시도 없음(none)은 0."""
    timed = sorted((r for r in rows if r.result != "none"), key=lambda r: r.time_sec or 0.0)
    for i, r in enumerate(timed, 1):
        r.attempt = i
    return timed + [r for r in rows if r.result == "none"]


# ---------------------------------------------------------------- 화면

class FrameCache:
    """영상을 처음부터 순서대로 한 번 읽어 메모리(JPEG)에 담는다. **되감기·건너뛰기에 OpenCV 탐색을 쓰지 않는다.**

    2차 촬영 영상(HEVC, 휴대폰)은 cv2.CAP_PROP_POS_FRAMES 로 탐색하면 엉뚱한 장면이 나온다 —
    C01 에서 598번(10초)을 요청하면 1168번(19.5초)이 나오고, 15초 넘게는 읽지도 못했다 (2026-11-04 측정).
    그래서 순서대로 읽은 번호만 믿는다. 읽는 동안에도 이미 읽은 장면은 바로 볼 수 있다 (뒤 스레드).
    """

    def __init__(self, path: Path, width: int) -> None:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(path)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.expected = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        self.width = width
        self.frames: list[np.ndarray] = []
        self.done = False
        self._cap = cap
        self._thread = threading.Thread(target=self._load, daemon=True)
        self._thread.start()

    def _load(self) -> None:
        try:
            while True:
                ok, frame = self._cap.read()
                if not ok:
                    break
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (self.width, int(h * self.width / w)), interpolation=cv2.INTER_AREA)
                self.frames.append(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])[1])
        finally:
            self._cap.release()
            self.done = True

    @property
    def n(self) -> int:
        return len(self.frames) if self.done else max(self.expected, len(self.frames))

    @property
    def progress(self) -> float:
        return 1.0 if self.done else min(len(self.frames) / self.expected, 0.99)

    def get(self, i: int) -> np.ndarray | None:
        """i 번째 장면. 아직 안 읽었으면 읽힐 때까지 기다린다."""
        while i >= len(self.frames) and not self.done:
            time.sleep(0.005)
        if not self.frames:
            return None
        return cv2.imdecode(self.frames[min(i, len(self.frames) - 1)], cv2.IMREAD_COLOR)


class PairPlayer:
    """전체 카메라를 기준으로 선반 카메라를 시차만큼 맞춰 보여준다."""

    def __init__(self, ov_path: Path, de_path: Path, offset_ms: float, width: int = 800) -> None:
        self.ov = FrameCache(ov_path, width)
        self.de = FrameCache(de_path, width) if de_path.exists() else None
        self.fps = self.ov.fps
        self.de_fps = self.de.fps if self.de else 30.0
        self.offset_ms = offset_ms
        self.width = width

    @property
    def n(self) -> int:
        return self.ov.n

    @property
    def de_n(self) -> int:
        return self.de.n if self.de else 0

    @property
    def progress(self) -> float:
        return min(self.ov.progress, self.de.progress if self.de else 1.0)

    def frame(self, idx: int) -> np.ndarray:
        left = self._fit(self.ov.get(idx))
        if self.de is None:
            return left
        # 공통 시각 = 선반 카메라 재생 시각 + offset → 선반 카메라 시각 = 공통 − offset
        de_t = idx / self.fps - self.offset_ms / 1000.0
        de_idx = int(round(de_t * self.de_fps))
        if 0 <= de_idx < self.de_n:
            right = self._fit(self.de.get(de_idx))
        else:
            right = np.zeros_like(left)
            cv2.putText(right, "angle0: out of range", (20, 40), FONT, 0.8, (180, 180, 180), 2)
        if right.shape[0] != left.shape[0]:
            right = cv2.resize(right, (int(right.shape[1] * left.shape[0] / right.shape[0]), left.shape[0]))
        return np.hstack([left, right])

    def _fit(self, frame):
        if frame is None:
            return np.zeros((int(self.width * 9 / 16), self.width, 3), np.uint8)
        h, w = frame.shape[:2]
        if w == self.width:
            return frame
        return cv2.resize(frame, (self.width, int(h * self.width / w)))

    def release(self) -> None:
        pass  # 읽기 스레드가 끝나면 스스로 닫는다. 메모리는 참조가 사라지면 풀린다.


def draw_hud(img, pair, idx, player: PairPlayer, rows, pending_items, playing, done, total, saved_offset=None):
    t = idx / player.fps
    bar = np.full((190, img.shape[1], 3), 30, np.uint8)
    put = lambda s, x, y, c=(230, 230, 230), sc=0.6: cv2.putText(bar, s, (x, y), FONT, sc, c, 1, cv2.LINE_AA)
    put(f"{pair}  [{done}/{total} labeled]   t={t:6.2f}s  frame {idx}/{player.n}   "
        f"{'PLAY' if playing else 'PAUSE'}   next items={pending_items}", 10, 24, (255, 255, 255), 0.65)
    if player.progress < 1.0:
        put(f"loading {player.progress * 100:3.0f}%", img.shape[1] - 560, 24, (0, 200, 255), 0.6)
    changed = saved_offset is not None and abs(player.offset_ms - saved_offset) > 0.5
    put(f"offset {player.offset_ms:+.0f}ms" + (f"  (saved {saved_offset:+.0f}ms, unsaved)" if changed else ""),
        img.shape[1] - 430, 24, (0, 200, 255) if changed else (200, 200, 200), 0.6)
    put("SPACE play | <- -> 1 frame | up/down -1s/+1s | [ ] offset 1 frame   { } 10 frames   0 reset offset",
        10, 50, (170, 170, 170), 0.45)
    put("1-9 items | p paid  f failed  i incomplete  x cancelled  n none | u undo | TAB next  BKSP prev | "
        "s save  q quit", 10, 70, (170, 170, 170), 0.45)
    y = 96
    if not rows:
        put("(no label yet)", 10, y, (120, 120, 255))
    for r in rows:
        txt = f"#{r.attempt} {r.result:<10} items={r.items}" + (f" at {r.time_sec:.2f}s" if r.time_sec is not None else "")
        put(txt, 10, y, RESULT_COLOR.get(r.result, (255, 255, 255)))
        y += 22
    # 타임라인
    x0, x1, ty = 10, img.shape[1] - 10, 178
    cv2.line(bar, (x0, ty), (x1, ty), (90, 90, 90), 2)
    cx = int(x0 + (x1 - x0) * idx / max(player.n - 1, 1))
    cv2.line(bar, (cx, ty - 8), (cx, ty + 4), (255, 255, 255), 2)
    for r in rows:
        if r.time_sec is not None:
            mx = int(x0 + (x1 - x0) * r.time_sec * player.fps / max(player.n - 1, 1))
            cv2.circle(bar, (mx, ty), 5, RESULT_COLOR.get(r.result, (255, 255, 255)), -1)
    return np.vstack([img, bar])


def normalize_key(raw: int) -> int:
    if raw in LEFT_KEYS:
        return ord("a")
    if raw in RIGHT_KEYS:
        return ord("d")
    if raw in UP_KEYS:
        return KEY_UP
    if raw in DOWN_KEYS:
        return KEY_DOWN
    return raw & 0xFF if raw >= 0 else 255


def save_offset(sync: SyncTable, row, offset_ms: float, checked_by: str) -> None:
    """눈으로 맞춘 시차를 sync.csv 에 기록한다. 바뀌지 않았으면 그대로 둔다."""
    if abs(offset_ms - row.offset_ms) <= 0.5:
        return
    before = row.offset_ms
    # 여러 번 저장해도 처음 값(자동 추정)을 메모에 남긴다
    if not row.note.startswith("눈으로 맞춤"):
        row.note = f"눈으로 맞춤 (처음 {row.method} {before:+.0f}ms)"
    row.offset_ms, row.method, row.checked_by = offset_ms, "manual", checked_by
    sync.upsert(row)
    sync.save(LABELS / "sync.csv")
    print(f"   시차 저장: {row.pair} {before:+.0f}ms -> {offset_ms:+.0f}ms")


def label_pair(pair: str, row, labels, checked_by: str, done: int, total: int, width: int,
               sync: SyncTable, player: PairPlayer | None = None) -> str:
    """한 짝을 라벨링한다. 돌려주는 값: 'next' / 'prev' / 'quit'."""
    player = player or open_player(row, width)
    rows = labels.setdefault(pair, [])
    idx, playing, pending = 0, False, 1
    win = "label_payments"
    hint = scenario_hint(pair)
    print(f"\n== {pair}  시나리오: {hint}   시차 {row.offset_ms:+.0f}ms ({row.method})")
    step_1s = int(round(player.fps))
    offset_step = 1000.0 / player.de_fps  # 선반 카메라 1 프레임
    try:
        while True:
            idx = max(0, min(idx, player.n - 1))  # 다 읽고 나면 실제 장면 수가 메타값보다 적을 수 있다
            img = player.frame(idx)  # OpenCV 는 한글을 못 그린다 — 시나리오 힌트는 터미널에 출력
            cv2.imshow(win, draw_hud(img, pair, idx, player, rows, pending, playing, done, total, row.offset_ms))
            key = normalize_key(cv2.waitKeyEx(int(1000 / player.fps) if playing else 0))
            t = round(idx / player.fps, 2)
            if playing and key == 255:
                idx = min(idx + 1, player.n - 1)
                if idx == player.n - 1:
                    playing = False
                continue
            if key == ord(" "):
                playing = not playing
            elif key == ord("d"):
                idx = min(idx + 1, player.n - 1)
            elif key == ord("a"):
                idx = max(idx - 1, 0)
            elif key == KEY_DOWN:
                idx = min(idx + step_1s, player.n - 1)
            elif key == KEY_UP:
                idx = max(idx - step_1s, 0)
            elif key in (ord("["), ord("]"), ord("{"), ord("}")):
                # 선반 카메라 시각 = 공통 시각 − offset → offset 을 키우면 오른쪽 화면이 더 이른 장면이 된다
                steps = 10 if key in (ord("{"), ord("}")) else 1
                sign = 1 if key in (ord("["), ord("{")) else -1
                player.offset_ms = round(player.offset_ms + sign * steps * offset_step, 1)
            elif key == ord("0"):
                player.offset_ms = row.offset_ms
            elif ord("1") <= key <= ord("9"):
                pending = key - ord("0")
            elif key in (ord("p"), ord("f"), ord("i"), ord("x")):
                result = {"p": "paid", "f": "failed", "i": "incomplete", "x": "cancelled"}[chr(key)]
                rows[:] = [r for r in rows if r.result != "none"]
                rows.append(PaymentLabel(pair, 0, t, pending if result == "paid" else 0, result, checked_by))
                rows[:] = renumber(rows)
                print(f"   {result} items={rows[-1].items if result == 'paid' else 0} at {t:.2f}s")
                pending = 1
            elif key == ord("n"):
                rows[:] = [PaymentLabel(pair, 0, None, 0, "none", checked_by)]
                print("   결제 시도 없음")
            elif key == ord("u") and rows:
                print(f"   삭제: {rows.pop()}")
                rows[:] = renumber(rows)
            elif key == ord("s"):
                save_labels(labels)
                save_offset(sync, row, player.offset_ms, checked_by)
                print(f"   저장: {OUT}")
            elif key == 9:  # TAB
                return "next"
            elif key == 8:  # BACKSPACE
                return "prev"
            elif key in (ord("q"), 27):
                return "quit"
    finally:
        save_offset(sync, row, player.offset_ms, checked_by)
        player.release()
        if not rows:
            labels.pop(pair, None)
        save_labels(labels)


def summary(labels: dict[str, list[PaymentLabel]], pairs: list[str]) -> None:
    missing = [p for p in pairs if p not in labels]
    print(f"라벨 {len(pairs) - len(missing)}/{len(pairs)}  (없음: {', '.join(missing) or '-'})")
    for p in pairs:
        if p not in labels:
            continue
        rows = labels[p]
        paid = sum(r.items for r in rows if r.result == "paid")
        desc = " / ".join(f"{r.result}" + (f"({r.items})" if r.result == "paid" else "")
                          + (f"@{r.time_sec:.1f}s" if r.time_sec is not None else "") for r in rows)
        print(f"  {p:<6} 결제 품목 {paid}  | {desc:<45} | 시나리오: {scenario_hint(p)}")


def open_player(row, width: int) -> PairPlayer:
    return PairPlayer(SOURCE / row.reference, SOURCE / row.video, row.offset_ms, width)


def main() -> None:
    # Windows 콘솔(cp949)이 '—' 같은 글자에서 멈추지 않게
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", nargs="*", help="짝 이름 (예: T4 N1). 비우면 아직 안 찍은 짝 전부")
    ap.add_argument("--all", action="store_true", help="이미 찍은 짝도 다시 돌아본다")
    ap.add_argument("--summary", action="store_true", help="요약만 출력")
    ap.add_argument("--by", default=os.environ.get("USERNAME", ""), help="checked_by 에 남길 이름")
    ap.add_argument("--width", type=int, default=800, help="카메라 한 화면 폭(px)")
    args = ap.parse_args()

    sync = SyncTable.load(LABELS / "sync.csv")
    table = dict(sync.rows)
    all_pairs = sorted(table, key=lambda p: (p[0], int(re.match(r"[A-Z](\d+)", p).group(1)), p))
    labels = load_labels()

    if args.summary:
        summary(labels, all_pairs)
        return
    unknown = [p for p in args.pairs if p not in table]
    if unknown:
        sys.exit(f"sync.csv 에 없는 짝: {unknown}")
    todo = args.pairs or [p for p in all_pairs if args.all or p not in labels]
    if not todo:
        print("모든 짝을 찍었다. 다시 보려면 --all 또는 짝 이름을 준다.")
        summary(labels, all_pairs)
        return

    i = 0
    players: dict[str, PairPlayer] = {}
    while 0 <= i < len(todo):
        done = sum(1 for p in all_pairs if p in labels)
        # 지금 짝 + 다음 짝을 미리 읽어 둔다 (짝 하나 읽는 데 약 17초). 나머지는 메모리에서 뺀다.
        keep = todo[i:i + 2]
        for p in keep:
            if p not in players:
                players[p] = open_player(table[p], args.width)
        for p in list(players):
            if p not in keep:
                players.pop(p).release()
        action = label_pair(todo[i], table[todo[i]], labels, args.by, done, len(all_pairs), args.width, sync,
                            players[todo[i]])
        if action == "quit":
            break
        i += 1 if action == "next" else -1
        i = max(i, 0)
    cv2.destroyAllWindows()
    summary(labels, all_pairs)


if __name__ == "__main__":
    main()
