"""3차 촬영(2026-11-11) 결제 라벨 초안 — 바코드 스캐너 삑 소리 + 촬영 시나리오.

3차는 바코드를 찍어 결제했다. 스캐너가 내는 삑 소리(약 3.4kHz 순음, 0.2~0.4초)는 두 카메라 녹음에 모두 남았고,
선반 카메라(angle0) 쪽이 더 크게 들린다. 결제 완료음은 따로 없다 (그 밖의 소리는 1.2kHz 아래 말소리·잡음).
그래서 **삑 한 번 = 물건 하나 스캔** 으로 보고, 몇 초 안에 이어진 삑들을 한 번의 계산으로 묶는다.

    python scripts/beep_payments.py              # 39편 삑 소리 찾기 + 초안 (확인한 짝은 그대로 둠)
    python scripts/beep_payments.py T04 N06      # 고른 짝만
    python scripts/beep_payments.py --force      # 확인한 짝도 초안으로 다시 씀 (주의)

결과
    labels/store01_shoot3/beeps.csv     짝별 삑 소리 (camera,time_sec=그 카메라 재생 시각, axis_sec=공통 시각, kind=scan|short)
    labels/store01_shoot3/payments.csv  label_payments.py 와 같은 형식, checked_by=beep-draft
초안은 **검증 전**이다: python scripts/label_payments.py --shoot 3 으로 영상을 보며 확인하고 v 로 확정한다.

규칙 (초안)
- 스캔 = 0.15초 이상 이어진 삑. 0.05~0.10초 짧은 소리(화면 누름 등)는 빼고 메모에 남긴다.
- 스캔 사이가 GROUP_GAP_SEC(5초) 이하면 같은 계산 → paid 한 줄, items = 스캔 수, 시각 = 그 계산의 마지막 스캔.
  두 사람이 따로 계산했는지 함께 했는지는 이 간격으로만 추정 — 확인 필요.
- 시나리오: T06(스캔·화면 조작 후 결제 미완료) → incomplete, N06(스캔 후 구매 취소) → cancelled,
  스캔이 없으면 none. 시나리오와 스캔 수가 어긋나면 메모에 적는다.
- 시각은 공통 시간축(전체 카메라 angle1 재생 시각 = angle0 시각 + sync.csv offset_ms) — dual_eval 판정과 같은 축.
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import stft

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import label_payments as LP  # noqa: E402
from src.inputs.sync import SyncTable  # noqa: E402

SR = 16000
BAND_HZ = (3250.0, 3550.0)
TONE_RATIO = 30.0
"""삑 대역 최대 세기 ÷ 1kHz 위 다른 대역 중앙값. 말소리·잡음은 이 비율이 수 배에 그친다."""
MIN_POWER = 1e-6
MIN_LEN_SEC = 0.05
SCAN_SEC = 0.15
GROUP_GAP_SEC = 5.0
BEEP_FIELDS = ("pair", "camera", "time_sec", "axis_sec", "dur", "db", "kind")
SCENARIO_RESULT = {"T6": "incomplete", "N6": "cancelled"}
"""스캔은 했지만 결제가 끝나지 않은 대본 — 삑이 있어도 paid 로 두지 않는다."""
EXPECTED_SCANS = {"N1": 1, "N2": None, "N3": None, "N4": 3, "N5": 3, "N9": 1, "N10": 1, "N12": None,
                  "T1": 0, "T2": 0, "T3": 0, "T4": 2, "T5": 1, "T8": 1, "T9": 0, "T10": 0, "N11": 0, "C10": 0, "C16": 0}
"""대본에서 정해지는 스캔 수 (None = 대본에 수량 없음). 어긋나면 메모로 알린다."""


@dataclass
class Beep:
    start: float
    """그 카메라 재생 시각 (초)."""
    dur: float
    db: float

    @property
    def kind(self) -> str:
        return "scan" if self.dur >= SCAN_SEC else "short"


def read_audio(path: Path) -> np.ndarray:
    """16kHz 모노, -1~1. 소리 줄이 없거나 ffmpeg 가 없으면 빈 배열 (sync_floor.read_audio 재사용)."""
    from sync_floor import read_audio as _read
    return _read(path) / 32768.0


def find_beeps(x: np.ndarray, sr: int = SR) -> list[Beep]:
    """삑 대역 순음 구간들."""
    if len(x) < 512:
        return []
    f, t, z = stft(x, sr, nperseg=512, noverlap=384)
    p = np.abs(z) ** 2
    band = (f >= BAND_HZ[0]) & (f <= BAND_HZ[1])
    other = (f > 1000) & ~band
    inb = p[band].max(0)
    on = (inb / (np.median(p[other], 0) + 1e-12) > TONE_RATIO) & (inb > MIN_POWER)
    out, i, n = [], 0, len(on)
    while i < n:
        if not on[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and (on[j + 1] or (j + 2 < n and on[j + 2])):   # 한 칸 끊김은 잇는다
            j += 1
        if t[j] - t[i] >= MIN_LEN_SEC:
            out.append(Beep(round(float(t[i]), 2), round(float(t[j] - t[i]), 2),
                            round(float(10 * np.log10(inb[i:j + 1].max())), 1)))
        i = j + 1
    return out


def group_scans(times: list[float], gap: float = GROUP_GAP_SEC) -> list[list[float]]:
    groups: list[list[float]] = []
    for s in sorted(times):
        if groups and s - groups[-1][-1] <= gap:
            groups[-1].append(s)
        else:
            groups.append([s])
    return groups


def draft(pair: str, beeps_axis: list[tuple[float, str]]) -> list[LP.PaymentLabel]:
    """공통 시각의 삑들 (axis_sec, kind) → 결제 라벨 초안."""
    sid = LP.scenario_id(pair)
    scans = [s for s, k in beeps_axis if k == "scan"]
    shorts = [s for s, k in beeps_axis if k != "scan"]
    notes = []
    if shorts:
        notes.append("짧은 소리(뺌) " + " ".join(f"{s:.2f}" for s in shorts))
    want = EXPECTED_SCANS.get(sid)
    if want is not None and want != len(scans):
        notes.append(f"대본 스캔 {want} / 삑 {len(scans)}")
    groups = group_scans(scans)
    if len(groups) > 1:
        notes.append(f"계산 {len(groups)}번으로 나눔 (스캔 간격 {GROUP_GAP_SEC:.0f}초 넘음) — 따로/함께 결제 확인")
    rows: list[LP.PaymentLabel] = []
    special = SCENARIO_RESULT.get(sid)
    if special:
        when = scans[-1] if scans else None
        if not scans:
            notes.append("삑 없음 — 시각을 영상에서 찍기")
        rows.append(LP.PaymentLabel(pair, 0, when, 0, special, LP.DRAFT_BY))
    elif not scans:
        rows.append(LP.PaymentLabel(pair, 0, None, 0, "none", LP.DRAFT_BY))
    else:
        for g in groups:
            rows.append(LP.PaymentLabel(pair, 0, g[-1], len(g), "paid", LP.DRAFT_BY,
                                        "스캔 " + " ".join(f"{s:.2f}" for s in g)))
    rows = LP.renumber(rows)
    if notes:
        rows[0].note = "; ".join([rows[0].note] + notes if rows[0].note else notes)
    return rows


def merge(labels: dict, pair: str, rows: list[LP.PaymentLabel], force: bool) -> bool:
    """초안을 넣는다. 사람이 손댄 줄(checked_by 가 초안이 아님)이 하나라도 있으면 force 일 때만 덮는다. 넣었으면 True."""
    if not force and any(r.checked_by != LP.DRAFT_BY for r in labels.get(pair) or []):
        return False
    labels[pair] = rows
    return True


def save_beeps(rows: list[dict], pairs: set[str], path: Path) -> None:
    keep = []
    if path.exists():
        with open(path, encoding="utf-8-sig", newline="") as f:
            keep = [r for r in csv.DictReader(f) if r["pair"] not in pairs]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BEEP_FIELDS)
        w.writeheader()
        for r in sorted(keep + rows, key=lambda r: (r["pair"], r["camera"], float(r["time_sec"]))):
            w.writerow(r)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pairs", nargs="*")
    ap.add_argument("--force", action="store_true", help="사람이 확인한 짝도 초안으로 다시 쓴다")
    args = ap.parse_args()
    LP.set_shoot(3)
    table = SyncTable.load(LP.LABELS / "sync.csv")
    pairs = args.pairs or sorted(p for p in table.rows if p != "area_setting")
    unknown = [p for p in pairs if p not in table.rows]
    if unknown:
        sys.exit(f"sync.csv 에 없는 짝: {unknown}")
    labels = LP.load_labels()
    beep_rows, silent, kept = [], [], []
    for pair in pairs:
        row = table.rows[pair]
        found = {}
        for cam, rel, off in (("angle0", row.video, row.offset_ms), ("angle1", row.reference, 0.0)):
            x = read_audio(LP.SOURCE / rel)
            if len(x) == 0:
                continue
            found[cam] = find_beeps(x)
            beep_rows += [{"pair": pair, "camera": cam, "time_sec": f"{b.start:.2f}",
                           "axis_sec": f"{b.start + off / 1000.0:.2f}", "dur": f"{b.dur:.2f}",
                           "db": f"{b.db:.1f}", "kind": b.kind} for b in found[cam]]
        if "angle0" not in found:   # 초안은 선반 카메라 소리로 (더 크게 들림)
            silent.append(pair)
            continue
        axis = [(round(b.start + row.offset_ms / 1000.0, 2), b.kind) for b in found["angle0"]]
        rows = draft(pair, axis)
        if not merge(labels, pair, rows, args.force):
            kept.append(pair)
        desc = " / ".join(f"{r.result}" + (f"({r.items})" if r.result == "paid" else "")
                          + (f"@{r.time_sec:.1f}" if r.time_sec is not None else "") for r in rows)
        n1 = sum(1 for b in found.get("angle1", []) if b.kind == "scan")
        n0 = sum(1 for b in found["angle0"] if b.kind == "scan")
        print(f"  {pair:<6} 스캔 a0 {n0} a1 {n1} | {desc:<32} | {LP.scenario_hint(pair)}"
              + (f" | {rows[0].note}" if rows[0].note else "") + ("  [확인됨 — 안 바꿈]" if pair in kept else ""))
    LP.save_labels(labels)
    save_beeps(beep_rows, set(pairs), LP.BEEPS)
    print(f"→ {LP.OUT}\n→ {LP.BEEPS}")
    if kept:
        print(f"확인한 짝이라 그대로 둠: {', '.join(kept)} (--force 로 덮기)")
    if silent:
        print(f"소리 없음 — 영상으로 직접 라벨: {', '.join(silent)}")
    print("다음: python scripts/label_payments.py --shoot 3  (초안 확인 후 v)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
