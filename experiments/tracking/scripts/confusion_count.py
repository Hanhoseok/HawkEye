"""두 카메라 가운데 한쪽만 사람을 헷갈린 순간이 몇 번인가 — 저장된 기록만으로 센다 (docs/multiview-progress.md 후속 67, 한계 C-26 · C-28).

    python scripts/confusion_count.py                    # 2 · 3 · 4차 전부
    python scripts/confusion_count.py --shoot 4          # 한 회차
    python scripts/confusion_count.py --shoot 4 --pairs C05 --list   # 그 짝의 구간을 하나씩 찍는다
    python scripts/confusion_count.py --by-pair          # 영상마다 몇 번인지 표로
    python scripts/confusion_count.py --solo             # 한 명만 나오는 영상에서 잡히는 구간 = 잘못 세는 양 (가짜 상자)

왜 세나: "헷갈리는 순간은 덜 헷갈리는 카메라가 정한다"를 만들기 전에, 그런 순간이 실제로 얼마나 있는지 본다.
양쪽이 함께 헷갈리면 그 규칙도 못 가른다.

한 카메라가 '헷갈린다' (사람 상자만 본다 — 외형 · 판정은 안 본다):
  겹침    두 사람의 상자가 OVERLAP 이상 겹친다 (겹친 넓이 ÷ 작은 상자 넓이 — fusion.swap_overlap 과 같은 식 · 같은 값)
  합쳐짐  한 사람의 상자가 사라졌는데, 사라지기 직전 상자의 OVERLAP 이상이 남은 다른 사람 상자에 덮여 있다.
          그 사람이 다시 보이거나 MERGED_MAX_S 가 지날 때까지 (4차 C05: 두 사람이 상자 하나가 됨)
그 카메라가 '갈라 본다' = 헷갈리지 않고 사람 상자가 둘 이상.

아무것도 고치지 않는다. 영상 · 모델이 필요 없다.

못 보는 것 (한계):
  - 사람 상자 수만 본다. 다른 카메라가 갈라 본 두 사람이 **같은 두 사람**인지는 안 본다 (세 명 이상인 영상에서 틀릴 수 있다).
  - 한 사람 위에 생긴 두 상자(한계 C-25)를 '두 사람'으로 셀 수 있다 — MIN_LIFE_S 보다 짧게 산 번호만 뺀다.
  - 번호가 실제로 틀렸는지는 안 본다. 헷갈릴 만한 순간을 셀 뿐이다.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import dual_eval as de  # noqa: E402
from sync_pair import SyncTable  # noqa: E402

OVERLAP = 0.5
"""상자 겹침 기준 — fusion.swap_overlap 과 같은 값. 정상 영상으로 맞춘 값이 아니다 (한계 A-15)."""
MERGED_MAX_S = 3.0
"""합쳐짐으로 치는 가장 긴 시간 — identity.stitch_max_seconds 와 같은 값."""
MIN_LIFE_S = 1.0
"""이보다 짧게 보인 번호는 사람으로 안 친다 (가짜 상자)."""
JOIN_S = 0.5
"""헷갈린 장면 사이가 이 안이면 한 구간으로 잇는다."""
MIN_EPISODE_S = 0.2
"""이보다 짧은 구간은 안 센다 (한두 장면의 스침)."""

CAMS = {"angle1": "전체", "angle0": "선반"}
OTHER = {"angle1": "angle0", "angle0": "angle1"}
KINDS = ("갈라 봄", "함께 헷갈림", "두 사람이 안 보임")


def overlap(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return w * h / small if small > 0 else 0.0


def covered(mine, other) -> float:
    """mine 상자 넓이 가운데 other 가 덮은 비율."""
    w = min(mine[2], other[2]) - max(mine[0], other[0])
    h = min(mine[3], other[3]) - max(mine[1], other[1])
    area = (mine[2] - mine[0]) * (mine[3] - mine[1])
    return w * h / area if w > 0 and h > 0 and area > 0 else 0.0


def camera_states(frames: list[tuple[float, list[tuple[int, tuple]]]]) -> list[tuple[float, str]]:
    """frames = [(ms, [(번호, 상자), …]), …] 시간순 → [(ms, "헷갈림" | "갈라 봄" | "한 명 이하"), …]."""
    seen: dict[int, list[float]] = collections.defaultdict(list)
    for ms, boxes in frames:
        for pid, _ in boxes:
            seen[pid].append(ms)
    real = {pid for pid, t in seen.items() if pid >= 0 and t[-1] - t[0] >= MIN_LIFE_S * 1000.0}

    out = []
    last: dict[int, tuple] = {}
    merged: dict[int, tuple[int, float]] = {}   # 사라진 번호 → (덮은 번호, 사라진 ms)
    for ms, boxes in frames:
        now = {pid: box for pid, box in boxes if pid in real}
        for pid in now:
            merged.pop(pid, None)
        for pid, box in last.items():
            if pid in now or pid in merged:
                continue
            into = [other for other, b in now.items() if covered(box, b) >= OVERLAP]
            if into:
                merged[pid] = (into[0], ms)
        for pid, (into, since) in list(merged.items()):
            if into not in now or ms - since > MERGED_MAX_S * 1000.0:
                merged.pop(pid)
        ids = list(now)
        over = any(overlap(now[a], now[b]) >= OVERLAP for i, a in enumerate(ids) for b in ids[i + 1:])
        out.append((ms, "헷갈림" if over or merged else "갈라 봄" if len(now) >= 2 else "한 명 이하"))
        last = now
    return out


def episodes(states: list[tuple[float, str]]) -> list[tuple[float, float]]:
    spans: list[list[float]] = []
    for ms, state in states:
        if state != "헷갈림":
            continue
        if spans and ms - spans[-1][1] <= JOIN_S * 1000.0:
            spans[-1][1] = ms
        else:
            spans.append([ms, ms])
    return [(a, b) for a, b in spans if b - a >= MIN_EPISODE_S * 1000.0]


def other_view(states: list[tuple[float, str]], start: float, end: float) -> str:
    """그 구간에 다른 카메라가 가장 오래 있던 상태."""
    count = collections.Counter(state for ms, state in states if start <= ms <= end)
    if not count:
        return "두 사람이 안 보임"
    top = count.most_common(1)[0][0]
    return {"갈라 봄": "갈라 봄", "헷갈림": "함께 헷갈림", "한 명 이하": "두 사람이 안 보임"}[top]


def pair_rows(pair: str, offset_ms: float) -> list[dict]:
    log = de.PairLog(pair)
    states = {}
    for cam in CAMS:
        shift = offset_ms if cam == "angle0" else 0.0
        frames = [(log.ms(cam, idx) + shift, [(o.person_id, tuple(o.bbox)) for o in obs])
                  for idx, obs in sorted(log.ids[cam].items())]
        # 사람이 없는 장면은 기록에 없다 — 상태는 '한 명 이하'이고 구간을 잇지도 끊지도 않는다
        states[cam] = camera_states(sorted(frames))
    rows = []
    for cam in CAMS:
        for start, end in episodes(states[cam]):
            rows.append({"pair": pair, "cam": cam, "start": start, "end": end,
                         "other": other_view(states[OTHER[cam]], start, end)})
    return rows


def pair_table(rows: list[dict], names: list[str], truth: dict) -> list[str]:
    """영상마다 한 줄 — 카메라별 헷갈린 구간 수와 그동안 다른 카메라의 상태 (갈라 봄 / 함께 헷갈림 / 두 사람이 안 보임)."""
    out = ["| 영상 | 손님 | 전체 카메라가 헷갈림 | 갈라 봄 / 함께 / 안 보임 | 선반 카메라가 헷갈림 | 갈라 봄 / 함께 / 안 보임 |",
           "| --- | --- | --- | --- | --- | --- |"]
    for pair in names:
        cells = [pair, str(truth.get(pair, {}).get("people", "?"))]
        for cam in CAMS:
            mine = [r for r in rows if r["pair"] == pair and r["cam"] == cam]
            secs = sum(r["end"] - r["start"] for r in mine) / 1000.0
            cells.append(f"{len(mine)}번 {secs:.1f}초" if mine else "0")
            cells.append(" / ".join(str(sum(r["other"] == k for r in mine)) for k in KINDS) if mine else "-")
        out.append("| " + " | ".join(cells) + " |")
    return out


def run(shoot: int, pattern: str | None, show: bool, solo: bool = False, table_too: bool = False) -> None:
    de.set_profile(shoot)
    table = SyncTable.load(de.LABELS / "sync.csv")
    truth = de.load_truth()
    names = [p for p in de.pairs(pattern) if (de.OUT / p / ".done").exists()]
    multi = [p for p in names if (truth.get(p, {}).get("people", 2) >= 2) != solo]
    rows = []
    for pair in multi:
        rows += pair_rows(pair, table.offset_ms(pair) or 0.0)

    print(f"\n== {shoot}차 — 기록 {de.OUT.name}, {'한 명만 나오는' if solo else '두 명 이상인'} 영상 {len(multi)}편 (전체 {len(names)}편)")
    print(f"{'헷갈린 카메라':<10}{'구간':>5}{'시간':>8}   " + "   ".join(f"다른 카메라 {k}" for k in KINDS))
    for cam, label in CAMS.items():
        mine = [r for r in rows if r["cam"] == cam]
        total = sum(r["end"] - r["start"] for r in mine) / 1000.0
        cells = []
        for kind in KINDS:
            part = [r for r in mine if r["other"] == kind]
            cells.append(f"{len(part):>4}번 {sum(r['end'] - r['start'] for r in part) / 1000.0:>6.1f}초")
        print(f"{label:<12}{len(mine):>5}번{total:>7.1f}초   " + "   ".join(cells))
    by_pair = collections.Counter(r["pair"] for r in rows if r["other"] == "갈라 봄")
    print("한쪽만 헷갈린 구간이 있는 영상:", len(by_pair), "편 —", ", ".join(f"{p} {n}" for p, n in by_pair.most_common(12)))
    none = [p for p in multi if p not in {r["pair"] for r in rows}]
    print("헷갈린 구간이 없는 영상:", len(none), "편")
    if table_too:
        for line in pair_table(rows, multi, truth):
            print(line)
    if show:
        for r in rows:
            print(f"   {r['pair']} {CAMS[r['cam']]} 카메라 {r['start'] / 1000:.2f}~{r['end'] / 1000:.2f}초 — 다른 카메라: {r['other']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shoot", type=int, choices=(2, 3, 4))
    ap.add_argument("--pairs", help="짝 이름 (쉼표, * 가능)")
    ap.add_argument("--list", action="store_true", help="구간을 하나씩 찍는다")
    ap.add_argument("--by-pair", action="store_true", help="영상마다 한 줄로 찍는다")
    ap.add_argument("--solo", action="store_true", help="한 명만 나오는 영상으로 센다 — 여기서 잡히는 구간은 잘못 센 것")
    args = ap.parse_args()
    for shoot in ([args.shoot] if args.shoot else [2, 3, 4]):
        run(shoot, args.pairs, args.list, args.solo, args.by_pair)


if __name__ == "__main__":
    main()
