"""정답표(truth.csv)를 영상으로 확인하는 표 — 시나리오 표로 만든 정답·인원을 사람이 영상을 보고 확정한다.

    python scripts/truth_check.py sheet          # 확인표 만들기 → labels/store01_shoot4/truth_check.csv (엑셀로 열어 채운다)
    python scripts/truth_check.py status         # 몇 편 확인했는지
    python scripts/truth_check.py apply          # 확인표 ↔ 정답표 차이만 보여준다 (아무것도 쓰지 않음)
    python scripts/truth_check.py apply --write  # 확인한 줄을 정답표에 반영

확인표 열
    pair · scenario · plan_people · plan_expected   시나리오 표 기준 (지금 정답표 값)
    pos_paid                                        그 영상 안의 실제 결제 기록 (영상 시각, 개수, 상품)
    seen_people      영상에 나온 손님 수 (촬영자·행인처럼 손님이 아닌 사람은 빼고)
    seen_expected    HIGH_RISK = 결제 안 한 물건을 한 개라도 들고 나갔다 / CLEAR = 아니다
    note             시나리오와 달랐던 점, 누가 절도였는지 (예: 'L1: 두 번째로 들어온 사람이 절도')
    checked_by       확인한 사람

**판정 결과는 일부러 넣지 않았다.** 정답은 영상만 보고 정한다 — 시스템이 낸 판정을 보고 고치면 4차가 시험 세트가 아니게 된다
(docs/shoot4-demo-plan.md §3). 그래서 확인표에는 시나리오와 결제 기록(사실)만 있다.

sheet 를 다시 돌려도 이미 채운 칸(seen_*, note, checked_by)과 plan_* (처음 만들 때의 시나리오 표 값)는 그대로 둔다 —
정답표에 반영한 뒤에도 '시나리오 표와 어디가 달랐는지'가 남는다. seen_people·seen_expected·checked_by 가 모두 찬 줄만
'확인함'으로 보고 반영하고, 일부만 채운 줄은 문제로 알려 반영하지 않는다. 정답표의 set 열(시나리오 분류)은 바꾸지 않는다.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from label_payments import load_labels as load_payments  # noqa: E402

LABELS = ROOT / "labels" / "store01_shoot4"
TRUTH_FIELDS = ("pair", "set", "people", "expected", "notes")
SHEET_FIELDS = ("pair", "scenario", "plan_people", "plan_expected", "pos_paid",
                "seen_people", "seen_expected", "note", "checked_by")
FILLED = ("seen_people", "seen_expected", "note", "checked_by")
"""사람이 채우는 칸 — sheet 를 다시 만들어도 보존한다."""
REQUIRED = ("seen_people", "seen_expected", "checked_by")
VERDICTS = ("CLEAR", "HIGH_RISK")
SCENARIO = re.compile(r"시나리오:\s*(.*?)\.\s*(?:정답·인원은|영상 확인)")
PRODUCTS = re.compile(r"\)\s*(.+)$")


def read_csv(path: Path) -> list[dict]:
    """엑셀이 다시 저장한 파일도 읽는다 — UTF-8(BOM 있든 없든) 먼저, 안 되면 한글 윈도우 기본(cp949)."""
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "cp949"):
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        return [{k: (v or "").strip() for k, v in row.items() if k} for row in csv.DictReader(io.StringIO(text, newline=""))]
    raise SystemExit(f"읽을 수 없는 글자 인코딩: {path}")


def line_ending(path: Path) -> str:
    """그 파일이 쓰던 줄바꿈 — 정답표를 고쳐 쓸 때 안 바뀐 줄까지 통째로 달라지지 않게 그대로 따른다."""
    return "\r\n" if path.exists() and b"\r\n" in path.read_bytes() else "\n"


def write_csv(path: Path, fields, rows: list[dict], encoding: str, newline: str = "\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding=encoding, newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator=newline)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in fields} for r in rows)


def scenario_of(notes: str) -> str:
    m = SCENARIO.search(notes)
    return m.group(1) if m else notes


def paid_summary(payments: list) -> str:
    """그 영상의 결제 기록 한 줄 — '13초 1개 (동과루×1) / 29초 1개 (칸타타×1)'. 결제 없으면 '결제 없음', 라벨이 없으면 빈칸."""
    paid = [p for p in payments if p.result == "paid" and p.time_sec is not None]
    if not paid:
        return "결제 없음" if payments else ""
    parts = []
    for p in sorted(paid, key=lambda p: p.time_sec):
        m = PRODUCTS.search(p.note)
        parts.append(f"{p.time_sec:.0f}초 {p.items}개" + (f" ({m.group(1)})" if m else ""))
    return " / ".join(parts)


def table_problems(sheet: list[dict], known) -> list[str]:
    """확인표 전체의 문제 — 같은 짝이 두 줄, 정답표에 없는 짝, 짝 이름 없이 입력만 있는 줄 (엑셀에서 줄을 복사·밀어 넣다 생긴다).

    이런 표는 어느 줄이 맞는지 알 수 없다: 다시 만들지도(채운 칸이 사라진다) 정답표에 반영하지도 않는다."""
    counts = collections.Counter(r.get("pair", "") for r in sheet if r.get("pair"))
    out = [f"{p}: 확인표에 {n}줄 — 한 줄만 남긴다" for p, n in counts.items() if n > 1]
    out += [f"{p}: 정답표에 없는 짝" for p in counts if p not in known]
    nameless = sum(1 for r in sheet if not r.get("pair") and any(r.get(k) for k in FILLED))
    if nameless:
        out.append(f"짝 이름이 빈 줄 {nameless}개에 입력이 있음")
    return out


def sheet_rows(truth: list[dict], payments: dict, old: list[dict]) -> list[dict]:
    """정답표 순서대로 확인표 줄. 이미 채운 칸과 plan_* (처음 만들 때의 시나리오 표 값)는 old 에서 가져온다."""
    kept = {r["pair"]: r for r in old if r.get("pair")}
    rows = []
    for t in truth:
        prev = kept.get(t["pair"], {})
        rows.append({"pair": t["pair"], "scenario": scenario_of(t["notes"]),
                     "plan_people": prev.get("plan_people") or t["people"],
                     "plan_expected": prev.get("plan_expected") or t["expected"],
                     "pos_paid": paid_summary(payments.get(t["pair"], [])),
                     **{k: prev.get(k, "") for k in FILLED}})
    return rows


def is_checked(row: dict) -> bool:
    return all(row.get(k) for k in REQUIRED)


def row_problems(row: dict) -> list[str]:
    """확인표 한 줄의 문제 (없으면 빈 목록). 아무것도 안 채운 줄은 문제가 아니다 — 아직 안 본 영상."""
    filled = [k for k in REQUIRED if row.get(k)]
    if not filled and not row.get("note"):
        return []
    out = [f"{k} 비어 있음" for k in REQUIRED if not row.get(k)]
    if row.get("seen_expected") and row["seen_expected"].upper() not in VERDICTS:
        out.append(f"seen_expected '{row['seen_expected']}' — {' 또는 '.join(VERDICTS)}")
    if row.get("seen_people") and not (row["seen_people"].isdigit() and int(row["seen_people"]) >= 1):
        out.append(f"seen_people '{row['seen_people']}' — 1 이상의 숫자")
    return out


def apply(truth: list[dict], sheet: list[dict], today: str) -> tuple[list[dict], list[str], list[str]]:
    """확인한 줄을 정답표에 반영한 새 줄들, 시나리오 표와 다른 점 목록, 문제 목록. 문제가 있는 줄·안 본 줄은 그대로 둔다.

    '시나리오 표와 다름'은 확인표의 plan_* (없으면 지금 정답표 값)과 견준다 — 반영한 뒤 다시 돌려도 같은 결과가 나온다."""
    by_pair = {r["pair"]: r for r in sheet if r.get("pair")}
    broken = table_problems(sheet, {t["pair"] for t in truth})
    changes, problems = [], list(broken)
    out = []
    for t in truth:
        row = by_pair.get(t["pair"])
        bad = row_problems(row) if row else []
        if bad:
            problems.append(f"{t['pair']}: " + ", ".join(bad))
        if not row or bad or not is_checked(row):
            out.append(dict(t))
            continue
        people, expected = str(int(row["seen_people"])), row["seen_expected"].upper()
        plan_expected, plan_people = row.get("plan_expected") or t["expected"], row.get("plan_people") or str(t["people"])
        differs = []
        if expected != plan_expected:
            differs.append(f"정답 {plan_expected}→{expected}")
        if people != plan_people:
            differs.append(f"인원 {plan_people}→{people}")
        notes = f"시나리오: {scenario_of(t['notes'])}. 영상 확인 ({row['checked_by']}, {today})"
        if differs:
            notes += " — 시나리오 표와 다름: " + ", ".join(differs)
            changes.append(f"{t['pair']}: " + ", ".join(differs) + (f"  ({row['note']})" if row.get("note") else ""))
        if row.get("note"):
            notes += f" · {row['note']}"
        out.append({**t, "people": people, "expected": expected, "notes": notes})
    if broken:   # 표 자체가 어긋났다(같은 짝 두 줄 등) — 어느 줄이 맞는지 모르니 하나도 반영하지 않는다
        return [dict(t) for t in truth], [], problems
    return out, changes, problems


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["sheet", "status", "apply"])
    ap.add_argument("--labels", type=Path, default=LABELS, help="라벨 폴더 (기본 labels/store01_shoot4)")
    ap.add_argument("--write", action="store_true", help="apply: 정답표에 실제로 쓴다 (없으면 차이만 출력)")
    args = ap.parse_args(argv)
    truth_path, sheet_path = args.labels / "truth.csv", args.labels / "truth_check.csv"
    if not truth_path.exists():
        raise SystemExit(f"정답표 없음: {truth_path}")
    truth = read_csv(truth_path)

    if args.stage == "sheet":
        old = read_csv(sheet_path) if sheet_path.exists() else []
        broken = table_problems(old, {t["pair"] for t in truth})
        if broken:
            raise SystemExit("확인표를 먼저 고친다 — 다시 만들면 채운 칸이 사라질 수 있어 그대로 뒀다: " + " / ".join(broken))
        rows = sheet_rows(truth, load_payments(args.labels / "payments.csv"), old)
        write_csv(sheet_path, SHEET_FIELDS, rows, "utf-8-sig")      # BOM — 엑셀이 한글을 바로 읽는다
        print(f"확인표 {len(rows)}줄 (이미 확인한 줄 {sum(is_checked(r) for r in rows)}) → {sheet_path}")
        return 0
    if not sheet_path.exists():
        raise SystemExit(f"확인표 없음: {sheet_path} — 먼저 python scripts/truth_check.py sheet")
    sheet = read_csv(sheet_path)
    new, changes, problems = apply(truth, sheet, dt.date.today().isoformat())
    checked = [r["pair"] for r in sheet if r.get("pair") and is_checked(r) and not row_problems(r)]
    left = [t["pair"] for t in truth if t["pair"] not in checked]
    print(f"확인 {len(checked)}/{len(truth)}" + (f"  |  남은 짝: {', '.join(left)}" if left else "  |  모두 확인했다"))
    for line in problems:
        print(f"  [문제] {line}")
    if args.stage == "status":
        return 1 if problems else 0
    print(f"시나리오 표와 다른 짝 {len(changes)}편" + (":" if changes else ""))
    for line in changes:
        print(f"  {line}")
    moved = [n["pair"] for n, t in zip(new, truth) if (n["people"], n["expected"]) != (t["people"], t["expected"])]
    print(f"정답표에서 정답·인원이 바뀌는 짝 {len(moved)}편" + (f": {', '.join(moved)}" if moved else ""))
    if not args.write:
        print("(차이만 봤다 — 반영하려면 --write)")
        return 1 if problems else 0
    if problems:
        raise SystemExit("문제가 있는 줄을 고친 뒤 다시 — 아무것도 쓰지 않았다")
    write_csv(truth_path, TRUTH_FIELDS, new, "utf-8", line_ending(truth_path))
    print(f"정답표에 반영: {truth_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
