"""3차 정답(takes.csv) 시각을 angle1 → angle0 영상 시각으로 옮긴다 (일회성).

정답 도구(label_takes.py)의 기준 화면을 angle1 에서 angle0(선반 정면)으로 바꾸기 전에 찍은 줄을 옮긴다.
angle0 시각 = angle1 시각 - offset_ms/1000 (labels/store01_shoot3/sync.csv, angle1 = angle0 + offset).

    python scripts/migrate_takes_axis.py

두 번 빼지 않게:
  - 처음 실행 때 takes.csv 를 takes_angle1_backup.csv 로 복사한다 (백업이 있으면 덮지 않음).
  - 결과는 늘 백업에서 다시 계산한다.
  - 지금 takes.csv 가 백업과 같으면 → 옮겨 쓴다. 예상 결과와 같으면 → 이미 옮김, 아무것도 안 함.
    둘 다 아니면(그 사이 손으로 고침) → 멈춘다. 손으로 확인할 것.
"""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "labels" / "store01_shoot3"


def read_rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def read_sync(path: Path) -> dict[str, float]:
    _, rows = read_rows(path)
    return {r["pair"]: float(r["offset_ms"]) for r in rows if (r.get("offset_ms") or "").strip()}


def convert(rows: list[dict], sync: dict[str, float]) -> list[dict]:
    """시각이 있는 줄만 옮긴다 (none 줄은 그대로). 시차 없는 짝이 있으면 ValueError."""
    missing = sorted({r["pair"] for r in rows if (r.get("time_sec") or "").strip() and r["pair"] not in sync})
    if missing:
        raise ValueError(f"시차 없는 짝: {', '.join(missing)}")
    out = []
    for r in rows:
        t = (r.get("time_sec") or "").strip()
        out.append({**r, "time_sec": f"{float(t) - sync[r['pair']] / 1000.0:.2f}"} if t else dict(r))
    return out


def migrate(takes: Path, backup: Path, sync_path: Path) -> str:
    """돌려주는 값: 'moved' / 'already' / 'nothing'. 손댄 파일이면 SystemExit."""
    if not takes.exists():
        return "nothing"
    if not backup.exists():
        shutil.copyfile(takes, backup)
    fields, base = read_rows(backup)
    expected = convert(base, read_sync(sync_path))
    _, current = read_rows(takes)
    if current == expected:
        return "already"
    if current != base:
        raise SystemExit(f"{takes.name} 가 백업({backup.name})과도, 옮긴 결과와도 다르다 — 그 사이 고친 줄이 있다. 손으로 확인.")
    tmp = takes.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(expected)
    tmp.replace(takes)
    return "moved"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    takes, backup, sync = LABELS / "takes.csv", LABELS / "takes_angle1_backup.csv", LABELS / "sync.csv"
    try:
        result = migrate(takes, backup, sync)
    except ValueError as e:
        print(f"멈춤: {e}")
        return 1
    print({"moved": f"옮김: {takes} (백업 {backup.name})", "already": "이미 angle0 시각 — 바꾸지 않음",
           "nothing": "takes.csv 없음"}[result])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
