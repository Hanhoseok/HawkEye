"""MERL Shopping 라벨(.mat)을 우리 평가 형식(CSV)으로 바꾼다.

우리가 손으로 찍은 라벨은 65초 영상 하나에서 33건뿐이었다.
MERL 은 106개 영상 / 41명 / 5,377건이고, subject 단위로 train/val/test 가 나뉘어 있다.
같은 데이터로 튜닝하고 평가하던 문제가 여기서 해소된다.

## 행동 대응

MERL 은 5개 클래스를 쓴다.

| MERL | 평균 | 우리 개념 |
|---|---|---|
| 1 Reach To Shelf      | 1.2초 | **TAKE** (선반으로 손을 뻗는 순간) |
| 2 Retract From Shelf  | 1.2초 | TAKE 의 끝 |
| 3 Hand In Shelf       | 2.1초 | 집는 중 |
| 4 Inspect Product     | 3.9초 | 집은 뒤 (물건 들고 봄) |
| 5 Inspect Shelf       | 4.3초 | **BROWSE** (보기만, 안 만짐) |

기본 대응은 `TAKE <- Reach To Shelf`, `BROWSE <- Inspect Shelf` 다.

**주의**: MERL 도 "실제로 물건을 가져갔는지"는 라벨하지 않는다.
"선반으로 손을 뻗었다"까지다. 우리 TakeCandidate 와 같은 한계이므로 대응이 자연스럽다.

`--take-span reach` (기본) 은 손 뻗는 순간만 TAKE 로 본다.
`--take-span interaction` 은 뻗기~빼기를 하나의 상호작용 구간으로 합친다.

## person_id 문제

우리 평가는 person 단위로 대조한다. MERL 영상은 쇼핑객이 한 명이므로,
파이프라인이 그 사람에게 붙인 번호를 `--person` 으로 지정하거나,
`--person auto` 로 관측이 가장 많은 사람에게 자동 배정한다.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REACH, RETRACT, HAND_IN, INSPECT_PRODUCT, INSPECT_SHELF = 0, 1, 2, 3, 4
FPS = 30.0


def load_tlabs(path: Path) -> list[list[tuple[int, int]]]:
    """MERL .mat 에서 5개 클래스의 (시작, 끝) 프레임 목록을 꺼낸다."""
    from scipy.io import loadmat

    raw = loadmat(path)["tlabs"]
    out: list[list[tuple[int, int]]] = []
    for i in range(5):
        cell = raw[i][0] if raw.shape[0] == 5 else raw[0][i]
        if cell is None or len(cell) == 0:
            out.append([])
            continue
        out.append([(int(s), int(e)) for s, e in cell])
    return out


def merge_interactions(
    reach: list[tuple[int, int]],
    hand_in: list[tuple[int, int]],
    retract: list[tuple[int, int]],
    max_gap: int = 30,
) -> list[tuple[int, int]]:
    """뻗기 -> (손 넣기) -> 빼기 를 하나의 상호작용 구간으로 합친다."""
    spans = sorted(reach + hand_in + retract)
    merged: list[list[int]] = []
    for s, e in spans:
        if merged and s - merged[-1][1] <= max_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def convert(mat_path: Path, person_id: int, take_span: str) -> list[dict]:
    tlabs = load_tlabs(mat_path)
    rows: list[dict] = []

    if take_span == "interaction":
        takes = merge_interactions(tlabs[REACH], tlabs[HAND_IN], tlabs[RETRACT])
    else:
        takes = tlabs[REACH]

    for action, spans in (("TAKE", takes), ("BROWSE", tlabs[INSPECT_SHELF])):
        for s, e in spans:
            rows.append(
                {
                    "person_id": person_id,
                    "action": action,
                    "start_frame": s,
                    "end_frame": e,
                    "start_sec": round(s / FPS, 3),
                    "end_sec": round(e / FPS, 3),
                }
            )
    rows.sort(key=lambda r: r["start_frame"])
    return rows


def dominant_person(identities: Path) -> int:
    from src.sinks.observation_log import read_identities

    counter = Counter(o.person_id for o in read_identities(identities) if o.person_id > 0)
    if not counter:
        raise SystemExit(f"{identities} 에 신원이 없습니다")
    return counter.most_common(1)[0][0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mat", help="xx_yy_label.mat")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--person",
        default="1",
        help="라벨에 붙일 person_id. 'auto' 면 --identities 에서 가장 관측이 많은 사람",
    )
    parser.add_argument("--identities", help="--person auto 일 때 참조할 identities.jsonl")
    parser.add_argument(
        "--take-span",
        choices=["reach", "interaction"],
        default="reach",
        help="reach=손 뻗는 순간만 / interaction=뻗기~빼기 전체",
    )
    args = parser.parse_args()

    if args.person == "auto":
        if not args.identities:
            raise SystemExit("--person auto 는 --identities 가 필요합니다")
        person_id = dominant_person(Path(args.identities))
        print(f"person_id 자동 지정: P{person_id}")
    else:
        person_id = int(args.person)

    rows = convert(Path(args.mat), person_id, args.take_span)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["person_id", "action", "start_frame", "end_frame", "start_sec", "end_sec"]
        )
        writer.writeheader()
        writer.writerows(rows)

    takes = sum(1 for r in rows if r["action"] == "TAKE")
    browses = len(rows) - takes
    print(f"저장: {out}  TAKE {takes}건 / BROWSE {browses}건")


if __name__ == "__main__":
    main()
