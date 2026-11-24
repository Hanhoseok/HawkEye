"""추적 도구가 '지금 설정으로 다시 낸' 판정을 보관함에 넣을 JSON 으로 꺼낸다 (표준 라이브러리만).

판정은 experiments/tracking/workbench/data.py 가 만든다 (저장된 추적 기록을 영상 없이 다시 판정 — 짝당 1초 안팎):

    cd experiments/tracking && python workbench/data.py --shoot 4      # → outputs/workbench/shoot4/pairs/<짝>.json

    python tools/export_verdicts.py --tracking experiments/tracking --shoot 4 <verdicts-4.json>

먼저 추적 도구 자신의 '다시 만들어야 할 짝' 확인(workbench.data.Builder.stale — 설정·기록·정답표 서명 비교)을 돌린다.
하나라도 최신이 아니면(다시 판정하다 실패해 예전 결과가 남았거나 아직 안 만듦) 내보내지 않고 그 짝을 알린다
(--allow-stale 이면 그 짝만 빼고 내보낸다). 추적 도구 코드는 읽기만 한다 (실행에 workbench 의존성이 필요할 수 있음).

출력: {"source": "...", "scenes": {"C01": [{"ms", "person", "level", "zone", "reason", "taken", "paid", "unpaid"}]}}
ms = 공통 시각 = 전체 카메라(cam1) 재생 시각. 장면 이름은 보관함과 같은 scene_key.
보관함에 넣기: python -m hawkeye_server.library import --folder "4차 촬영" --verdicts <verdicts-4.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from hawkeye_server.library import scene_key  # noqa: E402

FIELDS = ("ms", "person", "level", "zone", "reason", "taken", "paid", "unpaid")


def verdicts_of(pair: dict[str, Any]) -> list[dict[str, Any]]:
    """짝 JSON 의 events → 보관함 판정 목록 (시각 순)."""
    out = []
    for e in pair.get("events") or []:
        out.append({"ms": round(float(e["ms"]), 1), "person": int(e["pid"]), "level": str(e["level"]),
                    "zone": str(e.get("zone") or ""), "reason": str(e.get("reason") or ""),
                    "taken": int(e.get("taken") or 0), "paid": int(e.get("paid") or 0),
                    "unpaid": int(e.get("unpaid") or 0)})
    return sorted(out, key=lambda v: (v["ms"], v["person"]))


def export(workbench: Path, skip: set[str] | None = None, only: set[str] | None = None) -> dict[str, Any]:
    """짝 판정 폴더 → 보관함 판정. skip(짝 이름) 은 빼고 'skipped' 에 적는다.
    only 가 있으면 그 짝만 (최신 확인을 통과한 짝) — 나머지 판정 파일(원본 기록이 없어진 짝 등)은 'skipped' 에 적는다."""
    index = json.loads((workbench / "index.json").read_text(encoding="utf-8")) if (workbench / "index.json").is_file() else {}
    built = str(index.get("built") or "")
    scenes: dict[str, list[dict[str, Any]]] = {}
    skipped = []
    for p in sorted((workbench / "pairs").glob("*.json")):
        if (skip and p.stem in skip) or (only is not None and p.stem not in only):
            skipped.append(p.stem)
            continue
        pair = json.loads(p.read_text(encoding="utf-8"))
        scenes[scene_key(str(pair.get("pair") or p.stem))] = verdicts_of(pair)
    source = "추적 도구가 지금 설정으로 다시 낸 판정" + (f" · {built}" if built else "")
    out: dict[str, Any] = {"source": source, "scenes": scenes}
    if skipped:
        out["skipped"] = skipped
    return out


def freshness(tracking: Path, shoot: int) -> tuple[Path, list[str], list[str]]:
    """추적 도구의 판정 폴더, 확인한 짝(지금 기록이 있는 짝 — Builder.pairs), 그중 최신이 아닌 짝(Builder.stale —
    서명이 다르거나 결과가 없음). 추적 코드는 읽기만."""
    sys.path.insert(0, str(tracking.resolve()))
    from workbench.data import Builder  # noqa: PLC0415 — 추적 도구 코드

    b = Builder(shoot)
    return Path(b.out), sorted(b.pairs()), sorted(b.stale())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="추적 도구 판정 → 보관함 판정 JSON")
    ap.add_argument("out")
    ap.add_argument("--tracking", help="experiments/tracking 폴더 — 최신 확인 후 그 회차 판정 폴더를 읽는다")
    ap.add_argument("--shoot", type=int, help="회차 (2 · 3 · 4, --tracking 과 함께)")
    ap.add_argument("--workbench", help="판정 폴더를 직접 (최신 확인 없이 — --no-check 와 함께)")
    ap.add_argument("--no-check", action="store_true", help="최신 확인을 건너뛴다 (시험용)")
    ap.add_argument("--allow-stale", action="store_true", help="최신이 아닌 짝은 빼고 내보낸다")
    a = ap.parse_args(argv)
    stale: list[str] = []
    checked: set[str] | None = None
    if a.tracking:
        if a.shoot is None:
            ap.error("--tracking 에는 --shoot 이 필요합니다")
        wb, pairs, stale = freshness(Path(a.tracking), a.shoot)
        checked = set(pairs)
    elif a.workbench and a.no_check:
        wb = Path(a.workbench)
    else:
        ap.error("--tracking DIR --shoot N (또는 시험용 --workbench DIR --no-check)")
    if not (wb / "pairs").is_dir():
        ap.error(f"짝 판정 폴더가 없습니다: {wb / 'pairs'} (먼저 workbench/data.py 를 돌리세요)")
    if stale:
        print(f"[export_verdicts] 최신이 아닌 짝 {len(stale)}개: {', '.join(stale)}", file=sys.stderr)
        if not a.allow_stale:
            print("  → workbench/data.py --shoot N 으로 다시 판정한 뒤 실행하세요 (빼고 내보내려면 --allow-stale)", file=sys.stderr)
            return 1
    data = export(wb, set(stale), checked)
    orphans = [s for s in data.get("skipped", []) if s not in stale]
    if orphans:      # 판정 파일은 있는데 지금 기록이 없는 짝 — 확인할 수 없어 내보내지 않음
        print(f"[export_verdicts] 기록이 없어 확인 못 한 판정 {len(orphans)}개는 뺌: {', '.join(orphans)}", file=sys.stderr)
    Path(a.out).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    n = sum(len(v) for v in data["scenes"].values())
    print(f"[export_verdicts] 장면 {len(data['scenes'])}, 판정 {n} → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
