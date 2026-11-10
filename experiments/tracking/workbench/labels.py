"""정답 적기 — 영상을 보며 사람이 적은 정답을 읽고 쓴다.

정답표(truth.csv)에는 손님 수와 '경보가 나야 하나'만 있어, 두 화면 짝과 집은 개수는 눈으로만 볼 수 있었다 (한계 A-13·A-14).
여기서 짝마다 다음을 적는다 — labels/store01_dual/events/<짝>.json (커밋한다).

    takes     집음·놓음: 시각, 종류(take/put), 개수, 칸(선택), 누가(who)
    culprits  경보가 나야 하는 사람: 시각, 누가
    links     두 화면 같은 사람: 시각, 선반 카메라의 그 사람(detail), 전체 카메라의 그 사람(overview)
    takes_done  이 영상의 집음·놓음을 빠짐없이 적었나 — 참이어야 개수를 채점한다 (하나도 없는 것도 정답일 수 있어서)

'누가'는 프로그램의 손님 번호가 아니라 **그 시각 화면 위의 한 점**(0~1 비율)으로 적는다. 정답이 프로그램의 판단에 기대지 않게 하려는 것이고,
채점할 때 그 점을 품은 사람 상자로 프로그램의 손님 번호를 찾는다 (workbench/analyze.py locate).
시각은 전체 카메라 재생 시각(초) = 공통 시간축.

결제 정답은 예전부터 있는 labels/store01_dual/payments.csv (scripts/label_payments.py 의 형식) 그대로 읽고 쓴다.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "labels" / "store01_dual"
EVENTS = LABELS / "events"
PAYMENTS = LABELS / "payments.csv"
PAY_RESULTS = ("paid", "failed", "incomplete", "cancelled", "none")
LIMIT = 200


def empty(pair: str) -> dict:
    return {"pair": pair, "checked_by": "", "updated": "", "takes": [], "culprits": [], "links": [], "takes_done": False}


def path_of(pair: str, folder: Path | None = None) -> Path:
    if not pair or any(c in pair for c in "/\\.:") or len(pair) > 40:
        raise ValueError(f"짝 이름이 이상합니다: {pair!r}")
    return (folder or EVENTS) / f"{pair}.json"


def load(pair: str, folder: Path | None = None) -> dict:
    path = path_of(pair, folder)
    if not path.exists():
        return empty(pair)
    return {**empty(pair), **json.loads(path.read_text(encoding="utf-8"))}


def has_any(labels: dict) -> bool:
    return bool(labels["takes"] or labels["culprits"] or labels["links"] or labels["takes_done"])


def _time(value) -> float:
    try:
        t = round(float(value), 2)
    except (TypeError, ValueError):
        raise ValueError("시각이 숫자가 아닙니다") from None
    if not 0 <= t <= 36000:
        raise ValueError("시각이 범위를 벗어났습니다")
    return t


def _point(value) -> list[float]:
    try:
        x, y = (round(float(v), 4) for v in value)
    except (TypeError, ValueError):
        raise ValueError("화면 위 점이 [x, y] 가 아닙니다") from None
    if not (0 <= x <= 1 and 0 <= y <= 1):
        raise ValueError("화면 위 점이 0~1 을 벗어났습니다")
    return [x, y]


def _text(value, limit: int) -> str:
    return str(value or "").replace("\r", " ").replace("\n", " ").strip()[:limit]


def clean(pair: str, raw: dict) -> dict:
    """화면에서 온 정답을 확인해 저장할 모양으로. 틀린 값이면 ValueError."""
    if not isinstance(raw, dict):
        raise ValueError("정답이 올바른 모양이 아닙니다")
    out = empty(pair)
    out["checked_by"] = _text(raw.get("checked_by"), 40)
    out["takes_done"] = bool(raw.get("takes_done"))
    takes, culprits, links = raw.get("takes") or [], raw.get("culprits") or [], raw.get("links") or []
    if max(len(takes), len(culprits), len(links)) > LIMIT:
        raise ValueError("항목이 너무 많습니다")
    for t in takes:
        kind = t.get("kind")
        if kind not in ("take", "put"):
            raise ValueError("종류는 take(집음) 또는 put(놓음)")
        count = int(t.get("count") or 1)
        if not 1 <= count <= 20:
            raise ValueError("개수는 1~20")
        out["takes"].append({"time_sec": _time(t.get("time_sec")), "kind": kind, "count": count,
                             "slot": _text(t.get("slot"), 30), "who": _point(t.get("who")), "note": _text(t.get("note"), 200)})
    for c in culprits:
        out["culprits"].append({"time_sec": _time(c.get("time_sec")), "who": _point(c.get("who")), "note": _text(c.get("note"), 200)})
    for link in links:
        out["links"].append({"time_sec": _time(link.get("time_sec")), "detail": _point(link.get("detail")),
                             "overview": _point(link.get("overview")), "note": _text(link.get("note"), 200)})
    for key in ("takes", "culprits", "links"):
        out[key].sort(key=lambda r: r["time_sec"])
    return out


def save(pair: str, raw: dict, folder: Path | None = None) -> dict:
    """정답을 저장한다 → 저장한 것. 아무것도 없으면 파일을 지운다 (정답 없는 영상으로 돌아간다)."""
    labels = clean(pair, raw)
    path = path_of(pair, folder)
    if not has_any(labels):
        if path.exists():
            path.unlink()
        return labels
    labels["updated"] = time.strftime("%Y-%m-%d %H:%M")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(labels), encoding="utf-8")
    return labels


def dumps(labels: dict) -> str:
    """항목 하나가 한 줄 — 커밋했을 때 무엇이 바뀌었는지 줄 단위로 보이게."""
    lines = ["{"]
    for key in ("pair", "checked_by", "updated", "takes_done"):
        lines.append(f" {json.dumps(key)}: {json.dumps(labels[key], ensure_ascii=False)},")
    lists = ("takes", "culprits", "links")
    for n, key in enumerate(lists):
        items = [f"  {json.dumps(item, ensure_ascii=False)}" for item in labels[key]]
        body = ("\n" + ",\n".join(items) + "\n ") if items else ""
        lines.append(f" {json.dumps(key)}: [{body}]" + ("," if n < len(lists) - 1 else ""))
    lines.append("}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------- 결제 정답 (payments.csv)

def load_payments(pair: str, path: Path | None = None) -> list[dict]:
    from label_payments import load_labels
    rows = load_labels(path or PAYMENTS).get(pair, [])
    return [{"time_sec": r.time_sec, "items": r.items, "result": r.result, "checked_by": r.checked_by, "note": r.note} for r in rows]


def clean_payments(raw: list) -> list[dict]:
    if not isinstance(raw, list) or len(raw) > 50:
        raise ValueError("결제 정답이 올바른 모양이 아닙니다")
    out = []
    for r in raw:
        result = r.get("result")
        if result not in PAY_RESULTS:
            raise ValueError(f"결제 결과는 {', '.join(PAY_RESULTS)} 중 하나")
        items = int(r.get("items") or 0)
        if not 0 <= items <= 50:
            raise ValueError("결제 품목 수는 0~50")
        out.append({"time_sec": None if result == "none" else _time(r.get("time_sec")), "items": 0 if result == "none" else items,
                    "result": result, "checked_by": _text(r.get("checked_by"), 40), "note": _text(r.get("note"), 200)})
    if any(r["result"] == "none" for r in out) and len(out) > 1:
        raise ValueError("'결제 시도 없음'은 다른 결제와 함께 둘 수 없습니다")
    return out


def save_payments(pair: str, raw: list, path: Path | None = None) -> bool:
    """그 짝의 결제 정답을 바꿔 쓴다 → 파일이 바뀌었나. 다른 짝의 줄은 그대로 둔다."""
    from label_payments import PaymentLabel, load_labels, renumber, save_labels
    path_of(pair)
    path = path or PAYMENTS
    rows = clean_payments(raw)
    before = load_payments(pair, path)
    key = [(r["time_sec"], r["items"], r["result"], r["note"]) for r in rows]
    if key == [(r["time_sec"], r["items"], r["result"], r["note"]) for r in before]:
        return False
    everything = load_labels(path)
    new = renumber([PaymentLabel(pair, 0, r["time_sec"], r["items"], r["result"], r["checked_by"], r["note"]) for r in rows])
    if new:
        everything[pair] = new
    else:
        everything.pop(pair, None)
    save_labels(everything, path)
    return True
