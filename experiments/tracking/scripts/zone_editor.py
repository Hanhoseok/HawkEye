"""구역 편집 페이지(scripts/zone_editor.html)의 서버 쪽 — 초안 저장·검사·판정 돌리기·적용.

`review_serve.py` 가 /api/zones* · /api/judge 요청을 여기로 넘긴다. 내 컴퓨터(127.0.0.1)에서만 열린다.

    python scripts/review_serve.py
    → http://127.0.0.1:8765/experiments/tracking/scripts/zone_editor.html

흐름:  적용본(zones/store01_dual/) 불러오기 → 페이지에서 고치기 → 초안 저장(outputs/zone_editor/draft/)
      → 판정 돌리기(적용본·초안 둘 다 38쌍 다시 판정, 각 18초쯤) → 비교 → 적용(적용본을 backup/ 에 남기고 덮어씀)
초안·백업·판정 결과는 outputs/ 아래라 저장소에 올라가지 않는다. 적용한 구역 파일(zones/)만 커밋한다.

판정 다시 돌리기는 **기록(추적 결과)을 그대로 두고** 구역만 바꿔 다시 판정한다. 그래서 바닥 구역·띠는 바로 반영되지만,
탐지·추적 단계에서 쓰는 구역(사람 아님 영역, 선반 카메라의 선반 상자·칸, 전체 카메라 선반 앞 머묾으로 생기는 손 뻗기)은
영상을 다시 돌려야(`dual_eval.py run`) 반영된다. 페이지가 고친 구역마다 어느 쪽인지 알려 준다.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.types import ZoneType  # noqa: E402
from src.zones.floor import FloorMap  # noqa: E402
from src.zones.zone_map import ZoneMap  # noqa: E402

APPLIED = ROOT / "zones" / "store01_dual"
WORK = ROOT / "outputs" / "zone_editor"
DRAFT = WORK / "draft"
BACKUP = WORK / "backup"
RECORDS = "dual_eval_edgegap"
"""판정을 다시 돌릴 기록 폴더 (outputs/ 아래) — 지금 기본 설정의 기록."""
FOOT = {
    "bbox": (RECORDS, []),
    "ankle": ("dual_eval_pose", ["--set", "zones.foot_point=ankle"]),
}
"""발 점 방식 → (기록 폴더, 판정 추가 설정). 발목은 전체 카메라를 관절 모델로 돌린 기록(11-04)에만 있다 (B-15)."""
FILES = {"floor": "floor.yaml", "angle0": "angle0.yaml"}
INSIDE_ZONES = ("shelf", "counter", "aisle")
"""configs/store01_dual.yaml risk.inside_zones — 이 이름의 구역이 없으면 손님이 한 명도 안 생긴다."""
SHELF_ZONE = "shelf"
"""fusion.shelf_zone — 선반 앞 띠에 선 사람(선반 앞 한 명) 기준 구역."""

FLOOR_ORDER = ("camera", "note", "notes", "band", "calibration", "zones", "ignore_image", "shelf_image")
DETAIL_ORDER = ("note", "notes", "normalized", "shelves", "band_map", "zones")


# ---------------------------------------------------------------------- yaml 쓰기

class _Dumper(yaml.SafeDumper):
    pass


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _repr_list(d, data):
    """숫자 목록·점 목록은 한 줄로 — 사람이 읽고 git diff 로 보기 쉽게."""
    flat = all(_is_num(v) for v in data) and len(data) <= 4
    points = (data and all(isinstance(v, list) and v and all(_is_num(x) for x in v) and len(v) <= 4 for v in data)
              and len(data) <= 12)
    return d.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=bool(flat or points))


def _repr_str(d, s):
    if "\n" in s:
        return d.represent_scalar("tag:yaml.org,2002:str", s, style="|")
    return d.represent_scalar("tag:yaml.org,2002:str", s)


_Dumper.add_representer(list, _repr_list)
_Dumper.add_representer(str, _repr_str)


def dump_yaml(data: dict) -> str:
    return yaml.dump(data, Dumper=_Dumper, allow_unicode=True, sort_keys=False, width=200)


def _ordered(raw: dict, order: tuple[str, ...]) -> dict:
    out = {k: raw[k] for k in order if k in raw and raw[k] not in (None, "", [], {})}
    out.update({k: v for k, v in raw.items() if k not in out and k not in order})
    return out


def _r(v, nd):
    return round(float(v), nd)


def clean_floor(raw: dict) -> dict:
    """페이지가 보낸 floor 를 파일 형식으로 (숫자 자리수 맞춤, 순서 고정, 모르는 키는 그대로)."""
    raw = dict(raw)
    cal = raw.get("calibration") or {}
    raw["calibration"] = {"image": [[_r(x, 5), _r(y, 5)] for x, y in cal.get("image") or []],
                          "floor": [[_r(x, 1), _r(y, 1)] for x, y in cal.get("floor") or []]}
    if raw.get("band"):
        raw["band"] = {"start": _r(raw["band"]["start"], 1), "end": _r(raw["band"]["end"], 1)}
    raw["zones"] = [{"name": z["name"], "type": z.get("type", "OTHER"),
                     "polygon": [[_r(x, 1), _r(y, 1)] for x, y in z["polygon"]],
                     **{k: v for k, v in z.items() if k not in ("name", "type", "polygon")}}
                    for z in raw.get("zones") or []]
    raw["ignore_image"] = [[_r(v, 4) for v in r] for r in raw.get("ignore_image") or []]
    raw["shelf_image"] = [[[_r(x, 4), _r(y, 4)] for x, y in p] for p in raw.get("shelf_image") or []]
    return _ordered(raw, FLOOR_ORDER)


def clean_detail(raw: dict) -> dict:
    raw = dict(raw)
    raw["normalized"] = True
    shelves = []
    for s in raw.get("shelves") or []:
        s = dict(s)
        if s.get("polygon"):
            # 다각형이면 box 는 그 둘레 사각형 (box 만 읽는 코드·보기 페이지용)
            s["polygon"] = [[_r(x, 4), _r(y, 4)] for x, y in s["polygon"]]
            xs, ys = [p[0] for p in s["polygon"]], [p[1] for p in s["polygon"]]
            s["box"] = [min(xs), min(ys), max(xs), max(ys)]
        s["box"] = [_r(v, 4) for v in s["box"]]
        shelves.append({k: s[k] for k in ("name", "box", "polygon", "zone") if k in s}
                       | {k: v for k, v in s.items() if k not in ("name", "box", "polygon", "zone")})
    raw["shelves"] = shelves
    if raw.get("band_map"):
        raw["band_map"] = [_r(v, 4) for v in raw["band_map"]]
    raw["zones"] = [{"name": z["name"], "type": z.get("type", "SHELF"),
                     "polygon": [[_r(x, 4), _r(y, 4)] for x, y in z["polygon"]],
                     **{k: v for k, v in z.items() if k not in ("name", "type", "polygon")}}
                    for z in raw.get("zones") or []]
    return _ordered(raw, DETAIL_ORDER)


# ---------------------------------------------------------------------- 읽기·검사

def read_set(folder: Path) -> dict | None:
    if not all((folder / f).exists() for f in FILES.values()):
        return None
    return {k: yaml.safe_load((folder / f).read_text(encoding="utf-8")) or {} for k, f in FILES.items()}


def digest(folder: Path) -> str | None:
    if not all((folder / f).exists() for f in FILES.values()):
        return None
    h = hashlib.sha1()
    for f in FILES.values():
        h.update((folder / f).read_bytes())
    return h.hexdigest()[:12]


def validate(floor: dict, detail: dict) -> tuple[list[str], list[str]]:
    """(오류, 주의). 오류가 있으면 저장하지 않는다. 실제 읽는 코드(FloorMap.load·ZoneMap.load)로도 한 번 읽어 본다."""
    errors, warns = [], []
    names = [z.get("name") for z in floor.get("zones") or []]
    for z in floor.get("zones") or []:
        if len(z.get("polygon") or []) < 3:
            errors.append(f"바닥 구역 '{z.get('name')}' 꼭짓점이 3개 미만")
        try:
            ZoneType(z.get("type", "OTHER"))
        except ValueError:
            errors.append(f"바닥 구역 '{z.get('name')}' 종류 '{z.get('type')}' 를 모름")
    if len(set(names)) != len(names):
        errors.append("바닥 구역 이름이 겹침")
    for need in INSIDE_ZONES:
        if need not in names:
            warns.append(f"'{need}' 구역이 없음 — 매장 안 판단(risk.inside_zones)에 쓰는 이름이라 손님이 덜 잡힘")
    if SHELF_ZONE not in names:
        warns.append(f"'{SHELF_ZONE}' 구역이 없음 — '선반 앞 한 명' 주인 정하기(fusion.shelf_zone)가 꺼짐")
    types = {z.get("type") for z in floor.get("zones") or []}
    for t, what in (("EXIT", "출구 판정"), ("OUTSIDE", "문 밖으로 나감 확정"), ("CHECKOUT", "계산대 체류 = 결제")):
        if t not in types:
            warns.append(f"{t} 종류 구역이 없음 — {what}이 안 됨")
    band = floor.get("band")
    if band and not float(band["start"]) < float(band["end"]):
        errors.append("선반 앞 띠: 시작이 끝보다 작아야 함")
    for i, r in enumerate(floor.get("ignore_image") or []):
        if len(r) != 4 or not (0 <= r[0] < r[2] <= 1 and 0 <= r[1] < r[3] <= 1):
            errors.append(f"사람 아님 영역 {i + 1}: 화면 비율 x1<x2, y1<y2 (0~1) 이어야 함")
    for i, p in enumerate(floor.get("shelf_image") or []):
        if len(p) < 3:
            errors.append(f"전체 카메라 선반 영역 {i + 1}: 꼭짓점 3개 미만")
    for s in detail.get("shelves") or []:
        poly = s.get("polygon")
        if poly is not None and (len(poly) < 3 or not all(0 <= v <= 1 for p in poly for v in p)):
            errors.append(f"선반 영역 '{s.get('name')}': 꼭짓점 3개 이상, 화면 비율 0~1 이어야 함")
        b = s.get("box") or []
        if len(b) != 4 or not (0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1):
            errors.append(f"선반 상자 '{s.get('name')}': 화면 비율 x1<x2, y1<y2 (0~1) 이어야 함")
    slot_names = [z.get("name") for z in detail.get("zones") or []]
    if len(set(slot_names)) != len(slot_names):
        errors.append("선반 칸 이름이 겹침")
    for z in detail.get("zones") or []:
        if len(z.get("polygon") or []) < 3:
            errors.append(f"선반 칸 '{z.get('name')}' 꼭짓점이 3개 미만")
    if errors:
        return errors, warns
    with tempfile.TemporaryDirectory() as tmp:
        fp, dp = Path(tmp) / "floor.yaml", Path(tmp) / "angle0.yaml"
        fp.write_text(dump_yaml(floor), encoding="utf-8")
        dp.write_text(dump_yaml(detail), encoding="utf-8")
        try:
            fm = FloorMap.load(fp)
            for z in fm.zones:
                for x, y in z.polygon:
                    u, v, w = fm._h_inv @ [x, y, 1.0]
                    if w <= 0:
                        warns.append(f"바닥 구역 '{z.name}' 꼭짓점 ({x:.0f}, {y:.0f})cm 가 카메라 뒤쪽 — 화면에 못 그림")
                        break
            err = fm.reprojection_error_cm()
            if len(fm.image_points) > 4 and err > 15:
                warns.append(f"보정점 평균 오차 {err:.0f}cm — 점 하나가 틀렸을 수 있음")
        except Exception as e:  # noqa: BLE001 — 읽는 코드가 내는 오류를 그대로 보여 준다
            errors.append(f"floor.yaml 을 읽지 못함: {e}")
        try:
            ZoneMap.load(dp)
        except Exception as e:  # noqa: BLE001
            errors.append(f"angle0.yaml 을 읽지 못함: {e}")
    return errors, warns


# ---------------------------------------------------------------------- 상태·저장·적용

def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def backups() -> list[str]:
    if not BACKUP.exists():
        return []
    return sorted((p.name for p in BACKUP.iterdir() if p.is_dir()), reverse=True)[:20]


def state() -> dict:
    return {"applied": read_set(APPLIED), "draft": read_set(DRAFT),
            "applied_hash": digest(APPLIED), "draft_hash": digest(DRAFT), "backups": backups(),
            "paths": {"applied": _rel(APPLIED), "draft": _rel(DRAFT)}}


def save_draft(payload: dict) -> dict:
    floor, detail = clean_floor(payload["floor"]), clean_detail(payload["angle0"])
    errors, warns = validate(floor, detail)
    if errors:
        return {"ok": False, "errors": errors, "warnings": warns}
    DRAFT.mkdir(parents=True, exist_ok=True)
    (DRAFT / FILES["floor"]).write_text(dump_yaml(floor), encoding="utf-8")
    (DRAFT / FILES["angle0"]).write_text(dump_yaml(detail), encoding="utf-8")
    return {"ok": True, "errors": [], "warnings": warns, "draft_hash": digest(DRAFT)}


def apply_draft() -> dict:
    if digest(DRAFT) is None:
        return {"ok": False, "errors": ["초안이 없음"]}
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest, n = BACKUP / stamp, 1
    while dest.exists():
        n += 1
        dest = BACKUP / f"{stamp}-{n}"
    dest.mkdir(parents=True, exist_ok=True)
    for f in FILES.values():
        shutil.copy2(APPLIED / f, dest / f)
        shutil.copy2(DRAFT / f, APPLIED / f)
    shutil.rmtree(DRAFT)
    return {"ok": True, "backup": _rel(dest), "applied_hash": digest(APPLIED)}


def restore_backup(name: str) -> dict:
    """백업 하나를 초안으로 불러온다 (바로 덮어쓰지 않음 — 판정을 보고 적용)."""
    src = BACKUP / Path(name).name
    if not all((src / f).exists() for f in FILES.values()):
        return {"ok": False, "errors": [f"백업 없음: {name}"]}
    DRAFT.mkdir(parents=True, exist_ok=True)
    for f in FILES.values():
        shutil.copy2(src / f, DRAFT / f)
    return {"ok": True, "draft_hash": digest(DRAFT)}


def discard_draft() -> dict:
    if DRAFT.exists():
        shutil.rmtree(DRAFT)
    return {"ok": True}


# ---------------------------------------------------------------------- 판정 돌리기

class JudgeJob:
    """적용본·초안 구역으로 38쌍을 다시 판정한다 (dual_eval.py judge --zones). 구역 파일이 같으면 저장해 둔 결과를 쓴다."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = False
        self.step = ""
        self.error = ""
        self.started = 0.0
        self.foot = "bbox"

    def cache(self, h: str, foot: str = "bbox") -> Path:
        return WORK / "judge" / (f"{h}.json" if foot == "bbox" else f"{h}_{foot}.json")

    def result(self, h: str | None, foot: str = "bbox") -> dict | None:
        if not h or not self.cache(h, foot).exists():
            return None
        return json.loads(self.cache(h, foot).read_text(encoding="utf-8"))

    def start(self, foot: str = "bbox") -> dict:
        if foot not in FOOT:
            return {"ok": False, "errors": [f"모르는 발 점 방식: {foot}"]}
        with self.lock:
            if self.running:
                return {"ok": False, "errors": ["이미 돌리는 중"]}
            self.running, self.error, self.started, self.foot = True, "", time.time(), foot
        threading.Thread(target=self._run, daemon=True).start()
        return {"ok": True}

    def _run(self) -> None:
        try:
            records, extra = FOOT[self.foot]
            targets = [("적용본", APPLIED), ("초안", DRAFT)]
            for name, folder in targets:
                h = digest(folder)
                if h is None or self.cache(h, self.foot).exists():
                    continue
                self.step = f"{name} 구역으로 38쌍 판정 중 ({'발목' if self.foot == 'ankle' else '상자'} 발 점)"
                label = f"zone_{h}_{self.foot}"
                proc = subprocess.run(
                    [sys.executable, str(ROOT / "scripts" / "dual_eval.py"), "judge", "--out", records,
                     "--zones", str(folder), "--label", label, *extra],
                    cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
                    env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
                out = ROOT / "outputs" / records / f"results_{label}.json"
                if proc.returncode != 0 or not out.exists():
                    self.error = (proc.stderr or proc.stdout)[-2000:]
                    return
                rows = json.loads(out.read_text(encoding="utf-8"))
                out.unlink()
                keep = ("pair", "kind", "verdicts", "correct", "reasons", "people", "links", "payments", "cancels",
                        "unowned_takes", "actions")
                data = {"hash": h, "foot": self.foot, "records": records, "when": time.strftime("%Y-%m-%d %H:%M"),
                        "seconds": round(time.time() - self.started), "rows": [{k: r.get(k) for k in keep} for r in rows]}
                self.cache(h, self.foot).parent.mkdir(parents=True, exist_ok=True)
                self.cache(h, self.foot).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            self.error = repr(e)
        finally:
            self.step = ""
            self.running = False

    def status(self, foot: str = "bbox") -> dict:
        return {"running": self.running, "step": self.step, "error": self.error, "foot": foot,
                "elapsed": round(time.time() - self.started) if self.running else 0,
                "applied": self.result(digest(APPLIED), foot), "draft": self.result(digest(DRAFT), foot)}


JOB = JudgeJob()


def handle(method: str, path: str, body: dict | None) -> tuple[int, dict]:
    """review_serve.py 가 부른다. (HTTP 상태, JSON)."""
    if method == "GET" and path == "/api/zones":
        return 200, state()
    if method == "GET" and path in ("/api/judge", "/api/judge/ankle"):
        return 200, JOB.status("ankle" if path.endswith("/ankle") else "bbox")
    if method == "POST" and path == "/api/zones/draft":
        return 200, save_draft(body or {})
    if method == "POST" and path == "/api/zones/apply":
        return 200, apply_draft()
    if method == "POST" and path == "/api/zones/discard":
        return 200, discard_draft()
    if method == "POST" and path == "/api/zones/restore":
        return 200, restore_backup((body or {}).get("name", ""))
    if method == "POST" and path == "/api/judge":
        return 200, JOB.start((body or {}).get("foot", "bbox"))
    return 404, {"ok": False, "errors": [f"모르는 요청: {method} {path}"]}


if __name__ == "__main__":
    # 구역 파일을 편집 페이지가 쓰는 형식으로 다시 쓴다 (내용은 그대로) — 처음 한 번, 손으로 고친 뒤 정리할 때.
    for key, f in FILES.items():
        raw = yaml.safe_load((APPLIED / f).read_text(encoding="utf-8"))
        clean = clean_floor(raw) if key == "floor" else clean_detail(raw)
        (APPLIED / f).write_text(dump_yaml(clean), encoding="utf-8")
    print(validate(*[yaml.safe_load((APPLIED / f).read_text(encoding="utf-8")) for f in FILES.values()]))
