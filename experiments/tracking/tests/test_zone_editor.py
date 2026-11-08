"""구역 편집 페이지 서버 쪽 (scripts/zone_editor.py) — 파일 형식 그대로 쓰기, 검사, 초안·적용."""
import copy
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import zone_editor as ze  # noqa: E402
from src.zones.floor import FloorMap  # noqa: E402


def _applied():
    return ze.read_set(ze.APPLIED)


def test_zone_files_round_trip_unchanged():
    """적용본을 편집 페이지 형식으로 다시 써도 글자 하나 안 바뀐다 — 저장할 때마다 diff 가 생기지 않게."""
    for key, name in ze.FILES.items():
        text = (ze.APPLIED / name).read_text(encoding="utf-8")
        raw = yaml.safe_load(text)
        clean = ze.clean_floor(raw) if key == "floor" else ze.clean_detail(raw)
        assert ze.dump_yaml(clean) == text


def test_applied_zones_pass_validation():
    s = _applied()
    errors, warns = ze.validate(s["floor"], s["angle0"])
    assert errors == [] and warns == []


def test_validation_catches_broken_zones():
    s = _applied()
    floor, detail = copy.deepcopy(s["floor"]), copy.deepcopy(s["angle0"])
    floor["zones"][0]["polygon"] = floor["zones"][0]["polygon"][:2]
    floor["ignore_image"][0] = [0.5, 0.5, 0.4, 0.6]
    detail["zones"][1]["name"] = detail["zones"][0]["name"]
    errors, _ = ze.validate(floor, detail)
    assert any("3개 미만" in e for e in errors)
    assert any("사람 아님 영역 1" in e for e in errors)
    assert any("칸 이름이 겹침" in e for e in errors)


def test_validation_warns_when_config_names_disappear():
    """설정이 이름으로 찾는 구역(inside_zones·shelf_zone)이 없어지면 알려 준다."""
    s = _applied()
    floor = copy.deepcopy(s["floor"])
    for z in floor["zones"]:
        if z["name"] == "aisle":
            z["name"] = "hall"
    errors, warns = ze.validate(floor, s["angle0"])
    assert errors == []
    assert any("'aisle'" in w for w in warns)


def test_draft_apply_restore_cycle(tmp_path, monkeypatch):
    """초안 저장 → 적용(이전 것은 백업) → 백업을 초안으로 되돌리기. 실제 zones/ 는 건드리지 않게 임시 폴더에서."""
    applied, work = tmp_path / "zones", tmp_path / "work"
    applied.mkdir()
    for name in ze.FILES.values():
        (applied / name).write_text((ze.APPLIED / name).read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(ze, "APPLIED", applied)
    monkeypatch.setattr(ze, "WORK", work)
    monkeypatch.setattr(ze, "DRAFT", work / "draft")
    monkeypatch.setattr(ze, "BACKUP", work / "backup")
    before = ze.digest(applied)

    s = ze.read_set(applied)
    s["floor"]["zones"][1]["polygon"][2] = [100.0, 175.0]
    res = ze.save_draft(s)
    assert res["ok"] and res["draft_hash"] != before

    out = ze.apply_draft()
    assert out["ok"] and ze.digest(applied) == res["draft_hash"] and ze.digest(ze.DRAFT) is None
    assert FloorMap.load(applied / "floor.yaml").zones[1].polygon[2] == (100.0, 175.0)

    name = ze.backups()[0]
    assert ze.restore_backup(name)["ok"] and ze.digest(ze.DRAFT) == before


def test_api_rejects_unknown_route():
    code, data = ze.handle("POST", "/api/zones/delete-everything", {})
    assert code == 404 and not data["ok"]
