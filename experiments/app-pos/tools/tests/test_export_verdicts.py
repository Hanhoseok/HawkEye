"""추적 도구 판정 → 보관함 판정 JSON."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export_verdicts as ev  # noqa: E402


def test_export_reads_pairs_and_normalizes_scene_names(tmp_path):
    wb = tmp_path / "shoot4"
    (wb / "pairs").mkdir(parents=True)
    (wb / "index.json").write_text(json.dumps({"built": "2026-11-24 17:32"}), encoding="utf-8")
    (wb / "pairs" / "C5.json").write_text(json.dumps({"pair": "C5", "events": [
        {"pid": 1, "level": "HIGH_RISK", "zone": "door", "ms": 26666.04, "reason": "결제가 물건보다 적음", "bbox": [1, 2, 3, 4],
         "taken": 3, "paid": 1, "unpaid": 2},
        {"pid": 2, "level": "CLEAR", "zone": "door", "ms": 17066.0, "reason": "집은 물건 없음", "taken": 0, "paid": 0, "unpaid": 0},
    ]}, ensure_ascii=False), encoding="utf-8")
    (wb / "pairs" / "N08-1.json").write_text(json.dumps({"pair": "N8-1", "events": []}), encoding="utf-8")
    out = ev.export(wb)
    assert out["source"] == "추적 도구가 지금 설정으로 다시 낸 판정 · 2026-11-24 17:32"
    assert list(out["scenes"]) == ["C05", "N08-1"]
    c05 = out["scenes"]["C05"]
    assert [(v["ms"], v["person"], v["level"]) for v in c05] == [(17066.0, 2, "CLEAR"), (26666.0, 1, "HIGH_RISK")]
    assert set(c05[1]) == set(ev.FIELDS) and c05[1]["unpaid"] == 2
    assert out["scenes"]["N08-1"] == []


def test_cli_writes_file(tmp_path):
    wb = tmp_path / "wb"
    (wb / "pairs").mkdir(parents=True)
    (wb / "pairs" / "T1.json").write_text(json.dumps({"pair": "T1", "events": [
        {"pid": 1, "level": "WARNING", "ms": 1000, "reason": "r"}]}), encoding="utf-8")
    out = tmp_path / "v.json"
    assert ev.main([str(out), "--workbench", str(wb), "--no-check"]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["source"] == "추적 도구가 지금 설정으로 다시 낸 판정"
    assert data["scenes"]["T01"][0]["level"] == "WARNING"


def fake_tracking(tmp_path, stale, pairs=("C1", "C2")):
    """추적 도구 자리: workbench/data.py 의 Builder(shoot).out · pairs() · stale() 만 흉내 낸다."""
    root = tmp_path / "tracking"
    wb = root / "outputs" / "workbench" / "shoot4"
    (wb / "pairs").mkdir(parents=True)
    for name, level in (("C1", "CLEAR"), ("C2", "HIGH_RISK")):
        (wb / "pairs" / f"{name}.json").write_text(json.dumps({"pair": name, "events": [
            {"pid": 1, "level": level, "ms": 1000}]}), encoding="utf-8")
    (root / "workbench").mkdir()
    (root / "workbench" / "__init__.py").write_text("", encoding="utf-8")
    (root / "workbench" / "data.py").write_text(
        "from pathlib import Path\n"
        "class Builder:\n"
        "    def __init__(self, key=2):\n"
        f"        self.out = Path(r'{wb}')\n"
        f"    def pairs(self):\n        return {list(pairs)!r}\n"
        f"    def stale(self):\n        return {stale!r}\n", encoding="utf-8")
    return root


def drop_workbench_module():
    for name in [m for m in sys.modules if m == "workbench" or m.startswith("workbench.")]:
        del sys.modules[name]


def test_stale_pairs_block_export_unless_allowed(tmp_path, capsys):
    """다시 판정하다 실패해 예전 결과가 남은 짝(C2)은 '지금 판정' 으로 내보내지 않는다."""
    root = fake_tracking(tmp_path, ["C2"])
    out = tmp_path / "v.json"
    drop_workbench_module()
    try:
        assert ev.main([str(out), "--tracking", str(root), "--shoot", "4"]) == 1
        assert not out.exists() and "C2" in capsys.readouterr().err
        assert ev.main([str(out), "--tracking", str(root), "--shoot", "4", "--allow-stale"]) == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert list(data["scenes"]) == ["C01"] and data["skipped"] == ["C2"]
    finally:
        drop_workbench_module()
        sys.path[:] = [p for p in sys.path if not p.startswith(str(tmp_path))]


def test_verdicts_without_records_are_not_exported(tmp_path, capsys):
    """원본 기록이 없어져 최신 확인 대상에서 빠진 짝(C2)의 예전 판정은 내보내지 않는다."""
    root = fake_tracking(tmp_path, [], pairs=("C1",))
    out = tmp_path / "v.json"
    drop_workbench_module()
    try:
        assert ev.main([str(out), "--tracking", str(root), "--shoot", "4"]) == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert list(data["scenes"]) == ["C01"] and data["skipped"] == ["C2"]
        assert "C2" in capsys.readouterr().err
    finally:
        drop_workbench_module()
        sys.path[:] = [p for p in sys.path if not p.startswith(str(tmp_path))]


def test_fresh_pairs_export_all(tmp_path):
    root = fake_tracking(tmp_path, [])
    out = tmp_path / "v.json"
    drop_workbench_module()
    try:
        assert ev.main([str(out), "--tracking", str(root), "--shoot", "4"]) == 0
        assert list(json.loads(out.read_text(encoding="utf-8"))["scenes"]) == ["C01", "C02"]
    finally:
        drop_workbench_module()
        sys.path[:] = [p for p in sys.path if not p.startswith(str(tmp_path))]
