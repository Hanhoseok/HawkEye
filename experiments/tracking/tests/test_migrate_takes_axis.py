import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import migrate_takes_axis as M  # noqa: E402

TAKES = ("pair,seq,time_sec,lane,action,qty,checked_by,note\n"
         "C01,1,6.55,L1,take,1,h,\n"
         "C02,1,5.98,L2,take,1,h,\n"
         "C03,1,,,none,0,h,\n")
SYNC = ("pair,video,reference,offset_ms,method,checked_by,note\n"
        "C01,a,b,14.9,manual,h,\nC02,a,b,-255.0,manual,h,\nC03,a,b,10.0,manual,h,\n")


def _files(tmp_path, sync=SYNC):
    takes, backup, sync_p = tmp_path / "takes.csv", tmp_path / "backup.csv", tmp_path / "sync.csv"
    takes.write_text(TAKES, encoding="utf-8")
    sync_p.write_text(sync, encoding="utf-8")
    return takes, backup, sync_p


def times(path):
    return [r["time_sec"] for r in M.read_rows(path)[1]]


def test_first_run_moves_and_keeps_backup(tmp_path):
    takes, backup, sync = _files(tmp_path)
    assert M.migrate(takes, backup, sync) == "moved"
    assert times(takes) == ["6.54", "6.24", ""]          # 6.55-0.0149, 5.98+0.255, none 그대로
    assert times(backup) == ["6.55", "5.98", ""]


def test_rerun_does_not_subtract_twice(tmp_path):
    takes, backup, sync = _files(tmp_path)
    M.migrate(takes, backup, sync)
    assert M.migrate(takes, backup, sync) == "already"
    assert times(takes) == ["6.54", "6.24", ""]


def test_hand_edited_file_stops(tmp_path):
    takes, backup, sync = _files(tmp_path)
    M.migrate(takes, backup, sync)
    takes.write_text(takes.read_text(encoding="utf-8").replace("6.54", "7.00"), encoding="utf-8")
    with pytest.raises(SystemExit):
        M.migrate(takes, backup, sync)
    assert "7.00" in takes.read_text(encoding="utf-8")


def test_missing_offset_stops_without_writing(tmp_path):
    takes, backup, sync = _files(tmp_path, SYNC.replace("C02,a,b,-255.0,manual,h,\n", ""))
    with pytest.raises(ValueError):
        M.migrate(takes, backup, sync)
    assert times(takes) == ["6.55", "5.98", ""]
