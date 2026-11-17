"""분석 도구의 새 영상 추가(workbench/intake.py) — 이름·파일 받기·영상 확인·표에 덧붙이기. 실제 처리(탐지·추적)는 시험하지 않는다."""

import io

import pytest

from workbench import intake
from workbench.data import BATCH_NAME, totals

TAKEN = {"N1", "N2", "N8-1", "N12", "T1", "T10", "C1", "C21"}


def test_name_must_follow_set_letter_and_number_and_be_new():
    assert intake.check_name("N13", TAKEN) == "N13" and intake.check_name("C22-1", TAKEN) == "C22-1"
    for bad in ("", "X1", "n13", "N", "N1234", "N13 ", "../N13", "N13.mp4", "T11_a"):
        with pytest.raises(ValueError):
            intake.check_name(bad, TAKEN)
    with pytest.raises(ValueError, match="이미 있는"):
        intake.check_name("N8-1", TAKEN)


def test_suggested_names_continue_each_set():
    assert intake.suggest_names(TAKEN) == {"normal": "N13", "theft": "T11", "complex": "C22"}
    assert intake.suggest_names(set()) == {"normal": "N1", "theft": "T1", "complex": "C1"}


def test_append_csv_adds_one_line_and_keeps_the_rest(tmp_path):
    path = tmp_path / "truth.csv"
    original = b'pair,set,people,expected,notes\r\nT9,theft,2,HIGH_RISK,"a, b"\r\n'
    path.write_bytes(original)
    fields = ("pair", "set", "people", "expected", "notes")
    intake._append_csv(path, fields, {"pair": "N13", "set": "normal", "people": 1, "expected": "CLEAR", "notes": "메모, 쉼표"})
    raw = path.read_bytes()
    assert raw.startswith(original)                                   # 있던 줄은 그대로
    assert raw[len(original):].decode("utf-8") == 'N13,normal,1,CLEAR,"메모, 쉼표"\r\n'   # 그 파일의 줄바꿈을 따른다
    fresh = tmp_path / "added.csv"
    intake._append_csv(fresh, ("pair", "added"), {"pair": "N13", "added": "2026-11-10"})
    assert fresh.read_text(encoding="utf-8") == "pair,added\nN13,2026-11-10\n"          # 없던 파일이면 머리글부터
    no_newline = tmp_path / "x.csv"
    no_newline.write_bytes(b"pair,added\nN1,2026-01-01")
    intake._append_csv(no_newline, ("pair", "added"), {"pair": "N2", "added": "2026-01-02"})
    assert no_newline.read_text(encoding="utf-8").splitlines() == ["pair,added", "N1,2026-01-01", "N2,2026-01-02"]


def video(seconds=18.0, fps=60.0, w=1920, h=1080):
    return {"fps": fps, "frames": int(seconds * fps), "width": w, "height": h, "seconds": seconds}


def test_check_videos_rejects_what_the_zones_cannot_handle():
    assert intake.check_videos(video(18.3, 60), video(18.6, 30)) == []
    assert intake.check_videos(video(18, 60, 1280, 720), video(18, 30, 3840, 2160)) == []     # 크기는 달라도 16:9 면 된다
    with pytest.raises(ValueError, match="16:9"):
        intake.check_videos(video(18, 60, 1080, 1920), video(18, 30))                          # 세로로 찍은 영상
    with pytest.raises(ValueError, match="길이가 많이 다릅니다"):
        intake.check_videos(video(18, 60), video(40, 30))
    with pytest.raises(ValueError, match="5분"):
        intake.check_videos(video(400, 60), video(400, 30))
    notes = intake.check_videos(video(18, 30), video(18, 30))                                 # 전체 카메라가 30장
    assert len(notes) == 1 and "전체 카메라" in notes[0] and "30장" in notes[0]


def test_upload_writes_file_only_when_complete(tmp_path, monkeypatch):
    class FakeSyncPair:
        ADDED_DIR = tmp_path / "added"
        VIDEO_EXT = (".mp4", ".mov")

        @staticmethod
        def all_pairs():
            return ["N1"]

        @staticmethod
        def added_pairs():
            return []

    monkeypatch.setattr(intake, "_sync_pair", lambda: FakeSyncPair)
    monkeypatch.setattr(intake, "ADDED", tmp_path / "added.csv")
    monkeypatch.setattr(intake, "TRUTH", tmp_path / "truth.csv")
    out = intake.save_upload("N13", "angle1", ".MP4", io.BytesIO(b"x" * 5000), 5000)
    assert out == {"saved": "N13_angle1.mp4", "bytes": 5000}
    assert (tmp_path / "added" / "N13_angle1.mp4").read_bytes() == b"x" * 5000
    intake.save_upload("N13", "angle1", ".mov", io.BytesIO(b"y" * 10), 10)                    # 다시 올리면 바꿔 쓴다
    assert sorted(p.name for p in (tmp_path / "added").iterdir()) == ["N13_angle1.mov"]
    with pytest.raises(ValueError, match="끊겼습니다"):
        intake.save_upload("N13", "angle0", ".mp4", io.BytesIO(b"z" * 10), 100)               # 다 못 받음
    assert sorted(p.name for p in (tmp_path / "added").iterdir()) == ["N13_angle1.mov"]       # 받다 만 파일은 남지 않는다
    for name, cam, ext, size in (("N1", "angle1", ".mp4", 10), ("N13", "front", ".mp4", 10), ("N13", "angle1", ".exe", 10),
                                 ("N13", "angle1", ".mp4", 0), ("../N13", "angle1", ".mp4", 10)):
        with pytest.raises(ValueError):
            intake.save_upload(name, cam, ext, io.BytesIO(b"x" * 10), size)


def test_batch_of_and_batch_names():
    assert intake.batch_of("N13", {"N13": {}}) == "added" and intake.batch_of("N1", {"N13": {}}) == "shoot2"
    # 영상 추가는 2차 촬영 자리만 — 그 묶음들은 갤러리 묶음 이름에 모두 있어야 한다 (3차는 묶음이 하나: shoot3)
    assert [key for key, _ in intake.BATCHES] == [key for key in BATCH_NAME if key != "shoot3"]
    assert all(BATCH_NAME[key] == name for key, name in intake.BATCHES)


def test_totals_work_for_a_single_batch():
    cards = [{"pair": "N13", "set": "normal", "correct": True, "correct_last": True, "no_verdict": False, "people_ok": True, "labeled": None}]
    t = totals(cards)
    assert (t["n"], t["correct"], t["by_set"]["theft"]["n"], t["labeled"]["n"]) == (1, 1, 0, 0)
