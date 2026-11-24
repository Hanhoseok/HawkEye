"""보관함 — 폴더·장면·영상 (docs/design.md §11)."""

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hawkeye_server.app import Settings, create_app
from hawkeye_server.library import (Busy, Library, LibraryError, VideoInfo, clean_name, convert_args, load_sync,
                                    maintenance_lock, natural_key, plan_import, playable_as_is, run_import, run_import_ai,
                                    run_import_verdicts, clean_verdicts,
                                    scene_key, scene_offsets)

KEY = {"X-API-Key": "k"}
HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


# --- 순수 함수 -----------------------------------------------------------------

def test_scene_key_matches_names_across_cameras():
    assert scene_key("20261028_C1") == "C01"
    assert scene_key("C01") == "C01"
    assert scene_key("c04-1") == "C04-1"
    assert scene_key("N8") == "N08"
    assert scene_key("N08-1") == "N08-1"
    assert scene_key("C123") == "C123"
    assert scene_key("시연 장면") == "시연 장면"


def test_natural_sort():
    names = ["C10", "C2", "시연용", "1차 촬영", "0차 촬영", "C01", "10차 촬영", "2차 촬영"]
    assert sorted(names, key=natural_key) == ["0차 촬영", "1차 촬영", "2차 촬영", "10차 촬영", "C01", "C2", "C10", "시연용"]


def test_clean_name():
    assert clean_name("  1차 촬영 ") == "1차 촬영"
    for bad in ["", "   ", "a" * 41, "줄\n바꿈", None, 3]:
        with pytest.raises(LibraryError) as e:
            clean_name(bad)
        assert (e.value.status, e.value.code) == (422, "invalid_name")
    assert clean_name("a" * 40) == "a" * 40


def test_plan_import_pairs_cameras_and_keeps_singles():
    plan = plan_import({
        "cam1": [Path("C01.mp4"), Path("C02.mp4"), Path("C10.mp4"), Path("N01.mp4")],
        "cam2": [Path("20261028_C1.mp4"), Path("20261028_C2.mp4"), Path("20261028_C3.mp4")],
    })
    assert [name for name, _ in plan] == ["C01", "C02", "C03", "C10", "N01"]
    assert plan[0][1] == {"cam1": Path("C01.mp4"), "cam2": Path("20261028_C1.mp4")}
    assert plan[2][1] == {"cam2": Path("20261028_C3.mp4")}
    assert plan[3][1] == {"cam1": Path("C10.mp4")}


def test_convert_args_copies_playable_h264_and_reencodes_the_rest():
    ok = VideoInfo("h264", "yuv420p", 1280, 720, 30.0, 25.0)
    assert playable_as_is(ok)
    args = convert_args(Path("a.mp4"), Path("b.mp4"), ok, "libx264")
    assert args[args.index("-c:v") + 1] == "copy" and "-an" in args and "+faststart" in args

    hevc60 = VideoInfo("hevc", "yuv420p10le", 1920, 1080, 60.0, 25.0)
    assert not playable_as_is(hevc60)
    args = convert_args(Path("a.mp4"), Path("b.mp4"), hevc60, "h264_nvenc")
    vf = args[args.index("-vf") + 1]
    assert "scale='min(1280,iw)':-2" in vf and "fps=30" in vf and "format=yuv420p" in vf
    assert args[args.index("-c:v") + 1] == "h264_nvenc" and args[args.index("-profile:v") + 1] == "main"

    hevc30 = VideoInfo("hevc", "yuv420p", 1920, 1080, 30.0, 25.0)
    assert "fps=30" not in convert_args(Path("a"), Path("b"), hevc30, "libx264")[6:][1]


# --- 저장소 ----------------------------------------------------------------------

def fake_video(lib: Library, scene_id: int, camera: str, duration: float = 20.0) -> str:
    """ffmpeg 없이 영상 자리를 채운다 (파일 내용은 시험용 바이트)."""
    name = f"{uuid.uuid4().hex}.mp4"
    (lib.files / name).write_bytes(b"\x00\x00\x00\x18ftypmp42" + bytes(range(256)) * 40)
    lib.put_video(scene_id, camera, name, VideoInfo("h264", "yuv420p", 1280, 720, 30.0, duration))
    return name


@pytest.fixture
def lib(tmp_path):
    lib = Library(tmp_path / "library")
    yield lib
    lib.close()


def seed(lib: Library):
    f1 = lib.create_folder("1차 촬영")["id"]
    f2 = lib.create_folder("시연용")["id"]
    s1, _ = lib.add_scene(f1, "C01", "1차 촬영", "1차 촬영/C01")
    s2, _ = lib.add_scene(f1, "C02", "1차 촬영", "1차 촬영/C02")
    s10, _ = lib.add_scene(f1, "C10", "1차 촬영", "1차 촬영/C10")
    fake_video(lib, s1, "cam1", 26.0)
    fake_video(lib, s1, "cam2", 25.5)
    fake_video(lib, s2, "cam1", 24.0)
    return f1, f2, s1, s2, s10


def test_folder_listing_counts_scenes_and_length(lib):
    f1, f2, *_ = seed(lib)
    folders = lib.folders()
    assert [f["name"] for f in folders] == ["1차 촬영", "시연용"]
    assert folders[0]["scene_count"] == 3 and folders[0]["total_sec"] == 50.0   # 26 + 24 + (영상 없음 0)
    assert folders[1] == {"id": f2, "name": "시연용", "scene_count": 0, "total_sec": 0.0}


def test_folder_detail_sorts_scenes_naturally(lib):
    f1, _, s1, *_ = seed(lib)
    d = lib.folder(f1)
    assert [s["name"] for s in d["scenes"]] == ["C01", "C02", "C10"]
    first = d["scenes"][0]
    assert first["id"] == s1 and first["duration"] == 26.0 and first["origin"] == "1차 촬영"
    assert [v["camera_id"] for v in first["videos"]] == ["cam1", "cam2"]


def test_duplicate_folder_name_is_rejected_case_insensitively(lib):
    lib.create_folder("Demo")
    with pytest.raises(LibraryError) as e:
        lib.create_folder(" demo ")
    assert (e.value.status, e.value.code) == (409, "duplicate_name")
    other = lib.create_folder("other")["id"]
    with pytest.raises(LibraryError):
        lib.rename_folder(other, "DEMO")
    assert lib.rename_folder(other, "다른 이름")["name"] == "다른 이름"


def test_move_is_all_or_nothing(lib):
    f1, f2, s1, s2, _ = seed(lib)
    with pytest.raises(LibraryError) as e:
        lib.move_scenes([s1, 9999], f2)
    assert e.value.status == 404 and e.value.extra["missing"] == [9999]
    assert lib.folder(f2)["scenes"] == []
    with pytest.raises(LibraryError):
        lib.move_scenes([s1], 9999)
    assert lib.move_scenes([s1, s2, s1], f2) == {"moved": 2}
    assert [s["name"] for s in lib.folder(f2)["scenes"]] == ["C01", "C02"]
    assert lib.move_scenes([s1], f2) == {"moved": 0}      # 이미 그 폴더


def test_delete_scenes_removes_files_and_thumbs(lib):
    f1, _, s1, s2, _ = seed(lib)
    files = [p.name for p in lib.files.iterdir()]
    lib.thumb_file(s1).write_bytes(b"jpg")
    with pytest.raises(LibraryError):
        lib.delete_scenes([s1, 424242])
    assert len(list(lib.files.iterdir())) == 3            # 아무것도 안 지움
    assert lib.delete_scenes([s1]) == {"deleted": 1}
    assert len(list(lib.files.iterdir())) == 1 and not lib.thumb_file(s1).exists()
    assert len(files) == 3
    assert [s["id"] for s in lib.folder(f1)["scenes"]] == [s2, s2 + 1]


def test_delete_folder_needs_with_scenes_when_not_empty(lib):
    f1, f2, *_ = seed(lib)
    with pytest.raises(LibraryError) as e:
        lib.delete_folder(f1)
    assert (e.value.status, e.value.code, e.value.extra) == (409, "not_empty", {"scene_count": 3})
    assert lib.delete_folder(f2) == {"deleted_scenes": 0}
    assert lib.delete_folder(f1, with_scenes=True) == {"deleted_scenes": 3}
    assert lib.folders() == [] and list(lib.files.iterdir()) == []


def test_scene_ids_are_validated(lib):
    for bad in ([], ["1"], [True], list(range(1, 502))):
        with pytest.raises(LibraryError) as e:
            lib.delete_scenes(bad)
        assert e.value.code == "invalid_ids"


def test_gc_removes_old_orphans_but_not_temp_or_recent_files(lib):
    f1, _, s1, *_ = seed(lib)
    old = time.time() - 3600
    for p in (lib.files / "orphan.mp4", lib.files / "half.part.mp4", lib.thumbs / "999.jpg", lib.files / "fresh.mp4"):
        p.write_bytes(b"x")
        if p.name != "fresh.mp4":
            os.utime(p, (old, old))
    lib.thumb_file(s1).write_bytes(b"x")
    os.utime(lib.thumb_file(s1), (old, old))
    for p in lib.files.iterdir():
        if p.name not in ("fresh.mp4",):
            os.utime(p, (old, old))
    assert lib.gc() == 2                                  # orphan.mp4, 999.jpg
    names = {p.name for p in lib.files.iterdir()}
    assert {"half.part.mp4", "fresh.mp4"} <= names and "orphan.mp4" not in names
    assert lib.thumb_file(s1).exists() and len(names) == 5   # 장면 영상 3 + 임시 + 최근


@pytest.mark.skipif(sys.platform == "win32", reason="flock 은 Linux")
def test_maintenance_lock_lets_one_process_at_a_time(tmp_path):
    with maintenance_lock(tmp_path):
        with pytest.raises(Busy):
            with maintenance_lock(tmp_path):
                pass
    with maintenance_lock(tmp_path):     # 풀린 뒤에는 다시 잡힌다
        pass


def test_sync_table_gives_offsets_by_file_name(tmp_path):
    table = tmp_path / "sync.csv"
    table.write_text(
        "pair,video,reference,offset_ms,method,checked_by,note\n"
        "C20,1(angle0)/20261028_C20.mp4,1(angle1)/C20.mp4,-2863.9,manual,hoseok,\n"
        "C2,1(angle0)/20261028_C2.mp4,1(angle1)/C02.mp4,1.5,manual,,\n"
        "bad,x.mp4,y.mp4,not-a-number,,,\n", encoding="utf-8")
    sync = load_sync(table)
    assert len(sync) == 2
    files = {"cam1": Path("s/C20.mp4"), "cam2": Path("s/20261028_C20.mp4")}       # 이름이 달라 이름만으로 맞춤
    assert scene_offsets(files, sync) == {"cam2": -2863.9, "cam1": 0.0}
    assert scene_offsets({"cam1": Path("C03.mp4"), "cam2": Path("20261028_C3.mp4")}, sync) is None


def test_sync_table_with_same_file_names_uses_folder_names(tmp_path):
    """3·4차는 두 카메라 파일 이름이 같다 (2(angle0)/C01.mp4 ↔ 2(angle1)/C01.mp4)."""
    table = tmp_path / "sync.csv"
    table.write_text("pair,video,reference,offset_ms,method,checked_by,note\n"
                     "C01,2(angle0)/C01.mp4,2(angle1)/C01.mp4,14.9,manual,hoseok,\n"
                     "C02,2(angle0)/C02.mp4,2(angle1)/C02.mp4,-255.0,manual,hoseok,\n", encoding="utf-8")
    sync = load_sync(table)
    files = {"cam1": tmp_path / "up" / "2(angle1)" / "C02.mp4", "cam2": tmp_path / "up" / "2(angle0)" / "C02.mp4"}
    assert scene_offsets(files, sync) == {"cam2": -255.0, "cam1": 0.0}
    # 폴더 이름이 다르면(표와 다른 곳에 올림) 같은 이름끼리는 맞출 수 없다 → None (0 으로 둠)
    other = {"cam1": Path("a/C01.mp4"), "cam2": Path("b/C01.mp4")}
    assert scene_offsets(other, sync) is None


def test_import_does_not_leave_empty_scenes_when_conversion_fails(tmp_path, monkeypatch):
    import hawkeye_server.library as library

    cam1, cam2 = tmp_path / "1(angle1)", tmp_path / "1(angle0)"
    cam1.mkdir()
    cam2.mkdir()
    for d, n in ((cam1, "C01.mp4"), (cam2, "20261028_C1.mp4")):
        (d / n).write_bytes(b"not a video")

    def boom(*_a, **_k):
        raise RuntimeError("ffmpeg 실패")

    monkeypatch.setattr(library, "convert", boom)
    monkeypatch.setattr(library, "pick_encoder", lambda choice, ffmpeg="ffmpeg": "libx264")
    lib = Library(tmp_path / "library")
    c = run_import(lib, "2차 촬영", {"cam1": cam1, "cam2": cam2}, log=lambda _m: None)
    assert (c["added"], c["failed"]) == (0, 2)
    assert lib.folders()[0]["scene_count"] == 0                 # 빈 장면 없음

    # 이미 영상이 있는 장면은 다시 넣다 실패해도 지우지 않는다
    sid, _ = lib.add_scene(lib.folders()[0]["id"], "C01", "2차 촬영", "2차 촬영/C01")
    fake_video(lib, sid, "cam1")
    c = run_import(lib, "2차 촬영", {"cam1": cam1, "cam2": cam2}, replace=True, log=lambda _m: None)
    assert c["failed"] == 2 and lib.scene(sid)["videos"][0]["camera_id"] == "cam1"
    lib.close()


def test_media_at_changes_when_video_is_replaced(lib):
    _, _, s1, *_ = seed(lib)
    before = lib.scene(s1)["media_at"]
    time.sleep(0.01)
    fake_video(lib, s1, "cam2", 10.0)
    assert lib.scene(s1)["media_at"] > before


def test_put_video_replaces_old_file(lib):
    f1, _, s1, *_ = seed(lib)
    before = {p.name for p in lib.files.iterdir()}
    new = fake_video(lib, s1, "cam1", 30.0)
    after = {p.name for p in lib.files.iterdir()}
    assert new in after and len(after) == len(before)
    assert lib.scene(s1)["duration"] == 30.0


# --- API ----------------------------------------------------------------------------

@pytest.fixture
def api(tmp_path):
    app = create_app(Settings(db_path=tmp_path / "hawkeye.db", snapshot_dir=tmp_path / "snap", api_key="k",
                              library_dir=tmp_path / "library"))
    with TestClient(app) as c:
        yield c


def test_api_needs_key_and_library(tmp_path, api):
    assert api.get("/api/library").status_code == 401
    off = TestClient(create_app(Settings(db_path=tmp_path / "x.db", snapshot_dir=tmp_path / "s")))
    r = off.get("/api/library")
    assert r.status_code == 503 and r.json()["detail"].startswith("library_unavailable")


def test_api_folder_flow(api):
    r = api.post("/api/library/folders", json={"name": "시연용"}, headers=KEY)
    assert r.status_code == 201
    fid = r.json()["id"]
    r = api.post("/api/library/folders", json={"name": "시연용"}, headers=KEY)
    assert r.status_code == 409 and r.json()["detail"].startswith("duplicate_name")
    r = api.post("/api/library/folders", json={"name": " "}, headers=KEY)
    assert r.status_code == 422 and r.json()["detail"].startswith("invalid_name")
    assert api.patch(f"/api/library/folders/{fid}", json={"name": "발표"}, headers=KEY).json()["name"] == "발표"
    assert api.get("/api/library", headers=KEY).json() == {
        "folders": [{"id": fid, "name": "발표", "scene_count": 0, "total_sec": 0.0}]}
    assert api.get(f"/api/library/folders/{fid}", headers=KEY).json()["scenes"] == []
    assert api.get("/api/library/folders/999", headers=KEY).status_code == 404
    assert api.delete(f"/api/library/folders/{fid}", headers=KEY).json() == {"deleted_scenes": 0}


def test_api_scene_flow_and_video_range(api, tmp_path):
    lib = Library(tmp_path / "library")      # 가져오기 도구처럼 다른 연결로 넣는다
    f1, f2, s1, s2, _ = seed(lib)
    lib.thumb_file(s1).write_bytes(b"\xff\xd8thumb\xff\xd9")
    lib.close()

    d = api.get(f"/api/library/folders/{f1}", headers=KEY).json()
    assert [s["name"] for s in d["scenes"]] == ["C01", "C02", "C10"] and d["scenes"][0]["has_thumb"] is True
    vid = d["scenes"][0]["videos"][0]["id"]

    r = api.get(f"/api/library/videos/{vid}.mp4", headers={**KEY, "Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100 and r.headers["content-type"] == "video/mp4"
    assert api.get(f"/api/library/videos/{vid}.mp4").status_code == 401
    assert api.get("/api/library/videos/999.mp4", headers=KEY).status_code == 404
    assert api.get(f"/api/library/scenes/{s1}/thumb.jpg", headers=KEY).content.startswith(b"\xff\xd8")
    assert api.get(f"/api/library/scenes/{s2}/thumb.jpg", headers=KEY).status_code == 404

    assert api.patch(f"/api/library/scenes/{s1}", json={"name": "도난 장면"}, headers=KEY).json()["name"] == "도난 장면"
    r = api.post("/api/library/scenes/move", json={"scene_ids": [s1, 777], "folder_id": f2}, headers=KEY)
    assert r.status_code == 404 and r.json()["missing"] == [777]
    assert api.post("/api/library/scenes/move", json={"scene_ids": [s1], "folder_id": f2}, headers=KEY).json() == {"moved": 1}
    moved = api.get(f"/api/library/folders/{f2}", headers=KEY).json()["scenes"]
    assert [(s["name"], s["origin"]) for s in moved] == [("도난 장면", "1차 촬영")]

    r = api.delete(f"/api/library/folders/{f1}", headers=KEY)
    assert r.status_code == 409 and r.json()["scene_count"] == 2
    assert api.post("/api/library/scenes/delete", json={"scene_ids": [s1]}, headers=KEY).json() == {"deleted": 1}
    assert api.get(f"/api/library/videos/{vid}.mp4", headers=KEY).status_code == 404
    assert api.delete(f"/api/library/folders/{f1}?with_scenes=true", headers=KEY).json() == {"deleted_scenes": 2}


# --- 가져오기 (ffmpeg 이 있을 때만) ---------------------------------------------------

def make_clip(path: Path, size: str, rate: int, codec: list[str]) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    f"testsrc=size={size}:rate={rate}:duration=2", *codec, str(path)], check=True)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg 없음")
def test_import_converts_pairs_and_is_idempotent(tmp_path):
    cam1, cam2 = tmp_path / "1(angle1)", tmp_path / "1(angle0)"
    cam1.mkdir()
    cam2.mkdir()
    make_clip(cam1 / "C01.mp4", "1920x1080", 60, ["-c:v", "libx264", "-pix_fmt", "yuv420p"])   # 다시 인코딩
    make_clip(cam2 / "20261028_C1.mp4", "1280x720", 30, ["-c:v", "libx264", "-pix_fmt", "yuv420p"])  # 그대로
    make_clip(cam2 / "20261028_N2.mp4", "1280x720", 30, ["-c:v", "libx264", "-pix_fmt", "yuv420p"])
    (cam1 / "notes.txt").write_text("x")

    lib = Library(tmp_path / "library")
    logs: list[str] = []
    sync = [("20261028_C1.mp4", "C01.mp4", -1200.0)]
    c = run_import(lib, "2차 촬영", {"cam1": cam1, "cam2": cam2}, encoder="libx264", sync=sync, log=logs.append)
    assert c == {"scenes": 2, "added": 3, "skipped": 0, "failed": 0, "no_sync": 0}   # N02 는 카메라 1대라 시차 불필요
    folder = lib.folders()[0]
    d = lib.folder(folder["id"])
    assert [s["name"] for s in d["scenes"]] == ["C01", "N02"]
    c01 = d["scenes"][0]
    assert [(v["camera_id"], v["width"], v["offset_ms"]) for v in c01["videos"]] == [
        ("cam1", 1280, 0.0), ("cam2", 1280, -1200.0)]
    assert c01["has_thumb"] and 1.5 < c01["duration"] < 2.5 and c01["media_at"]

    # 다시 실행: 영상은 건너뛰고, 고친 시차는 반영
    again = run_import(lib, "2차 촬영", {"cam1": cam1, "cam2": cam2}, encoder="libx264",
                       sync=[("20261028_C1.mp4", "C01.mp4", 300.0)], log=logs.append)
    assert again == {"scenes": 2, "added": 0, "skipped": 3, "failed": 0, "no_sync": 0}
    assert len(lib.folders()) == 1 and len(list(lib.files.iterdir())) == 3
    assert lib.folder(folder["id"])["scenes"][0]["videos"][1]["offset_ms"] == 300.0
    lib.close()


# --- AI 화면 ---------------------------------------------------------------------------

def fake_ai(lib: Library, video_id: int, duration: float = 26.0) -> str:
    name = f"{uuid.uuid4().hex}.mp4"
    (lib.files / name).write_bytes(b"ai-video" * 50)
    lib.put_ai(video_id, name, VideoInfo("h264", "yuv420p", 1280, 720, 30.0, duration))
    return name


def test_ai_video_attaches_to_raw_video_and_follows_its_life(lib):
    _, _, s1, s2, _ = seed(lib)
    raw = lib.video_id(s1, "cam1")
    assert not lib.has_ai(raw)
    ai = fake_ai(lib, raw)
    videos = lib.scene(s1)["videos"]
    assert [(v["camera_id"], v["has_ai"]) for v in videos] == [("cam1", True), ("cam2", False)]
    assert lib.ai_file(raw).name == ai

    with pytest.raises(LibraryError) as e:                     # 길이가 원본(26초)과 다르면 넣지 않음
        fake_ai(lib, raw, duration=40.0)
    assert e.value.code == "length_mismatch"
    assert lib.ai_file(raw).name == ai

    fake_video(lib, s1, "cam1", 26.0)                           # 원본을 바꾸면 AI 는 지워짐
    assert lib.ai_file(raw) is None and not (lib.files / ai).exists()
    assert lib.video_id(s1, "cam1") == raw                      # 영상 id 는 그대로

    ai2 = fake_ai(lib, raw)
    lib.delete_scenes([s1])                                     # 장면을 지우면 AI 파일도
    assert not (lib.files / ai2).exists()


def test_gc_keeps_ai_files(lib):
    _, _, s1, *_ = seed(lib)
    ai = fake_ai(lib, lib.video_id(s1, "cam1"))
    old = time.time() - 3600
    for p in lib.files.iterdir():
        os.utime(p, (old, old))
    assert lib.gc() == 0 and (lib.files / ai).exists()


def test_api_serves_ai_video(api, tmp_path):
    lib = Library(tmp_path / "library")
    _, _, s1, s2, _ = seed(lib)
    raw = lib.video_id(s1, "cam1")
    fake_ai(lib, raw)
    lib.close()
    r = api.get(f"/api/library/videos/{raw}/ai.mp4", headers={**KEY, "Range": "bytes=0-9"})
    assert r.status_code == 206 and r.content == b"ai-videoai"
    other = api.get(f"/api/library/folders/1", headers=KEY).json()["scenes"][1]["videos"][0]
    assert other["has_ai"] is False
    assert api.get(f"/api/library/videos/{other['id']}/ai.mp4", headers=KEY).status_code == 404


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg 없음")
def test_import_ai_attaches_only_to_existing_scenes(tmp_path):
    raw1, raw2 = tmp_path / "raw1", tmp_path / "raw2"
    ai1, ai2 = tmp_path / "ai1", tmp_path / "ai2"
    for d in (raw1, raw2, ai1, ai2):
        d.mkdir()
    mp4 = ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    make_clip(raw1 / "C01.mp4", "1280x720", 30, mp4)
    make_clip(raw2 / "20261028_C1.mp4", "1280x720", 30, mp4)
    make_clip(ai1 / "C01.mp4", "1280x720", 30, mp4)
    make_clip(ai2 / "20261028_C1.mp4", "1280x720", 30, mp4)
    make_clip(ai1 / "C09.mp4", "1280x720", 30, mp4)               # 원본이 없는 장면
    lib = Library(tmp_path / "library")
    run_import(lib, "2차 촬영", {"cam1": raw1, "cam2": raw2}, encoder="libx264", log=lambda _m: None)
    c = run_import_ai(lib, "2차 촬영", {"cam1": ai1, "cam2": ai2}, encoder="libx264", log=lambda _m: None)
    assert (c["added"], c["no_source"], c["failed"]) == (2, 1, 0)
    scenes = lib.folder(lib.folders()[0]["id"])["scenes"]
    assert [s["name"] for s in scenes] == ["C01"]                # AI 가져오기는 장면을 만들지 않음
    assert all(v["has_ai"] for v in scenes[0]["videos"])
    again = run_import_ai(lib, "2차 촬영", {"cam1": ai1, "cam2": ai2}, encoder="libx264", log=lambda _m: None)
    assert again["skipped"] == 2 and again["added"] == 0
    lib.close()


# --- AI 판정 ---------------------------------------------------------------------------

def test_clean_verdicts_keeps_valid_items_sorted():
    good, dropped = clean_verdicts([
        {"ms": 26666.0, "person": 1, "level": "HIGH_RISK", "zone": "door", "reason": "결제가 물건보다 적음", "taken": 3, "paid": 1, "unpaid": 2},
        {"ms": 17066, "person": 2, "level": "CLEAR", "reason": "집은 물건 없음"},
        {"ms": -1, "person": 1, "level": "CLEAR"},
        {"ms": 5, "person": 1, "level": "PANIC"},
        {"ms": 5, "person": True, "level": "CLEAR"},
        {"person": 1, "level": "CLEAR"},
        {"ms": float("nan"), "person": 1, "level": "CLEAR"},
    ])
    assert [(v["ms"], v["person"], v["level"]) for v in good] == [(17066.0, 2, "CLEAR"), (26666.0, 1, "HIGH_RISK")]
    assert good[0]["taken"] == 0 and good[1]["unpaid"] == 2 and dropped == 5
    assert clean_verdicts("x") == ([], 1)
    base = {"ms": 1, "level": "CLEAR"}
    bad_people = [1.9, 0, -3, 2_147_483_648, "2", None, True, float("inf")]
    good2, dropped2 = clean_verdicts([{**base, "person": p} for p in bad_people] + [{**base, "person": 7}])
    assert [v["person"] for v in good2] == [7] and dropped2 == len(bad_people)          # 1.9 를 1 로 만들지 않음
    huge = 10 ** 400                                                                      # float() 하면 OverflowError
    good3, dropped3 = clean_verdicts([{"ms": huge, "person": 1, "level": "CLEAR"},
                                      {"ms": 1, "person": 1, "level": "CLEAR", "taken": 2.5},
                                      {"ms": 1, "person": 1, "level": "CLEAR", "unpaid": -1},
                                      {"ms": True, "person": 1, "level": "CLEAR"},
                                      {"ms": 1, "person": 1, "level": "CLEAR", "taken": 3}])
    assert [v["taken"] for v in good3] == [3] and dropped3 == 4
    long, _ = clean_verdicts([{"ms": 1, "person": 1, "level": "CLEAR", "reason": "가" * 900}])
    assert len(long[0]["reason"]) == 500


def test_import_verdicts_attaches_by_scene_name_and_follows_scene(tmp_path):
    lib = Library(tmp_path / "library")
    f = lib.create_folder("4차 촬영")["id"]
    s1, _ = lib.add_scene(f, "C05", "4차 촬영", "4차 촬영/C05")
    fake_video(lib, s1, "cam1")
    path = tmp_path / "v.json"
    path.write_text(json.dumps({"source": "추적 도구 판정 · 2026-11-24", "scenes": {
        "C05": [{"ms": 26666.0, "person": 1, "level": "HIGH_RISK", "reason": "r", "taken": 3, "paid": 1, "unpaid": 2},
                {"ms": 17066.0, "person": 2, "level": "CLEAR", "reason": "c"}],
        "C99": []}}, ensure_ascii=False), encoding="utf-8")
    c = run_import_verdicts(lib, "4차 촬영", path, log=lambda _m: None)
    assert (c["set"], c["no_scene"], c["dropped"]) == (1, 1, 0)
    scene = lib.scene(s1)
    assert [v["level"] for v in scene["verdicts"]] == ["CLEAR", "HIGH_RISK"]
    assert scene["verdict_source"] == "추적 도구 판정 · 2026-11-24"
    # 다시 넣으면 바뀜
    path.write_text(json.dumps({"scenes": {"C5": []}}), encoding="utf-8")
    run_import_verdicts(lib, "4차 촬영", path, log=lambda _m: None)
    assert lib.scene(s1)["verdicts"] == [] and lib.scene(s1)["verdict_source"] is None
    # 원본 영상을 바꾸면 판정도 지운다 (새 영상 기준으로 다시 넣어야 함)
    lib.set_verdicts(s1, [{"ms": 1.0, "person": 1, "level": "CLEAR", "zone": "", "reason": "", "taken": 0, "paid": 0,
                           "unpaid": 0}], "src")
    fake_video(lib, s1, "cam2")                                  # 새 카메라 추가는 그대로
    assert len(lib.scene(s1)["verdicts"]) == 1
    fake_video(lib, s1, "cam1")                                  # 있던 원본 교체 → 판정 지움
    assert lib.scene(s1)["verdicts"] == [] and lib.scene(s1)["verdict_source"] is None
    # 판정 없는 장면은 빈 목록, 장면을 지우면 판정도
    lib.set_verdicts(s1, [], "src")
    lib.delete_scenes([s1])
    assert lib._db.execute("SELECT COUNT(*) FROM scene_verdicts").fetchone()[0] == 0
    with pytest.raises(LibraryError):
        path.write_text("[]", encoding="utf-8")
        run_import_verdicts(lib, "4차 촬영", path, log=lambda _m: None)
    lib.close()


def test_verdicts_only_dry_run_does_not_write(tmp_path):
    from hawkeye_server.library import main as library_main

    root = tmp_path / "library"
    lib = Library(root)
    f = lib.create_folder("4차 촬영")["id"]
    s1, _ = lib.add_scene(f, "C05", "4차 촬영", "4차 촬영/C05")
    fake_video(lib, s1, "cam1")
    lib.set_verdicts(s1, [{"ms": 5.0, "person": 1, "level": "WARNING", "zone": "", "reason": "옛 판정",
                           "taken": 0, "paid": 0, "unpaid": 0}], "옛")
    lib.close()
    path = tmp_path / "v.json"
    path.write_text(json.dumps({"source": "새", "scenes": {"C05": []}}), encoding="utf-8")
    assert library_main(["--library", str(root), "import", "--folder", "4차 촬영", "--verdicts", str(path), "--dry-run"]) == 0
    lib = Library(root)
    assert lib.scene(s1)["verdicts"][0]["reason"] == "옛 판정" and lib.scene(s1)["verdict_source"] == "옛"
    lib.close()
    assert library_main(["--library", str(root), "import", "--folder", "4차 촬영", "--verdicts", str(path)]) == 0
    lib = Library(root)
    assert lib.scene(s1)["verdicts"] == [] and lib.scene(s1)["verdict_source"] == "새"
    lib.close()


def test_api_scene_has_verdicts_field(api, tmp_path):
    lib = Library(tmp_path / "library")
    f1, _, s1, *_ = seed(lib)
    lib.set_verdicts(s1, [{"ms": 1000.0, "person": 1, "level": "WARNING", "zone": "", "reason": "x",
                           "taken": 1, "paid": 0, "unpaid": 1}], "src")
    lib.close()
    scenes = api.get(f"/api/library/folders/{f1}", headers=KEY).json()["scenes"]
    assert scenes[0]["verdicts"][0]["level"] == "WARNING" and scenes[0]["verdict_source"] == "src"
    assert scenes[1]["verdicts"] == [] and scenes[1]["verdict_source"] is None
