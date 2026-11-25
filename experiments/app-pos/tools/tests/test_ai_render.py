"""AI 화면 그리기 — 기록 읽기 · 보간 · 번호 맞추기 · 장면 짝 (영상 없이 도는 것만, 하나는 짧은 합성 영상으로)."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ai_render as ar  # noqa: E402

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def write_identities(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_track_interpolates_close_samples_and_holds_briefly(tmp_path):
    times = [i * 100.0 for i in range(20)]          # 장면 번호 → ms
    write_identities(tmp_path / "id.jsonl", [
        {"person_id": 1, "frame": 2, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"person_id": 1, "frame": 4, "bbox": [10, 0, 20, 10], "score": 0.9},
        {"person_id": 1, "frame": 4, "bbox": [99, 99, 99, 99], "score": 0.1},   # 같은 장면 점수 낮은 것은 버림
        {"person_id": 2, "frame": 10, "bbox": [5, 5, 15, 15], "score": 0.8},
        {"person_id": 2, "frame": 99, "bbox": [0, 0, 1, 1], "score": 0.8},      # 기록 밖 장면은 무시
    ])
    tr = ar.load_track(tmp_path / "id.jsonl", times, period_ms=200.0)
    assert tr.boxes_at(150) == []                                           # 처음 기록 전
    assert tr.boxes_at(200) == [(1, (0.0, 0.0, 10.0, 10.0))]
    assert tr.boxes_at(300) == [(1, (5.0, 0.0, 15.0, 10.0))]                 # 앞뒤 기록 사이 보간
    assert tr.boxes_at(650) == [(1, (10.0, 0.0, 20.0, 10.0))]                # 마지막 기록 뒤 1.5 간격(300ms)까지 유지
    assert tr.boxes_at(800) == []
    assert tr.boxes_at(1000) == [(2, (5.0, 5.0, 15.0, 15.0))]


def test_unassigned_people_stay_apart_and_get_no_label(tmp_path):
    """신원이 아직 안 정해진 관측(-1)은 추적 번호마다 따로 — 한 사람으로 이어 그리지 않고, 이름표도 없다."""
    times = [i * 100.0 for i in range(10)]
    write_identities(tmp_path / "id.jsonl", [
        {"person_id": -1, "track_id": 11, "frame": 2, "bbox": [0, 0, 10, 10]},
        {"person_id": -1, "track_id": 22, "frame": 2, "bbox": [100, 0, 110, 10]},
        {"person_id": -1, "track_id": 22, "frame": 3, "bbox": [102, 0, 112, 10]},
        {"person_id": -1, "track_id": 11, "frame": 4, "bbox": [4, 0, 14, 10]},
        {"person_id": -1, "frame": 4, "bbox": [50, 50, 60, 60]},                 # 추적 번호도 없으면 버림
    ])
    tr = ar.load_track(tmp_path / "id.jsonl", times, period_ms=100.0)
    at = dict(tr.boxes_at(200))
    assert len(at) == 2                                                       # 같은 장면 두 사람 모두
    assert at[ar.track_key(-1, 11)] == (0.0, 0.0, 10.0, 10.0) and at[ar.track_key(-1, 22)] == (100.0, 0.0, 110.0, 10.0)
    assert dict(tr.boxes_at(300))[ar.track_key(-1, 11)] == (2.0, 0.0, 12.0, 10.0)   # 11 은 11 끼리만 보간
    swaps = ar.Swaps([])
    assert ar.person_number("cam1", ar.track_key(-1, 11), 200, 0.0, lambda p: p, swaps, None) is None
    assert ar.person_number("cam2", ar.track_key(-1, 22), 200, 0.0, lambda p: p, swaps, ar.Links([])) is None
    assert ar.track_key(3, 11) == 3 and ar.track_key(-1, None) is None


def test_grid_follows_real_frame_times():
    # 30fps 인데 0.067 장면이 빠진 영상: 출력 k 번째(시각 k/30)는 그때 떠 있는 원본 장면
    pts = [0.0, 1 / 30, 3 / 30, 4 / 30]
    assert ar.grid_sources(pts, 30.0) == [0, 1, 1, 2, 3]
    even = [i / 30 for i in range(5)]
    assert ar.grid_sources(even, 30.0) == [0, 1, 2, 3, 4]
    assert ar.nominal_fps(pts, 25.0) == 30.0 and ar.nominal_fps([0.0], 25.0) == 25.0


def test_far_samples_are_not_interpolated(tmp_path):
    times = [i * 100.0 for i in range(40)]
    write_identities(tmp_path / "id.jsonl", [
        {"person_id": 3, "frame": 0, "bbox": [0, 0, 10, 10]},
        {"person_id": 3, "frame": 30, "bbox": [100, 0, 110, 10]},               # 3초 뒤 — 사이를 잇지 않음
    ])
    tr = ar.load_track(tmp_path / "id.jsonl", times, period_ms=200.0)
    assert tr.boxes_at(1500) == []
    assert tr.boxes_at(3000) == [(3, (100.0, 0.0, 110.0, 10.0))]


def test_links_map_shelf_person_to_overview_number():
    links = ar.Links([(1000, 2000, 1, 7, 0.8), (1000, 2000, 1, 9, 0.95), (3000, 4000, 2, 5, 0.9)])
    assert links.overview_for(1, 1500) == 9                                   # 같은 시각 둘이면 점수 높은 것
    assert links.overview_for(1, 2300) == 9                                   # 400ms 안이면 가장 가까운 것
    assert links.overview_for(1, 2600) is None
    assert links.overview_for(3, 1500) is None


def test_times_fall_back_to_frame_over_fps_for_shoot2(tmp_path):
    rec = tmp_path / "C1"
    rec.mkdir()
    assert ar.load_times(rec, "angle1", {"fps": 50, "frames": 4}) == [0.0, 20.0, 40.0, 60.0]
    (rec / "times_angle1.json").write_text("[0, 21.2, 37.9]", encoding="utf-8")
    assert ar.load_times(rec, "angle1", {"fps": 50, "frames": 4}) == [0.0, 21.2, 37.9]
    with pytest.raises(RuntimeError):
        ar.load_times(rec, "angle0", {})


def test_merges_resolve_to_the_end(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text('{"from": 3, "into": 2, "frame": 858}\n{"from": 5, "into": 3, "frame": 900}\n', encoding="utf-8")
    r = ar.load_merges(p)
    assert (r(3), r(5), r(2), r(7)) == (2, 2, 2, 7)
    assert ar.load_merges(tmp_path / "없음.jsonl")(4) == 4


def test_overview_swap_renames_from_when_it_happened(tmp_path):
    # 4차 C05: 9.6초부터 화면 1·2 가 맞바뀐 것을 되돌림 — 그때부터 화면 2 = 손님 1
    p = tmp_path / "s.jsonl"
    p.write_text('{"ms": 9599.8, "found_ms": 10133.1, "screen": [1, 2], "store": [1, 2]}\n', encoding="utf-8")
    sw = ar.load_swaps(p)
    assert (sw.store_id(1, 9000), sw.store_id(2, 9000)) == (1, 2)
    assert (sw.store_id(1, 14000), sw.store_id(2, 14000)) == (2, 1)
    assert sw.store_id(3, 14000) == 3
    seen = ar.load_swaps(tmp_path / "x.jsonl")
    assert seen.store_id(1, 1e9) == 1
    p.write_text('{"ms": 5000, "seen_ms": 7000, "screen": [1, 2], "store": [1, 2]}\n', encoding="utf-8")
    assert ar.load_swaps(p).store_id(1, 6000) == 1                          # 바뀐 때(seen_ms)부터


def test_both_cameras_show_the_same_number_after_merge_and_swap():
    swaps = ar.Swaps([(9600.0, (1, 2), (1, 2))])
    resolve0 = lambda pid: {3: 2}.get(pid, pid)                             # 선반 화면 3 은 2 로 합쳐짐
    links = ar.Links([(9000, 20000, 3, 1, 0.9)], resolve0)                   # 짝은 맺을 때 번호(3), 전체는 매장 신원(1)
    # 14초: 전체 카메라 화면 2 = 손님 1, 선반 화면 2(합쳐진 3) = 손님 1
    assert ar.person_number("cam1", 2, 14000, 0.0, lambda p: p, swaps, None) == 1
    assert ar.person_number("cam2", 2, 14000 - 500, 500.0, resolve0, swaps, links) == 1
    assert ar.person_number("cam2", 9, 14000, 0.0, resolve0, swaps, links) is None   # 짝 없음 → 상자만


def test_scale_box_and_colors():
    assert ar.scale_box((300, 150, 900, 1050), 1280 / 1920, 720 / 1080) == (200, 100, 600, 700)
    assert ar.color_for(1) == ar.color_for(1 + len(ar.PALETTE))
    assert ar.color_for(None) == ar.UNLINKED


def make_record(root: Path, name: str, angles=("angle1", "angle0")):
    d = root / name
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({a: {"fps": 30, "frames": 60, "width": 1920, "height": 1080, "stride": 2}
                                             for a in angles}), encoding="utf-8")
    for a in angles:
        (d / f"times_{a}.json").write_text(json.dumps([i * 33.333 for i in range(60)]), encoding="utf-8")
        write_identities(d / a / "identities.jsonl", [{"person_id": 1, "frame": f, "bbox": [600, 300, 900, 900]}
                                                       for f in range(0, 60, 2)])
    return d


def test_plan_jobs_matches_scene_names_and_offsets(tmp_path):
    rec = tmp_path / "rec"
    make_record(rec, "C1")
    make_record(rec, "N8-1", angles=("angle1",))
    cam1, cam2 = tmp_path / "1(angle1)", tmp_path / "1(angle0)"
    cam1.mkdir()
    cam2.mkdir()
    for p in (cam1 / "C01.mp4", cam2 / "20261028_C1.mp4", cam1 / "N08-1.mp4", cam2 / "20261028_N8-1.mp4", cam1 / "T01.mp4"):
        p.write_bytes(b"x")
    sync = [("1(angle0)/20261028_C1.mp4", "1(angle1)/C01.mp4", -1200.0)]
    jobs, skipped = ar.plan_jobs(rec, {"cam1": cam1, "cam2": cam2}, tmp_path / "out", sync)
    assert [(j.scene, j.camera, j.offset_ms) for j in jobs] == [("C01", "cam1", 0.0), ("C01", "cam2", -1200.0),
                                                                ("N08-1", "cam1", 0.0)]
    assert jobs[1].out == tmp_path / "out" / "cam2" / "20261028_C1.mp4"
    assert skipped == ["N08-1 cam2: 이 카메라 기록 없음", "T01: 추적 기록 없음"]


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg 없음")
def test_render_accepts_60fps_record_on_30fps_video_without_times(tmp_path):
    """2차처럼 times 가 없고, 기록은 원본 60fps(120장면)인데 영상은 보관함 변환본 30fps(60장면)."""
    rec = tmp_path / "rec" / "C1"
    rec.mkdir(parents=True)
    (rec / "meta.json").write_text(json.dumps({"angle1": {"fps": 60, "frames": 120, "width": 1920, "height": 1080, "stride": 4}}),
                                   encoding="utf-8")
    write_identities(rec / "angle1" / "identities.jsonl", [{"person_id": 1, "frame": f, "bbox": [600, 300, 900, 900]}
                                                            for f in range(0, 120, 4)])
    cam1 = tmp_path / "v1"
    cam1.mkdir()
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=1280x720:rate=30:duration=2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(cam1 / "C01.mp4")], check=True)
    jobs, _ = ar.plan_jobs(tmp_path / "rec", {"cam1": cam1}, tmp_path / "out", None)
    assert ar.render(jobs[0]).startswith("C01 cam1: 60장면")


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg 없음")
def test_render_keeps_length_and_size(tmp_path):
    rec = tmp_path / "rec"
    make_record(rec, "C01")
    cam1 = tmp_path / "v1"
    cam1.mkdir()
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=1280x720:rate=30:duration=2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(cam1 / "C01.mp4")], check=True)
    jobs, _ = ar.plan_jobs(rec, {"cam1": cam1}, tmp_path / "out", None)
    line = ar.render(jobs[0])
    assert line.startswith("C01 cam1: 60장면") and "상자 " in line
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,nb_read_frames,codec_name", "-of", "csv=p=0", str(jobs[0].out)],
                         capture_output=True, text=True).stdout.strip()
    assert out == "h264,1280,720,60"


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg 없음")
def test_render_keeps_real_time_on_uneven_video(tmp_path):
    """한 장면이 빠진 가변 간격 영상: 출력은 고른 간격이고 길이 · 시각이 원본과 같다."""
    rec = tmp_path / "rec"
    make_record(rec, "C01")
    cam1 = tmp_path / "v1"
    cam1.mkdir()
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=1280x720:rate=30:duration=2", "-vf", "select='not(eq(n\\,30))'",
                    "-fps_mode", "passthrough", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(cam1 / "C01.mp4")], check=True)
    pts = ar.probe_pts(cam1 / "C01.mp4")
    assert len(pts) == 59 and abs(pts[-1] - 59 / 30) < 0.01
    jobs, _ = ar.plan_jobs(rec, {"cam1": cam1}, tmp_path / "out", None)
    line = ar.render(jobs[0])
    assert "60장면" in line and "고르지 않은 간격 1곳" in line
    out_pts = ar.probe_pts(jobs[0].out)
    assert len(out_pts) == 60 and abs(out_pts[-1] - pts[-1]) < 0.01


def test_render_refuses_mismatched_length(tmp_path):
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg 없음")
    rec = tmp_path / "rec"
    make_record(rec, "C01")
    cam1 = tmp_path / "v1"
    cam1.mkdir()
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=1280x720:rate=30:duration=6", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(cam1 / "C01.mp4")], check=True)
    jobs, _ = ar.plan_jobs(rec, {"cam1": cam1}, tmp_path / "out", None)
    with pytest.raises(RuntimeError, match="길이"):
        ar.render(jobs[0])
    assert not jobs[0].out.exists() and not list((tmp_path / "out").rglob("*.part.mp4"))
