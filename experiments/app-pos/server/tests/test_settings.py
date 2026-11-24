"""환경변수로 서버 설정 읽기."""

from pathlib import Path

from hawkeye_server.__main__ import parse_streams, settings_from_env


def test_parse_streams_reads_camera_url_pairs():
    assert parse_streams("cam1=rtsp://10.0.0.2:8554/cam1, cam2=rtsp://10.0.0.2:8554/cam2") == [
        {"camera_id": "cam1", "url": "rtsp://10.0.0.2:8554/cam1"},
        {"camera_id": "cam2", "url": "rtsp://10.0.0.2:8554/cam2"},
    ]


def test_parse_streams_empty_is_no_streams():
    assert parse_streams("") == []


def test_settings_from_env_defaults_to_local_data_dir():
    s = settings_from_env({})
    assert (s.db_path, s.snapshot_dir, s.api_key, s.streams) == (
        Path("data/hawkeye.db"), Path("data/snapshots"), None, [])


def test_settings_from_env_reads_values():
    s = settings_from_env({
        "HAWKEYE_DB": "x/a.db", "HAWKEYE_SNAPSHOTS": "x/snap", "HAWKEYE_API_KEY": "k",
        "HAWKEYE_STREAMS": "cam1=rtsp://h/cam1",
    })
    assert (s.db_path, s.snapshot_dir, s.api_key) == (Path("x/a.db"), Path("x/snap"), "k")
    assert s.streams == [{"camera_id": "cam1", "url": "rtsp://h/cam1"}]


def test_settings_from_env_reads_recordings_and_ai_streams():
    s = settings_from_env({
        "HAWKEYE_STREAMS": "cam1=rtsp://h/cam1,cam2=rtsp://h/cam2",
        "HAWKEYE_AI_STREAMS": "cam1=rtsp://h/cam1_ai",
        "HAWKEYE_PLAYBACK_URL": "http://127.0.0.1:9996",
        "HAWKEYE_CLIPS": "x/clips", "HAWKEYE_RECORDINGS": "x/rec", "HAWKEYE_MIN_FREE_GB": "50",
    })
    assert s.streams == [{"camera_id": "cam1", "url": "rtsp://h/cam1", "ai_url": "rtsp://h/cam1_ai"},
                         {"camera_id": "cam2", "url": "rtsp://h/cam2"}]
    assert (s.playback_url, s.clips_dir, s.disk_path, s.min_free_gb) == (
        "http://127.0.0.1:9996", Path("x/clips"), Path("x/rec"), 50.0)


def test_recordings_off_by_default():
    s = settings_from_env({})
    assert (s.playback_url, s.disk_path) == (None, None)
