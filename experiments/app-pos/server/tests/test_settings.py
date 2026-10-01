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
