"""관찰 층 저장·재생 (src/replay/cache.py) — 가짜 탐지기·추적기·외형 모델로, 영상 없이."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.camera_worker import CameraWorker  # noqa: E402
from src.core.config import AppConfig, CameraConfig, IdentityConfig  # noqa: E402
from src.core.types import Detection, Frame, TrackObservation  # noqa: E402
from src.identity.registry import IdentityRegistry  # noqa: E402
from src.replay import cache as rc  # noqa: E402

W, H = 640, 480


class FakeDetector:
    """사람 둘: 하나는 왼쪽에 서 있고 하나는 오른쪽으로 걸어간다. 화면(image)이 있어야만 답한다."""

    def detect(self, frame: Frame):
        assert frame.image is not None and frame.image.shape[:2] == (H, W)
        x = 300.0 + frame.index * 2.0
        return [Detection((50.0, 100.0, 120.0, 400.0), 0.9, 0, "person"),
                Detection((x, 120.0, x + 70.0, 420.0), 0.8, 0, "person")]


class FakeTracker:
    def reset(self):
        pass

    def update(self, frame: Frame, detections):
        return [TrackObservation(track_id=i + 1, bbox=d.bbox, frame=frame.index, timestamp=frame.timestamp, score=d.score)
                for i, d in enumerate(detections)]


class FakeEmbedder:
    """상자 x 위치로 정해지는 벡터 — 화면이 진짜(0 이 아닌 값)일 때만 뽑는다. 호출 수를 센다."""

    def __init__(self):
        self.calls = 0

    def embed(self, image, boxes):
        assert image is not None and image[0, 0, 0] == 7, "재생 중에는 진짜 외형 모델을 부르면 안 된다"
        self.calls += 1
        out = []
        for b in np.asarray(boxes, dtype=np.float32):
            v = np.zeros(8, dtype=np.float32)
            v[0 if b[0] < 200 else 1] = 1.0
            v[2] = 0.01 * float(b[0] % 7)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


def frames(n=30):
    image = np.full((H, W, 3), 7, dtype=np.uint8)
    return [Frame(index=i * 2, timestamp=f"t{i}", pts_ms=i * 66.7, image=image, camera_id="angle1", wall_ms=1000.0 + i)
            for i in range(n)]


class ListSource:
    width, height, fps = W, H, 30.0

    def __init__(self, items):
        self.items = items

    def __iter__(self):
        return iter(self.items)


def make_pipe(embedder, identity: IdentityConfig | None = None):
    cam = CameraConfig("angle1", "none.mp4", "overview", stride=2)
    registry = IdentityRegistry(identity or IdentityConfig(min_tensor_height=0, edge_margin=0.0), embedder)
    worker = CameraWorker("angle1", FakeDetector(), FakeTracker(), registry)
    pipe = SimpleNamespace(overview_cam=cam, overview=worker, detail_cam=None, detail=None,
                           _source=lambda cam, start_s=None: ListSource(frames()))
    return pipe, cam, worker


def run(pipe, cam, worker):
    out = []
    for frame in pipe._source(cam):
        step = worker.step(frame)
        out.append([(o.person_id, o.track_id, tuple(o.bbox)) for o in step.identities])
    return out


def record(tmp_path, fp=None):
    pipe, cam, worker = make_pipe(FakeEmbedder())
    caches = rc.attach_recording(pipe, {"angle1": fp or {"k": 1}})
    direct = run(pipe, cam, worker)
    path = tmp_path / "angle1.pkl.gz"
    caches["angle1"].save(path)
    return direct, path


def test_replay_reproduces_direct_run_without_images_or_models(tmp_path):
    direct, path = record(tmp_path)
    cache = rc.CameraCache.load(path)
    assert (cache.width, cache.height, cache.fps) == (W, H, 30.0) and len(cache.frames) == 30
    pipe, cam, worker = make_pipe(None)                       # 외형 모델 없음
    worker.detector = worker.tracker = None                    # 탐지기·추적기도 없음
    rc.attach_replay(pipe, {"angle1": cache}, {"angle1": {"k": 1}})
    assert run(pipe, cam, worker) == direct
    assert cache.spare_used == 0                               # 설정이 같으면 요청분만으로 끝난다


def test_replay_frames_are_placeholders_with_right_shape(tmp_path):
    _, path = record(tmp_path)
    src = rc.ReplaySource(rc.CameraCache.load(path), stride=2)
    first = next(iter(src))
    assert first.image.shape == (H, W, 3) and first.index == 0 and first.timestamp == "t0" and first.wall_ms == 1000.0
    assert first.image.base is not None and first.image.base.size <= 3   # 메모리를 잡지 않는 자리표시


def test_missing_frame_raises_instead_of_silently_empty(tmp_path):
    _, path = record(tmp_path)
    cache = rc.CameraCache.load(path)
    del cache.tracks[10]
    pipe, cam, worker = make_pipe(None)
    rc.attach_replay(pipe, {"angle1": cache}, {"angle1": {"k": 1}})
    with pytest.raises(rc.CacheMiss, match="추적"):
        run(pipe, cam, worker)


def test_changed_fingerprint_is_refused(tmp_path):
    _, path = record(tmp_path, fp={"ignore_image": [[0.1, 0.1, 0.2, 0.2]], "tracker": {"name": "botsort"}})
    cache = rc.CameraCache.load(path)
    pipe, _, _ = make_pipe(None)
    with pytest.raises(rc.FingerprintMismatch, match="ignore_image"):
        rc.attach_replay(pipe, {"angle1": cache}, {"angle1": {"ignore_image": [], "tracker": {"name": "botsort"}}})
    with pytest.raises(rc.FingerprintMismatch, match="tracker.name"):
        rc.attach_replay(pipe, {"angle1": cache},
                         {"angle1": {"ignore_image": [[0.1, 0.1, 0.2, 0.2]], "tracker": {"name": "bytetrack"}}})


def test_spare_vectors_serve_requests_the_recording_run_never_made(tmp_path):
    """신원 설정이 달라져 기록 때 요청하지 않은 상자를 요청해도 예비(모든 추적 상자)로 답한다 — 쓴 횟수를 센다."""
    _, path = record(tmp_path)
    cache = rc.CameraCache.load(path)
    assert len(cache.embed_spare) == 60 and 0 < len(cache.embed_asked) <= 60
    cache.embed_asked.clear()
    pipe, cam, worker = make_pipe(None)
    rc.attach_replay(pipe, {"angle1": cache}, {"angle1": {"k": 1}})
    run(pipe, cam, worker)
    assert cache.spare_used > 0
    cache.current = 0
    with pytest.raises(rc.CacheMiss, match="외형 벡터"):
        rc.CachedEmbedder(cache).embed(None, np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))


def write_zones(tmp_path, ignore, polygon_x):
    floor = tmp_path / "floor.yaml"
    floor.write_text(f"ignore_image: {ignore}\nzones:\n- name: aisle\n  polygon: [[{polygon_x}, 0], [1, 0], [1, 1]]\n", encoding="utf-8")
    return floor


def test_fingerprint_tracks_ignore_regions_but_not_floor_zones(tmp_path):
    """사람 아님 영역은 탐지 결과를 바꾸므로 유효 조건에 들어가고, 바닥 구역 좌표는 들어가지 않는다 (다시 계산하려는 것)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import yaml

    cfg = AppConfig()
    video = tmp_path / "a.mp4"
    video.write_bytes(b"x" * 10)

    def fp(ignore, polygon_x, **identity):
        floor = write_zones(tmp_path, ignore, polygon_x)
        cam = CameraConfig("angle1", str(video), "overview", str(floor), 0.0, 4, "yolo-pose")
        for k, v in identity.items():
            setattr(cfg.identity, k, v)
        raw = yaml.safe_load(floor.read_text(encoding="utf-8"))
        return rc.fingerprint(cfg, cam, ignore_image=raw.get("ignore_image") or [])

    base = fp([[0.1, 0.1, 0.2, 0.2]], 0.0)
    assert rc.diff_fingerprint(base, fp([[0.1, 0.1, 0.2, 0.2]], 0.5, late_merge_threshold=0.5)) == []
    assert any("ignore_image" in d for d in rc.diff_fingerprint(base, fp([[0.1, 0.1, 0.3, 0.2]], 0.0)))
    cfg.tracker.name = "botsort" if cfg.tracker.name != "botsort" else "bytetrack"
    assert any("tracker.name" in d for d in rc.diff_fingerprint(base, fp([[0.1, 0.1, 0.2, 0.2]], 0.0)))


def test_run_recorded_without_appearance_model_replays_without_one(tmp_path):
    pipe, cam, worker = make_pipe(None)
    caches = rc.attach_recording(pipe, {"angle1": {"k": 1}})
    direct = run(pipe, cam, worker)
    path = tmp_path / "angle1.pkl.gz"
    caches["angle1"].save(path)
    cache = rc.CameraCache.load(path)
    assert cache.has_embedder is False and not cache.embed_asked
    pipe, cam, worker = make_pipe(None)
    worker.detector = worker.tracker = None
    rc.attach_replay(pipe, {"angle1": cache}, {"angle1": {"k": 1}})
    assert worker.registry.embedder is None
    assert run(pipe, cam, worker) == direct


def test_detail_interaction_override_reaches_the_config(tmp_path, monkeypatch):
    """기본 config 에는 detail_interaction 절이 없다 — dual_config 가 만든 뒤에 --set 이 얹혀야 한다."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import dual_eval

    cfg = AppConfig()
    assert cfg.detail_interaction is None
    dual_eval._apply_set(cfg, ["detail_interaction.dwell_seconds=0.5", "identity.late_merge_threshold=0.9"], early=True)
    assert cfg.identity.late_merge_threshold == 0.9
    from src.core.config import InteractionConfig
    cfg.detail_interaction = InteractionConfig()
    dual_eval._apply_set(cfg, ["detail_interaction.dwell_seconds=0.5"])
    assert cfg.detail_interaction.dwell_seconds == 0.5
