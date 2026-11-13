"""관찰 층의 '화면을 보는 계산'을 저장해 두고, 화면 없이 다시 돌린다 (docs/replay-cache.md).

화면(화소)을 쓰는 곳은 다섯 군데뿐이다 — ① 사람 탐지 ② 추적(BoT-SORT 는 외형 비교에 화면 사용) ③ 신원의 외형 벡터
④ 선반 물건 탐지 ⑤ 칸 그림 비교. 이 다섯의 결과를 장면 번호로 저장해 두면, 신원 규칙·손 뻗기·선반 지도·합치기·판정을
바꿔 다시 계산할 때 영상을 읽지도 모델을 돌리지도 않는다 (촬영 한 회 1시간 → 몇 분).

- 기록(attach_recording): 작업자의 부품을 '진짜를 부르고 결과를 적는' 감싸개로 바꾼다.
- 재생(attach_replay): 부품을 '저장분을 돌려주는' 감싸개로, 영상 입력을 ReplaySource 로 바꾼다. 없는 것을 찾으면 CacheMiss.
- 유효 조건(fingerprint): 저장분을 만든 영상·탐지기·추적기·선반 설정·사람 아님 영역이 지금과 같아야 한다. 다르면 FingerprintMismatch.
  신원·손 뻗기·위험·합치기·바닥 구역은 넣지 않는다 — 그걸 바꿔 다시 계산하려는 것이다.

외형 벡터는 두 가지를 저장한다. '요청분' = 신원 층이 실제로 요청한 호출의 결과 그대로, '예비' = 그 장면의 모든 추적 상자를
따로 뽑은 것. 재생은 요청분을 먼저 쓴다 — 설정이 같으면 직접 실행과 값이 완전히 같다. 신원 설정을 바꿔 요청이 달라지면 예비를 쓰는데,
한 번에 뽑는 묶음이 달라 부동소수 끝자리(1e-6 수준)가 직접 실행과 다를 수 있다.

파일은 pickle 이다 (np.float32 같은 값의 형을 그대로 지켜 직접 실행과 계산이 같게). 자기가 만든 outputs/ 아래 파일만 읽는다.
"""

from __future__ import annotations

import gzip
import pickle
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

from ..core.types import Frame

VERSION = 1


class CacheMiss(KeyError):
    """저장분에 없는 것을 찾았다 — 조용히 빈 값으로 넘어가지 않는다."""


class FingerprintMismatch(ValueError):
    """저장분을 만든 조건이 지금 설정과 다르다."""


def _plain(value):
    """설정 값을 비교·저장할 수 있는 모양으로 (dataclass → dict, tuple → list)."""
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def fingerprint(config, cam, *, ignore_image=None, shelf_areas=None, slots=None) -> dict:
    """이 카메라의 저장분이 유효한 조건. 영상(이름·크기·수정 시각)·처리 간격·탐지기·추적기(외형 모델 포함)·선반 설정·
    사람 아님 영역·선반 영역·칸 좌표."""
    src = Path(str(cam.source))
    stat = src.stat() if src.exists() else None
    shelves = _plain(config.shelves)
    shelves.pop("areas", None)   # 영역은 구역 파일에서 온다 (shelf_areas)
    return _plain({
        "version": VERSION,
        "video": {"name": src.name, "size": stat.st_size if stat else None, "mtime": int(stat.st_mtime) if stat else None},
        "stride": cam.stride, "start_time": config.video.start_time, "max_frames": config.video.max_frames,
        "detector": {**_plain(config.detector), "kind": cam.detector},
        "tracker": config.tracker,
        "identity_imgsz": config.detector.imgsz,
        "shelves": shelves,
        "ignore_image": ignore_image or [],
        "shelf_areas": shelf_areas or [],
        "slots": slots or [],
    })


def diff_fingerprint(saved: dict, now: dict, prefix: str = "") -> list[str]:
    """다른 항목 이름들 (사람이 읽을 수 있게)."""
    out = []
    for key in sorted(set(saved) | set(now)):
        a, b = saved.get(key), now.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            out += diff_fingerprint(a, b, f"{prefix}{key}.")
        elif a != b:
            out.append(f"{prefix}{key}: 저장 {a!r} ≠ 지금 {b!r}")
    return out


class CameraCache:
    """카메라 하나의 저장분."""

    def __init__(self, camera: str) -> None:
        self.camera = camera
        self.fingerprint: dict = {}
        self.width = self.height = 0
        self.fps = 0.0
        self.frames: list[tuple[int, float, float | None, str]] = []
        """처리한 장면 순서대로 (번호, pts_ms, wall_ms, timestamp)."""
        self.detections: dict[int, list] = {}
        self.tracks: dict[int, list] = {}
        self.embed_asked: dict[tuple[int, bytes], np.ndarray] = {}
        self.embed_spare: dict[tuple[int, bytes], np.ndarray] = {}
        self.items: dict[int, list] = {}
        self.slot_changes: dict[int, list] = {}
        self.has_embedder = True
        """기록 때 외형 모델이 있었는지 — 없이 기록했으면 재생도 없이 돈다 (예전 저장분에는 이 값이 없어 True)."""
        self.current: int | None = None
        """지금 처리 중인 장면 번호 — 장면을 모르는 부품(외형 벡터·칸 그림 비교)이 이것으로 찾는다. 저장하지 않는다."""
        self.spare_used = 0
        """재생 중 요청분이 없어 예비를 쓴 횟수 (신원 설정이 달라졌다는 뜻)."""

    # -- 파일
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v for k, v in self.__dict__.items() if k not in ("current", "spare_used")}
        with gzip.open(path, "wb", compresslevel=3) as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "CameraCache":
        if not path.exists():
            raise CacheMiss(f"저장분 파일이 없습니다: {path}")
        with gzip.open(path, "rb") as f:
            data = pickle.load(f)
        cache = cls(data["camera"])
        cache.__dict__.update(data)
        cache.current, cache.spare_used = None, 0
        return cache

    def check(self, now: dict) -> None:
        bad = diff_fingerprint(self.fingerprint, now)
        if bad:
            raise FingerprintMismatch(f"{self.camera}: 저장분을 만든 조건이 지금과 다릅니다 — " + "; ".join(bad[:6]))

    def _need(self, table: dict, key, what: str):
        if key not in table:
            raise CacheMiss(f"{self.camera}: {what} 저장분 없음 (장면 {key})")
        return table[key]


def _box_key(box) -> bytes:
    return np.asarray(box, dtype=np.float32).tobytes()


# ---------------------------------------------------------------------- 기록용 감싸개

class _Wrapper:
    def __init__(self, inner, cache: CameraCache) -> None:
        self.inner, self.cache = inner, cache

    def __getattr__(self, name):
        if name in ("inner", "cache"):
            raise AttributeError(name)
        return getattr(self.inner, name)


class RecordingDetector(_Wrapper):
    def detect(self, frame: Frame):
        c = self.cache
        c.current = frame.index
        c.frames.append((frame.index, frame.pts_ms, frame.wall_ms, frame.timestamp))
        if frame.image is not None and not c.width:
            c.height, c.width = frame.image.shape[:2]
        out = self.inner.detect(frame)
        c.detections[frame.index] = list(out)
        return out


class RecordingTracker(_Wrapper):
    def __init__(self, inner, cache: CameraCache, embedder=None) -> None:
        super().__init__(inner, cache)
        self.embedder = embedder
        """진짜 외형 모델 — 그 장면의 모든 추적 상자를 예비로 뽑아 둔다."""

    def update(self, frame: Frame, detections):
        out = self.inner.update(frame, detections)
        c = self.cache
        c.current = frame.index
        c.tracks[frame.index] = list(out)
        if self.embedder is not None and out and frame.image is not None:
            boxes = np.array([o.bbox for o in out], dtype=np.float32)
            vecs = self.embedder.embed(frame.image, boxes)
            if vecs is not None:
                for box, vec in zip(boxes, vecs):
                    c.embed_spare[(frame.index, box.tobytes())] = np.array(vec, copy=True)
        return out


class RecordingEmbedder(_Wrapper):
    def embed(self, image, boxes):
        out = self.inner.embed(image, boxes)
        if out is not None:
            for box, vec in zip(np.asarray(boxes, dtype=np.float32), out):
                self.cache.embed_asked[(self.cache.current, box.tobytes())] = np.array(vec, copy=True)
        return out


class RecordingItemDetector(_Wrapper):
    def detect(self, frame: Frame):
        out = self.inner.detect(frame)
        self.cache.items[frame.index] = list(out)
        return out


class RecordingSlotDiff(_Wrapper):
    def update(self, now_ms, image, people_boxes):
        out = self.inner.update(now_ms, image, people_boxes)
        self.cache.slot_changes[self.cache.current] = list(out)
        return out


# ---------------------------------------------------------------------- 재생용 감싸개

class CachedDetector:
    def __init__(self, cache: CameraCache) -> None:
        self.cache = cache

    def detect(self, frame: Frame):
        self.cache.current = frame.index
        return list(self.cache._need(self.cache.detections, frame.index, "사람 탐지"))


class CachedTracker:
    def __init__(self, cache: CameraCache) -> None:
        self.cache = cache

    def reset(self) -> None:
        pass

    def update(self, frame: Frame, detections):
        self.cache.current = frame.index
        return list(self.cache._need(self.cache.tracks, frame.index, "추적"))


class CachedEmbedder:
    def __init__(self, cache: CameraCache) -> None:
        self.cache = cache

    def embed(self, image, boxes):
        if boxes is None or len(boxes) == 0:
            return None
        c = self.cache
        rows = []
        for box in np.asarray(boxes, dtype=np.float32):
            key = (c.current, box.tobytes())
            if key in c.embed_asked:
                rows.append(c.embed_asked[key])
            elif key in c.embed_spare:
                c.spare_used += 1
                rows.append(c.embed_spare[key])
            else:
                raise CacheMiss(f"{c.camera}: 외형 벡터 저장분 없음 (장면 {c.current}, 상자 {box.tolist()})")
        return np.stack(rows).astype(np.float32, copy=False)


class CachedItemDetector:
    def __init__(self, cache: CameraCache) -> None:
        self.cache = cache

    def detect(self, frame: Frame):
        return list(self.cache._need(self.cache.items, frame.index, "선반 물건 탐지"))


class CachedSlotDiff:
    def __init__(self, cache: CameraCache) -> None:
        self.cache = cache

    def reset(self) -> None:
        pass

    def update(self, now_ms, image, people_boxes):
        return list(self.cache._need(self.cache.slot_changes, self.cache.current, "칸 그림 비교"))


class ReplaySource:
    """저장된 장면 목록을 그대로 낸다. 화면은 크기만 맞는 자리표시(메모리 없음) — 화소를 보는 부품은 모두 저장분으로 바뀌어 있다."""

    def __init__(self, cache: CameraCache, stride: int = 1) -> None:
        self.cache = cache
        self.camera_id = cache.camera
        self.width, self.height = cache.width, cache.height
        self.fps = cache.fps
        self.total_frames = len(cache.frames) * max(1, stride)
        self._blank = np.broadcast_to(np.zeros((1, 1, 3), dtype=np.uint8), (max(1, cache.height), max(1, cache.width), 3))

    def __iter__(self):
        for index, pts_ms, wall_ms, timestamp in self.cache.frames:
            yield Frame(index=index, timestamp=timestamp, pts_ms=pts_ms, image=self._blank,
                        camera_id=self.camera_id, wall_ms=wall_ms)

    def release(self) -> None:
        pass


# ---------------------------------------------------------------------- 파이프라인에 끼우기

def _workers(pipe):
    out = [(pipe.overview_cam, pipe.overview)]
    if pipe.detail is not None:
        out.append((pipe.detail_cam, pipe.detail))
    return out


def _wrap_slot_diff(shelves, make) -> None:
    """slot_diff 는 ShelfWatcher.resolve 에서 만들어진다 — 만든 직후 감싼다."""
    resolve = shelves.resolve

    def resolved(width, height, fps):
        resolve(width, height, fps)
        if getattr(shelves, "slot_diff", None) is not None:
            shelves.slot_diff = make(shelves.slot_diff)
    shelves.resolve = resolved


def attach_recording(pipe, fingerprints: dict[str, dict]) -> dict[str, CameraCache]:
    """파이프라인의 부품을 기록용으로 바꾼다. 돌린 뒤 돌려받은 저장분을 save 한다."""
    caches = {}
    for cam, worker in _workers(pipe):
        cache = caches[cam.name] = CameraCache(cam.name)
        cache.fingerprint = fingerprints[cam.name]
        real_embedder = worker.registry.embedder if worker.registry is not None else None
        cache.has_embedder = real_embedder is not None
        worker.detector = RecordingDetector(worker.detector, cache)
        worker.tracker = RecordingTracker(worker.tracker, cache, real_embedder)
        if real_embedder is not None:
            worker.registry.embedder = RecordingEmbedder(real_embedder, cache)
        if worker.shelves is not None:
            worker.shelves.detector = RecordingItemDetector(worker.shelves.detector, cache)
            _wrap_slot_diff(worker.shelves, lambda inner, cache=cache: RecordingSlotDiff(inner, cache))
    source = pipe._source

    def recording_source(cam, start_s=None):
        src = source(cam, start_s)
        caches[cam.name].fps = float(src.fps)
        caches[cam.name].width, caches[cam.name].height = int(src.width), int(src.height)
        return src
    pipe._source = recording_source
    return caches


def attach_replay(pipe, caches: dict[str, CameraCache], fingerprints: dict[str, dict]) -> None:
    """파이프라인의 부품을 저장분으로 바꾼다. 유효 조건이 다르면 FingerprintMismatch."""
    for cam, worker in _workers(pipe):
        cache = caches[cam.name]
        cache.check(fingerprints[cam.name])
        worker.detector = CachedDetector(cache)
        worker.tracker = CachedTracker(cache)
        if worker.registry is not None:
            worker.registry.embedder = CachedEmbedder(cache) if cache.has_embedder else None
        if worker.shelves is not None:
            worker.shelves.detector = CachedItemDetector(cache)
            _wrap_slot_diff(worker.shelves, lambda inner, cache=cache: CachedSlotDiff(cache))
    pipe._source = lambda cam, start_s=None: ReplaySource(caches[cam.name], cam.stride)
