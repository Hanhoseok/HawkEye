"""카메라 2대 파이프라인 — 전체 카메라(angle1) + 선반 카메라(angle0)를 함께 돌린다 (docs/multiview-design.md §4).

    전체 카메라 → 카메라 작업자(사람·신원·바닥 위치·선반 앞 머묾) ─┐
    선반 카메라 → 카메라 작업자(사람+관절·신원·손-칸 접촉·선반 지도) ┼→ ⑦ 합치기(DualJudge) → ⑤ 손님 장부 → ⑥ 위험 판정
    선반 센서(있으면) ──────────────────────────────────────────────┘

두 영상의 장면을 공통 시각(pts_ms, 시차를 반영한 값) 순서로 섞어 하나씩 처리한다.
판단 층과 합치기 층은 영상을 모르고, 여기서 쓰는 기록 파일만으로 다시 돌릴 수 있다 (scripts/dual_eval.py judge).
"""

from __future__ import annotations

import heapq
import json
import time
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

from .camera_worker import CameraWorker
from .core.config import AppConfig, CameraConfig, ShelfConfig
from .core.types import RiskLevel
from .fusion.judge import DualJudge
from .fusion.shelf_events import TimeAxis, load_sensor_events
from .inputs.video_source import VideoSource
from .sinks.observation_log import ObservationLog
from .zones.floor import FloorMap, FloorZoneMap
from .zones.zone_map import ZoneMap


@dataclass
class DetailSetup:
    """선반 카메라 구역 파일(angle0.yaml)에서 읽는 것: 칸(SHELF 구역), 선반 지도 영역, 띠 축 변환."""

    zone_map: ZoneMap
    shelves: list[dict]
    band_map: tuple[float, float]
    slot_shelf: dict[str, str]
    empty_slots: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "DetailSetup":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        zm = ZoneMap.load(path)
        shelves = raw.get("shelves") or []
        first = shelves[0]["name"] if shelves else "shelf"
        slot_shelf = dict(raw.get("slot_shelf") or {})
        for z in zm.zones:
            slot_shelf.setdefault(z.name, first)
        a, b = raw.get("band_map") or (1.0, 0.0)
        return cls(zm, shelves, (float(a), float(b)), slot_shelf, dict(raw.get("empty_slots") or {}))


@dataclass
class DualStats:
    frames: dict = field(default_factory=dict)
    elapsed_sec: float = 0.0
    events: list = field(default_factory=list)
    links: int = 0
    summary_lines: list = field(default_factory=list)

    def summary(self) -> str:
        lines = ["─" * 52]
        for cam, n in self.frames.items():
            lines.append(f"{cam:8s} 처리 장면 : {n}")
        total = sum(self.frames.values())
        if self.elapsed_sec:
            lines.append(f"처리 속도        : {total / self.elapsed_sec:.2f} 장면/초 (두 카메라 합, {self.elapsed_sec:.1f}s)")
        lines.append(f"카메라 사이 짝   : {self.links}건")
        lines += self.summary_lines
        lines.append("─" * 52)
        return "\n".join(lines)


class MultiCameraPipeline:
    def __init__(self, config: AppConfig, overview: CameraWorker, detail: CameraWorker | None,
                 overview_cam: CameraConfig, detail_cam: CameraConfig | None, judge: DualJudge,
                 out_dir: str | Path) -> None:
        self.config = config
        self.overview, self.detail = overview, detail
        self.overview_cam, self.detail_cam = overview_cam, detail_cam
        self.judge = judge
        self.out_dir = Path(out_dir)

    def _source(self, cam: CameraConfig):
        """녹화본은 VideoSource(장면을 빼지 않음, 시차 반영), 실시간 카메라는 LiveSource(최신 장면만, 재연결)."""
        v = self.config.video
        src = str(cam.source)
        if src.isdigit() or src.lower().startswith(("rtsp://", "rtsps://", "rtmp://", "http://", "https://")):
            from .inputs.live_source import LiveSource
            return LiveSource(src, cam.name, max_frames=v.max_frames)
        return VideoSource(cam.source, start_time=v.start_time, max_frames=v.max_frames, stride=cam.stride,
                           camera_id=cam.name, offset_ms=cam.offset_ms)

    def run(self) -> DualStats:
        out = self.out_dir
        out.mkdir(parents=True, exist_ok=True)
        stats = DualStats()
        started = time.perf_counter()
        workers = {self.overview_cam.name: self.overview}
        cams = [self.overview_cam]
        if self.detail is not None:
            workers[self.detail_cam.name] = self.detail
            cams.append(self.detail_cam)
        logs: dict[str, dict[str, ObservationLog]] = {}
        sources = {c.name: self._source(c) for c in cams}
        self.judge.reset()
        for c in cams:
            w = workers[c.name]
            w.reset()
            src = sources[c.name]
            w.resolve(src.width, src.height, src.fps / (1 if hasattr(src, "reconnects") else c.stride))
            d = out / c.name
            logs[c.name] = {k: ObservationLog(d / f"{k}.jsonl") for k in ("identities", "take_candidates")}
            stats.frames[c.name] = 0
            print(f"입력 {c.name} ({c.role}): {c.source} ({src.width}x{src.height}, {src.fps:.1f}fps, "
                  f"{c.stride}장에 1장, 시차 {c.offset_ms:+.0f}ms)")
        detail_visits = (out / (self.detail_cam.name if self.detail_cam else "_") / "shelf_visits.jsonl")
        visit_file = detail_visits.open("w", encoding="utf-8") if self.detail is not None else None
        slot_file = (detail_visits.with_name("slot_changes.jsonl").open("w", encoding="utf-8")
                     if self.detail is not None else None)
        appearance_file = (out / "appearance.jsonl").open("w", encoding="utf-8")
        risk_log = ObservationLog(out / self.config.output.risk_name)
        pay_log = ObservationLog(out / self.config.output.payments_name)
        last_app: dict[tuple[str, int], float] = {}
        last_overview = None

        def note_appearance(cam: str, worker: CameraWorker, pid: int, ms: float):
            if worker.registry is None or ms - last_app.get((cam, pid), -1e9) < 500.0:
                return
            vec = worker.registry.appearance(pid)
            if vec is None:
                return
            last_app[(cam, pid)] = ms
            appearance_file.write(json.dumps({"camera": cam, "person_id": pid, "ms": round(ms, 1),
                                              "vec": [round(float(x), 4) for x in vec]}) + "\n")

        try:
            merged = heapq.merge(*(iter(s) for s in sources.values()), key=lambda f: f.pts_ms)
            for frame in merged:
                cam = frame.camera_id
                worker = workers[cam]
                step = worker.step(frame)
                stats.frames[cam] += 1
                ids = step.identities or []
                logs[cam]["identities"].write_many(ids)
                logs[cam]["take_candidates"].write_many(step.takes)
                resolve = worker.registry.resolve if worker.registry is not None else None
                appearance = worker.registry.appearance if worker.registry is not None else None
                for o in ids:
                    if o.person_id > 0:
                        note_appearance(cam, worker, resolve(o.person_id), frame.pts_ms)
                if self.detail is not None and cam == self.detail_cam.name:
                    visits = []
                    for visit, change in zip(step.visits, step.changes):
                        visits.append((visit, change.start_ms, change.end_ms))
                        visit_file.write(json.dumps(visit.to_dict() | {"start_ms": round(change.start_ms, 1),
                                                                         "end_ms": round(change.end_ms, 1)},
                                                    ensure_ascii=False) + "\n")
                    for sc in step.slot_changes:
                        slot_file.write(json.dumps(sc.to_dict(), ensure_ascii=False) + "\n")
                    self.judge.detail_step(frame.pts_ms, ids, step.takes, visits, appearance, resolve,
                                           step.slot_changes)
                else:
                    events, records = self.judge.overview_step(frame, ids, step.takes, worker.registry.merges,
                                                               appearance, resolve)
                    risk_log.write_many(events)
                    pay_log.write_many(records)
                    for a in self.judge.risk.new_actions:
                        if a.kind != "LOOK":
                            print(f"  [행동] {frame.pts_ms / 1000:6.1f}s  {a.describe()}")
                    for e in events:
                        self._report(e)
                    last_overview = frame
        finally:
            for c in cams:
                for log in logs[c.name].values():
                    log.close()
            if visit_file is not None:
                visit_file.close()
                slot_file.close()
            appearance_file.close()

        # 영상 끝: 열린 손 뻗기를 닫고, 대기 중인 선반 사건을 넘기고, 출구 쪽에서 막 사라진 손님을 확정
        if self.detail is not None:
            rest = self.detail.flush()
            self.judge.detail_step(float("inf"), [], rest, [])
        rest = self.overview.flush()
        if last_overview is not None:
            if rest:
                self.judge.risk.add_late_takes(rest, last_overview.pts_ms)
            for e in self.judge.finish(last_overview):
                self._report(e)
                risk_log.write_many([e])
        risk_log.close()
        pay_log.close()
        with ObservationLog(out / "cross_view_links.jsonl") as f:
            f.write_many(self.judge.linker.links)
        with ObservationLog(out / "shelf_events.jsonl") as f:
            f.write_many(self.judge.fuser.records)
        with ObservationLog(out / self.config.output.actions_name) as f:
            f.write_many(self.judge.risk.all_actions())
        for c in cams:
            reg = workers[c.name].registry
            if reg is not None and reg.merges:
                (out / c.name / "identity_merges.jsonl").write_text(
                    "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in reg.merges), encoding="utf-8")
        stats.elapsed_sec = time.perf_counter() - started
        stats.events = list(self.judge.events)
        stats.links = len(self.judge.linker.links)
        stats.summary_lines.append(self.judge.risk.summary())
        if self.judge.unowned_takes:
            stats.summary_lines.append(f"주인을 못 정한 손 접촉 : {self.judge.unowned_takes}건")
        return stats

    @staticmethod
    def _report(event) -> None:
        mark = {RiskLevel.HIGH_RISK: "!!", RiskLevel.REVIEW: "??", RiskLevel.WARNING: "!?"}.get(event.level, "ok")
        print(f"  [{mark}] {event.time_sec:7.2f}s  손님 {event.person_id}  {event.level.value}  {event.reason}")


def build_multicam(config: AppConfig, out_dir: str | Path, *, build_detector, build_tracker, build_embedder,
                   build_item_detector, IdentityRegistry, InteractionDetector, ShelfWatcher, RiskEngine,
                   PaymentFeed) -> MultiCameraPipeline:
    """config.cameras 로 카메라 2대 파이프라인을 조립한다 (모델 생성 함수는 run_tracking 이 넘겨준다)."""
    from .detectors.masked import MaskedDetector

    overview_cam = next((c for c in config.cameras if c.role == "overview"), None)
    detail_cam = next((c for c in config.cameras if c.role == "detail"), None)
    if overview_cam is None or not overview_cam.zones:
        raise SystemExit("cameras 에 role: overview (전체 카메라) 와 그 바닥 지도(zones: floor.yaml)가 필요합니다")
    floor = FloorMap.load(overview_cam.zones)
    floor_raw = yaml.safe_load(Path(overview_cam.zones).read_text(encoding="utf-8")) or {}
    embedder = build_embedder(config.tracker.reid)

    def registry(cam: CameraConfig):
        # 카메라마다 화각이 달라 신원 계층 값 일부(예: 늘 잘리는 변)를 따로 준다
        ident = replace(config.identity, **{k: tuple(v) if isinstance(v, list) else v for k, v in cam.identity.items()})
        return IdentityRegistry(ident, embedder, imgsz=config.detector.imgsz)

    # 전체 카메라: 사람 상자만, 바닥 구역, 선반 앞 띠에서 머묾(보조)
    floor_zones = FloorZoneMap(floor)
    ov_detector = build_detector(config.detector)
    ignore = [tuple(map(float, r)) for r in floor_raw.get("ignore_image") or []]
    if ignore:
        ov_detector = MaskedDetector(ov_detector, ignore)
    ov_registry = registry(overview_cam)
    overview = CameraWorker(overview_cam.name, ov_detector, build_tracker(config.tracker), ov_registry, floor_zones,
                            InteractionDetector(config.interaction, floor_zones))

    detail = None
    setup = None
    if detail_cam is not None and detail_cam.zones:
        setup = DetailSetup.load(detail_cam.zones)
        dcfg = deepcopy(config.detector)
        if detail_cam.detector == "yolo-pose":
            dcfg.name, dcfg.model_path = "yolo-pose", "yolov8n-pose.pt"
        icfg = config.detail_interaction or deepcopy(config.interaction)
        scfg = ShelfConfig(**{**config.shelves.__dict__, "areas": setup.shelves})
        shelves = ShelfWatcher(scfg, build_item_detector(scfg.model_path, scfg.imgsz, scfg.conf,
                                                         config.detector.device, scfg.ignore),
                               slots=[(z.name, z.polygon) for z in setup.zone_map.zones]) if setup.shelves else None
        detail = CameraWorker(detail_cam.name, build_detector(dcfg), build_tracker(config.tracker), registry(detail_cam),
                              setup.zone_map, InteractionDetector(icfg, setup.zone_map), shelves)

    payments = PaymentFeed.load(config.risk.payments) if config.risk.payments else PaymentFeed.empty()
    risk = RiskEngine(config.risk, floor_zones, payments, consistency=ov_registry.recent_consistency)
    sensors = load_sensor_events(config.fusion.sensor_path) if config.fusion.sensor_path else []

    import cv2
    def probe(cam):
        cap = cv2.VideoCapture(cam.source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        return fps, size

    ov_fps, ov_size = probe(overview_cam)
    floor_zones.resolve(*ov_size)
    if detail_cam is not None:
        d_fps, d_size = probe(detail_cam)
    else:
        d_fps, d_size = 30.0, (1920, 1080)
    judge = DualJudge(config.fusion, risk, floor, setup.band_map if setup else (1.0, 0.0),
                      TimeAxis(d_fps, detail_cam.offset_ms if detail_cam else 0.0), TimeAxis(ov_fps, overview_cam.offset_ms),
                      d_size[0], ov_size, setup.slot_shelf if setup else {}, sensors)
    return MultiCameraPipeline(config, overview, detail, overview_cam, detail_cam, judge, out_dir)
