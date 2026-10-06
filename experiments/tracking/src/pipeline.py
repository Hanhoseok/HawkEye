"""파이프라인 조립: 영상 입력 -> 탐지 -> 추적 -> 표준 출력 -> 시각화/저장.

이 모듈은 구체 라이브러리가 아니라 core.protocols 의 인터페이스에만 의존한다.
1차 구현의 범위가 여기서 한눈에 보인다.
근거: 05 아키텍처 §1 파이프라인
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from .core.config import AppConfig
from .core.protocols import Detector, Tracker
from .core.types import IdentityObservation, RiskLevel, TakeCandidate, TrackObservation
from .inputs.video_source import VideoSource
from .sinks.observation_log import ObservationLog
from .sinks.video_writer import VideoWriter
from .viz.overlay import TrackOverlay, draw_alerts, draw_zones
from .zones.zone_map import ZoneMap


@dataclass
class RunStats:
    """실행 결과 요약. Phase 별 '완료 확인' 항목을 숫자로 남긴다."""

    frames: int = 0
    detections: int = 0
    observations: int = 0
    unique_track_ids: set[int] = field(default_factory=set)
    unique_person_ids: set[int] = field(default_factory=set)
    unassigned: int = 0
    take_candidates: list = field(default_factory=list)
    elapsed_sec: float = 0.0
    video_path: Path | None = None
    log_path: Path | None = None
    identity_path: Path | None = None
    takes_path: Path | None = None
    risk_events: list = field(default_factory=list)
    risk_summary: str = ""
    alert_summary: str = ""
    risk_path: Path | None = None

    @property
    def fps(self) -> float:
        return self.frames / self.elapsed_sec if self.elapsed_sec else 0.0

    def summary(self) -> str:
        lines = [
            "─" * 52,
            f"처리 프레임      : {self.frames}",
            f"총 detection     : {self.detections}",
            f"총 observation   : {self.observations}",
            f"고유 track_id    : {len(self.unique_track_ids)}",
            f"처리 속도        : {self.fps:.2f} FPS ({self.elapsed_sec:.1f}s)",
        ]
        if self.unique_person_ids:
            lines.append(f"고유 person_id   : {len(self.unique_person_ids)}")
            merged = len(self.unique_track_ids) - len(self.unique_person_ids)
            if merged > 0:
                lines.append(f"  └ 신원으로 이어붙인 track : {merged}건")
            if self.unassigned:
                lines.append(f"  └ 크기 미달로 신원 보류   : {self.unassigned}건")
        if self.take_candidates:
            people = {c.person_id for c in self.take_candidates}
            total = sum(c.duration_sec for c in self.take_candidates)
            lines.append(f"TAKE 후보       : {len(self.take_candidates)}건 / {len(people)}명")
            lines.append(f"  └ 총 체류 {total:.1f}초, 평균 {total/len(self.take_candidates):.1f}초")
        if self.risk_summary:
            lines.append(self.risk_summary)
        if self.alert_summary:
            lines.append(self.alert_summary)
        if self.risk_path:
            lines.append(f"위험 판정 로그   : {self.risk_path}")
        if self.video_path:
            lines.append(f"결과 영상        : {self.video_path}")
        if self.log_path:
            lines.append(f"observation 로그 : {self.log_path}")
        lines.append("─" * 52)
        return "\n".join(lines)


class TrackingPipeline:
    """1차 구현 범위 전체를 실행한다."""

    def __init__(
        self,
        config: AppConfig,
        detector: Detector,
        tracker: Tracker,
        registry=None,
        zone_map: ZoneMap | None = None,
        interactions=None,
        risk=None,
        alert_sink=None,
    ) -> None:
        self.config = config
        self.detector = detector
        self.tracker = tracker
        self.registry = registry
        self.zone_map = zone_map or ZoneMap.empty()
        self.interactions = interactions
        self.risk = risk
        self.alert_sink = alert_sink
        self._recent: deque = deque()
        """(재생 시각 ms, 줄인 장면, 줄인 비율) — 경보 사진용 최근 장면."""

    ALERT_SHOW_SECONDS = 3.0
    """HIGH_RISK 판정 뒤 결과 영상에 빨간 박스를 몇 초간 띄울지."""

    def run(self, on_frame=None) -> RunStats:
        cfg = self.config
        out_dir = Path(cfg.output.dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        stats = RunStats()
        overlay = TrackOverlay(cfg.output.draw_trail, cfg.output.trail_length)
        self.tracker.reset()
        if self.registry is not None:
            self.registry.reset()
        if self.interactions is not None:
            self.interactions.reset()
        if self.risk is not None:
            self.risk.reset()
        alert_until: dict[int, tuple[float, str]] = {}  # 손님 -> (표시 종료 시각, 문구)
        last_pts_ms = 0.0

        writer: VideoWriter | None = None
        log: ObservationLog | None = None
        identity_log: ObservationLog | None = None
        takes_log: ObservationLog | None = None
        risk_log: ObservationLog | None = None
        payments_log: ObservationLog | None = None
        started = time.perf_counter()

        try:
            with VideoSource(
                cfg.video.source,
                start_time=cfg.video.start_time,
                start_frame=cfg.video.start_frame,
                max_frames=cfg.video.max_frames,
                stride=cfg.video.stride,
                scale=cfg.video.scale,
                scale_mode=cfg.video.scale_mode,
                blackout=cfg.video.blackout,
            ) as source:
                effective_fps = source.fps / cfg.video.stride

                if cfg.output.write_video:
                    writer = VideoWriter(out_dir / cfg.output.video_name, effective_fps)
                    stats.video_path = writer.path
                if cfg.output.write_observations:
                    log = ObservationLog(out_dir / cfg.output.observations_name)
                    stats.log_path = log.path
                if self.registry is not None and cfg.output.write_identities:
                    identity_log = ObservationLog(out_dir / cfg.output.identities_name)
                    stats.identity_path = identity_log.path
                if self.interactions is not None and cfg.output.write_takes:
                    takes_log = ObservationLog(out_dir / cfg.output.takes_name)
                    stats.takes_path = takes_log.path
                if self.risk is not None and cfg.output.write_risk:
                    risk_log = ObservationLog(out_dir / cfg.output.risk_name)
                    payments_log = ObservationLog(out_dir / cfg.output.payments_name)
                    stats.risk_path = risk_log.path

                if self.zone_map:
                    self.zone_map.resolve(source.width, source.height)
                    print(f"구역: {self.zone_map.summary()}")

                print(
                    f"입력: {cfg.video.source} "
                    f"({source.width}x{source.height}, {source.fps:.1f}fps, "
                    f"{source.total_frames} frames)"
                )

                for frame in source:
                    detections = self.detector.detect(frame)
                    observations: list[TrackObservation] = self.tracker.update(frame, detections)

                    stats.frames += 1
                    stats.detections += len(detections)
                    stats.observations += len(observations)
                    stats.unique_track_ids.update(o.track_id for o in observations)

                    if log is not None:
                        log.write_many(observations)

                    # 계층 2: 매장 단위 신원 배정. 이후 로직은 track_id 가 아니라 person_id 를 쓴다.
                    drawable = observations
                    if self.registry is not None:
                        identities: list[IdentityObservation] = self.registry.assign(
                            frame, observations
                        )
                        stats.unique_person_ids.update(
                            o.person_id for o in identities if o.person_id > 0
                        )
                        stats.unassigned += sum(1 for o in identities if o.person_id == -1)
                        if identity_log is not None:
                            identity_log.write_many(identities)
                        drawable = identities

                        # 계층 3: 선반 앞 '멈춤' 구간 -> TAKE 후보
                        done: list[TakeCandidate] = []
                        if self.interactions is not None:
                            done = self.interactions.update(frame, identities)
                            if done:
                                stats.take_candidates.extend(done)
                                if takes_log is not None:
                                    takes_log.write_many(done)

                        # 손님 상태 · 결제 · 출구 판정
                        if self.risk is not None:
                            events, records = self.risk.update(
                                frame, identities, done, self.registry.merges
                            )
                            if risk_log is not None:
                                risk_log.write_many(events)
                                payments_log.write_many(records)
                            if self.alert_sink is not None:
                                self._remember(frame)
                            for event in events:
                                self._report(event, frame, out_dir, alert_until)
                                if self.alert_sink is not None:
                                    self._send_alert(event, frame)
                        last_pts_ms = frame.pts_ms

                    if writer is not None or cfg.output.show_window or on_frame is not None:
                        header = (
                            f"frame {frame.index}  t={frame.pts_ms / 1000:.2f}s  "
                            f"det={len(detections)}  track={len(observations)}"
                            + (
                                f"  person={len(stats.unique_person_ids)}"
                                if self.registry is not None
                                else ""
                            )
                        )
                        base = frame.image
                        if self.zone_map and cfg.output.draw_zones:
                            base = draw_zones(base, self.zone_map)
                        canvas = overlay.draw(base, drawable, header)
                        if alert_until and self.risk is not None:
                            canvas = draw_alerts(canvas, self._live_alerts(drawable, frame, alert_until))
                        if writer is not None:
                            writer.write(canvas)
                        if cfg.output.show_window:
                            cv2.imshow("tracking", canvas)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                print("사용자 중단(q)")
                                break
                        if on_frame is not None:
                            on_frame(frame, observations, canvas)

                    if stats.frames % 50 == 0:
                        print(f"  ... {stats.frames} frames 처리, track_id {len(stats.unique_track_ids)}개")
        finally:
            if writer is not None:
                writer.release()
            if log is not None:
                log.close()
            if self.interactions is not None:
                remaining = self.interactions.flush()
                if remaining:
                    stats.take_candidates.extend(remaining)
                    if takes_log is not None:
                        takes_log.write_many(remaining)
                    if self.risk is not None:
                        self.risk.add_late_takes(remaining, last_pts_ms)
            if self.risk is not None:
                stats.risk_events = list(self.risk.events)
                stats.risk_summary = self.risk.summary()
            if risk_log is not None:
                risk_log.close()
                payments_log.close()
            if self.alert_sink is not None:
                stats.alert_summary = self.alert_sink.close()
            if identity_log is not None:
                identity_log.close()
                if self.registry.merges:
                    # 이미 내보낸 관측의 person_id 는 바뀌지 않는다. 뒤 계층이 이 기록으로 이력을 합친다.
                    merges_path = identity_log.path.with_name("identity_merges.jsonl")
                    merges_path.write_text(
                        "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in self.registry.merges),
                        encoding="utf-8",
                    )
            if takes_log is not None:
                takes_log.close()
            if cfg.output.show_window:
                cv2.destroyAllWindows()
            stats.elapsed_sec = time.perf_counter() - started

        return stats

    def _report(self, event, frame, out_dir: Path, alert_until: dict) -> None:
        """위험 판정 한 건을 알린다. HIGH_RISK 면 그 순간의 장면을 이미지로 남긴다."""
        mark = {RiskLevel.HIGH_RISK: "!!", RiskLevel.REVIEW: "??", RiskLevel.WARNING: "!?"}.get(event.level, "ok")
        print(f"  [{mark}] {event.time_sec:7.2f}s  손님 {event.person_id}  {event.level.value}  {event.reason}")
        if event.level != RiskLevel.HIGH_RISK:
            return
        text = f"HIGH RISK P{event.person_id} unpaid {event.unpaid}"
        alert_until[event.person_id] = (frame.pts_ms + self.ALERT_SHOW_SECONDS * 1000.0, text)
        if frame.image is None or not self.config.output.alerts_dir:
            return
        base = draw_zones(frame.image, self.zone_map) if self.zone_map else frame.image.copy()
        image = draw_alerts(base, [(event.bbox, text)])
        path = out_dir / self.config.output.alerts_dir / f"{frame.index:06d}_p{event.person_id}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image)

    def _remember(self, frame) -> None:
        """경보 사진용으로 최근 장면을 줄여 보관한다."""
        if frame.image is None:
            return
        acfg = self.config.alerts
        h, w = frame.image.shape[:2]
        scale = min(1.0, acfg.snapshot_max_side / float(max(h, w)))
        small = frame.image if scale >= 1.0 else cv2.resize(frame.image, (int(w * scale), int(h * scale)))
        self._recent.append((frame.pts_ms, small.copy() if small is frame.image else small, scale))
        limit = acfg.snapshot_seconds * 1000.0
        while self._recent and frame.pts_ms - self._recent[0][0] > limit:
            self._recent.popleft()

    def _send_alert(self, event, frame) -> None:
        """경보 서버로 보낸다. 사진은 그 손님이 마지막으로 보인 장면 (사라진 뒤 확정된 경보 대비)."""
        customer = self.risk.book.customers.get(event.person_id) if self.risk is not None else None
        target = customer.last_seen_ms if customer is not None else frame.pts_ms
        best = None
        for pts, image, scale in self._recent:
            if pts <= target + 1e-6:
                best = (image, scale)
        if best is None and self._recent:
            best = self._recent[0][1:]
        image, scale = best if best is not None else (frame.image, 1.0)
        self.alert_sink.send(event, image, scale)

    def _live_alerts(self, identities, frame, alert_until: dict) -> list:
        """판정 뒤 몇 초간, 지금 위치의 그 손님에게 빨간 박스를 띄운다."""
        alerts = []
        for obs in identities:
            if getattr(obs, "person_id", 0) <= 0:
                continue
            pid = self.risk.book.resolve(obs.person_id)
            until = alert_until.get(pid)
            if until and frame.pts_ms <= until[0]:
                alerts.append((obs.bbox, until[1]))
        return alerts
