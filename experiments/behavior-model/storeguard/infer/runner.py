"""카메라 1대에 대한 실시간 파이프라인 조립.

    VideoSource(스레드)  →  프레임 버퍼  →  CameraRunner(스레드)
                                              ├ 샘플링(3fps) → 클립 버퍼(16프레임)
                                              ├ ClipClassifier 추론 (1초마다)
                                              ├ AlertEngine 사건 판정
                                              └ ClipRecorder 스냅샷 + 전후 영상

CLI 로도 쓸 수 있다(웹서버 없이 MP4/RTSP 시험).

PowerShell 예:
    D:\\computervision\\.venv\\Scripts\\python.exe -m storeguard.infer.runner ^
        --source "D:\\computervision\\data\\raw\\val\\videos\\어떤파일.mp4" --run baseline_r2p1d --print
    $env:CAM1_RTSP="rtsp://user:pass@192.168.0.10:554/stream1"
    D:\\computervision\\.venv\\Scripts\\python.exe -m storeguard.infer.runner --source-env CAM1_RTSP --run baseline_r2p1d
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from storeguard.config import Config
from storeguard.infer.alerting import AlertEngine, AlertEvent
from storeguard.infer.engine import ClipClassifier, ModelNotLoaded
from storeguard.infer.recorder import ClipRecorder
from storeguard.infer.video_source import VideoSource, resolve_source
from storeguard.utils import get_logger, mask_url


class CameraRunner:
    def __init__(self, camera_id: str, source_spec: str, cfg: Config,
                 classifier: ClipClassifier, *, display_name: str | None = None,
                 on_event: Callable[[str, AlertEvent], None] | None = None,
                 loop_file: bool = False, realtime_file: bool = True):
        self.camera_id = camera_id
        self.display_name = display_name or camera_id
        self.cfg = cfg
        self.clf = classifier
        self.on_event = on_event
        self.log = get_logger(f"cam.{camera_id}")

        self.loop_file = loop_file
        self.realtime_file = realtime_file
        self._switch_lock = threading.Lock()
        self.source_label = source_spec
        self.source = self._make_source(source_spec, loop_file, realtime_file)
        self.sample_fps = float(cfg["data.sample_fps"])
        self.clip_len = int(cfg["data.clip_len"])
        self.infer_every = int(cfg["runtime.infer_every_frames"])
        self.cache_h = int(cfg["data.cache_height"])
        self.cache_w = int(cfg["data.cache_width"])
        self.stale_after = float(cfg["runtime.stale_after_sec"])

        self.clip_buf: deque[np.ndarray] = deque(maxlen=self.clip_len)
        self.alerts = AlertEngine(camera_id, cfg.action_classes,
                                  {a: cfg.alert_params(a) for a in cfg.action_classes})
        self.recorder = ClipRecorder(
            cfg.path("paths.events") / camera_id,
            pre_seconds=float(cfg["recording.pre_seconds"]),
            post_seconds=float(cfg["recording.post_seconds"]),
            fps=float(cfg["recording.fps"]), width=int(cfg["recording.width"]),
            enabled=bool(cfg["recording.enabled"]),
        )

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._preview_lock = threading.Lock()
        self._preview_jpeg: bytes | None = None
        self._preview_ts: float = 0.0
        self._last_bgr: np.ndarray | None = None

        self.last_scores: dict[str, float] = {}
        self.last_score_ts: float | None = None
        self.last_score_media_ts: float | None = None
        self.n_inferences = 0
        self.n_samples = 0
        self.inference_errors = 0
        self._sample_next_ts: float | None = None
        self._src_epoch = -1
        self._analysis_state = "IDLE"
        self._infer_lat: deque[float] = deque(maxlen=50)
        self.started_at: float | None = None

    # --- 소스 전환 ---
    def _make_source(self, spec: str, loop_file: bool, realtime_file: bool) -> VideoSource:
        url, kind = resolve_source(spec)
        return VideoSource(
            url, kind=kind,
            buffer_size=int(self.cfg["runtime.frame_buffer_size"]),
            drop_oldest=bool(self.cfg["runtime.drop_oldest_when_full"]),
            reconnect_delay=float(self.cfg["runtime.reconnect_delay_sec"]),
            reconnect_max_delay=float(self.cfg["runtime.reconnect_max_delay_sec"]),
            loop_file=loop_file, realtime_file=realtime_file, name=self.camera_id,
            rtsp_transport=str(self.cfg.get("runtime.rtsp_transport", "tcp")),
            rtsp_timeout_sec=float(self.cfg.get("runtime.rtsp_timeout_sec", 8.0)),
        )

    def switch_source(self, spec: str, *, loop_file: bool | None = None,
                      label: str | None = None) -> dict:
        """실행 중에 입력 소스를 바꾼다(웹에서 다음 영상 / RTSP 전환용).

        이전 소스를 완전히 정지시키고 클립 버퍼와 사건 상태를 버린다.
        옛 프레임과 새 프레임이 한 클립에 섞이면 존재하지 않는 움직임이 만들어진다.
        """
        with self._switch_lock:
            new_src = self._make_source(
                spec,
                self.loop_file if loop_file is None else loop_file,
                self.realtime_file,
            )
            old = self.source
            self.source = new_src
            self.source_label = label or spec
            new_src.start()
            try:
                old.stop()
            except Exception:
                pass
            # 상태 초기화 (epoch 변화로 _loop 에서도 한 번 더 정리된다)
            self.clip_buf.clear()
            self._sample_next_ts = None
            self._src_epoch = -1
            for ev in self.alerts.reset():
                self._emit_end(ev)
            with self._preview_lock:
                self._preview_jpeg = None
            self.last_scores = {}
            self.last_score_ts = None
            self._analysis_state = "IDLE"
        self.log.info("소스 전환: %s", mask_url(spec))
        return {"ok": True, "camera_id": self.camera_id,
                "source": self.source.status.url_masked, "label": self.source_label}

    # --- 수명주기 ---
    def start(self) -> "CameraRunner":
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self.started_at = time.time()
        self.source.start()
        self._thread = threading.Thread(target=self._loop, name=f"run-{self.camera_id}", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self.source.stop()
        if self._thread:
            self._thread.join(timeout=5.0)
        self.recorder.stop()

    # --- 내부 루프 ---
    def _loop(self) -> None:
        period = 1.0 / self.sample_fps
        while not self._stop.is_set():
            frame = self.source.read(timeout=0.5)
            if frame is None:
                if not self.source.status.connected:
                    self._on_disconnect()
                continue

            if self.source.epoch != self._src_epoch:
                # 재접속(또는 최초 접속): 오래된 프레임/사건 상태를 버린다
                self._src_epoch = self.source.epoch
                self.clip_buf.clear()
                self._sample_next_ts = None
                for ev in self.alerts.reset():
                    self._emit_end(ev)
                self.log.info("소스 epoch=%d 시작, 클립/사건 상태 초기화", self._src_epoch)

            self._last_bgr = frame.image
            self._maybe_preview(frame.image)

            if self._sample_next_ts is None:
                self._sample_next_ts = frame.media_ts
            if frame.media_ts + period < self._sample_next_ts - period:
                # 시각이 과거로 되돌아갔다(반복 재생, 카메라 타임스탬프 리셋 등).
                # 기준을 다시 잡지 않으면 이후 프레임이 전부 버려진다.
                self.log.info("소스 시각 역행 감지(%.1fs → %.1fs). 샘플링 기준 재설정",
                              self._sample_next_ts, frame.media_ts)
                self.clip_buf.clear()
                self._sample_next_ts = frame.media_ts
                for ev in self.alerts.reset():
                    self._emit_end(ev)
            if frame.media_ts + 1e-6 < self._sample_next_ts:
                continue                       # 이 프레임은 샘플링 간격에 안 맞음 → 버린다
            # 지연이 크게 누적된 경우 따라잡기(오래된 시각으로 계속 밀리지 않게)
            if frame.media_ts - self._sample_next_ts > 5 * period:
                self._sample_next_ts = frame.media_ts
            self._sample_next_ts += period
            self.n_samples += 1

            rgb = cv2.resize(frame.image, (self.cache_w, self.cache_h),
                             interpolation=cv2.INTER_AREA)[:, :, ::-1]
            self.clip_buf.append(np.ascontiguousarray(rgb))
            self.recorder.add(frame.image)

            if len(self.clip_buf) < self.clip_len:
                self._analysis_state = "BUFFERING"
                continue
            if self.n_samples % max(1, self.infer_every) != 0:
                continue
            self._infer(frame)

        self._analysis_state = "STOPPED"

    def _infer(self, frame) -> None:
        if self.clf.state != "READY":
            self._analysis_state = "MODEL_NOT_LOADED"
            return
        clip = np.stack(list(self.clip_buf), axis=0)
        t0 = time.perf_counter()
        try:
            scores = self.clf.predict(clip)
        except ModelNotLoaded:
            self._analysis_state = "MODEL_NOT_LOADED"
            return
        except Exception as exc:
            self.inference_errors += 1
            self._analysis_state = "INFER_ERROR"
            self.log.warning("추론 실패: %s", exc)
            return
        lat = (time.perf_counter() - t0) * 1000.0
        self._infer_lat.append(lat)
        self.n_inferences += 1
        self._analysis_state = "ANALYZING"
        self.last_scores = scores
        self.last_score_ts = time.time()
        self.last_score_media_ts = frame.media_ts

        started, ended = self.alerts.step(
            {a: scores.get(a, 0.0) for a in self.cfg.action_classes}, frame.media_ts)
        for ev in started:
            self._emit_start(ev, frame)
        for ev in ended:
            self._emit_end(ev)

    def _emit_start(self, ev: AlertEvent, frame) -> None:
        # 스냅샷은 즉시(작고 빠름), 영상 저장은 예약만 하고 기다리지 않는다.
        ev.snapshot_path = self.recorder.snapshot(frame.image, ev.event_id)
        ev.video_path = self.recorder.start_clip(ev.event_id)
        ev.extra = {
            "camera_name": self.display_name,
            "scores": {k: round(v, 4) for k, v in self.last_scores.items()},
            "alert_latency_ms": round((time.time() - ev.started_at) * 1000.0, 1),
            "source": self.source.status.url_masked,
            "점수_주의": "보정되지 않은 모델 점수. 실제 발생 확률이 아니다.",
        }
        self.log.info("[사건 시작] %s %s score=%.3f media_ts=%.1fs",
                      ev.event_id, ev.action, ev.score, ev.started_media_ts)
        if self.on_event:
            self.on_event("started", ev)

    def _emit_end(self, ev: AlertEvent) -> None:
        self.log.info("[사건 종료] %s %s peak=%.3f", ev.event_id, ev.action, ev.peak_score)
        if self.on_event:
            self.on_event("ended", ev)

    def _on_disconnect(self) -> None:
        if self._analysis_state != "DISCONNECTED":
            self._analysis_state = "DISCONNECTED"
            self.clip_buf.clear()
            self._sample_next_ts = None
            for ev in self.alerts.reset():
                self._emit_end(ev)
            with self._preview_lock:
                self._preview_jpeg = None      # 끊긴 뒤 마지막 화면을 계속 보여주지 않는다
            self.last_scores = {}

    def _maybe_preview(self, bgr: np.ndarray) -> None:
        fps = float(self.cfg.get("runtime.mjpeg_fps", 8))
        now = time.time()
        if now - self._preview_ts < 1.0 / max(fps, 0.1):
            return
        w = int(self.cfg.get("runtime.mjpeg_width", 640))
        h, ow = bgr.shape[:2]
        if ow > w:
            bgr = cv2.resize(bgr, (w, int(round(h * w / ow))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", bgr,
                               [cv2.IMWRITE_JPEG_QUALITY, int(self.cfg.get("runtime.jpeg_quality", 80))])
        if ok:
            with self._preview_lock:
                self._preview_jpeg = buf.tobytes()
                self._preview_ts = now

    # --- 외부 조회 ---
    def preview_jpeg(self) -> bytes | None:
        with self._preview_lock:
            if self._preview_jpeg is None:
                return None
            if time.time() - self._preview_ts > self.stale_after:
                return None          # 오래된 프레임은 라이브가 아니다
            return self._preview_jpeg

    def status(self) -> dict:
        src = self.source.status.public(self.stale_after)
        lat = list(self._infer_lat)
        uptime = None if self.started_at is None else round(time.time() - self.started_at, 1)
        eff_fps = round(self.n_inferences / uptime, 3) if uptime else None
        return {
            "camera_id": self.camera_id,
            "name": self.display_name,
            "source": src,
            "source_label": self.source_label,
            "analysis_state": ("MODEL_NOT_LOADED" if self.clf.state != "READY"
                               else self._analysis_state),
            "model": self.clf.status(),
            "buffered_frames": self.source.buffered(),
            "clip_fill": f"{len(self.clip_buf)}/{self.clip_len}",
            "samples": self.n_samples,
            "inferences": self.n_inferences,
            "inference_errors": self.inference_errors,
            "infer_latency_ms": {
                "last": round(lat[-1], 1) if lat else None,
                "mean": round(float(np.mean(lat)), 1) if lat else None,
                "p90": round(float(np.percentile(lat, 90)), 1) if lat else None,
            },
            "inferences_per_sec": eff_fps,
            "uptime_sec": uptime,
            "last_scores": {k: round(v, 4) for k, v in self.last_scores.items()},
            "last_score_age_sec": (None if self.last_score_ts is None
                                   else round(time.time() - self.last_score_ts, 2)),
            "alert_rules": self.alerts.status(),
            "pending_recordings": self.recorder.pending(),
            "점수_주의": "보정되지 않은 모델 점수. 실제 발생 확률이 아니다.",
        }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--source", help="MP4 경로 또는 rtsp:// URL")
    ap.add_argument("--source-env", help="RTSP URL 이 든 환경변수 이름(비밀번호 보호)")
    ap.add_argument("--camera-id", default="cam1")
    ap.add_argument("--name", default="1번 카메라")
    ap.add_argument("--run", default=None, help="runs/ 아래 학습 실행 이름(없으면 모델 미로딩 상태)")
    ap.add_argument("--runs-dir", type=Path, default=Path(r"D:\computervision\runs"))
    ap.add_argument("--seconds", type=float, default=0, help="이 시간만 돌고 종료")
    ap.add_argument("--loop", action="store_true", help="파일 입력 반복 재생")
    ap.add_argument("--no-realtime", action="store_true", help="파일을 최대 속도로 처리")
    ap.add_argument("--print", dest="do_print", action="store_true", help="주기적으로 상태 출력")
    ap.add_argument("--status-out", type=Path, default=None,
                    help="종료 시 최종 상태와 사건 목록을 JSON 으로 저장(리포트용)")
    a = ap.parse_args(argv)

    cfg = Config.load(a.config, a.overrides)
    log = get_logger("runner")

    spec = a.source
    if a.source_env:
        spec = os.environ.get(a.source_env)
        if not spec:
            log.error("환경변수 %s 가 비어 있다. .env 또는 PowerShell 에서 설정할 것.", a.source_env)
            return 2
    if not spec:
        log.error("--source 또는 --source-env 중 하나가 필요하다.")
        return 2
    log.info("소스: %s", mask_url(spec))

    run_dir = (a.runs_dir / a.run) if a.run else None
    clf = ClipClassifier(run_dir, cfg=cfg)
    if clf.state != "READY":
        log.warning("모델 미로딩 상태로 시작한다: %s", clf.error)
        log.warning("가짜 탐지를 만들지 않는다. 학습 후 --run 으로 체크포인트를 지정할 것.")

    events: list[dict] = []

    def on_event(kind: str, ev: AlertEvent):
        if kind == "started":
            text = cfg.get(f"classes.alert_text.{ev.action}", "{camera}에서 이상행동이 감지되었습니다.")
            print("[알림] " + text.format(camera=a.name)
                  + f"  (event_id={ev.event_id}, score={ev.score:.3f}, t={ev.started_media_ts:.1f}s)")
        events.append({"kind": kind, **ev.to_dict()})

    runner = CameraRunner(a.camera_id, spec, cfg, clf, display_name=a.name,
                          on_event=on_event, loop_file=a.loop,
                          realtime_file=not a.no_realtime).start()
    t0 = time.time()
    try:
        while True:
            time.sleep(2.0)
            if a.do_print:
                st = runner.status()
                print(json.dumps({
                    "live": st["source"]["live"], "state": st["analysis_state"],
                    "recv_fps": st["source"]["measured_fps"], "buffered": st["buffered_frames"],
                    "infer": st["inferences"], "lat_ms": st["infer_latency_ms"]["mean"],
                    "scores": st["last_scores"],
                }, ensure_ascii=False))
            if a.seconds and time.time() - t0 >= a.seconds:
                break
            if runner.source.status.eof and not a.loop and runner.source.buffered() == 0:
                time.sleep(1.0)
                break
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop()

    st = runner.status()
    n_started = len([e for e in events if e["kind"] == "started"])
    if a.status_out:
        from storeguard.utils import write_json
        write_json(a.status_out, {**st, "events": events, "n_events": n_started})
        print(f"상태 저장: {a.status_out}")
    print(json.dumps({"최종상태": st, "사건수": n_started},
                     ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
