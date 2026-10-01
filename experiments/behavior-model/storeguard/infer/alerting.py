"""알림 판정 상태 기계.

한 번의 높은 점수로 알림을 내지 않는다. 클래스별로 다음을 따로 둔다.
  threshold            모델 점수 임계값
  window / min_hits    최근 window 회 추론 중 min_hits 회 이상 임계값을 넘어야 사건 시작
  end_below_threshold  연속 이 횟수만큼 임계값 미만이면 사건 종료
  cooldown_sec         사건 종료 후 이 시간 동안 같은 클래스 재알림 금지(중복 억제)

전도처럼 짧은 사건은 min_hits / window 를 작게 잡아 놓침을 줄인다(configs/default.yaml).

이 모듈은 I/O 를 하지 않는다. 실시간 추론과 오프라인 평가가 **같은 코드**를 쓰기 위해서다.

한계 (정직하게 기록)
  - 이 시스템은 클립 전체를 분류한다. 사람 추적 ID가 없다.
    따라서 '사람별' 중복 억제는 불가능하고, 카메라·클래스 단위 억제만 한다.
  - 모델 점수는 보정(calibration)되지 않았다. 확률처럼 해석하면 안 된다.
"""
from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Callable


@dataclass
class AlertEvent:
    event_id: str
    camera_id: str
    action: str
    started_at: float          # epoch seconds
    started_media_ts: float    # 소스 기준 시각(초). 파일 입력이면 영상 내 시각.
    score: float               # 시작 시점 점수
    peak_score: float = 0.0
    ended_at: float | None = None
    ended_media_ts: float | None = None
    n_hits: int = 0
    snapshot_path: str | None = None
    video_path: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ClassState:
    action: str
    threshold: float
    min_hits: int
    window: int
    cooldown_sec: float
    end_below: int
    recent: deque = field(default_factory=deque)
    active: AlertEvent | None = None
    below_streak: int = 0
    cooldown_until: float = 0.0


class AlertEngine:
    """카메라 1대에 대한 클래스별 알림 상태 기계."""

    def __init__(self, camera_id: str, actions: list[str], params: dict[str, dict],
                 clock: Callable[[], float] = time.time):
        self.camera_id = camera_id
        self.clock = clock
        self.states: dict[str, ClassState] = {}
        for a in actions:
            p = params.get(a, {})
            self.states[a] = ClassState(
                action=a,
                threshold=float(p.get("threshold", 0.6)),
                min_hits=int(p.get("min_hits", 3)),
                window=int(p.get("window", 5)),
                cooldown_sec=float(p.get("cooldown_sec", 20.0)),
                end_below=int(p.get("end_below_threshold", 4)),
            )

    def update_params(self, action: str, **kw) -> None:
        st = self.states[action]
        for k, v in kw.items():
            if k == "end_below_threshold":
                st.end_below = int(v)
            elif hasattr(st, k):
                setattr(st, k, type(getattr(st, k))(v))

    def reset(self) -> list[AlertEvent]:
        """연결 끊김 등으로 상태를 버릴 때. 진행 중이던 사건은 종료 처리해 반환한다."""
        closed = []
        now = self.clock()
        for st in self.states.values():
            if st.active is not None:
                st.active.ended_at = now
                closed.append(st.active)
                st.active = None
            st.recent.clear()
            st.below_streak = 0
        return closed

    def step(self, scores: dict[str, float], media_ts: float,
             now: float | None = None) -> tuple[list[AlertEvent], list[AlertEvent]]:
        """추론 1회 결과를 넣는다.

        returns (새로 시작된 사건, 종료된 사건)
        """
        now = self.clock() if now is None else now
        started: list[AlertEvent] = []
        ended: list[AlertEvent] = []

        for action, st in self.states.items():
            s = float(scores.get(action, 0.0))
            st.recent.append(s >= st.threshold)
            while len(st.recent) > st.window:
                st.recent.popleft()

            if st.active is not None:
                st.active.peak_score = max(st.active.peak_score, s)
                if s >= st.threshold:
                    st.active.n_hits += 1
                    st.below_streak = 0
                else:
                    st.below_streak += 1
                    if st.below_streak >= st.end_below:
                        st.active.ended_at = now
                        st.active.ended_media_ts = media_ts
                        ended.append(st.active)
                        st.active = None
                        st.below_streak = 0
                        st.cooldown_until = now + st.cooldown_sec
                        st.recent.clear()
                continue

            if now < st.cooldown_until:
                continue
            if sum(st.recent) >= st.min_hits:
                ev = AlertEvent(
                    event_id=uuid.uuid4().hex[:12],
                    camera_id=self.camera_id,
                    action=action,
                    started_at=now,
                    started_media_ts=media_ts,
                    score=round(s, 4),
                    peak_score=round(s, 4),
                    n_hits=int(sum(st.recent)),
                )
                st.active = ev
                st.below_streak = 0
                started.append(ev)
        return started, ended

    def active_events(self) -> list[AlertEvent]:
        return [st.active for st in self.states.values() if st.active is not None]

    def status(self) -> dict:
        return {
            a: {"threshold": st.threshold, "min_hits": st.min_hits, "window": st.window,
                "cooldown_sec": st.cooldown_sec, "end_below_threshold": st.end_below,
                "recent_hits": int(sum(st.recent)), "active": st.active.event_id if st.active else None,
                "cooldown_remaining": max(0.0, round(st.cooldown_until - self.clock(), 1))}
            for a, st in self.states.items()
        }
