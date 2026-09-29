"""계층 3 · TAKE 후보 판정.

`IdentityObservation` + 구역 정보를 받아, 사람이 선반 앞에서 **멈춰 있던 구간**을 찾는다.

## 왜 근접만으로는 안 되는가

전체 영상에서 신원별 선반 접촉 비율을 세어 보니 **모든 사람이 30~90%** 였다
(docs/zones.md). 통로를 걷는다는 것 자체가 선반 옆에 있다는 뜻이므로,
"선반에 가까웠다"는 신호는 지나간 사람과 집은 사람을 전혀 구분하지 못한다.

그래서 조건을 셋으로 둔다.

    1. 선반과 min_overlap 이상 겹친다
    2. 속도가 max_speed 이하다 (= 멈췄다)
    3. 위 상태가 dwell_seconds 이상 이어진다

## 속도를 체구 높이로 정규화하는 이유

한 화면 안에서 사람 크기가 4~10배 변한다(docs/scale-limits.md).
픽셀 속도를 쓰면 같은 걸음이라도 멀리 있을 때는 느린 것으로 잡혀,
**통로 안쪽을 그냥 지나가는 사람이 전부 '멈춤'으로 오인된다.**
bbox 높이로 나누면 거리와 무관한 값이 된다.

## 이 계층이 알 수 없는 것

상품이 실제로 옮겨졌는지는 **전혀 모른다.** 그래서 출력 이름이 Event 가 아니라 Candidate 다.
상품 개별 추적은 측정 결과 실패했고(같은 상품끼리 ID 건너뜀), ReID 로도 고칠 수 없다.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ..core.config import InteractionConfig
from ..core.types import Frame, IdentityObservation, TakeCandidate, ZoneType
from ..zones.zone_map import ZoneMap
from .pose_features import hand_height, hands_in_zones, wrist_zone_distance


@dataclass
class _Motion:
    """사람 한 명의 최근 움직임. 속도를 시간 창으로 평활하기 위해 보관한다."""

    samples: deque = field(default_factory=lambda: deque(maxlen=64))
    """(pts_ms, 발 x, 발 y, bbox 높이)"""

    def add(self, pts_ms: float, x: float, y: float, height: float) -> None:
        self.samples.append((pts_ms, x, y, height))

    def speed(self, window_ms: float) -> float | None:
        """체구 높이 대비 초당 이동량. 표본이 부족하면 None."""
        if len(self.samples) < 2:
            return None
        now_ms, nx, ny, height = self.samples[-1]
        if height <= 0:
            return None
        oldest = None
        for sample in reversed(self.samples):
            oldest = sample
            if now_ms - sample[0] >= window_ms:
                break
        if oldest is None:
            return None
        dt = (now_ms - oldest[0]) / 1000.0
        if dt <= 0:
            return None
        dist = ((nx - oldest[1]) ** 2 + (ny - oldest[2]) ** 2) ** 0.5
        return (dist / height) / dt


@dataclass
class _Episode:
    """한 사람이 한 선반 앞에서 멈춰 있던 진행 중인 구간."""

    person_id: int
    zone: str
    start_frame: int
    start_ms: float
    start_timestamp: str
    end_frame: int
    end_ms: float
    end_timestamp: str
    observations: int = 0
    max_overlap: float = 0.0
    min_speed: float = float("inf")
    last_ok_ms: float = 0.0

    # 2단계용. 이 체류 구간 안에서 손이 올라가 있던 구간들.
    # 체류가 자격을 갖추는지는 구간이 끝나야 알 수 있으므로 일단 모아둔다.
    raises: list = field(default_factory=list)
    open_raise: dict | None = None

    def to_candidate(self) -> TakeCandidate:
        return TakeCandidate(
            person_id=self.person_id,
            zone=self.zone,
            start_frame=self.start_frame,
            end_frame=self.end_frame,
            start_timestamp=self.start_timestamp,
            end_timestamp=self.end_timestamp,
            duration_sec=round((self.end_ms - self.start_ms) / 1000.0, 3),
            observations=self.observations,
            max_overlap=round(self.max_overlap, 4),
            min_speed=round(self.min_speed, 4) if self.min_speed != float("inf") else 0.0,
        )


class InteractionDetector:
    """선반 앞 '멈춤' 구간을 찾아 TAKE 후보로 내보낸다."""

    def __init__(self, config: InteractionConfig, zone_map: ZoneMap) -> None:
        self.config = config
        self.zone_map = zone_map
        self.reset()

    def reset(self) -> None:
        self._last_height: float | None = None
        self._motion: dict[int, _Motion] = {}
        self._episodes: dict[tuple[int, str], _Episode] = {}
        self.completed: list[TakeCandidate] = []

    # ------------------------------------------------------------------

    def update(
        self, frame: Frame, identities: list[IdentityObservation]
    ) -> list[TakeCandidate]:
        """이 프레임에서 **끝난** 후보들을 돌려준다."""
        cfg = self.config
        window_ms = cfg.speed_window_seconds * 1000.0
        finished: list[TakeCandidate] = []
        touched: set[tuple[int, str]] = set()

        use_hand = cfg.signal in ("hand", "both")
        use_dwell = cfg.signal in ("dwell", "both", "raise", "stage")
        use_raise = cfg.signal == "raise"
        use_stage = cfg.signal == "stage"
        use_reach = cfg.signal == "reach"

        for obs in identities:
            if obs.person_id <= 0:
                continue
            hand_zones = (
                hands_in_zones(self.zone_map, obs.keypoints, cfg.hand_conf)
                if use_hand
                else set()
            )
            near_shelf = True
            if use_reach:
                distance = wrist_zone_distance(
                    self.zone_map, obs.keypoints, cfg.hand_conf, obs.bbox[3] - obs.bbox[1]
                )
                near_shelf = distance is not None and distance <= cfg.wrist_zone_max

            raised = True
            self._last_height = None
            if use_raise or use_stage:
                height = hand_height(
                    obs.keypoints, cfg.hand_conf, obs.bbox[3] - obs.bbox[1]
                )
                self._last_height = height
                raised = height is not None and height >= cfg.hand_height_min
            x1, y1, x2, y2 = obs.bbox
            height = y2 - y1
            motion = self._motion.setdefault(obs.person_id, _Motion())
            motion.add(frame.pts_ms, (x1 + x2) / 2.0, y2, height)
            speed = motion.speed(window_ms)
            if speed is None:
                continue  # 아직 속도를 판단할 표본이 부족하다
            slow = speed <= cfg.max_speed

            for hit in self.zone_map.evaluate(obs.bbox):
                if hit.zone_type is not ZoneType.SHELF:
                    continue
                # 몸 겹침 + 저속 (dwell) / 손이 구역 안 (hand) / 둘 다 (both)
                ok_dwell = hit.overlap >= cfg.min_overlap and slow
                if use_raise:
                    ok_dwell = ok_dwell and raised
                if use_reach:
                    # 수직 시점에서는 몸 겹침·속도가 아니라 손-선반 거리가 신호다.
                    ok_dwell = near_shelf
                ok_hand = hit.zone in hand_zones
                if use_dwell and use_hand:
                    passed = ok_dwell and ok_hand
                elif use_hand:
                    passed = ok_hand
                else:
                    passed = ok_dwell
                if not passed:
                    continue
                key = (obs.person_id, hit.zone)
                touched.add(key)
                episode = self._episodes.get(key)
                if episode is None:
                    episode = _Episode(
                        person_id=obs.person_id,
                        zone=hit.zone,
                        start_frame=obs.frame,
                        start_ms=frame.pts_ms,
                        start_timestamp=obs.timestamp,
                        end_frame=obs.frame,
                        end_ms=frame.pts_ms,
                        end_timestamp=obs.timestamp,
                    )
                    self._episodes[key] = episode
                episode.end_frame = obs.frame
                episode.end_ms = frame.pts_ms
                episode.end_timestamp = obs.timestamp
                episode.last_ok_ms = frame.pts_ms
                episode.observations += 1
                episode.max_overlap = max(episode.max_overlap, hit.overlap)
                episode.min_speed = min(episode.min_speed, speed)

                if use_stage:
                    self._track_raise(episode, obs, frame, raised, hit.overlap, speed)

        # 조건이 끊긴 지 오래된 구간을 마감한다.
        # 곧바로 끊지 않는 이유: 한 프레임 튄 값 때문에 구간이 쪼개지면
        # 체류 시간이 과소평가되어 후보를 놓친다.
        tolerance_ms = cfg.gap_tolerance_seconds * 1000.0
        for key, episode in list(self._episodes.items()):
            if key in touched:
                continue
            if frame.pts_ms - episode.last_ok_ms > tolerance_ms:
                del self._episodes[key]
                finished.extend(self._close(episode))

        self.completed.extend(finished)
        return finished

    def flush(self) -> list[TakeCandidate]:
        """영상이 끝났을 때 남아 있는 구간을 마감한다."""
        out: list[TakeCandidate] = []
        for episode in self._episodes.values():
            out.extend(self._close(episode))
        self._episodes.clear()
        self.completed.extend(out)
        return out

    def _track_raise(self, episode, obs, frame, raised, overlap, speed) -> None:
        """2단계 — 체류 구간 안에서 손이 올라가 있던 구간을 기록한다.

        여기서는 자격 판정을 하지 않는다. 체류가 dwell_seconds 를 채울지는
        구간이 끝나야 알 수 있으므로, 일단 모아두고 _close 에서 걸러낸다.
        """
        run = episode.open_raise
        if raised:
            if run is None:
                episode.open_raise = {
                    "start_frame": obs.frame,
                    "start_ms": frame.pts_ms,
                    "start_timestamp": obs.timestamp,
                    "end_frame": obs.frame,
                    "end_ms": frame.pts_ms,
                    "end_timestamp": obs.timestamp,
                    "observations": 1,
                    "max_overlap": overlap,
                    "min_speed": speed,
                    "peak": self._last_height or 0.0,
                }
            else:
                run["end_frame"] = obs.frame
                run["end_ms"] = frame.pts_ms
                run["end_timestamp"] = obs.timestamp
                run["observations"] += 1
                run["max_overlap"] = max(run["max_overlap"], overlap)
                run["min_speed"] = min(run["min_speed"], speed)
                run["peak"] = max(run["peak"], self._last_height or 0.0)
        elif run is not None:
            episode.raises.append(run)
            episode.open_raise = None

    def _close(self, episode: _Episode) -> list[TakeCandidate]:
        """체류 구간을 마감한다.

        signal="stage" 면 체류 구간 자체가 아니라, 그 안의 '손 올림' 구간들을 내보낸다.
        체류는 문지기이고(정밀도), 손 올림이 시점을 찍는다(국소화).
        """
        duration = (episode.end_ms - episode.start_ms) / 1000.0
        if duration < self.config.dwell_seconds:
            return []

        if self.config.signal != "stage":
            return [episode.to_candidate()]

        if episode.open_raise is not None:
            episode.raises.append(episode.open_raise)
            episode.open_raise = None

        out: list[TakeCandidate] = []
        min_ms = self.config.raise_min_seconds * 1000.0
        runs = [r for r in episode.raises if r["end_ms"] - r["start_ms"] >= min_ms]
        if self.config.best_raise_only and runs:
            # 구경 중에도 손은 여러 번 올라간다. 가장 높이 든 한 번만 남긴다.
            runs = [max(runs, key=lambda r: r["peak"])]
        for run in runs:
            out.append(
                TakeCandidate(
                    person_id=episode.person_id,
                    zone=episode.zone,
                    start_frame=run["start_frame"],
                    end_frame=run["end_frame"],
                    start_timestamp=run["start_timestamp"],
                    end_timestamp=run["end_timestamp"],
                    duration_sec=round((run["end_ms"] - run["start_ms"]) / 1000.0, 3),
                    observations=run["observations"],
                    max_overlap=round(run["max_overlap"], 4),
                    min_speed=round(run["min_speed"], 4),
                )
            )
        return out

    # ------------------------------------------------------------------

    def active_people(self) -> set[int]:
        """지금 어떤 선반 앞에 멈춰 있는 사람들. 시각화용."""
        return {pid for pid, _ in self._episodes}

    def summary(self) -> str:
        if not self.completed:
            return "TAKE 후보 없음"
        people = {c.person_id for c in self.completed}
        total = sum(c.duration_sec for c in self.completed)
        return (
            f"TAKE 후보 {len(self.completed)}건 / {len(people)}명 "
            f"(총 체류 {total:.1f}초, 평균 {total/len(self.completed):.1f}초)"
        )
