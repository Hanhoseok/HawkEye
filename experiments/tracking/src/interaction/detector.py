"""계층 3 · 손 뻗기 후보 판정.

`IdentityObservation`(관절 포함) + 구역 정보를 받아, 사람의 **손목이 선반 구역에 닿아 있던 구간**을 찾는다.

## 근거 (`InteractionConfig.signal`)

    hand  : 손목이 선반 구역 안에 들어감 (선반 카메라의 칸 · 전체 카메라의 선반 영역)
    reach : 손목-선반 거리가 wrist_zone_max 이하 (수직 천장 시점 전용, MERL)

둘 다 관절 탐지기(yolo-pose)가 낸 손목이 있어야 한다. 관절이 없으면 후보가 나오지 않는다.
그 상태가 `min_seconds` 이상 이어지면 후보 하나다 (잠깐 끊긴 것은 `gap_tolerance_seconds` 안에서 잇는다).

## 지운 근거 — 선반 앞 머묾 (2026-11-21, 후속 60)

'몸이 선반 구역과 겹침 + 저속 + 일정 시간'(signal: dwell)과 그 위에 얹은 both · raise · stage 는 지웠다.
상품 · 관절 인식 전의 임시 근거였다. 집기 · 되돌려놓기 · 구경이 모두 같은 자리에서 일어나 몸의 위치로는
갈리지 않는다 (docs/take-candidates.md, docs/limitations.md D-1). 예전 값을 주면 만들 때 멈춘다.

## 이 계층이 알 수 없는 것

상품이 실제로 옮겨졌는지는 **전혀 모른다.** 그래서 출력 이름이 Event 가 아니라 Candidate 다.
개수는 선반 지도 · 전체 카메라 상품 기록 · 센서가 센다.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.config import InteractionConfig
from ..core.types import Frame, IdentityObservation, TakeCandidate, ZoneType
from ..zones.zone_map import ZoneMap
from .pose_features import hands_in_zones, wrist_zone_distance

SIGNALS = ("hand", "reach")
"""쓸 수 있는 근거. 머묾 계열(dwell · both · raise · stage)은 지웠다."""


@dataclass
class _Episode:
    """한 사람의 손이 한 선반 구역에 닿아 있던 진행 중인 구간."""

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
    last_ok_ms: float = 0.0

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
        )


class InteractionDetector:
    """손이 선반 구역에 닿아 있던 구간을 찾아 TAKE 후보로 내보낸다."""

    def __init__(self, config: InteractionConfig, zone_map: ZoneMap) -> None:
        if config.signal not in SIGNALS:
            raise ValueError(
                f"interaction.signal={config.signal!r} 은 쓸 수 없다 — {' · '.join(SIGNALS)} 만 남았다 "
                "(선반 앞 머묾 계열은 지웠다, docs/limitations.md D-1)"
            )
        self.config = config
        self.zone_map = zone_map
        self.reset()

    def reset(self) -> None:
        self._seen: set[int] = set()
        self._episodes: dict[tuple[int, str], _Episode] = {}
        self.completed: list[TakeCandidate] = []

    # ------------------------------------------------------------------

    def update(
        self, frame: Frame, identities: list[IdentityObservation]
    ) -> list[TakeCandidate]:
        """이 프레임에서 **끝난** 후보들을 돌려준다."""
        cfg = self.config
        finished: list[TakeCandidate] = []
        touched: set[tuple[int, str]] = set()
        use_reach = cfg.signal == "reach"

        for obs in identities:
            if obs.person_id <= 0:
                continue
            if obs.person_id not in self._seen:
                # 그 사람이 처음 보인 장면 하나는 건너뛴다 (지금까지의 기록과 구간 시작을 같게 둔다)
                self._seen.add(obs.person_id)
                continue
            hand_zones: set[str] = set()
            near_shelf = False
            if use_reach:
                distance = wrist_zone_distance(
                    self.zone_map, obs.keypoints, cfg.hand_conf, obs.bbox[3] - obs.bbox[1]
                )
                near_shelf = distance is not None and distance <= cfg.wrist_zone_max
            else:
                hand_zones = hands_in_zones(self.zone_map, obs.keypoints, cfg.hand_conf)

            for hit in self.zone_map.evaluate(obs.bbox):
                if hit.zone_type is not ZoneType.SHELF:
                    continue
                # 손목이 구역 안 (hand) / 손목-선반 거리 (reach, 수직 시점)
                if not (near_shelf if use_reach else hit.zone in hand_zones):
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

        # 조건이 끊긴 지 오래된 구간을 마감한다.
        # 곧바로 끊지 않는 이유: 한 프레임 튄 값 때문에 구간이 쪼개지면
        # 길이가 과소평가되어 후보를 놓친다.
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

    def _close(self, episode: _Episode) -> list[TakeCandidate]:
        """구간을 마감한다. min_seconds 보다 짧으면 버린다."""
        duration = (episode.end_ms - episode.start_ms) / 1000.0
        if duration < self.config.min_seconds:
            return []
        return [episode.to_candidate()]

    # ------------------------------------------------------------------

    def active_people(self) -> set[int]:
        """지금 손이 어떤 선반 구역에 닿아 있는 사람들. 시각화용."""
        return {pid for pid, _ in self._episodes}

    def summary(self) -> str:
        if not self.completed:
            return "TAKE 후보 없음"
        people = {c.person_id for c in self.completed}
        total = sum(c.duration_sec for c in self.completed)
        return (
            f"TAKE 후보 {len(self.completed)}건 / {len(people)}명 "
            f"(총 {total:.1f}초, 평균 {total/len(self.completed):.1f}초)"
        )
