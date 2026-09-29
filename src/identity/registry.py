"""계층 2: 매장 단위 신원 레지스트리.

tracker 의 `lost_track_buffer` 는 **초 단위 짧은 공백**을 메우는 장치다.
고객이 매장에 5분 머무르며 선반 뒤로 스무 번 사라지는 것을 버티게 하려고
버퍼를 키우면, 이미 나간 사람까지 붙잡고 있어 오병합만 늘어난다.
(실측: buffer 30 -> 90 -> 150 으로 키워도 재식별은 복구되지 않았다.
 docs/tracker-comparison.md)

그래서 tracker 위에 계층을 하나 더 둔다.

    [계층 1] tracker            프레임 간 연결. 초 단위 공백 담당.
                 | TrackObservation
    [계층 2] IdentityRegistry   매장 안에 있는 사람들의 외형 템플릿을 보관.
                                새 track 이 생기면 먼저 대조 -> 기존 신원에 잇거나 신규 입장.
                 | IdentityObservation

이 계층은 **tracker 구현을 전혀 모른다.** `TrackObservation` 만 받으므로
ByteTrack 이든 BoT-SORT 든 그대로 동작한다.

설계상 중요한 점:

* 임베딩은 **매 프레임 뽑지 않는다.** 새 track 이 생겼을 때와 주기적 갱신 때만 뽑는다.
  tracker 내부 ReID 가 매 프레임 계산하는 것과 달리 비용이 훨씬 적다.
* 한 사람당 **템플릿을 여러 개** 보관한다. 같은 카메라 안에서도 사람 크기가
  4~10배 변하므로(docs/scale-limits.md), 하나만 두면 멀리 있을 때와 가까이 있을 때를 잇지 못한다.
* 너무 작은 관측은 **신원 판단에 쓰지 않는다.** 텐서 65px 미만에서는
  탐지 재현율이 43% 이하로 떨어지고 신원이 유실되기 시작한다(docs/scale-limits.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.config import IdentityConfig
from ..core.types import Frame, IdentityObservation, TrackObservation


@dataclass
class _Person:
    """매장 안에 있는(또는 있었던) 한 사람."""

    person_id: int
    templates: list[np.ndarray] = field(default_factory=list)
    template_heights: list[float] = field(default_factory=list)
    first_seen: int = 0
    last_seen: int = 0
    # 시간 판단은 프레임 번호가 아니라 영상 재생 시각(ms)으로 한다.
    # 프레임 번호는 원본 fps 와 stride 에 따라 같은 값이 다른 시간을 뜻하기 때문이다.
    last_seen_ms: float = 0.0
    last_template_ms: float = -1e18

    def similarity(self, embedding: np.ndarray) -> float:
        """보관 중인 템플릿 중 가장 닮은 것과의 코사인 유사도."""
        if not self.templates:
            return -1.0
        return float(max(float(t @ embedding) for t in self.templates))

    def add_template(self, embedding: np.ndarray, height: float, max_templates: int) -> None:
        """템플릿을 보관한다. 가득 차면 크기가 가장 비슷한 것을 대체한다.

        크기가 비슷한 것을 버려야 '멀 때'와 '가까울 때'의 템플릿이 함께 남는다.
        오래된 것부터 버리면 최근 크기로만 쏠려 스케일 변화를 못 견딘다.
        """
        if len(self.templates) < max_templates:
            self.templates.append(embedding)
            self.template_heights.append(height)
            return
        closest = min(
            range(len(self.template_heights)),
            key=lambda i: abs(self.template_heights[i] - height),
        )
        self.templates[closest] = embedding
        self.template_heights[closest] = height


class IdentityRegistry:
    """TrackObservation 에 매장 단위 person_id 를 붙인다.

    embedder 는 `embed(frame, boxes) -> (N, D) 정규화된 ndarray` 만 제공하면 된다.
    None 이면 외형 대조 없이 track_id 를 그대로 신원으로 쓴다(비교 실험용).
    """

    def __init__(
        self,
        config: IdentityConfig,
        embedder=None,
        imgsz: int = 640,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.imgsz = imgsz
        self.reset()

    def reset(self) -> None:
        self._people: dict[int, _Person] = {}
        self._track_to_person: dict[int, int] = {}
        self._next_person_id = 1
        self._min_bbox_px: float | None = None

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _ensure_threshold(self, frame: Frame) -> None:
        """텐서 기준 최소 크기를 이 영상의 화면 픽셀 기준으로 환산한다.

        텐서 픽셀 = 화면 픽셀 x imgsz / 프레임의 긴 변  (docs/scale-limits.md)
        """
        if self._min_bbox_px is not None:
            return
        if frame.image is None:
            self._min_bbox_px = 0.0
            return
        h, w = frame.image.shape[:2]
        longest = max(h, w) or 1
        self._min_bbox_px = self.config.min_tensor_height * longest / float(self.imgsz)

    def _usable(self, obs: TrackObservation) -> bool:
        """이 관측을 신원 판단에 써도 되는 크기인가."""
        height = obs.bbox[3] - obs.bbox[1]
        return height >= (self._min_bbox_px or 0.0)

    def _active_people(self, now_ms: float) -> list[_Person]:
        """아직 매장 안에 있다고 보는 사람들.

        기준은 '마지막으로 보인 뒤 얼마나 지났는가'이며, 체류 시간과는 무관하다.
        """
        limit_ms = self.config.retire_after_seconds * 1000.0
        return [p for p in self._people.values() if now_ms - p.last_seen_ms <= limit_ms]

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------

    def assign(
        self, frame: Frame, observations: list[TrackObservation]
    ) -> list[IdentityObservation]:
        self._ensure_threshold(frame)

        known: list[tuple[TrackObservation, int]] = []
        pending: list[TrackObservation] = []  # 신원을 새로 정해야 하는 관측
        refresh: list[TrackObservation] = []  # 템플릿만 갱신할 관측

        for obs in observations:
            person_id = self._track_to_person.get(obs.track_id)
            if person_id is None:
                pending.append(obs)
                continue
            known.append((obs, person_id))
            person = self._people[person_id]
            person.last_seen = frame.index
            person.last_seen_ms = frame.pts_ms
            refresh_ms = self.config.template_refresh_seconds * 1000.0
            due = frame.pts_ms - person.last_template_ms >= refresh_ms
            if due and self._usable(obs):
                refresh.append(obs)

        # 이번 프레임에 이미 살아 있는 track 이 점유한 신원은 대조 대상에서 뺀다.
        # 한 사람이 같은 프레임에 두 개의 track 으로 존재할 수는 없기 때문이다.
        occupied = {pid for _, pid in known}

        results: list[IdentityObservation] = [
            IdentityObservation(
                person_id=pid,
                track_id=obs.track_id,
                bbox=obs.bbox,
                frame=obs.frame,
                timestamp=obs.timestamp,
                score=obs.score,
                class_name=obs.class_name,
                keypoints=obs.keypoints,
            )
            for obs, pid in known
        ]

        embeddings = self._embed(frame, refresh + [o for o in pending if self._usable(o)])
        cursor = 0

        # 1) 살아 있는 track 의 템플릿 갱신
        for obs in refresh:
            emb = embeddings[cursor] if embeddings is not None else None
            cursor += 1
            if emb is None:
                continue
            person = self._people[self._track_to_person[obs.track_id]]
            person.add_template(emb, obs.bbox[3] - obs.bbox[1], self.config.max_templates)
            person.last_template_ms = frame.pts_ms

        # 2) 새로 생긴 track 에 신원을 배정
        for obs in pending:
            if not self._usable(obs):
                # 너무 작아 신원을 판단할 수 없다. 배정을 보류하고 다음 프레임에 다시 시도한다.
                results.append(
                    IdentityObservation(
                        person_id=-1,
                        track_id=obs.track_id,
                        bbox=obs.bbox,
                        frame=obs.frame,
                        timestamp=obs.timestamp,
                        score=obs.score,
                        class_name=obs.class_name,
                        keypoints=obs.keypoints,
                    )
                )
                continue

            emb = embeddings[cursor] if embeddings is not None else None
            cursor += 1
            person_id, similarity = self._match(emb, frame.pts_ms, occupied)

            if person_id is None:
                person_id = self._next_person_id
                self._next_person_id += 1
                self._people[person_id] = _Person(
                    person_id=person_id,
                    first_seen=frame.index,
                    last_seen=frame.index,
                    last_seen_ms=frame.pts_ms,
                )
                similarity = None

            person = self._people[person_id]
            person.last_seen = frame.index
            person.last_seen_ms = frame.pts_ms
            if emb is not None:
                person.add_template(emb, obs.bbox[3] - obs.bbox[1], self.config.max_templates)
                person.last_template_ms = frame.pts_ms

            self._track_to_person[obs.track_id] = person_id
            occupied.add(person_id)

            results.append(
                IdentityObservation(
                    person_id=person_id,
                    track_id=obs.track_id,
                    bbox=obs.bbox,
                    frame=obs.frame,
                    timestamp=obs.timestamp,
                    score=obs.score,
                    class_name=obs.class_name,
                    keypoints=obs.keypoints,
                    is_new=similarity is None,
                    matched_similarity=similarity,
                )
            )

        results.sort(key=lambda o: o.track_id)
        return results

    def _match(
        self, embedding: np.ndarray | None, now_ms: float, occupied: set[int]
    ) -> tuple[int | None, float | None]:
        """가장 닮은 기존 신원을 찾는다. 임계값에 못 미치면 (None, None)."""
        if embedding is None:
            return None, None
        best_id, best_sim = None, -1.0
        for person in self._active_people(now_ms):
            if person.person_id in occupied:
                continue
            sim = person.similarity(embedding)
            if sim > best_sim:
                best_id, best_sim = person.person_id, sim
        if best_id is not None and best_sim >= self.config.match_threshold:
            return best_id, best_sim
        return None, None

    def _embed(self, frame: Frame, observations: list[TrackObservation]):
        if not observations or self.embedder is None or frame.image is None:
            return None
        boxes = np.array([o.bbox for o in observations], dtype=np.float32)
        return self.embedder.embed(frame.image, boxes)

    # ------------------------------------------------------------------
    # 요약
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [f"등록된 신원 : {len(self._people)}명", f"track -> 신원 매핑 : {len(self._track_to_person)}건"]
        merged = len(self._track_to_person) - len(self._people)
        if merged > 0:
            lines.append(f"이어붙인 track : {merged}건 (tracker 가 쪼갠 것을 신원으로 복원)")
        return "\n".join(lines)
