"""계층 2: 매장 단위 신원 레지스트리.

tracker 의 `lost_track_buffer` 는 **초 단위 짧은 공백**을 메우는 장치다.
고객이 매장에 5분 머무르며 선반 뒤로 스무 번 사라지는 것을 버티게 하려고
버퍼를 키우면, 이미 나간 사람까지 붙잡고 있어 오병합만 늘어난다.
(실측: buffer 30 -> 90 -> 150 으로 키워도 재식별은 복구되지 않았다.
 docs/tracker-comparison.md)

그래서 tracker 위에 계층을 하나 더 둔다.

    [계층 1] tracker            프레임 간 연결. 초 단위 공백 담당.
                 | TrackObservation
    [계층 2] IdentityRegistry   매장 안에 있는 사람들의 외형을 보관.
                                새 track 이 생기면 먼저 대조 -> 기존 신원에 잇거나 신규 입장.
                 | IdentityObservation

이 계층은 **tracker 구현을 전혀 모른다.** `TrackObservation` 만 받으므로
ByteTrack 이든 BoT-SORT 든 그대로 동작한다.

설계상 중요한 점:

* 임베딩은 **매 프레임 뽑지 않는다.** 새 track 의 판정 기간과 주기적 갱신 때만 뽑는다.
* 한 사람당 **템플릿을 여러 개** 보관한다. 같은 카메라 안에서도 사람 크기가
  4~10배 변하므로(docs/scale-limits.md), 하나만 두면 멀리 있을 때와 가까이 있을 때를 잇지 못한다.
* 너무 작은 관측은 **신원 판단에 쓰지 않는다.** 텐서 65px 미만에서는
  탐지 재현율이 43% 이하로 떨어지고 신원이 유실되기 시작한다(docs/scale-limits.md).

## 즉시 판정 + 나중에 합치기 (late merge)

새 track 은 사진 한 장으로 즉시 판정한다. 천장 시점에서는 한 장 비교가 자주 틀려
같은 사람이 새 신원으로 등록된다(MERL train: 같은 사람을 57% 만 이어줌).

판정을 몇 초 보류하고 사진을 모아 평균하면 정확해지지만, 보류하는 동안(person_id = -1)의
행동은 계층 3 이 볼 수 없다. 게다가 연속 프레임은 자세가 거의 같아서 평균해도 별 도움이 안 됐다
(연속 8장 = 0.5초 모아도 67%). 잡음이 상쇄되려면 사진 사이에 시간 간격이 있어야 한다.

그래서 판정은 즉시 하고, 신원마다 0.5초 간격으로 사진을 계속 모은다. 새로 등록된 신원이
10장(약 5초)을 모으면 '그 직전에 사라진 사람'과 평균끼리 다시 비교해 같으면 합친다(88%).

합친 기록은 `merges` 에 남는다. 이미 내보낸 관측의 person_id 는 바뀌지 않으므로,
뒤 계층(상태·위험 판정)이 이 기록을 받아 두 신원의 이력을 하나로 합쳐야 한다.
`resolve(person_id)` 가 최종 신원을 알려준다.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np

from ..core.config import IdentityConfig
from ..core.types import Frame, IdentityObservation, TrackObservation


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


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
    first_seen_ms: float = 0.0
    last_seen_ms: float = 0.0
    last_template_ms: float = -1e18
    last_bbox: tuple | None = None
    """마지막으로 보인 위치. 위치·시간 근거(stitching)에 쓴다."""

    embedding_sum: np.ndarray | None = None
    embedding_count: int = 0
    recent: deque = field(default_factory=lambda: deque(maxlen=3))
    """최근 사진 몇 장. 판정 직전 신원 확인(recent_consistency)에 쓴다."""

    def similarity(self, embedding: np.ndarray) -> float:
        """보관 중인 템플릿 중 가장 닮은 것과의 코사인 유사도 (새 track 즉시 판정용)."""
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

    def add_embedding(self, embedding: np.ndarray) -> None:
        """평균 대표값에 임베딩 하나를 보탠다."""
        if self.embedding_sum is None:
            self.embedding_sum = embedding.astype(np.float64).copy()
        else:
            self.embedding_sum = self.embedding_sum + embedding
        self.embedding_count += 1
        self.recent.append(embedding)

    @property
    def mean(self) -> np.ndarray | None:
        """지금까지 모은 임베딩 전체의 평균. '나중에 합치기'에서 쓰는 대표 외형."""
        if self.embedding_sum is None:
            return None
        return _unit(self.embedding_sum)


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
        self._alias: dict[int, int] = {}
        self.merges: list[dict] = []
        """나중에 합친 기록. {"from": 없어진 신원, "into": 남은 신원, "frame", "similarity"}"""
        self._next_person_id = 1
        self._min_bbox_px: float | None = None

    def resolve(self, person_id: int) -> int:
        """합쳐진 신원이면 최종적으로 남은 신원을 돌려준다."""
        while person_id in self._alias:
            person_id = self._alias[person_id]
        return person_id

    def recent_consistency(self, person_id: int) -> float | None:
        """최근 사진 몇 장의 평균과 그 이전 평균의 유사도.

        추적 번호가 다른 사람에게 옮겨 붙었으면 최근 모습이 이전 모습과 달라져 낮아진다
        (UCF-Crime Shoplifting047: 집기 기록을 가진 번호가 문가의 다른 사람으로 옮겨 붙어 오경보).
        기록이 모자라면 None — 확인할 수 없다는 뜻이지, 괜찮다는 뜻이 아니다.
        """
        person = self._people.get(self.resolve(person_id))
        cfg = self.config
        if person is None or person.embedding_sum is None:
            return None
        k = len(person.recent)
        if k < cfg.swap_check_samples or person.embedding_count - k < cfg.swap_check_min_older:
            return None
        recent = np.sum(np.stack(list(person.recent)), axis=0)
        older = person.embedding_sum - recent
        return float(_unit(older) @ _unit(recent))

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

    def _at_edge(self, obs: TrackObservation, frame: Frame) -> bool:
        """bbox 가 화면 가장자리에 걸려 몸이 잘려 있는가."""
        margin = self.config.edge_margin
        if margin <= 0 or frame.image is None:
            return False
        h, w = frame.image.shape[:2]
        x1, y1, x2, y2 = obs.bbox
        return x1 <= margin * w or y1 <= margin * h or x2 >= (1 - margin) * w or y2 >= (1 - margin) * h

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

    @staticmethod
    def _observation(obs: TrackObservation, person_id: int, **extra) -> IdentityObservation:
        return IdentityObservation(
            person_id=person_id,
            track_id=obs.track_id,
            bbox=obs.bbox,
            frame=obs.frame,
            timestamp=obs.timestamp,
            score=obs.score,
            class_name=obs.class_name,
            keypoints=obs.keypoints,
            **extra,
        )

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------

    def assign(
        self, frame: Frame, observations: list[TrackObservation]
    ) -> list[IdentityObservation]:
        self._ensure_threshold(frame)
        cfg = self.config

        known: list[tuple[TrackObservation, int]] = []
        pending: list[TrackObservation] = []  # 신원을 새로 정해야 하는 관측
        refresh: list[TrackObservation] = []  # 템플릿만 갱신할 관측

        refresh_ms = cfg.template_refresh_seconds * 1000.0
        for obs in observations:
            person_id = self._track_to_person.get(obs.track_id)
            if person_id is None:
                pending.append(obs)
                continue
            known.append((obs, person_id))
            person = self._people[person_id]
            person.last_seen = frame.index
            person.last_seen_ms = frame.pts_ms
            person.last_bbox = obs.bbox
            due = frame.pts_ms - person.last_template_ms >= refresh_ms
            if due and self._usable(obs):
                refresh.append(obs)

        # 이번 프레임에 이미 살아 있는 track 이 점유한 신원은 대조 대상에서 뺀다.
        # 한 사람이 같은 프레임에 두 개의 track 으로 존재할 수는 없기 때문이다.
        occupied = {pid for _, pid in known}

        results = [self._observation(obs, pid) for obs, pid in known]

        usable_pending = [o for o in pending if self._usable(o)]
        embeddings = self._embed(frame, refresh + usable_pending)
        cursor = 0

        grown: list[_Person] = []  # 사진이 한 장 늘어난 신원 (나중에 합치기 확인 대상)

        # 1) 살아 있는 track 의 템플릿 갱신
        for obs in refresh:
            emb = embeddings[cursor] if embeddings is not None else None
            cursor += 1
            if emb is None:
                continue
            person = self._people[self._track_to_person[obs.track_id]]
            person.add_template(emb, obs.bbox[3] - obs.bbox[1], cfg.max_templates)
            person.add_embedding(emb)
            person.last_template_ms = frame.pts_ms
            grown.append(person)

        # 2) 새로 생긴 track 에 신원을 배정
        for obs in pending:
            if not self._usable(obs):
                # 너무 작아 신원을 판단할 수 없다. 배정을 보류하고 다음 프레임에 다시 시도한다.
                results.append(self._observation(obs, -1))
                continue

            emb = embeddings[cursor] if embeddings is not None else None
            cursor += 1

            if self._at_edge(obs, frame):
                # 몸이 잘린 사진으로는 누구든 닮아 보인다. 합치지 않고 새 신원으로 두면
                # 몸이 다 보인 뒤 '나중에 합치기'가 판단한다. 잘못 합치는 것보다 잠깐 쪼개지는 편이 낫다.
                person_id, similarity, stitched = None, None, False
            else:
                person_id, similarity, stitched = self._match(emb, obs.bbox, frame.pts_ms, occupied)

            if person_id is None:
                person_id = self._next_person_id
                self._next_person_id += 1
                self._people[person_id] = _Person(
                    recent=deque(maxlen=cfg.swap_check_samples),
                    person_id=person_id,
                    first_seen=frame.index,
                    last_seen=frame.index,
                    first_seen_ms=frame.pts_ms,
                    last_seen_ms=frame.pts_ms,
                )
                similarity = None

            person = self._people[person_id]
            person.last_seen = frame.index
            person.last_seen_ms = frame.pts_ms
            person.last_bbox = obs.bbox
            if emb is not None:
                person.add_template(emb, obs.bbox[3] - obs.bbox[1], cfg.max_templates)
                person.add_embedding(emb)
                person.last_template_ms = frame.pts_ms
                grown.append(person)

            self._track_to_person[obs.track_id] = person_id
            occupied.add(person_id)

            results.append(
                self._observation(
                    obs,
                    person_id,
                    is_new=similarity is None,
                    matched_similarity=similarity,
                    stitched=stitched,
                )
            )

        # 3) 나중에 합치기
        merged = False
        for person in grown:
            if person.person_id in self._people and self._late_merge(person, frame) is not None:
                merged = True
        if merged:
            results = [
                replace(o, person_id=self.resolve(o.person_id)) if o.person_id > 0 else o
                for o in results
            ]

        results.sort(key=lambda o: o.track_id)
        return results

    def _late_merge(self, person: _Person, frame: Frame) -> int | None:
        """새로 등록됐던 신원이 사진을 충분히 모았으면, 직전에 사라진 사람과 다시 비교해 합친다.

        합칠 수 있는 상대의 조건:
          - 이 신원이 처음 나타나기 **전에** 사라졌다. 같은 시각에 함께 보인 적이 있다면
            한 사람일 수 없다. 이 조건이 오병합을 막는 가장 강한 근거다.
          - 사라진 지 retire_after_seconds 이내였다.
          - 평균끼리의 유사도 >= late_merge_threshold.
        """
        cfg = self.config
        if not cfg.late_merge_enabled or person.embedding_count not in cfg.late_merge_samples:
            return None
        mean = person.mean
        if mean is None:
            return None
        retire_ms = cfg.retire_after_seconds * 1000.0
        best_id, best_sim = None, cfg.late_merge_threshold
        for other in self._people.values():
            if other is person or other.mean is None:
                continue
            if other.last_seen_ms >= person.first_seen_ms:
                continue  # 함께 보인 적이 있다
            if person.first_seen_ms - other.last_seen_ms > retire_ms:
                continue
            sim = float(other.mean @ mean)
            if sim >= best_sim:
                best_id, best_sim = other.person_id, sim
        if best_id is None:
            return None
        self._absorb(best_id, person)
        self.merges.append(
            {"from": person.person_id, "into": best_id, "frame": frame.index, "similarity": best_sim}
        )
        return best_id

    def _absorb(self, into_id: int, person: _Person) -> None:
        """person 을 into 신원에 합친다. 이후 person 의 track 은 into 로 나간다."""
        into = self._people[into_id]
        into.embedding_sum = into.embedding_sum + person.embedding_sum
        into.embedding_count += person.embedding_count
        for t, h in zip(person.templates, person.template_heights):
            into.add_template(t, h, self.config.max_templates)
        into.last_seen = person.last_seen
        into.last_seen_ms = person.last_seen_ms
        into.last_bbox = person.last_bbox
        into.last_template_ms = person.last_template_ms
        into.recent = person.recent  # 지금 보이는 쪽(합쳐진 신원)의 최근 모습
        for track_id, pid in self._track_to_person.items():
            if pid == person.person_id:
                self._track_to_person[track_id] = into_id
        self._alias[person.person_id] = into_id
        del self._people[person.person_id]

    def _near(self, person: _Person, bbox, now_ms: float) -> bool:
        """방금 가까운 곳에서 사라진 사람인가 (위치·시간 근거).

        위치는 발 위치(bbox 아래쪽 중앙)로 본다. 팔을 뻗으면 bbox 위쪽은 크게 움직이지만
        아래쪽은 거의 그대로여서, 가운데보다 흔들림이 적다.
        """
        cfg = self.config
        if not cfg.stitch_enabled or person.last_bbox is None:
            return False
        if now_ms - person.last_seen_ms > cfg.stitch_max_seconds * 1000.0:
            return False
        px1, py1, px2, py2 = person.last_bbox
        nx1, ny1, nx2, ny2 = bbox
        body = max(py2 - py1, ny2 - ny1, 1.0)
        dist = (((px1 + px2) / 2 - (nx1 + nx2) / 2) ** 2 + (py2 - ny2) ** 2) ** 0.5
        return dist / body <= cfg.stitch_max_distance

    def _match(
        self, query: np.ndarray | None, bbox, now_ms: float, occupied: set[int]
    ) -> tuple[int | None, float | None, bool]:
        """가장 그럴듯한 기존 신원을 찾는다. 없으면 (None, None, False).

        두 가지 기준이 있다.
          - 생김새만으로:           유사도 >= match_threshold
          - 위치·시간이 맞을 때:    유사도 >= stitch_threshold (완화)

        후보가 여럿이면 '위치·시간이 맞는 쪽'에 가산점을 주어 고른다.
        방금 같은 자리에서 사라진 사람이, 20초 전 멀리서 사라진 사람보다
        생김새가 조금 덜 닮았더라도 훨씬 그럴듯하기 때문이다.
        """
        if query is None:
            return None, None, False
        cfg = self.config
        bonus = cfg.match_threshold - cfg.stitch_threshold
        best: tuple[float, int, float, bool] | None = None  # (유효점수, id, 유사도, stitched)
        for person in self._active_people(now_ms):
            if person.person_id in occupied:
                continue
            sim = person.similarity(query)
            near = self._near(person, bbox, now_ms)
            effective = sim + (bonus if near else 0.0)
            if effective < cfg.match_threshold:
                continue
            if best is None or effective > best[0]:
                # 생김새만으로도 기준을 넘었다면 stitching 덕분이 아니다
                best = (effective, person.person_id, sim, near and sim < cfg.match_threshold)
        if best is None:
            return None, None, False
        _, pid, sim, stitched = best
        return pid, sim, stitched

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
        if self.merges:
            lines.append(f"나중에 합친 신원 : {len(self.merges)}건")
        return "\n".join(lines)
