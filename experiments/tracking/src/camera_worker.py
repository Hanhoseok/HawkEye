"""카메라 한 대의 관찰 층 (① 사람 찾기·잇기 → ② 화면 신원 → ③ 손 뻗기 → ④ 선반 지도).

카메라가 둘 이상이 되면 카메라마다 이 작업자를 하나씩 두고, 판단(⑤ 손님 장부, ⑥ 위험 판정)은 합친 뒤에 한다
(후기 융합, docs/multiview-design.md M-2). 작업자는 '관찰'만 내고 판단하지 않는다.
카메라가 한 대일 때는 작업자 하나가 예전 파이프라인의 ①~④를 그대로 한다 — 결과가 같아야 한다 (design 1-1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .core.protocols import Detector, Tracker
from .core.types import Detection, Frame, IdentityObservation, TakeCandidate, TrackObservation
from .zones.zone_map import ZoneMap


@dataclass
class CameraStep:
    """한 장면을 작업자가 처리한 결과. 판단 층으로 넘어가는 관찰 묶음."""

    frame: Frame
    detections: list[Detection]
    observations: list[TrackObservation]
    identities: list[IdentityObservation] | None = None
    """신원 계층이 없으면 None (추적만 하는 실행)."""
    takes: list[TakeCandidate] = field(default_factory=list)
    """이 장면에서 끝난 손 뻗기 구간."""
    visits: list = field(default_factory=list)
    """이 장면에서 끝난 선반 방문 (ShelfVisit)."""
    changes: list = field(default_factory=list)
    """visits 와 짝을 이루는 선반 변화 (ShelfChange)."""
    slot_changes: list = field(default_factory=list)
    """칸 그림 비교로 확정된 칸 변화 (SlotChange, 설계 M-7 보조). 칸을 지정한 선반 카메라에서만."""

    @property
    def drawable(self) -> list:
        """화면에 그릴 사람들 — 신원이 있으면 신원, 없으면 추적 결과."""
        return self.identities if self.identities is not None else self.observations


class CameraWorker:
    """카메라 한 대 몫의 탐지기·추적기·신원·손 뻗기·선반. 상태는 카메라마다 따로 가진다."""

    def __init__(
        self,
        name: str,
        detector: Detector,
        tracker: Tracker,
        registry=None,
        zone_map: ZoneMap | None = None,
        interactions=None,
        shelves=None,
    ) -> None:
        self.name = name
        self.detector = detector
        self.tracker = tracker
        self.registry = registry
        self.zone_map = zone_map or ZoneMap.empty()
        self.interactions = interactions
        self.shelves = shelves

    def reset(self) -> None:
        self.tracker.reset()
        for part in (self.registry, self.interactions, self.shelves):
            if part is not None:
                part.reset()

    def resolve(self, width: int, height: int, fps: float) -> None:
        """프레임 크기·처리 속도를 알게 된 시점에 구역·선반 좌표를 픽셀로 확정한다."""
        if self.shelves is not None and self.registry is not None:
            self.shelves.resolve(width, height, fps)
        if self.zone_map:
            self.zone_map.resolve(width, height)

    @property
    def watches_shelves(self) -> bool:
        """선반 지도는 신원이 있어야 '누가 가렸나'를 기록할 수 있다."""
        return self.shelves is not None and self.registry is not None

    def step(self, frame: Frame) -> CameraStep:
        detections = self.detector.detect(frame)
        observations = self.tracker.update(frame, detections)
        step = CameraStep(frame, detections, observations)
        if self.registry is None:
            return step

        # ② 매장 단위(카메라가 둘이면 화면 단위) 신원. 이후 로직은 track_id 가 아니라 person_id 를 쓴다.
        step.identities = self.registry.assign(frame, observations)
        # ③ 선반 앞 '멈춤' 구간 -> TAKE 후보
        if self.interactions is not None:
            step.takes = self.interactions.update(frame, step.identities)
        # ④ 손님이 다녀간 선반에서 몇 개가 비었/다시 찼는지
        if self.shelves is not None:
            step.visits, step.changes = self.shelves.update(frame, step.identities, self.registry.resolve)
            step.slot_changes = list(getattr(self.shelves, "new_slot_changes", []))
        return step

    def flush(self) -> list[TakeCandidate]:
        """영상이 끝났을 때 아직 열려 있던 손 뻗기 구간을 닫는다."""
        return self.interactions.flush() if self.interactions is not None else []
