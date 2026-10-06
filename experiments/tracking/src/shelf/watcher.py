"""선반 지켜보기 — 파이프라인에서 선반 지도(monitor.py)를 돌려, 손님 기록에 붙일 방문 결과를 만든다.

    매 장면:  물건 탐지(큰 해상도) + 이미 찾은 사람(매장 단위 신원)
              -> 선반마다 ShelfMonitor.update
              -> 사람이 다녀가고 선반 지도가 다시 확정되면 방문 결과(ShelfVisit)
                 "손님 3: 1단 2번째 사라짐" / "손님 4: 변화 없음(구경)"

탐지기는 Detection 목록만 돌려주면 무엇이든 된다(지금은 일반 YOLO, 2단계에서 상품 모델).
"""

from __future__ import annotations

from typing import Callable

from ..core.config import ShelfConfig
from ..core.types import Frame, IdentityObservation
from ..risk.customers import ShelfVisit
from .monitor import ShelfChange, ShelfItem, ShelfMonitor


class ShelfWatcher:
    def __init__(self, config: ShelfConfig, detector) -> None:
        self.config = config
        self.detector = detector
        self.monitors: list[tuple[dict, ShelfMonitor]] = []
        self.per_window = 1
        self._seen: list[int] = []

    def resolve(self, width: int, height: int, fps: float) -> None:
        """화면 비율로 준 선반 영역을 픽셀로 바꾸고 선반마다 지켜보기를 만든다."""
        self.monitors = []
        for i, area in enumerate(self.config.areas):
            x1, y1, x2, y2 = (float(v) for v in area["box"])
            name = area.get("name") or f"shelf{i + 1}"
            box = (x1 * width, y1 * height, x2 * width, y2 * height)
            self.monitors.append((area, ShelfMonitor(name, box, stable_seconds=self.config.stable_seconds)))
        self.per_window = max(1, int(round(self.config.stable_seconds * fps)))
        self._seen = [0] * len(self.monitors)

    def reset(self) -> None:
        for _, monitor in self.monitors:
            monitor.reset()
        self._seen = [0] * len(self.monitors)

    @property
    def boxes(self) -> list[tuple[str, tuple[float, float, float, float]]]:
        return [(m.name, m.box) for _, m in self.monitors]

    def update(
        self,
        frame: Frame,
        identities: list[IdentityObservation],
        resolve: Callable[[int], int] | None = None,
    ) -> tuple[list[ShelfVisit], list[ShelfChange]]:
        """이번 장면에서 끝난 방문들. (손님 기록용 ShelfVisit, 원본 ShelfChange) 를 같은 순서로 돌려준다.

        resolve: 합쳐진 신원을 최종 번호로 (IdentityRegistry.resolve). 합쳐지기 전 번호와 후 번호가
        한 방문에 섞이면 '두 사람'으로 보여 누구 것인지 정하지 못하기 때문이다.
        신원 보류(-1)인 사람도 선반을 가리기는 하므로 가림에는 넣는다.
        """
        if not self.monitors:
            return [], []
        items = [ShelfItem(d.class_name, d.bbox, d.score) for d in self.detector.detect(frame)]
        people = [
            ((resolve(o.person_id) if resolve else o.person_id) if o.person_id > 0 else -1, o.bbox)
            for o in identities
        ]
        visits: list[ShelfVisit] = []
        changes: list[ShelfChange] = []
        for i, (area, monitor) in enumerate(self.monitors):
            monitor.update(frame.pts_ms, items, people, self.per_window, frame.index)
            for change in monitor.visits[self._seen[i]:]:
                pid = change.person_id
                visits.append(ShelfVisit(
                    person_id=pid if pid is not None else -1,
                    shelf=monitor.name,
                    start_frame=change.start_frame,
                    end_frame=change.end_frame,
                    removed=len(change.removed),
                    added=len(change.added),
                    zone=area.get("zone"),
                    detail=change.describe(),
                ))
                changes.append(change)
            self._seen[i] = len(monitor.visits)
        return visits, changes


def parse_area(text: str, index: int) -> dict:
    """명령줄 값 'name=x1,y1,x2,y2[@zone]' 또는 'x1,y1,x2,y2' 를 선반 영역 설정으로."""
    name, _, rest = text.rpartition("=")
    rest, _, zone = rest.partition("@")
    values = [float(v) for v in rest.split(",")]
    if len(values) != 4 or not all(0.0 <= v <= 1.0 for v in values):
        raise ValueError(f"선반 영역은 화면 비율 x1,y1,x2,y2 (0~1) 이어야 합니다: {text}")
    area = {"name": name or f"shelf{index + 1}", "box": values}
    if zone:
        area["zone"] = zone
    return area
