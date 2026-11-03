"""카메라별 '사람 아님 영역' — 늘 같은 자리에서 사람으로 잘못 잡히는 물건을 뺀다.

높은 카메라(angle1)는 의자 위 외투를 사람으로 잡았다 (점수 0.30~0.33, limitations B-11).
그 자리를 화면 비율 상자로 적어 두고, 탐지 상자가 거의 그 상자와 같으면(겹침 비율 기준) 버린다.
사람이 그 앞을 지나가면 상자가 훨씬 커지거나 어긋나므로 남는다.
"""

from __future__ import annotations

from ..core.types import Detection, Frame


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class MaskedDetector:
    def __init__(self, detector, regions: list[tuple[float, float, float, float]], min_iou: float = 0.5) -> None:
        self.detector = detector
        self.regions = regions
        """화면 비율 (x1, y1, x2, y2)."""
        self.min_iou = min_iou
        self.dropped = 0

    def __getattr__(self, name):
        return getattr(self.detector, name)

    def detect(self, frame: Frame) -> list[Detection]:
        found = self.detector.detect(frame)
        if not self.regions or frame.image is None:
            return found
        h, w = frame.image.shape[:2]
        boxes = [(x1 * w, y1 * h, x2 * w, y2 * h) for x1, y1, x2, y2 in self.regions]
        kept = [d for d in found if all(_iou(d.bbox, r) < self.min_iou for r in boxes)]
        self.dropped += len(found) - len(kept)
        return kept
