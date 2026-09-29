"""tracker 교체 지점.

새 tracker 를 추가할 때 손대는 곳은 여기 한 곳뿐이다.
무거운 라이브러리는 실제로 선택됐을 때만 import 한다
(ByteTrack 만 쓰는 사람이 boxmot 을 설치하지 않아도 되도록).
"""

from __future__ import annotations

from ..core.config import TrackerConfig


def build_tracker(config: TrackerConfig):
    if config.name == "bytetrack":
        from .bytetrack_tracker import ByteTrackTracker

        return ByteTrackTracker(config)
    if config.name == "botsort":
        from .botsort_tracker import BotSortTracker

        return BotSortTracker(config)
    raise ValueError(f"알 수 없는 tracker: {config.name} (사용 가능: bytetrack, botsort)")
