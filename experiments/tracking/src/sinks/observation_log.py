"""TrackObservation 을 JSONL 로 남긴다.

Phase 4 의 확인 항목: 뒤 모듈이 tracker 라이브러리를 직접 참조하지 않고
이 로그(=표준 인터페이스)만 읽어도 동작할 수 있어야 한다.
근거: 04 인터페이스 §6
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from ..core.types import IdentityObservation, TrackObservation


class ObservationLog:
    """한 줄에 TrackObservation 하나씩 기록하는 writer."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
        self.count = 0

    def write_many(self, observations: Iterable[TrackObservation]) -> None:
        for obs in observations:
            self._file.write(json.dumps(obs.to_dict(), ensure_ascii=False) + "\n")
            self.count += 1

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "ObservationLog":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_observations(path: str | Path) -> Iterator[TrackObservation]:
    """저장된 JSONL 을 다시 TrackObservation 으로 읽는다. 이후 단계의 입력원."""
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            raw["bbox"] = tuple(raw["bbox"])
            if raw.get("keypoints"):
                raw["keypoints"] = tuple(tuple(k) for k in raw["keypoints"])
            yield TrackObservation(**raw)


def read_identities(path: str | Path) -> Iterator[IdentityObservation]:
    """identities.jsonl 을 IdentityObservation 으로 읽는다. 계층 3 의 입력원."""
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            raw["bbox"] = tuple(raw["bbox"])
            if raw.get("keypoints"):
                raw["keypoints"] = tuple(tuple(k) for k in raw["keypoints"])
            yield IdentityObservation(**raw)
