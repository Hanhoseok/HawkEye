"""Phase 1: 사람 탐지 계층.

사전학습 YOLO 로 person bbox 를 얻는다. 1차 구현에서는 재학습하지 않는다.
ultralytics 의존은 이 파일 안에서 끝나고, 바깥으로는 core.types.Detection 만 나간다.
근거: 02 기능 명세 F2(일부), 05 아키텍처 §4
"""

from __future__ import annotations

from ..core.config import DetectorConfig
from ..core.types import Detection, Frame


class YoloDetector:
    """ultralytics YOLO 를 core.types.Detection 으로 감싼 어댑터."""

    def __init__(self, config: DetectorConfig) -> None:
        from ultralytics import YOLO  # 무거운 import 는 생성 시점으로 미룬다

        self.config = config
        self.model = YOLO(config.model_path)
        self.names: dict[int, str] = dict(self.model.names)
        self.class_ids = self._resolve_class_ids(config.classes)

    def _resolve_class_ids(self, class_names: list[str] | None) -> list[int] | None:
        """설정의 클래스 이름을 모델의 클래스 id 로 바꾼다. 빈 값이면 전체 클래스."""
        if not class_names:
            return None
        lookup = {name.lower(): idx for idx, name in self.names.items()}
        ids: list[int] = []
        for name in class_names:
            key = name.lower()
            if key not in lookup:
                raise ValueError(
                    f"모델이 모르는 클래스입니다: {name} (사용 가능: {sorted(lookup)[:10]} ...)"
                )
            ids.append(lookup[key])
        return ids

    def detect(self, frame: Frame) -> list[Detection]:
        results = self.model.predict(
            source=frame.image,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.imgsz,
            device=self.config.device,
            classes=self.class_ids,
            half=self.config.half,
            verbose=False,
        )
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        return [
            Detection(
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                score=float(score),
                class_id=int(cls),
                class_name=self.names.get(int(cls), str(cls)),
            )
            for (x1, y1, x2, y2), score, cls in zip(xyxy, scores, class_ids)
        ]


def build_detector(config: DetectorConfig):
    """config.detector.name 으로 detector 를 고른다. 교체 지점은 여기 한 곳뿐이다."""
    if config.name == "yolo":
        return YoloDetector(config)
    if config.name == "yolo-pose":
        from .yolo_pose_detector import YoloPoseDetector

        return YoloPoseDetector(config)
    raise ValueError(f"알 수 없는 detector: {config.name} (사용 가능: yolo, yolo-pose)")
