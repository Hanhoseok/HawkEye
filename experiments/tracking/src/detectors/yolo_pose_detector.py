"""pose detector — bbox 와 손 위치를 한 번에 얻는다.

왜 별도 pose 모델을 추가로 돌리지 않고 detector 를 통째로 교체하는가:
`yolov8n-pose` 는 person bbox 와 17개 keypoint 를 **한 번의 추론으로** 준다.
detection 과 pose 를 따로 돌리면 108.7 + 112.4 = 221.1 ms/frame 이지만,
pose detector 하나면 112.4 ms/frame 으로 끝난다.

pose 가 필요한 이유는 오탐 분석에서 나왔다(docs/take-candidates.md §6-E).
집기 · 되돌려놓기 · 구경이 전부 같은 자리에서 일어나므로,
**몸의 위치가 아니라 손의 움직임**을 봐야 한다.
"""

from __future__ import annotations

from ..core.config import DetectorConfig
from ..core.types import Detection, Frame


class YoloPoseDetector:
    """ultralytics YOLO-pose 어댑터. person bbox + COCO 17 keypoint."""

    def __init__(self, config: DetectorConfig) -> None:
        from ultralytics import YOLO

        self.config = config
        self.model = YOLO(config.model_path)
        self.names: dict[int, str] = dict(self.model.names)

    def detect(self, frame: Frame) -> list[Detection]:
        results = self.model.predict(
            source=frame.image,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.imgsz,
            device=self.config.device,
            half=self.config.half,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        keypoints = None
        if result.keypoints is not None and result.keypoints.data is not None:
            keypoints = result.keypoints.data.cpu().numpy()  # (N, 17, 3)

        detections: list[Detection] = []
        for i, ((x1, y1, x2, y2), score, cls) in enumerate(zip(xyxy, scores, class_ids)):
            kp = None
            if keypoints is not None and i < len(keypoints):
                # 소수점 1자리로 줄인다. 관측마다 17x3 값이라 로그가 빠르게 커진다.
                kp = tuple(
                    (round(float(x), 1), round(float(y), 1), round(float(c), 3))
                    for x, y, c in keypoints[i]
                )
            detections.append(
                Detection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    score=float(score),
                    class_id=int(cls),
                    class_name=self.names.get(int(cls), str(cls)),
                    keypoints=kp,
                )
            )
        return detections
