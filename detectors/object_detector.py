"""General object detection using YOLOv8n (Ultralytics, Apache 2.0).

Closed-vocabulary: only the 80 COCO classes. `match` is filtered against the
class name after inference, so "pick up the cup" works but "pick up the
widget" will not - that needs the open-vocabulary path (Grounding DINO)
noted in the project plan.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from core.base_detector import BaseDetector, Detection

# Weights live at the project root. Resolved from __file__ rather than the cwd
# so the detector works no matter where the app is launched from.
_LOCAL_WEIGHTS = Path(__file__).resolve().parents[1] / "yolov8n.pt"


def _weights_path() -> str:
    if _LOCAL_WEIGHTS.exists():
        return str(_LOCAL_WEIGHTS)
    return "yolov8n.pt"  # not there - let ultralytics download it


class ObjectDetector(BaseDetector):
    NAME = "Objects"
    BOX_COLOR = (80, 200, 120)  # green
    MATCH_HINT = ("match must be a COCO class name (person, cup, bottle, bowl, "
                  "cell phone, scissors, book, apple, banana, orange, ...). "
                  "Use for everyday objects named by their kind, not color.")
    PARAM_SPEC = [
        {"key": "conf", "label": "Confidence threshold (%)", "type": "slider",
         "min": 5, "max": 95, "default": 40},
        {"key": "max_det", "label": "Max detections", "type": "slider",
         "min": 1, "max": 50, "default": 20},
    ]

    def __init__(self) -> None:
        super().__init__()
        # Lazy import: ultralytics pulls in torch, which is slow to load and
        # shouldn't be paid for unless this mode is actually used.
        from ultralytics import YOLO
        self._model = YOLO(_weights_path())
        self._names = self._model.names

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self._model(
            frame,
            conf=self.params["conf"] / 100.0,
            max_det=self.params["max_det"],
            verbose=False,
        )[0]

        detections: list[Detection] = []
        for box in results.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            w, h = x2 - x1, y2 - y1
            detections.append(Detection(
                bbox=(x1, y1, w, h),
                centroid=(x1 + w // 2, y1 + h // 2),
                label=self._names[int(box.cls[0])],
                confidence=float(box.conf[0]),
            ))
        return detections
