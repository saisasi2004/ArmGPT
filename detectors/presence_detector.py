"""Hand and face presence detection using MediaPipe (Google, Apache 2.0).

Doubles as the cell's soft safety check: `router` refuses to emit a motion
command while a hand is in frame. That is a convenience interlock, NOT a
safety-rated system — it must never be the only thing between a person and
the arm.
"""
from __future__ import annotations

import numpy as np

import cv2

from core.base_detector import BaseDetector, Detection


class PresenceDetector(BaseDetector):
    NAME = "Presence"
    BOX_COLOR = (200, 120, 255)  # pink
    MATCH_HINT = ("match may be \"hand\" or \"face\", or omit it for both. "
                  "Use for \"is anyone there?\", \"do you see my hand?\".")
    PARAM_SPEC = [
        {"key": "targets", "label": "Detect", "type": "combo",
         "options": ["hands + faces", "hands only", "faces only"],
         "default": "hands + faces"},
        {"key": "min_conf", "label": "Min confidence (%)", "type": "slider",
         "min": 10, "max": 95, "default": 50},
    ]

    def __init__(self) -> None:
        super().__init__()
        import mediapipe as mp  # lazy: heavy import, only paid when mode is used
        self._mp_hands = mp.solutions.hands
        self._hand_connections = list(self._mp_hands.HAND_CONNECTIONS)
        self._hands = self._mp_hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        self._faces = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.3,
        )

    def prepare(self, match: str | None) -> None:
        if not match:
            return
        m = match.lower()
        if "hand" in m:
            self.params["targets"] = "hands only"
        elif "face" in m or "person" in m or "people" in m:
            self.params["targets"] = "faces only"

    def _detect_hands(self, rgb: np.ndarray, w: int, h: int) -> list[Detection]:
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return []
        detections = []
        handedness = result.multi_handedness or []
        for i, landmarks in enumerate(result.multi_hand_landmarks):
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x, y = min(xs), min(ys)
            bw, bh = max(xs) - x, max(ys) - y
            label, conf = "hand", None
            if i < len(handedness):
                cls = handedness[i].classification[0]
                label = f"{cls.label.lower()} hand"
                conf = cls.score
            if conf is not None and conf < self.params["min_conf"] / 100.0:
                continue
            detections.append(Detection(
                bbox=(x, y, bw, bh),
                centroid=(x + bw // 2, y + bh // 2),
                label=label,
                confidence=conf,
                points=pts,
                connections=self._hand_connections,
            ))
        return detections

    def _detect_faces(self, rgb: np.ndarray, w: int, h: int) -> list[Detection]:
        result = self._faces.process(rgb)
        if not result.detections:
            return []
        detections = []
        for det in result.detections:
            conf = det.score[0]
            if conf < self.params["min_conf"] / 100.0:
                continue
            box = det.location_data.relative_bounding_box
            x, y = int(box.xmin * w), int(box.ymin * h)
            bw, bh = int(box.width * w), int(box.height * h)
            detections.append(Detection(
                bbox=(x, y, bw, bh),
                centroid=(x + bw // 2, y + bh // 2),
                label="face",
                confidence=conf,
            ))
        return detections

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        target = self.params["targets"]
        detections: list[Detection] = []
        if target in ("hands + faces", "hands only"):
            detections.extend(self._detect_hands(rgb, w, h))
        if target in ("hands + faces", "faces only"):
            detections.extend(self._detect_faces(rgb, w, h))
        return detections
