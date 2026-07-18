"""Motion detection: frame-differencing boxes and dense optical flow.

Stateful across frames (it needs a previous frame to difference against), so
one-shot use from a chat command is inherently weak — the router warms it up
with a couple of frames first. Primarily a live-preview mode.
"""
from __future__ import annotations

import cv2
import numpy as np

from core.base_detector import BaseDetector, Detection

FLOW_SCALE = 0.5  # compute flow on a downscaled frame for speed
ARROW_STEP = 24   # px between drawn flow vectors


class MotionDetector(BaseDetector):
    NAME = "Motion"
    BOX_COLOR = (0, 0, 255)  # red
    MATCH_HINT = ("no match value — returns every moving region. Use for "
                  "\"is anything moving?\".")
    PARAM_SPEC = [
        {"key": "mode", "label": "Method", "type": "combo",
         "options": ["diff boxes", "optical flow"], "default": "diff boxes"},
        {"key": "sensitivity", "label": "Sensitivity", "type": "slider",
         "min": 1, "max": 100, "default": 50},
        {"key": "min_area", "label": "Min region area (px)", "type": "slider",
         "min": 100, "max": 20000, "default": 1200},
        {"key": "vectors", "label": "Draw flow vectors", "type": "check",
         "default": True},
    ]

    def __init__(self) -> None:
        super().__init__()
        self._prev_gray: np.ndarray | None = None       # for frame differencing
        self._prev_small: np.ndarray | None = None      # for Farneback flow
        self._flow: np.ndarray | None = None            # kept for annotate_extra

    def _mask_to_detections(self, mask: np.ndarray) -> list[Detection]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            if cv2.contourArea(cnt) < self.params["min_area"]:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append(Detection(
                bbox=(x, y, w, h),
                centroid=(x + w // 2, y + h // 2),
                label="motion",
            ))
        return detections

    def _detect_diff(self, gray: np.ndarray) -> list[Detection]:
        if self._prev_gray is None:
            self._prev_gray = gray
            return []
        delta = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray
        # Higher sensitivity -> lower pixel-difference threshold
        thresh_val = int(np.interp(self.params["sensitivity"], [1, 100], [60, 5]))
        _, mask = cv2.threshold(delta, thresh_val, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, None, iterations=3)
        return self._mask_to_detections(mask)

    def _detect_flow(self, gray: np.ndarray) -> list[Detection]:
        small = cv2.resize(gray, None, fx=FLOW_SCALE, fy=FLOW_SCALE)
        if self._prev_small is None:
            self._prev_small = small
            return []
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_small, small, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        self._prev_small = small
        self._flow = flow

        magnitude = np.linalg.norm(flow, axis=2)
        # Higher sensitivity -> lower magnitude threshold (in downscaled px)
        mag_thresh = float(np.interp(self.params["sensitivity"], [1, 100], [8.0, 0.8]))
        mask = (magnitude > mag_thresh).astype(np.uint8) * 255
        mask = cv2.dilate(mask, None, iterations=2)
        mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        return self._mask_to_detections(mask)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        self._flow = None
        if self.params["mode"] == "optical flow":
            return self._detect_flow(gray)
        return self._detect_diff(gray)

    def annotate_extra(self, frame: np.ndarray) -> None:
        """Draw a sparse arrow field of the optical flow over the frame."""
        if self._flow is None or not self.params["vectors"]:
            return
        h, w = frame.shape[:2]
        step = int(ARROW_STEP * FLOW_SCALE)
        for fy in range(step // 2, self._flow.shape[0], step):
            for fx in range(step // 2, self._flow.shape[1], step):
                dx, dy = self._flow[fy, fx]
                if dx * dx + dy * dy < 1.0:
                    continue
                x0, y0 = int(fx / FLOW_SCALE), int(fy / FLOW_SCALE)
                x1 = int(np.clip(x0 + dx / FLOW_SCALE, 0, w - 1))
                y1 = int(np.clip(y0 + dy / FLOW_SCALE, 0, h - 1))
                cv2.arrowedLine(frame, (x0, y0), (x1, y1), (0, 220, 255), 1,
                                tipLength=0.3)
