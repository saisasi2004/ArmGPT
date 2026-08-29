"""Geometric shape classification via contour approximation.

Edges -> contours -> cv2.approxPolyDP; classified as triangle / rectangle /
square / pentagon / hexagon / circle from vertex count and circularity.
"""
from __future__ import annotations

import cv2
import numpy as np

from core.base_detector import BaseDetector, Detection

SHAPES = ["triangle", "square", "rectangle", "pentagon", "hexagon", "circle"]


class ShapeDetector(BaseDetector):
    NAME = "Shapes"
    BOX_COLOR = (255, 200, 0)  # cyan-blue
    MATCH_HINT = ("match must be one of: " + ", ".join(SHAPES)
                  + ". Use when the object is identified by its geometry "
                    "(\"the round one\", \"the square block\").")
    PARAM_SPEC = [
        {"key": "canny_lo", "label": "Edge sensitivity", "type": "slider",
         "min": 10, "max": 150, "default": 50},
        {"key": "min_area", "label": "Min shape area (px)", "type": "slider",
         "min": 200, "max": 30000, "default": 1500},
        {"key": "epsilon", "label": "Approx epsilon (% of perim)", "type": "slider",
         "min": 1, "max": 10, "default": 3},
    ]

    def _classify(self, contour: np.ndarray) -> str | None:
        peri = cv2.arcLength(contour, True)
        if peri == 0:
            return None
        approx = cv2.approxPolyDP(contour, (self.params["epsilon"] / 100.0) * peri, True)
        vertices = len(approx)
        area = cv2.contourArea(contour)
        circularity = 4 * np.pi * area / (peri * peri)

        if vertices == 3:
            return "triangle"
        if vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / float(h)
            return "square" if 0.9 <= aspect <= 1.1 else "rectangle"
        if vertices == 5:
            return "pentagon"
        if vertices == 6:
            return "hexagon"
        if circularity > 0.8:
            return "circle"
        return None  # irregular blob - don't guess

    def detect(self, frame: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        lo = self.params["canny_lo"]
        edges = cv2.Canny(blurred, lo, lo * 3)
        edges = cv2.dilate(edges, None, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        detections: list[Detection] = []
        for cnt in contours:
            if cv2.contourArea(cnt) < self.params["min_area"]:
                continue
            shape = self._classify(cnt)
            if shape is None:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            m = cv2.moments(cnt)
            if m["m00"] == 0:
                continue
            cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
            detections.append(Detection(
                bbox=(x, y, w, h), centroid=(cx, cy), label=shape,
            ))
        return detections
