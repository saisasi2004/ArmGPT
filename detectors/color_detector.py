"""Color-blob detection via OpenCV HSV thresholding.

The workhorse for pick-and-place: "the red object" resolves here. Fast,
deterministic, no GPU. `prepare()` pushes the LLM's requested color straight
into the HSV preset, so `match="red"` configures the threshold band.
"""
from __future__ import annotations

import cv2
import numpy as np

from core.base_detector import BaseDetector, Detection

# OpenCV HSV: H in [0,179], S/V in [0,255]. Red wraps around hue 0, so it
# needs two ranges that get OR-ed together.
PRESET_RANGES: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "red":    [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (179, 255, 255))],
    "green":  [((40, 70, 70), (80, 255, 255))],
    "blue":   [((100, 150, 50), (130, 255, 255))],
    "yellow": [((20, 100, 100), (35, 255, 255))],
    "orange": [((10, 100, 100), (20, 255, 255))],
    "purple": [((130, 50, 50), (160, 255, 255))],
}

# Words the LLM may emit that mean one of our presets.
COLOR_ALIASES = {
    "crimson": "red", "scarlet": "red", "maroon": "red",
    "lime": "green", "olive": "green",
    "navy": "blue", "cyan": "blue", "teal": "blue", "azure": "blue",
    "gold": "yellow", "amber": "yellow",
    "violet": "purple", "magenta": "purple", "pink": "purple",
}


def normalize_color(word: str | None) -> str | None:
    """Map a free-text color word onto a preset key, or None if unknown."""
    if not word:
        return None
    w = word.lower().strip()
    if w in PRESET_RANGES:
        return w
    if w in COLOR_ALIASES:
        return COLOR_ALIASES[w]
    # "red block" / "the blue plate" -> first recognizable color token
    for token in w.replace("-", " ").split():
        if token in PRESET_RANGES:
            return token
        if token in COLOR_ALIASES:
            return COLOR_ALIASES[token]
    return None


class ColorDetector(BaseDetector):
    NAME = "Color"
    BOX_COLOR = (0, 165, 255)  # orange
    MATCH_HINT = ("match must be one of: "
                  + ", ".join(PRESET_RANGES)
                  + ". Use this detector whenever the object is identified by "
                    "its color (\"the red one\", \"blue plate\").")
    PARAM_SPEC = [
        {"key": "target", "label": "Target color", "type": "combo",
         "options": list(PRESET_RANGES), "default": "red"},
        {"key": "min_area", "label": "Min blob area (px)", "type": "slider",
         "min": 100, "max": 20000, "default": 800},
    ]

    def prepare(self, match: str | None) -> None:
        color = normalize_color(match)
        if color:
            self.params["target"] = color

    def matches(self, det: Detection, match: str | None) -> bool:
        # detect() already thresholded to exactly one color; everything it
        # returns is by definition the requested one.
        return True

    def detect(self, frame: np.ndarray) -> list[Detection]:
        ranges = PRESET_RANGES.get(self.params["target"])
        if not ranges:
            return []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))

        # Clean speckle noise before contour extraction
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        label = self.params["target"]
        detections: list[Detection] = []
        for cnt in contours:
            if cv2.contourArea(cnt) < self.params["min_area"]:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            m = cv2.moments(cnt)
            if m["m00"] == 0:
                continue
            cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
            detections.append(Detection(
                bbox=(x, y, w, h), centroid=(cx, cy), label=label,
            ))
        return detections
