"""Shared drawing helpers: bounding box, centroid marker, labels, FPS counter.

Implemented once here and reused by the live preview stream and by the
single-shot snapshots attached to chat replies, so both look identical.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from core.base_detector import Detection

FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR = (240, 240, 240)
HIGHLIGHT = (0, 255, 170)  # BGR: the "this is the one I picked" accent


class FpsCounter:
    """Exponentially smoothed frames-per-second counter."""

    def __init__(self, smoothing: float = 0.9) -> None:
        self._smoothing = smoothing
        self._last = time.perf_counter()
        self._fps = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        if dt > 0:
            instant = 1.0 / dt
            self._fps = self._smoothing * self._fps + (1 - self._smoothing) * instant
        return self._fps


def _draw_label(frame: np.ndarray, text: str, x: int, y: int,
                bg_color: tuple[int, int, int]) -> None:
    """Text with a filled background bar so it stays readable on any frame."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, 0.5, 1)
    y = max(y, th + baseline)
    cv2.rectangle(frame, (x, y - th - baseline), (x + tw + 4, y + 2), bg_color, -1)
    cv2.putText(frame, text, (x + 2, y - baseline // 2), FONT, 0.5, (10, 10, 10), 1,
                cv2.LINE_AA)


def draw_detection(frame: np.ndarray, det: Detection,
                   color: tuple[int, int, int], highlight: bool = False) -> None:
    """Draw one detection: bbox, centroid dot + coords, label (+confidence).

    `highlight` thickens the box and switches to the accent color — used to
    mark the detection actually chosen as a pick or place point.
    """
    if highlight:
        color = HIGHLIGHT
    x, y, w, h = det.bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3 if highlight else 2)

    # Centroid: filled dot with pixel coordinates printed next to it
    cx, cy = det.centroid
    cv2.circle(frame, (cx, cy), 4, color, -1)
    if highlight:
        # Crosshair makes the exact pixel being sent to the robot unambiguous
        cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 22, 1)
    cv2.putText(frame, f"({cx}, {cy})", (cx + 8, cy + 4), FONT, 0.45, TEXT_COLOR, 1,
                cv2.LINE_AA)

    # Label above the box, with confidence when the detector provides one
    text = det.label
    if det.confidence is not None:
        text += f" {det.confidence:.2f}"
    _draw_label(frame, text, x, y - 4, color)

    # Optional landmark skeleton (e.g. MediaPipe hands)
    for i, j in det.connections:
        cv2.line(frame, det.points[i], det.points[j], color, 1, cv2.LINE_AA)
    for px, py in det.points:
        cv2.circle(frame, (px, py), 2, TEXT_COLOR, -1)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """Running FPS counter in the top-left corner."""
    _draw_label(frame, f"FPS: {fps:.1f}", 8, 24, (60, 60, 60))


def draw_banner(frame: np.ndarray, text: str) -> None:
    """Caption strip across the bottom, e.g. the command being visualized."""
    h = frame.shape[0]
    cv2.rectangle(frame, (0, h - 26), (frame.shape[1], h), (18, 18, 20), -1)
    cv2.putText(frame, text, (8, h - 8), FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)


def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()
