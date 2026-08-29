"""Detector registry.

Modules are imported on first use so process startup doesn't pay for torch or
mediapipe until a command actually needs them. Instances are cached and shared
process-wide, guarded by a lock because Flask serves requests on many threads
and the detectors are stateful (motion keeps a previous frame, YOLO holds a
model).
"""
from __future__ import annotations

import importlib
import threading

from core.base_detector import BaseDetector

# key -> (module path, class name)
_REGISTRY: dict[str, tuple[str, str]] = {
    "objects":  ("detectors.object_detector",   "ObjectDetector"),
    "color":    ("detectors.color_detector",    "ColorDetector"),
    "shapes":   ("detectors.shape_detector",    "ShapeDetector"),
    "markers":  ("detectors.marker_detector",   "MarkerDetector"),
    "motion":   ("detectors.motion_detector",   "MotionDetector"),
    "presence": ("detectors.presence_detector", "PresenceDetector"),
}

DETECTOR_KEYS = list(_REGISTRY)

_instances: dict[str, BaseDetector] = {}
_lock = threading.RLock()


class DetectorUnavailable(RuntimeError):
    """Raised when a detector's optional dependency isn't installed."""


def get(key: str) -> BaseDetector:
    """Return the shared instance for `key`, importing it on first use."""
    if key not in _REGISTRY:
        raise KeyError(f"Unknown detector: {key}")
    with _lock:
        if key not in _instances:
            module_path, class_name = _REGISTRY[key]
            try:
                module = importlib.import_module(module_path)
                _instances[key] = getattr(module, class_name)()
            except Exception as exc:
                raise DetectorUnavailable(
                    f"'{key}' could not be loaded: {exc}. "
                    f"Its dependency is probably not installed."
                ) from exc
        return _instances[key]


def lock() -> threading.RLock:
    """Held around detect() calls so the preview stream and a chat command
    never run the same stateful detector concurrently."""
    return _lock


def loaded() -> list[str]:
    with _lock:
        return sorted(_instances)


def catalog() -> list[dict]:
    """Metadata for every detector WITHOUT importing it.

    Used to build the LLM's system prompt and the UI's mode list, both of
    which must work before torch has ever been loaded.
    """
    return [{"key": key, **_STATIC_META[key]} for key in _REGISTRY]


# Kept in sync with each detector's NAME / MATCH_HINT. Duplicated here on
# purpose: reading them off the classes would import torch and mediapipe just
# to render the mode list.
_STATIC_META: dict[str, dict] = {
    "objects": {
        "name": "Objects",
        "hint": "YOLOv8n, 80 COCO classes. match = a COCO class name "
                "(cup, bottle, bowl, book, cell phone, scissors, apple, ...). "
                "For everyday objects named by kind.",
    },
    "color": {
        "name": "Color",
        "hint": "HSV thresholding. match = one of: red, green, blue, yellow, "
                "orange, purple. For objects identified by color - this is the "
                "usual choice for pick-and-place.",
    },
    "shapes": {
        "name": "Shapes",
        "hint": "Contour approximation. match = one of: triangle, square, "
                "rectangle, pentagon, hexagon, circle. For objects identified "
                "by geometry.",
    },
    "markers": {
        "name": "Markers",
        "hint": "ArUco tags + QR codes. match = a marker id as a bare number "
                "(\"3\") or QR payload text. Most reliable when available.",
    },
    "motion": {
        "name": "Motion",
        "hint": "Frame differencing / optical flow. No match value. For "
                "\"is anything moving?\".",
    },
    "presence": {
        "name": "Presence",
        "hint": "MediaPipe hands + faces. match = \"hand\" or \"face\", or omit "
                "for both. For \"is anyone there?\".",
    },
}
