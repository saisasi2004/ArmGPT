"""Abstract base class and shared Detection dataclass for all detectors.

Ported from the vision_suite testbed, with the Qt-era standalone runner
dropped and a `to_dict()` added so detections cross the JSON boundary to the
browser and the robot layer unchanged.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    """A single detected region of interest.

    bbox        -- (x, y, w, h) in pixel coordinates
    centroid    -- (cx, cy) center of the region; this is what the robot gets
    label       -- human-readable name (class / color / shape / marker id)
    confidence  -- 0..1 score, or None when the method has no notion of one
    points      -- optional landmark list [(x, y), ...] (e.g. hand skeleton)
    connections -- optional index pairs into `points` to draw as skeleton lines
    """
    bbox: tuple[int, int, int, int]
    centroid: tuple[int, int]
    label: str
    confidence: float | None = None
    points: list[tuple[int, int]] = field(default_factory=list)
    connections: list[tuple[int, int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-safe view. Landmarks are omitted: the UI never needs the 21
        hand points, and they bloat every chat response that touches them."""
        return {
            "bbox": list(self.bbox),
            "centroid": list(self.centroid),
            "label": self.label,
            "confidence": self.confidence,
        }

    def area(self) -> int:
        return self.bbox[2] * self.bbox[3]


class BaseDetector(ABC):
    """Common interface every detection mode implements.

    Subclasses declare:
      NAME        -- display name shown in the UI
      BOX_COLOR   -- BGR tuple used for this detector's overlays
      PARAM_SPEC  -- parameter descriptors the frontend renders as controls:
          {"key": str, "label": str, "type": "slider",
           "min": int, "max": int, "default": int}
          {"key": str, "label": str, "type": "combo",
           "options": [str, ...], "default": str}
          {"key": str, "label": str, "type": "check", "default": bool}
      MATCH_HINT  -- one line telling the LLM what `match` means for this
                     detector, and what values are legal.
    """

    NAME: str = "Base"
    BOX_COLOR: tuple[int, int, int] = (0, 255, 0)
    PARAM_SPEC: list[dict] = []
    MATCH_HINT: str = ""

    def __init__(self) -> None:
        self.params: dict = {spec["key"]: spec["default"] for spec in self.PARAM_SPEC}

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a BGR frame and return all detections."""

    def set_param(self, key: str, value) -> None:
        self.params[key] = value

    def annotate_extra(self, frame: np.ndarray) -> None:
        """Optional hook to draw detector-specific extras (e.g. a flow field)
        onto the already-annotated frame. Default: nothing."""

    def prepare(self, match: str | None) -> None:
        """Configure this detector to look for `match` before a one-shot run.

        Detectors whose search target is a *parameter* (color) override this to
        push `match` into self.params. Detectors that always return everything
        and get filtered afterwards (objects, shapes) can ignore it.
        """

    def matches(self, det: Detection, match: str | None) -> bool:
        """Does this detection satisfy the requested `match`? Default is a
        case-insensitive substring test against the label."""
        if not match:
            return True
        return match.lower().strip() in det.label.lower()
