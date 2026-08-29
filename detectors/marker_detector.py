"""Fiducial marker + QR/barcode detection.

The most reliable localization source in the cell - an ArUco tag beats natural
object detection every time, so "pick up marker 3" is the highest-confidence
command this system can serve.
"""
from __future__ import annotations

import cv2
import numpy as np

from core.base_detector import BaseDetector, Detection

try:
    from pyzbar import pyzbar
    HAS_PYZBAR = True
except Exception:  # pyzbar needs the zbar shared library; treat as optional
    HAS_PYZBAR = False

ARUCO_DICTS = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "5x5_100": cv2.aruco.DICT_5X5_100,
    "6x6_250": cv2.aruco.DICT_6X6_250,
    "original": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


def _quad_to_detection(corners: np.ndarray, label: str) -> Detection:
    """Convert a 4-corner polygon into an axis-aligned Detection."""
    xs, ys = corners[:, 0], corners[:, 1]
    x, y = int(xs.min()), int(ys.min())
    w, h = int(xs.max()) - x, int(ys.max()) - y
    return Detection(
        bbox=(x, y, w, h),
        centroid=(int(xs.mean()), int(ys.mean())),
        label=label,
        points=[(int(px), int(py)) for px, py in corners],
        connections=[(0, 1), (1, 2), (2, 3), (3, 0)],
    )


class MarkerDetector(BaseDetector):
    NAME = "Markers"
    BOX_COLOR = (255, 0, 255)  # magenta
    MATCH_HINT = ("match is an ArUco marker id as a bare number (\"3\") or QR "
                  "payload text. Use for \"marker 3\", \"the tag\", \"the QR code\".")
    PARAM_SPEC = [
        {"key": "dict", "label": "ArUco dictionary", "type": "combo",
         "options": list(ARUCO_DICTS), "default": "4x4_50"},
        {"key": "qr", "label": "Decode QR codes", "type": "check", "default": True},
        {"key": "pyzbar", "label": "Use pyzbar fallback", "type": "check",
         "default": HAS_PYZBAR},
    ]

    def __init__(self) -> None:
        super().__init__()
        self._dict_name: str | None = None
        self._aruco: cv2.aruco.ArucoDetector | None = None
        self._qr = cv2.QRCodeDetector()

    def matches(self, det: Detection, match: str | None) -> bool:
        if not match:
            return True
        m = match.lower().strip()
        # "3" must match "ArUco 3" but not "ArUco 13" - compare the id token.
        if m.isdigit() and det.label.lower().startswith("aruco"):
            return det.label.split()[-1] == m
        return m in det.label.lower()

    def _aruco_detector(self) -> cv2.aruco.ArucoDetector:
        if self._dict_name != self.params["dict"]:
            self._dict_name = self.params["dict"]
            dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[self._dict_name])
            self._aruco = cv2.aruco.ArucoDetector(dictionary,
                                                  cv2.aruco.DetectorParameters())
        return self._aruco

    def detect(self, frame: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections: list[Detection] = []

        corners_list, ids, _ = self._aruco_detector().detectMarkers(gray)
        if ids is not None:
            for corners, marker_id in zip(corners_list, ids.flatten()):
                detections.append(_quad_to_detection(corners.reshape(4, 2),
                                                     f"ArUco {marker_id}"))

        if self.params["qr"]:
            detections.extend(self._detect_qr(frame))

        if self.params["pyzbar"] and HAS_PYZBAR:
            detections.extend(self._detect_pyzbar(frame))

        return detections

    def _detect_qr(self, frame: np.ndarray) -> list[Detection]:
        try:
            ok, payloads, points, _ = self._qr.detectAndDecodeMulti(frame)
        except cv2.error:
            return []
        if not ok or points is None:
            return []
        out = []
        for payload, quad in zip(payloads, points):
            label = f"QR: {payload}" if payload else "QR (unreadable)"
            out.append(_quad_to_detection(quad.reshape(4, 2), label))
        return out

    def _detect_pyzbar(self, frame: np.ndarray) -> list[Detection]:
        out = []
        for symbol in pyzbar.decode(frame):
            x, y, w, h = (symbol.rect.left, symbol.rect.top,
                          symbol.rect.width, symbol.rect.height)
            out.append(Detection(
                bbox=(x, y, w, h),
                centroid=(x + w // 2, y + h // 2),
                label=f"{symbol.type}: {symbol.data.decode('utf-8', 'replace')}",
            ))
        return out
