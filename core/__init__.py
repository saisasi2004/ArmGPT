"""Core runtime helpers (camera, detection primitives, drawing).

Importing this package configures OpenCV's logging *before* cv2 is loaded
anywhere else in the process. Probing camera indices on Windows makes the
videoio backends print a wall of text straight to stderr:

    [ WARN:0] global cap.cpp VIDEOIO(MSMF): backend is generally available
              but can't be used to capture by index
    [ERROR:0] global obsensor_uvc_stream_channel.cpp Camera index out of range

None of that is actionable - it is the normal sound of asking "is there a
camera on index 4?" - but it drowns the application log and reads like a
crash. OpenCV only honours these settings before its videoio module
initialises, which is why they live in the package __init__ rather than in
camera.py.

Set ARMGPT_CV_VERBOSE=1 to get the raw OpenCV chatter back when debugging a
capture problem.
"""
from __future__ import annotations

import os

_VERBOSE = os.environ.get("ARMGPT_CV_VERBOSE", "").strip().lower() in (
    "1", "true", "yes", "on")

if not _VERBOSE:
    # Read by OpenCV at import time; must be set before `import cv2`.
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")

import cv2  # noqa: E402  (deliberately after the env vars above)

if not _VERBOSE:
    # The env vars cover most of it; this silences the videoio backend probes
    # that write through OpenCV's own logger after initialisation.
    try:
        cv2.setLogLevel(0)  # 0 == LOG_LEVEL_SILENT
    except Exception:       # very old builds have no setLogLevel
        pass
