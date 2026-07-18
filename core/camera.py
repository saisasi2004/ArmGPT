"""Threaded webcam capture shared by the preview stream and command execution.

The camera is opened once and kept in a daemon thread for the app's lifetime.
It only captures — it does not run detection. Detection happens in two places:

  * the MJPEG preview stream, which runs whatever detector the UI selected;
  * one-shot command execution, which grabs `latest()` and runs the detector
    the LLM chose.

Both read the same buffer, so a chat command and the preview always agree
about what the camera sees.
"""
from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

import config

log = logging.getLogger("armgpt.camera")

# Backend preference on Windows. MSMF (Media Foundation) is the modern, stable
# path; DSHOW (DirectShow) is the classic fallback; ANY lets OpenCV choose.
# Forcing DSHOW alone is what left the feed stuck on "starting…": on machines
# with an Orbbec/virtual sensor, DSHOW enumeration grabs that phantom device
# and cap.read() hangs or returns nothing. Trying MSMF first sidesteps it.
_BACKENDS: list[tuple[str, int]] = [
    ("MSMF", getattr(cv2, "CAP_MSMF", cv2.CAP_ANY)),
    ("DSHOW", getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)),
    ("ANY", cv2.CAP_ANY),
]

# A frame whose mean pixel value is at or below this is treated as "black" —
# i.e. a phantom/IR/depth device or a webcam that hasn't warmed up yet, not a
# usable image. Real scenes (even dim rooms) sit comfortably above this.
_MIN_BRIGHTNESS = 5.0


class CameraStream:
    """Single-producer latest-frame buffer over cv2.VideoCapture."""

    def __init__(self, index: int | None = None) -> None:
        self._index = config.CAMERA_INDEX if index is None else index
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._error: str | None = None
        self._frame_id = 0
        # Set by switch(); the capture loop picks it up and reopens. None means
        # "no pending switch". -1 is never a valid request.
        self._switch_to: int | None = None

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)

    def _open(self, exact: bool = False
              ) -> tuple[cv2.VideoCapture | None, np.ndarray | None]:
        """Find a camera that actually delivers a *non-black* frame.

        Opening a device is not proof it works, and getting a frame back is
        not proof it's usable — a phantom/virtual sensor (e.g. an Orbbec IR or
        depth stream) can report `isOpened()` and hand back all-zero frames,
        which is exactly what shows up as a live-but-black preview. So each
        candidate is validated by pulling frames until one has real image
        content (mean brightness above a small threshold). Real webcams also
        emit a few dark frames before auto-exposure settles, so we read for up
        to a few seconds before giving up on a device.

        When `exact` is set (a manual switch from the UI), only `self._index`
        is tried — no falling back to a different camera. The user picked that
        device on purpose; silently opening a different one is exactly the
        surprise this whole feature exists to remove. A black frame from the
        chosen device is still accepted so the switch "takes".

        Returns (cap, first_good_frame) or (None, None) if nothing worked.
        A device that opens but only ever yields black frames is remembered as
        a last-resort fallback, so we still show *something* rather than fail
        outright when the only camera present is genuinely dark.
        """
        if exact:
            indices = [self._index]
        else:
            # Preferred index first, then the usual low indices as a fallback.
            indices = []
            for idx in (self._index, 0, 1, 2):
                if idx not in indices:
                    indices.append(idx)

        # Last-resort: a device that opened + produced frames, but all black.
        fallback: tuple[int, str, int] | None = None

        for idx in indices:
            for name, backend in _BACKENDS:
                log.info("probing camera index %s via %s", idx, name)
                cap = cv2.VideoCapture(idx, backend)
                if not cap.isOpened():
                    cap.release()
                    continue

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

                good = None
                got_any = False
                for _ in range(30):          # warm-up: up to ~3s of reads
                    ok, f = cap.read()
                    if ok and f is not None:
                        got_any = True
                        if float(f.mean()) > _MIN_BRIGHTNESS:
                            good = f
                            break
                    time.sleep(0.1)

                if good is not None:
                    h, w = good.shape[:2]
                    log.info("camera opened: index %s via %s (%dx%d)",
                             idx, name, w, h)
                    self._index = idx
                    return cap, good

                if got_any and fallback is None:
                    log.warning("index %s via %s only gave black frames; "
                                "keeping as fallback", idx, name)
                    fallback = (idx, name, backend)
                else:
                    log.warning("index %s via %s opened but gave no frame",
                                idx, name)
                cap.release()

        # Nothing produced a real image. If some device at least streamed
        # (black) frames, reopen it so the app runs instead of erroring out.
        if fallback is not None:
            idx, name, backend = fallback
            log.warning("no non-black camera found; falling back to index %s "
                        "via %s (frames may be dark)", idx, name)
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
                for _ in range(10):
                    ok, f = cap.read()
                    if ok and f is not None:
                        self._index = idx
                        return cap, f
                    time.sleep(0.1)
            cap.release()

        return None, None

    def _loop(self) -> None:
        cap, frame = self._open()
        if cap is None:
            self._error = (
                f"No working camera found (tried index {self._index} and 0–2 "
                f"on MSMF/DSHOW/ANY). It may be unplugged or held by another app."
            )
            log.error(self._error)
            self._running = False
            return

        self._error = None
        with self._lock:                     # publish the validation frame
            self._frame = frame
            self._frame_id += 1

        while self._running:
            # Honour a pending switch before reading the next frame.
            with self._lock:
                target = self._switch_to
                self._switch_to = None
            if target is not None and target != self._index:
                log.info("switching camera %s -> %s", self._index, target)
                cap.release()
                self._index = target
                cap, frame = self._open(exact=True)
                if cap is None:
                    self._error = (f"Camera index {target} could not be opened "
                                   f"(unplugged, or held by another app).")
                    log.error(self._error)
                    self._running = False
                    return
                self._error = None
                with self._lock:
                    self._frame = frame
                    self._frame_id += 1
                continue

            ok, frame = cap.read()
            if not ok:
                self._error = "Camera read failed — stream stopped."
                log.error(self._error)
                break
            with self._lock:
                self._frame = frame
                self._frame_id += 1
        cap.release()
        self._running = False

    # ----------------------------------------------------------------- reads
    def latest(self) -> np.ndarray | None:
        """A private copy of the most recent frame, safe to annotate."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def frame_id(self) -> int:
        with self._lock:
            return self._frame_id

    def wait_for_frame(self, timeout: float = 5.0) -> np.ndarray | None:
        """Block until a frame is available. Covers the startup window where
        a command arrives before the capture thread has produced anything."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            frame = self.latest()
            if frame is not None:
                return frame
            if not self._running:
                return None
            time.sleep(0.02)
        return None

    # --------------------------------------------------------------- switching
    def switch(self, index: int) -> None:
        """Request a reopen on a different index. Applied by the capture loop
        before its next read, so callers never touch cv2 off the camera thread
        (doing so from a Flask worker is how you get two handles fighting over
        one device)."""
        with self._lock:
            self._switch_to = int(index)

    def list_devices(self, max_index: int = 6) -> list[dict]:
        """Enumerate camera indices that open and stream.

        The live device is reported without re-probing — on Windows a second
        handle to the in-use camera often fails, and a false "unavailable" on
        the camera you're currently watching would be absurd. Every other index
        is opened briefly, sampled, and released. `dark` flags a device that
        streamed only black frames (a depth/IR sensor, usually).
        """
        active = self._index
        out: list[dict] = []
        for idx in range(max_index):
            if idx == active:
                out.append({"index": idx, "active": True, "dark": False})
                continue
            opened = False
            dark = True
            for _name, backend in _BACKENDS:
                cap = cv2.VideoCapture(idx, backend)
                if not cap.isOpened():
                    cap.release()
                    continue
                opened = True
                for _ in range(3):
                    ok, f = cap.read()
                    if ok and f is not None and float(f.mean()) > _MIN_BRIGHTNESS:
                        dark = False
                        break
                cap.release()
                if not dark:
                    break
            if opened:
                out.append({"index": idx, "active": False, "dark": dark})
        return out

    # ---------------------------------------------------------------- status
    @property
    def error(self) -> str | None:
        return self._error

    @property
    def running(self) -> bool:
        return self._running

    def status(self) -> dict:
        return {
            "running": self._running,
            "error": self._error,
            "index": self._index,
            "has_frame": self.latest() is not None,
        }


# One instance for the whole process; app.py starts it.
camera = CameraStream()
