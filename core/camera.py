"""Threaded webcam capture shared by the preview stream and command execution.

The camera is opened once and kept in a daemon thread for the app's lifetime.
It only captures - it does not run detection. Detection happens in two places:

  * the MJPEG preview stream, which runs whatever detector the UI selected;
  * one-shot command execution, which grabs `latest()` and runs the detector
    the LLM chose.

Both read the same buffer, so a chat command and the preview always agree
about what the camera sees.

Everything that touches cv2.VideoCapture runs on the capture thread. That is
the single rule this module is built around, and it is not stylistic: a webcam
is an exclusive resource, and a second handle opened from a Flask worker (to
enumerate devices, say) can invalidate the stream the capture thread is in the
middle of reading. On Windows that shows up as

    OnReadSample() is called with error status: -1072873821  (device invalidated)
    CvCapture_MSMF::grabFrame videoio(MSMF): can't grab frame

and the feed dies. So device scans and source switches are *requests*: the
capture loop picks them up between frames, when it can safely release the
device first. See `list_devices()` and `switch()`.
"""
from __future__ import annotations

import logging
import platform
import threading
import time

import core  # noqa: F401  - configures OpenCV logging before cv2 loads
import cv2
import numpy as np

import config

log = logging.getLogger("armgpt.camera")

_IS_WINDOWS = platform.system() == "Windows"


def _backends() -> list[tuple[str, int]]:
    """Capture backends to try, best first.

    Windows: MSMF (Media Foundation) is the modern, stable path; DSHOW
    (DirectShow) is the classic fallback. CAP_ANY is deliberately NOT in the
    list - on Windows it routes through OpenCV's obsensor probe, which prints
    `getStreamChannelGroup Camera index out of range` for every index that
    isn't an Orbbec depth sensor. It has never opened a device that MSMF and
    DSHOW both refused, so it is pure log noise.

    Elsewhere CAP_ANY resolves to V4L2 (Linux) / AVFoundation (macOS), so one
    entry is enough.
    """
    if _IS_WINDOWS:
        return [("MSMF", getattr(cv2, "CAP_MSMF", cv2.CAP_ANY)),
                ("DSHOW", getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY))]
    return [("ANY", cv2.CAP_ANY)]


# A frame whose mean pixel value is at or below this is treated as "black" -
# i.e. a phantom/IR/depth device or a webcam that hasn't warmed up yet, not a
# usable image. Real scenes (even dim rooms) sit comfortably above this.
_MIN_BRIGHTNESS = 5.0

# Consecutive failed reads before we stop hoping and reopen the device. A
# webcam drops the occasional frame; five in a row means it's gone.
_READ_FAIL_LIMIT = 5

# Reconnect backoff, seconds. The last value repeats forever - a camera that
# was unplugged should be picked up when it comes back, without a restart and
# without a hot loop hammering the driver in the meantime.
_RECONNECT_BACKOFF = (1.0, 2.0, 4.0, 8.0, 15.0)

# How long a cached device list stays fresh. Scanning releases and reopens the
# live camera, so it is not something to do on every poll.
_DEVICE_CACHE_TTL = 60.0


class CameraStream:
    """Single-producer latest-frame buffer over cv2.VideoCapture."""

    def __init__(self, index: int | None = None) -> None:
        self._index = config.CAMERA_INDEX if index is None else index
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._error: str | None = None
        self._reconnecting = False
        # True when the open device only ever streamed black frames. Tracked
        # separately from _error because it is not a failure - we have a live
        # stream - but reporting it as plain "running" is a lie the operator
        # pays for: the sidebar goes green while the panel shows a black
        # rectangle, and the arm is told to look at nothing.
        self._dark = False
        self._frame_id = 0
        # Set by switch(); the capture loop picks it up and reopens. None means
        # "no pending switch".
        self._switch_to: int | None = None
        # Device enumeration, also serviced by the capture loop.
        self._scan_requested = False
        self._scan_done = threading.Event()
        self._devices: list[dict] = []
        self._devices_ts = 0.0
        self._max_scan_index = 6

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
            self._thread = None

    def set_index(self, index: int) -> None:
        """Set the startup index before start(). After start(), use switch()."""
        self._index = int(index)

    # ------------------------------------------------------------- capturing
    def _configure(self, cap: cv2.VideoCapture) -> None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        # One frame of driver buffering. Without it OpenCV can hand back a
        # frame queued seconds ago, and the arm gets told to pick up an object
        # that has already moved. Not every backend honours it; harmless if not.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    def _try_open(self, idx: int, backend: int
                  ) -> tuple[cv2.VideoCapture | None, np.ndarray | None, bool]:
        """Open one (index, backend) pair and grade what comes out.

        Returns (cap, frame, is_dark). `cap` is None when the device did not
        open or never produced a frame. `is_dark` marks a device that streams
        only black frames - a depth/IR sensor, usually.
        """
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            return None, None, False

        self._configure(cap)
        last: np.ndarray | None = None
        # Real webcams emit a few dark frames while auto-exposure settles, so
        # read for up to ~3s before writing a device off as dark.
        for _ in range(30):
            ok, frame = cap.read()
            if ok and frame is not None:
                last = frame
                if float(frame.mean()) > _MIN_BRIGHTNESS:
                    return cap, frame, False
            time.sleep(0.1)

        if last is None:
            cap.release()
            return None, None, False
        return cap, last, True

    def _open(self, exact: bool = False
              ) -> tuple[cv2.VideoCapture | None, np.ndarray | None]:
        """Find a camera that actually delivers a usable frame.

        Opening a device is not proof it works, and getting a frame back is not
        proof it's usable - a phantom/virtual sensor (e.g. an Orbbec IR or
        depth stream) can report `isOpened()` and hand back all-zero frames,
        which is what shows up as a live-but-black preview. So each candidate
        is validated by pulling frames until one has real image content.

        When `exact` is set (a manual switch from the UI, or a reconnect to the
        device already in use), only `self._index` is tried - no falling back
        to a different camera. The user picked that device on purpose; silently
        opening a different one is exactly the surprise this feature exists to
        remove. A black frame from the chosen device is still accepted so the
        switch "takes".

        Returns (cap, first_good_frame), or (None, None) if nothing worked.
        """
        if exact:
            indices = [self._index]
        else:
            indices = []
            for idx in (self._index, 0, 1, 2):
                if idx not in indices:
                    indices.append(idx)

        # Last resort: a device that opened + streamed, but all black.
        fallback: tuple[cv2.VideoCapture, np.ndarray, int, str] | None = None

        for idx in indices:
            for name, backend in _backends():
                log.debug("probing camera index %s via %s", idx, name)
                cap, frame, dark = self._try_open(idx, backend)
                if cap is None:
                    continue
                if not dark:
                    h, w = frame.shape[:2]
                    log.info("camera %s open via %s (%dx%d)", idx, name, w, h)
                    self._index = idx
                    self._dark = False
                    if fallback is not None:
                        fallback[0].release()
                    return cap, frame
                if fallback is None:
                    log.debug("index %s via %s streams only black frames; "
                              "keeping as fallback", idx, name)
                    fallback = (cap, frame, idx, name)
                else:
                    cap.release()

        if fallback is not None:
            cap, frame, idx, name = fallback
            log.warning("no camera produced a lit image; using index %s via %s "
                        "(frames are black - depth/IR sensor, lens cap, "
                        "privacy shutter, or the device is held by another "
                        "app)", idx, name)
            self._index = idx
            self._dark = True
            return cap, frame

        return None, None

    def _publish(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame
            self._frame_id += 1

    def _no_camera_error(self) -> str:
        tried = "/".join(name for name, _ in _backends())
        return (f"No working camera found (tried index {self._index} and 0-2 "
                f"on {tried}). It may be unplugged, blocked by Windows camera "
                f"privacy settings, or held by another app (Teams, Zoom, the "
                f"Camera app). Close that app, then press rescan.")

    def _loop(self) -> None:
        cap: cv2.VideoCapture | None = None
        attempt = 0            # 0 = first open, >0 = reconnect attempt number
        read_failures = 0

        while self._running:
            # --- (re)connect ------------------------------------------------
            if cap is None:
                # A reconnect targets the device we were already on; only the
                # very first open is allowed to go hunting for another index.
                cap, frame = self._open(exact=attempt > 0)
                if cap is None:
                    self._error = self._no_camera_error()
                    self._reconnecting = attempt > 0
                    if attempt == 0:
                        log.error("%s", self._error)
                    elif attempt == 1:
                        log.warning("retrying every %.0fs until a camera "
                                    "appears", _RECONNECT_BACKOFF[-1])
                    delay = _RECONNECT_BACKOFF[min(attempt,
                                                   len(_RECONNECT_BACKOFF) - 1)]
                    attempt += 1
                    # Sleep in slices so stop() and switch() stay responsive.
                    waited = 0.0
                    while (self._running and waited < delay
                           and self._switch_to is None):
                        time.sleep(0.1)
                        waited += 0.1
                    continue

                if attempt:
                    log.info("camera reconnected on index %s", self._index)
                attempt = 0
                read_failures = 0
                self._error = None
                self._reconnecting = False
                self._publish(frame)

            # --- pending source switch --------------------------------------
            with self._lock:
                target = self._switch_to
                self._switch_to = None
            if target is not None:
                if target != self._index:
                    log.info("switching camera %s -> %s", self._index, target)
                cap.release()
                cap = None
                self._index = target
                new_cap, new_frame = self._open(exact=True)
                if new_cap is None:
                    # Don't kill the thread: fall back to whatever worked
                    # before, so one bad pick in a dropdown doesn't leave the
                    # operator with a dead feed.
                    self._error = (f"Camera {target} could not be opened "
                                   f"(unplugged, or held by another app).")
                    log.error("%s", self._error)
                    cap, new_frame = self._open()
                    if cap is None:
                        attempt = 1
                        continue
                    log.info("fell back to camera index %s", self._index)
                else:
                    cap = new_cap
                    self._error = None
                    self._reconnecting = False
                read_failures = 0
                self._publish(new_frame)
                continue

            # --- pending device scan ----------------------------------------
            if self._scan_requested:
                cap.release()
                cap = None
                try:
                    self._scan(self._index)
                finally:
                    self._scan_requested = False
                    self._scan_done.set()
                continue

            # --- normal read ------------------------------------------------
            ok, frame = cap.read()
            if not ok or frame is None:
                read_failures += 1
                if read_failures < _READ_FAIL_LIMIT:
                    time.sleep(0.05)
                    continue
                self._error = "Camera stopped delivering frames - reconnecting."
                self._reconnecting = True
                log.warning("camera %s stopped delivering frames; reopening",
                            self._index)
                cap.release()
                cap = None
                attempt = 1
                continue

            read_failures = 0
            self._publish(frame)

        if cap is not None:
            cap.release()
        self._running = False

    # -------------------------------------------------------------- scanning
    def _scan(self, active: int) -> None:
        """Probe every index and cache the result. CAPTURE THREAD ONLY.

        The caller has already released the live device, so `active` is probed
        like any other index - which is the point: a scan that skips the camera
        you are watching cannot tell you it has gone away.
        """
        started = time.perf_counter()
        found: list[dict] = []
        for idx in range(self._max_scan_index):
            opened = False
            dark = True
            for _name, backend in _backends():
                cap, _frame, is_dark = self._try_open(idx, backend)
                if cap is None:
                    continue
                cap.release()
                opened = True
                if not is_dark:
                    dark = False
                    break
            if opened:
                found.append({"index": idx, "active": idx == active,
                              "dark": dark})
        with self._lock:
            self._devices = found
            self._devices_ts = time.monotonic()
        log.info("camera scan: %s (%.1fs)",
                 ", ".join(f"{d['index']}{' dark' if d['dark'] else ''}"
                           for d in found) or "no devices",
                 time.perf_counter() - started)

    def list_devices(self, refresh: bool = False,
                     timeout: float = 25.0) -> list[dict]:
        """Camera indices that open and stream.

        Served from cache unless `refresh` is set or the cache has expired,
        because a scan briefly takes the live camera away. When the capture
        thread is running the scan is delegated to it; otherwise it runs here.
        """
        with self._lock:
            cached = list(self._devices)
            fresh = (time.monotonic() - self._devices_ts) < _DEVICE_CACHE_TTL
        if cached and fresh and not refresh:
            # Keep the "active" flag honest even on a cache hit - the source
            # may have been switched since the scan.
            return [{**d, "active": d["index"] == self._index} for d in cached]

        if not self._running:
            self._scan(self._index)
        else:
            self._scan_done.clear()
            self._scan_requested = True
            if not self._scan_done.wait(timeout):
                log.warning("camera scan timed out after %.0fs", timeout)
                self._scan_requested = False
                return cached
        with self._lock:
            return list(self._devices)

    # ----------------------------------------------------------------- reads
    def latest(self) -> np.ndarray | None:
        """A private copy of the most recent frame, safe to annotate."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def frame_id(self) -> int:
        with self._lock:
            return self._frame_id

    def wait_for_frame(self, timeout: float = 5.0) -> np.ndarray | None:
        """Block until a fresh frame is available.

        Covers the startup window where a command arrives before the capture
        thread has produced anything, and the gap while a dropped camera is
        reconnecting - during which the buffer still holds a stale frame, and
        acting on an old view of the table is worse than admitting we can't
        see.
        """
        deadline = time.monotonic() + timeout
        start_id = self.frame_id()
        while time.monotonic() < deadline:
            frame = self.latest()
            if frame is not None and not (self._reconnecting
                                          and self.frame_id() == start_id):
                return frame
            if not self._running:
                return None
            time.sleep(0.02)
        return None if self._reconnecting else self.latest()

    # ------------------------------------------------------------- switching
    def switch(self, index: int) -> None:
        """Request a reopen on a different index. Applied by the capture loop
        before its next read, so callers never touch cv2 off the camera
        thread."""
        with self._lock:
            self._switch_to = int(index)

    def retry(self) -> None:
        """Force a reconnect attempt now - what the UI's retry button calls.

        Restarts the thread if it stopped, otherwise asks the running loop to
        reopen the current index.
        """
        if not self._running:
            self._error = None
            self.start()
        else:
            self.switch(self._index)

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
            "reconnecting": self._reconnecting,
            "dark": self._dark,
            "index": self._index,
            "has_frame": self.latest() is not None,
        }


# One instance for the whole process; app.py starts it.
camera = CameraStream()
