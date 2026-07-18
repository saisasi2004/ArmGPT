"""ArmGPT — Flask entry point.

    python app.py            # http://127.0.0.1:5050

Routes split three ways:
  /api/chat        the command pipeline (LLM -> vision -> TCP)
  /api/preview/*   live camera preview + detector tuning (the old testbed's job)
  /api/sessions/*  chat history in MongoDB
"""
from __future__ import annotations

import logging
import threading
import time

from flask import Flask, Response, jsonify, render_template, request

import config
import detectors
from core.camera import camera
from core.overlay_utils import FpsCounter, draw_detection, draw_fps, encode_jpeg
from services import llm, robot, router, store

logging.basicConfig(
    level=logging.DEBUG if config.DEBUG else logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("armgpt")

app = Flask(__name__)

# Which detector the live preview overlays. None = raw feed (the default, so
# opening the page doesn't load torch).
_preview_mode: str | None = None

PREVIEW_FRAME_INTERVAL = 1 / 15  # seconds between preview frames


# --------------------------------------------------------------------- page
@app.get("/")
def index():
    return render_template("index.html")


# --------------------------------------------------------------------- chat
@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "message is required"}), 400

    session_id = store.ensure_session(data.get("session_id"))
    started = time.perf_counter()
    result = router.handle(message, session_id)
    result["session_id"] = session_id
    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
    log.info("[%s] %r -> %s in %dms", session_id, message, result["status"],
             result["elapsed_ms"])
    return jsonify(result)


# ----------------------------------------------------------------- sessions
@app.get("/api/sessions")
def list_sessions():
    return jsonify({"sessions": store.list_sessions(),
                    "persistent": store.available()})


@app.post("/api/sessions")
def create_session():
    return jsonify({"session_id": store.new_session()})


@app.get("/api/sessions/<session_id>/messages")
def session_messages(session_id: str):
    return jsonify({"messages": store.get_messages(session_id)})


@app.delete("/api/sessions/<session_id>")
def remove_session(session_id: str):
    store.delete_session(session_id)
    return jsonify({"ok": True})


# ------------------------------------------------------------------ preview
@app.get("/api/detectors")
def detector_catalog():
    return jsonify({"detectors": detectors.catalog(),
                    "loaded": detectors.loaded(),
                    "preview": _preview_mode})


@app.post("/api/preview")
def set_preview():
    """Switch the overlay on the live feed. Loading happens here (not in the
    stream loop) so a missing dependency surfaces as a clean error."""
    global _preview_mode
    data = request.get_json(silent=True) or {}
    mode = data.get("mode")

    if mode in (None, "", "none", "raw"):
        _preview_mode = None
        return jsonify({"preview": None, "params": []})
    if mode not in detectors.DETECTOR_KEYS:
        return jsonify({"error": f"unknown detector: {mode}"}), 400

    try:
        detector = detectors.get(mode)
    except detectors.DetectorUnavailable as exc:
        return jsonify({"error": str(exc)}), 503

    _preview_mode = mode
    return jsonify({"preview": mode, "name": detector.NAME,
                    "params": detector.PARAM_SPEC, "values": detector.params})


@app.post("/api/preview/param")
def set_preview_param():
    data = request.get_json(silent=True) or {}
    mode, key, value = data.get("mode"), data.get("key"), data.get("value")
    if mode not in detectors.DETECTOR_KEYS:
        return jsonify({"error": f"unknown detector: {mode}"}), 400
    try:
        detector = detectors.get(mode)
    except detectors.DetectorUnavailable as exc:
        return jsonify({"error": str(exc)}), 503
    if key not in detector.params:
        return jsonify({"error": f"unknown param: {key}"}), 400
    detector.set_param(key, value)
    return jsonify({"ok": True, "values": detector.params})


class _PreviewBroadcaster:
    """One detect + JPEG-encode pass per frame, fanned out to every viewer.

    The old design let only the newest /video_feed survive: browsers routinely
    leave the previous request hanging on reload, and if every orphan kept
    encoding JPEGs (plus a full detector pass) they'd pile up and starve the
    CPU-bound LLM. This keeps that guarantee while allowing many *live* viewers
    — the work is done once here and shared, so N browsers cost the same as one.

    The producer thread runs only while at least one viewer is connected; the
    last one to leave stops it, so an idle page loads no torch and burns no
    cores.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._seq = 0
        self._viewers = 0
        self._thread: threading.Thread | None = None
        self._running = False

    # ----------------------------------------------------------- viewers
    def subscribe(self) -> None:
        with self._cond:
            self._viewers += 1
            if not self._running:
                self._running = True
                self._thread = threading.Thread(target=self._produce,
                                                name="preview", daemon=True)
                self._thread.start()

    def unsubscribe(self) -> None:
        with self._cond:
            self._viewers = max(0, self._viewers - 1)
            if self._viewers == 0:
                self._running = False       # producer sees this and exits
                self._cond.notify_all()

    def wait(self, last_seq: int, timeout: float = 1.0):
        """Block until a JPEG newer than `last_seq` exists. Returns
        (seq, jpeg); jpeg is None when the wait timed out with nothing new."""
        with self._cond:
            if self._seq <= last_seq or self._jpeg is None:
                self._cond.wait(timeout)
            fresh = self._seq > last_seq and self._jpeg is not None
            return self._seq, (self._jpeg if fresh else None)

    # ---------------------------------------------------------- producer
    def _produce(self) -> None:
        fps = FpsCounter()
        last_frame_id = -1
        while self._running:
            frame = camera.latest()
            if frame is None:
                time.sleep(0.05)
                continue

            # Don't re-run detection on a frame the capture thread hasn't
            # refreshed yet — pure waste on the same cores the LLM wants.
            fid = camera.frame_id()
            if fid == last_frame_id:
                time.sleep(0.005)
                continue
            last_frame_id = fid

            mode = _preview_mode
            if mode is not None:
                try:
                    detector = detectors.get(mode)
                    with detectors.lock():
                        found = detector.detect(frame)
                        for det in found:
                            draw_detection(frame, det, detector.BOX_COLOR)
                        detector.annotate_extra(frame)
                except Exception:
                    # Never kill the feed over a detector fault — the raw feed
                    # is more useful than a dead <img>.
                    log.exception("preview detector %s failed", mode)

            draw_fps(frame, fps.tick())
            jpeg = encode_jpeg(frame, quality=75)
            with self._cond:
                self._jpeg = jpeg
                self._seq += 1
                self._cond.notify_all()

            # A preview doesn't need 50fps, and every frame costs a JPEG encode
            # (plus a detector pass when an overlay is on). 15fps looks live and
            # leaves the cores alone.
            time.sleep(PREVIEW_FRAME_INTERVAL)


_preview = _PreviewBroadcaster()


@app.get("/api/camera/devices")
def camera_devices():
    """List switchable camera indices. Probing releases each device, so this
    is safe to call while the preview is streaming the active one."""
    return jsonify({"active": camera.status()["index"],
                    "devices": camera.list_devices()})


@app.post("/api/camera/switch")
def camera_switch():
    data = request.get_json(silent=True) or {}
    try:
        index = int(data.get("index"))
    except (TypeError, ValueError):
        return jsonify({"error": "index must be a number"}), 400
    camera.switch(index)
    # Remember it so the next launch opens the same feed instead of falling
    # back to the config default (which Windows may have reassigned anyway).
    try:
        store.save_setting("camera", {"index": index})
    except Exception:
        pass
    return jsonify({"ok": True, "index": index})


def _mjpeg():
    """Multipart MJPEG for a single viewer. Detection and encoding happen once
    in the shared `_preview` producer, so any number of browsers can watch the
    feed concurrently."""
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    _preview.subscribe()
    try:
        last_seq = 0
        while True:
            last_seq, jpeg = _preview.wait(last_seq)
            if jpeg is None:          # timed out with no new frame — keep waiting
                continue
            yield boundary + jpeg + b"\r\n"
    finally:
        _preview.unsubscribe()


@app.get("/video_feed")
def video_feed():
    return Response(_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# -------------------------------------------------------------------- robot
@app.get("/api/robot/config")
def robot_config():
    return jsonify({**robot.get_settings(), "history": robot.history()})


@app.post("/api/robot/config")
def set_robot_config():
    data = request.get_json(silent=True) or {}
    try:
        current = robot.update_settings(
            mode=data.get("mode"),
            host=data.get("host"),
            port=data.get("port"),
            dry_run=data.get("dry_run"),
            timeout=data.get("timeout"),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(current)


@app.post("/api/robot/test")
def robot_test():
    """Connect and disconnect. Sends no bytes — cannot move the arm."""
    data = request.get_json(silent=True) or {}
    return jsonify(robot.test_connection(host=data.get("host"),
                                         port=data.get("port")))


@app.post("/api/robot/send")
def robot_send():
    """Send a hand-written CSV line.

    This is the one endpoint that can command real motion without the LLM or
    the vision layer in the loop, so it validates the line against the two
    formats the controller understands rather than piping arbitrary bytes at
    it. Honours the dry-run flag exactly like a chat command does.
    """
    data = request.get_json(silent=True) or {}
    line = (data.get("line") or "").strip()
    if not line:
        return jsonify({"error": "line is required"}), 400

    parts = line.split(",")
    verb = parts[0].strip().upper()
    expected = {"PICKPLACE": 5, "LOCATE": 3}
    if verb not in expected:
        return jsonify({"error": f"Unknown command {verb!r}. "
                                 f"Expected PICKPLACE or LOCATE."}), 400
    if len(parts) != expected[verb]:
        return jsonify({"error": f"{verb} needs {expected[verb] - 1} numbers, "
                                 f"got {len(parts) - 1}."}), 400
    try:
        coords = [int(p) for p in parts[1:]]
    except ValueError:
        return jsonify({"error": "Coordinates must be whole numbers."}), 400
    if any(c < 0 for c in coords):
        return jsonify({"error": "Pixel coordinates cannot be negative."}), 400

    return jsonify(robot.send(f"{verb},{','.join(str(c) for c in coords)}\n"))


@app.delete("/api/robot/history")
def robot_clear_history():
    robot.clear_history()
    return jsonify({"ok": True})


# ------------------------------------------------------------------- status
@app.get("/api/status")
def status():
    return jsonify({
        "llm": llm.status(),
        "camera": camera.status(),
        "robot": robot.status(),
        "mongo": {"available": store.available(), "uri": config.MONGO_URI,
                  "db": config.MONGO_DB},
        "safety_check": config.SAFETY_CHECK,
    })


def main() -> None:
    store.init()
    robot.load_persisted()  # after store.init(): it reads from Mongo
    robot.apply_mode()      # start the TCP server now if mode == server

    # A camera index picked in the UI last time wins over the config default,
    # unless an explicit env var is set. Windows may have renumbered the
    # devices since, but the saved index is still a better guess than the
    # hardcoded default, and the UI picker is one click away either way.
    import os
    if "ARMGPT_CAMERA_INDEX" not in os.environ:
        saved = store.load_setting("camera")
        if saved and "index" in saved:
            camera._index = int(saved["index"])
            log.info("restored camera index %s", camera._index)

    camera.start()
    try:
        log.info("LLM model: %s", llm.resolve_model())
    except llm.LLMError as exc:
        log.warning("%s", exc)  # not fatal: the UI shows it and chat 503s

    if config.LLM_WARMUP:
        # Background: loading the model takes ~40s and there is no reason to
        # make the page wait for it. Anything sent before it finishes just
        # queues behind the load, which is what would have happened anyway.
        threading.Thread(target=llm.warmup, name="llm-warmup",
                         daemon=True).start()

    _rcfg = robot.get_settings()
    log.info("ArmGPT on http://%s:%s  (robot %s %s:%s%s)", config.HOST,
             config.PORT, _rcfg["mode"], _rcfg["host"], _rcfg["port"],
             ", DRY RUN" if _rcfg["dry_run"] else "")
    # threaded: the MJPEG stream holds a worker for its whole lifetime, so a
    # single-threaded server would deadlock the moment the page loads.
    # use_reloader off: it would spawn a second process fighting for the camera.
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG,
            threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
