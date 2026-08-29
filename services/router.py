"""Intent -> detector -> pixel coordinates -> robot.

This is the stage that decides whether a command actually reaches the arm. It
refuses in three cases, and each refusal is a normal chat reply rather than an
error:

  not_found  - the detector saw nothing matching. Nothing to send.
  ambiguous  - several things matched. Ask which one instead of guessing;
               silently picking one is how you get the arm grabbing the wrong
               object.
  blocked    - a hand is in frame. See the safety note on the interlock below.
"""
from __future__ import annotations

import base64
import logging
import time

import numpy as np

import config
import detectors
from core.base_detector import BaseDetector, Detection
from core.camera import camera
from core.overlay_utils import draw_banner, draw_detection, encode_jpeg
from services import llm, robot, store

log = logging.getLogger(__name__)


class SlotResult:
    """What one detector run found for one slot of a command."""

    def __init__(self, slot: dict, candidates: list[Detection],
                 detector: BaseDetector | None, error: str | None = None) -> None:
        self.slot = slot
        self.candidates = candidates
        self.detector = detector
        self.error = error

    @property
    def chosen(self) -> Detection | None:
        return self.candidates[0] if len(self.candidates) == 1 else None


def _run_detector(key: str, match: str | None,
                  frame: np.ndarray) -> tuple[list[Detection], BaseDetector]:
    detector = detectors.get(key)
    with detectors.lock():
        detector.prepare(match)
        # Motion is differential - one frame gives it nothing to compare
        # against, so prime it with the current frame before the real read.
        if key == "motion":
            detector.detect(frame)
            time.sleep(0.08)
            frame = camera.latest() if camera.latest() is not None else frame
        found = detector.detect(frame)
    kept = [d for d in found if detector.matches(d, match)]
    # Largest first: for a pick, the biggest blob is the most likely intent
    # and the least likely to be sensor noise.
    kept.sort(key=lambda d: d.area(), reverse=True)
    return kept, detector


def _resolve_slot(slot: dict, frame: np.ndarray) -> SlotResult:
    try:
        candidates, detector = _run_detector(slot["detector"], slot.get("match"),
                                             frame)
        return SlotResult(slot, candidates, detector)
    except detectors.DetectorUnavailable as exc:
        return SlotResult(slot, [], None, error=str(exc))
    except Exception as exc:
        log.exception("detector %s failed", slot["detector"])
        return SlotResult(slot, [], None,
                          error=f"{slot['detector']} failed: {exc}")


def _safety_block(frame: np.ndarray) -> str | None:
    """Refuse arm motion while a hand is visible.

    NOT a safety-rated interlock - MediaPipe misses hands, and a missed hand
    here means the arm moves anyway. It is a convenience check on top of the
    cell's real safety system, never a replacement for one. Fails open (with a
    logged warning) if mediapipe isn't installed, because a hard dependency on
    it would make the whole app unusable without it.
    """
    if not config.SAFETY_CHECK:
        return None
    try:
        hands, _ = _run_detector("presence", "hand", frame)
    except detectors.DetectorUnavailable as exc:
        log.warning("safety check skipped - %s", exc)
        return None
    except Exception as exc:
        log.warning("safety check errored, skipping - %s", exc)
        return None
    if hands:
        return (f"I can see a hand in the workspace, so I'm not moving the arm. "
                f"Clear the area and ask again.")
    return None


def _snapshot(frame: np.ndarray, results: list[SlotResult], caption: str) -> str:
    """Annotated JPEG as a data URI: every candidate boxed, chosen ones
    highlighted, so the transcript shows exactly what the arm was told."""
    canvas = frame.copy()
    for result in results:
        if result.detector is None:
            continue
        chosen = result.chosen
        for det in result.candidates:
            draw_detection(canvas, det, result.detector.BOX_COLOR,
                           highlight=det is chosen)
    if caption:
        draw_banner(canvas, caption)
    return "data:image/jpeg;base64," + base64.b64encode(encode_jpeg(canvas)).decode()


def _describe(dets: list[Detection]) -> str:
    return ", ".join(f"{d.label} at ({d.centroid[0]}, {d.centroid[1]})"
                     for d in dets)


def handle(message: str, session_id: str) -> dict:
    """Full pipeline for one user message. Never raises: every failure comes
    back as a chat reply with a `status` the UI can style."""
    # Read history BEFORE persisting this turn - otherwise the message the
    # model is being asked to parse also shows up in its own context.
    history = store.recent_turns(session_id, count=6)
    store.add_message(session_id, "user", message)
    result = _handle(message, history)
    store.add_message(session_id, "assistant", result["reply"], meta={
        "intent": result.get("intent"),
        "status": result.get("status"),
        "robot": result.get("robot"),
        "detections": result.get("detections"),
    })
    return result


def _reply(text: str, status: str, **extra) -> dict:
    return {"reply": text, "status": status, **extra}


def _handle(message: str, history: list[dict]) -> dict:
    # 1 - parse
    try:
        intent = llm.parse_intent(message, history)
    except llm.LLMError as exc:
        return _reply(f"I couldn't reach the language model. {exc}", "error")

    base = {"intent": intent}

    if intent["action"] == "chat":
        # `note` means validation overrode the model (e.g. a pick_place with no
        # slots). Say that plainly rather than paying a second call to have the
        # model improvise around a failure it just caused.
        if intent.get("note"):
            return _reply(intent["note"], "ok", **base)
        return _reply(llm.reply_for_chat(message, history), "ok", **base)

    # 2 - one frame drives the whole command, so source and target are
    # guaranteed to come from the same instant.
    frame = camera.wait_for_frame(timeout=5)
    if frame is None:
        return _reply(
            f"The camera isn't available ({camera.error or 'no frames yet'}), "
            f"so I can't look for anything.", "error", **base)

    slots = [("source", intent["source"])]
    if intent.get("target"):
        slots.append(("target", intent["target"]))

    results: dict[str, SlotResult] = {}
    for name, slot in slots:
        results[name] = _resolve_slot(slot, frame)

    base["detections"] = {
        name: [d.to_dict() for d in r.candidates] for name, r in results.items()
    }
    ordered = list(results.values())

    # 3 - detector-level failures
    for r in results.values():
        if r.error:
            return _reply(r.error, "error", **base)

    # 4 - nothing found
    missing = [n for n, r in results.items() if not r.candidates]
    if missing:
        names = " and ".join(llm.describe(results[n].slot) for n in missing)
        return _reply(
            f"I couldn't find the {names} in the current view. "
            f"Check it's on the table and not occluded, then ask again.",
            "not_found", snapshot=_snapshot(frame, ordered, message), **base)

    # 5 - count is answerable now; no motion, no disambiguation needed
    if intent["action"] == "count":
        dets = results["source"].candidates
        noun = llm.describe(results["source"].slot)
        plural = noun if len(dets) == 1 else f"{noun}s"
        return _reply(
            f"I can see {len(dets)} {plural}: {_describe(dets)}.", "ok",
            snapshot=_snapshot(frame, ordered, message), **base)

    # 6 - ambiguity. Ask rather than guess.
    ambiguous = [n for n, r in results.items() if len(r.candidates) > 1]
    if ambiguous:
        lines = []
        for n in ambiguous:
            r = results[n]
            lines.append(f"{len(r.candidates)} things match "
                         f"\"{llm.describe(r.slot)}\": {_describe(r.candidates)}")
        return _reply(
            "I need you to narrow that down before I move the arm.\n\n"
            + "\n".join(lines)
            + "\n\nTell me which one - by position, or move the others out of frame.",
            "ambiguous", snapshot=_snapshot(frame, ordered, message), **base)

    # 7 - locate: report and stop
    if intent["action"] == "locate":
        det = results["source"].chosen
        payload = robot.format_locate(det.centroid)
        reply = (f"Found the {llm.describe(results['source'].slot)} at pixel "
                 f"({det.centroid[0]}, {det.centroid[1]}).")
        if intent.get("note"):
            reply = f"{intent['note']}\n\n{reply}"
        return _reply(
            reply, "ok",
            snapshot=_snapshot(frame, ordered, message),
            robot={"line": payload.rstrip(), "sent": False,
                   "note": "locate is read-only - nothing sent to the arm"},
            **base)

    # 8 - pick_place: safety, then send
    blocked = _safety_block(frame)
    if blocked:
        return _reply(blocked, "blocked",
                      snapshot=_snapshot(frame, ordered, message), **base)

    src = results["source"].chosen
    dst = results["target"].chosen
    record = robot.send(robot.format_pick_place(src.centroid, dst.centroid))
    base["robot"] = record

    src_name = llm.describe(results["source"].slot)
    dst_name = llm.describe(results["target"].slot)

    if record["error"]:
        return _reply(
            f"I found both objects - {src_name} at "
            f"({src.centroid[0]}, {src.centroid[1]}) and {dst_name} at "
            f"({dst.centroid[0]}, {dst.centroid[1]}) - but couldn't reach the "
            f"controller at {record['host']}:{record['port']}.\n\n{record['error']}",
            "error", snapshot=_snapshot(frame, ordered, message), **base)

    verb = "Would send" if record["dry_run"] else "Sent"
    tail = " (dry run - no socket opened)" if record["dry_run"] else ""
    reply = (f"Picking up the {src_name} and placing it on the {dst_name}.\n\n"
             f"Pick ({src.centroid[0]}, {src.centroid[1]}) → "
             f"place ({dst.centroid[0]}, {dst.centroid[1]}).\n"
             f"{verb} `{record['line']}` to {record['host']}:{record['port']}{tail}.")
    if record.get("reply"):
        reply += f"\nController replied: `{record['reply']}`"

    return _reply(reply, "ok", snapshot=_snapshot(frame, ordered, message), **base)
