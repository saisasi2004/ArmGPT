"""Ollama client: natural language -> structured robot intent.

Two design choices worth knowing:

1. The JSON schema is passed to Ollama's `format` parameter, which constrains
   token sampling. The model *cannot* emit invalid JSON — we don't rely on
   prompt instructions and then hope.

2. Thinking is disabled by default (config.LLM_THINK). Qwen3 is a hybrid
   reasoning model, but intent parsing is narrow extraction and the robot
   pipeline blocks on this call, so latency beats reasoning depth here.
"""
from __future__ import annotations

import copy
import json
import logging
import re
import time

import requests

import config
from detectors import DETECTOR_KEYS, catalog

log = logging.getLogger(__name__)

ACTIONS = ["pick_place", "locate", "count", "chat"]

# Constrains sampling: every field and enum below is guaranteed in the output.
#
# Two things are load-bearing here, both measured rather than guessed.
#
# 1. Property ORDER. The schema compiles to a grammar that emits keys in this
#    order, so whatever comes first is decided with nothing else on the page.
#    `action` used to be first, and the 4B model would guess "chat" for any
#    question and then fill the rest to match — 5/8 on the eval set.
#    `needs_camera` is one cheap token that forces the distinction it was
#    getting wrong, and `action` is now conditioned on having answered it.
#
# 2. Every field costs ~150ms. Generation runs at ~6.4 tok/s on this CPU, so
#    the schema is exactly what the router cannot derive itself — nothing more.
#    `description` and `reply` used to live here and account for ~60 of 105
#    output tokens (~9s per command) despite the router overwriting the reply
#    on almost every path. Both are gone; see reply_for_chat() for the one
#    case that genuinely needs the model to write prose.
_SLOT = {
    "type": ["object", "null"],
    "properties": {
        "detector": {"type": "string", "enum": DETECTOR_KEYS},
        "match": {"type": ["string", "null"]},
    },
    "required": ["detector"],
}

INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "needs_camera": {"type": "boolean"},
        "action": {"type": "string", "enum": ACTIONS},
        "source": copy.deepcopy(_SLOT),
        "target": copy.deepcopy(_SLOT),
    },
    "required": ["needs_camera", "action"],
}

CAMERA_ACTIONS = ["pick_place", "locate", "count"]


def _forced_camera_schema() -> dict:
    """INTENT_SCHEMA with `chat` removed and `source` mandatory.

    Used only for the retry described in parse_intent(). Deleting the option
    from the enum is what makes this work: the grammar physically cannot emit
    "chat", so the model has to spend its tokens on the decision it was
    dodging. Asking for the same thing in prose did not move the eval at all.
    """
    schema = copy.deepcopy(INTENT_SCHEMA)
    schema["properties"]["action"]["enum"] = CAMERA_ACTIONS
    schema["properties"]["source"]["type"] = "object"  # no longer nullable
    schema["required"] = ["needs_camera", "action", "source"]
    return schema

# Kept deliberately terse. Every prompt token is re-evaluated on each call
# (~4.6s for the 1200-token version this replaced), so anything that doesn't
# change an output is pure latency. The needs_camera rule and the "match is a
# single value, not a phrase" rule stay verbose on purpose — those are the two
# the model actually gets wrong when they aren't spelled out.
SYSTEM_PROMPT = """Parse commands for ArmGPT, a SCARA robot arm with an \
overhead camera. Output structured intent only. Never invent coordinates.

1. `needs_camera`: must you LOOK to answer? Questions about the scene are
   true ("how many circles?", "where is the cup?", "do you see a bottle?").
   Questions about yourself are false ("what can you do?").
2. `action`: needs_camera=true -> pick_place | locate | count (never chat).
   needs_camera=false -> chat, with source and target null.

pick_place = move A onto B; needs source AND target.
locate = find one thing. count = how many. chat = no camera needed.

DETECTORS:
{catalog}

`match` is ONE value from the detector's list, never a phrase:
"the big red block" -> detector=color, match="red".

COLOUR WINS. If the phrase contains a colour word, detector=color and
match=<the colour> — even when the noun sounds like an object:
  "blue plate"  -> color/blue   NOT objects/plate
  "green block" -> color/green  NOT objects/block
`objects` is ONLY for the COCO class list above, with no colour word present.
Resolve back-references ("put it on the green one") against the history.

EXAMPLES
"place the red object on the blue plate" -> true, pick_place,
  source{{detector:color, match:red}}, target{{detector:color, match:blue}}
"where's the cup?" -> true, locate, source{{detector:objects, match:cup}}
"how many circles?" -> true, count, source{{detector:shapes, match:circle}}
"do you see a bottle?" -> true, locate, source{{detector:objects, match:bottle}}
"count the red blocks" -> true, count, source{{detector:color, match:red}}
"pick up marker 3, put it on the yellow square" -> true, pick_place,
  source{{detector:markers, match:3}}, target{{detector:color, match:yellow}}
"what can you do?" -> false, chat, source=null, target=null
"""

# Separate, tiny prompt for the one path that needs real prose. Small on
# purpose: it is a second call, and its cost is its own prompt plus ~20
# tokens, not the parser's whole detector catalog.
CHAT_SYSTEM = """You are ArmGPT. You find objects with an overhead camera and \
make a robot arm pick them up and place them. Reply in ONE short, friendly \
sentence. If asked what you can do, give an example like "put the red block \
on the blue plate"."""


class LLMError(RuntimeError):
    pass


def _catalog_block() -> str:
    return "\n".join(f"- {d['key']}: {d['hint']}" for d in catalog())


def system_prompt() -> str:
    return SYSTEM_PROMPT.format(catalog=_catalog_block())


# ------------------------------------------------------------------- model
_resolved_model: str | None = None


def available_models() -> list[str]:
    try:
        resp = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=5)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception as exc:
        raise LLMError(f"Ollama not reachable at {config.OLLAMA_HOST}: {exc}")


def resolve_model(force: bool = False) -> str:
    """First preferred tag that is actually pulled.

    Re-resolves when `force` is set, so finishing an `ollama pull` mid-session
    is picked up without restarting the server.
    """
    global _resolved_model
    if _resolved_model and not force:
        return _resolved_model

    previous = _resolved_model
    installed = available_models()
    # Exact tag first, then allow "qwen3.5:4b" to match "qwen3.5:4b-instruct-q4"
    for want in config.LLM_MODEL_PREFERENCES:
        if want in installed:
            _resolved_model = want
            break
        prefix = [m for m in installed if m.startswith(want)]
        if prefix:
            _resolved_model = prefix[0]
            break
    else:
        raise LLMError(
            f"None of the preferred models {config.LLM_MODEL_PREFERENCES} are "
            f"installed. Available: {installed or '(none)'}. "
            f"Run: ollama pull {config.LLM_MODEL_PREFERENCES[0]}"
        )

    # Only on an actual change — status() re-resolves every few seconds and
    # would otherwise fill the log with identical lines.
    if _resolved_model != previous:
        log.info("Using LLM model: %s", _resolved_model)
    return _resolved_model


def warmup() -> None:
    """Force the model resident so the first command doesn't pay the load.

    Sends the real system prompt, not a bare "hi": that primes Ollama's
    prefix KV cache too, so the ~900 prompt tokens are already evaluated when
    the first command arrives and only the user's own words need processing.
    """
    try:
        model = resolve_model()
    except LLMError as exc:
        log.warning("warmup skipped — %s", exc)
        return
    started = time.perf_counter()
    try:
        requests.post(f"{config.OLLAMA_HOST}/api/chat", timeout=300, json={
            "model": model,
            "messages": [{"role": "system", "content": system_prompt()},
                         {"role": "user", "content": "hello"}],
            "stream": False,
            "think": config.LLM_THINK,
            "keep_alive": config.LLM_KEEP_ALIVE,
            "options": {"num_predict": 1},
        }).raise_for_status()
        log.info("%s warm in %.0fs (keep_alive=%s)", model,
                 time.perf_counter() - started, config.LLM_KEEP_ALIVE)
    except Exception as exc:
        log.warning("warmup failed (first command will be slow): %s", exc)


def status() -> dict:
    try:
        installed = available_models()
    except LLMError as exc:
        return {"ok": False, "error": str(exc), "model": None, "installed": []}
    try:
        model = resolve_model(force=True)
    except LLMError as exc:
        return {"ok": False, "error": str(exc), "model": None,
                "installed": installed}
    return {"ok": True, "error": None, "model": model, "installed": installed,
            "thinking": config.LLM_THINK}


# ------------------------------------------------------------------ parsing
def parse_intent(message: str, history: list[dict] | None = None) -> dict:
    """Turn a user message into a validated intent dict.

    Two calls at most. The second only happens when the model contradicts
    itself — it answers needs_camera=true (which it gets right essentially
    every time) and then picks action=chat anyway, leaving source null so
    there is nothing to repair locally. Re-asking with `chat` removed from the
    grammar resolves it. On a CPU-bound box the retry costs real seconds, so
    it is deliberately narrow: only on that exact contradiction, never
    speculatively.
    """
    model = resolve_model()
    messages = [{"role": "system", "content": system_prompt()}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": message})

    intent = _call(model, messages, INTENT_SCHEMA)

    if intent.get("needs_camera") and intent.get("action") == "chat":
        log.info("needs_camera=true but action=chat — retrying without 'chat'")
        retry = _call(model, messages, _forced_camera_schema())
        retry["needs_camera"] = True
        intent = retry

    return _validate(intent, message)


def reply_for_chat(message: str, history: list[dict] | None = None) -> str:
    """Free-text reply for the `chat` action only.

    A second call, but a cheap one: the parser's detector catalog is not in
    scope here, so this is a ~60-token prompt plus a one-line answer. Camera
    commands never reach this — the router composes those replies from the
    detections, which is both faster and more honest than letting the model
    narrate an outcome it cannot see.
    """
    model = resolve_model()
    messages = [{"role": "system", "content": CHAT_SYSTEM}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": message})
    try:
        resp = requests.post(f"{config.OLLAMA_HOST}/api/chat", timeout=60, json={
            "model": model,
            "messages": messages,
            "stream": False,
            "think": config.LLM_THINK,
            "keep_alive": config.LLM_KEEP_ALIVE,
            "options": {"temperature": 0.4, "num_predict": 60},
        })
        resp.raise_for_status()
        text = resp.json().get("message", {}).get("content", "").strip()
        return text or "I'm here — tell me what to pick up."
    except Exception as exc:
        log.warning("chat reply failed: %s", exc)
        return ("I can find objects with the camera and have the arm pick and "
                "place them. Try: put the red block on the blue plate.")


def _call(model: str, messages: list[dict], schema: dict) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": schema,
        "think": config.LLM_THINK,
        "keep_alive": config.LLM_KEEP_ALIVE,
        # num_predict caps a runaway; a valid intent is well under 200 tokens
        # and every token costs ~150ms on CPU.
        "options": {"temperature": 0.1, "num_predict": 256},
    }

    try:
        resp = requests.post(f"{config.OLLAMA_HOST}/api/chat", json=payload,
                             timeout=config.LLM_TIMEOUT_S)
        resp.raise_for_status()
    except requests.Timeout:
        raise LLMError(f"{model} timed out after {config.LLM_TIMEOUT_S}s.")
    except requests.RequestException as exc:
        # Older Ollama builds reject `think` on non-thinking models; retry once.
        if "think" in str(exc).lower():
            payload.pop("think")
            resp = requests.post(f"{config.OLLAMA_HOST}/api/chat", json=payload,
                                 timeout=config.LLM_TIMEOUT_S)
            resp.raise_for_status()
        else:
            raise LLMError(f"Ollama request failed: {exc}")

    content = resp.json().get("message", {}).get("content", "").strip()
    if not content:
        raise LLMError("Model returned an empty response.")
    try:
        intent = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Model returned non-JSON despite schema: {exc}\n{content[:300]}")

    return intent  # raw; parse_intent validates once, after any retry


# YOLOv8n's COCO vocabulary. Hardcoded rather than read off the model so
# validation works without importing torch.
COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
}


def _repair_vocabulary(source, target, message: str) -> None:
    """Fix slots whose `match` the chosen detector could never return.

    The 4B reliably picks detector=objects for any object-shaped noun, even
    ones outside COCO — "blue plate" becomes objects/plate, and YOLO has no
    "plate" class, so that lookup is guaranteed to find nothing. Prompting
    against it (including an explicit "COLOUR WINS" rule) did not hold.

    So: if a slot names something YOLO cannot detect but the user's message
    mentions a colour, re-aim it at the colour detector. Colours are assigned
    in the order they appear, and a colour already taken by the other slot is
    not reused — "place the red object on the blue plate" gives red to source
    and blue to target.
    """
    from detectors.color_detector import normalize_color

    words = re.findall(r"[a-z]+", message.lower())
    colours = [c for c in (normalize_color(w) for w in words) if c]
    if not colours:
        return

    taken = {s["match"] for s in (source, target)
             if s and s["detector"] == "color" and s["match"]}

    for slot in (source, target):
        if not slot or slot["detector"] != "objects":
            continue
        if (slot["match"] or "").lower() in COCO_CLASSES:
            continue  # YOLO can actually find this one
        spare = next((c for c in colours if c not in taken), None)
        if spare:
            log.info("repair: %s/%s -> color/%s (not a COCO class)",
                     slot["detector"], slot["match"], spare)
            slot["detector"], slot["match"] = "color", spare
            taken.add(spare)


def _clean_slot(slot) -> dict | None:
    if not isinstance(slot, dict):
        return None
    detector = slot.get("detector")
    if detector not in DETECTOR_KEYS:
        return None
    match = slot.get("match")
    if isinstance(match, str):
        match = match.strip() or None
    elif match is not None:
        match = str(match)  # marker ids come back as ints often enough
    return {"detector": detector, "match": match}


def _validate(intent: dict, message: str) -> dict:
    """Repair whatever the schema couldn't guarantee.

    The schema pins types and enums, but it cannot enforce *relationships* —
    that pick_place has both slots filled, or that a slot the model marked
    null isn't needed. Those get checked here rather than trusted.
    """
    action = intent.get("action")
    if action not in ACTIONS:
        action = "chat"

    source = _clean_slot(intent.get("source"))
    target = _clean_slot(intent.get("target"))
    needs_camera = bool(intent.get("needs_camera"))
    note = None  # set when we override the model; the router shows it verbatim

    _repair_vocabulary(source, target, message)

    # needs_camera scored 8/8 on the eval set while `action` did not, so where
    # the two disagree, needs_camera wins. It is decided first, on the plain
    # question, before any slot-filling pressure.
    if needs_camera and action == "chat" and source is not None:
        # (The chat+source contradiction. The chat+null-source case can't be
        # repaired here — parse_intent already retried with a grammar that
        # can't emit "chat".)
        action = "pick_place" if target is not None else "locate"
    elif not needs_camera:
        # No camera needed: any non-chat action would send the arm somewhere
        # over a greeting. Drop the slots too, so nothing downstream acts.
        action = "chat"
        source = target = None

    # A pick_place missing a slot is unexecutable — downgrade rather than
    # half-execute a motion command.
    if action == "pick_place" and (source is None or target is None):
        if source is not None:
            action = "locate"
            note = (f"I caught what to pick up but not where to put it, so I'm "
                    f"only locating the {describe(source)} for now. Tell me the "
                    f"destination and I'll do the full move.")
        else:
            action = "chat"
            source = target = None
            note = ("I couldn't tell what to pick up and where to put it. "
                    "Try: put the red block on the blue plate.")
    if action in ("locate", "count") and source is None:
        action = "chat"
        note = "I'm not sure what you want me to look for."

    return {"action": action, "source": source, "target": target,
            "needs_camera": needs_camera, "note": note,
            "raw_message": message}


def describe(slot: dict | None) -> str:
    """Human-readable name for a slot, built from detector+match.

    Replaces the `description` field the model used to generate — same output,
    zero tokens.
    """
    if not slot:
        return "object"
    match, detector = slot.get("match"), slot.get("detector")
    if detector == "markers":
        return f"marker {match}" if match else "marker"
    if detector == "color":
        return f"{match} object" if match else "coloured object"
    if detector == "motion":
        return "moving object"
    return match or "object"
