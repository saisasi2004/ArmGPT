"""Central configuration. Every value can be overridden by an environment
variable, so nothing here needs editing to move between dev and the robot cell.
"""
from __future__ import annotations

import os


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# --- LLM (Ollama) --------------------------------------------------------
OLLAMA_HOST = os.environ.get("ARMGPT_OLLAMA_HOST", "http://localhost:11434")
# Preferred model first; the first tag actually present in `ollama list` wins.
# This lets the app run today on qwen3:8b and pick up qwen3.5:4b the moment
# its download finishes, with no code change.
LLM_MODEL_PREFERENCES = [
    m.strip() for m in os.environ.get(
        "ARMGPT_LLM_MODELS",
        "qwen3.5:4b,qwen3.5:latest,qwen3:4b,qwen3:8b",
    ).split(",") if m.strip()
]
LLM_TIMEOUT_S = _int("ARMGPT_LLM_TIMEOUT", 120)
# Ollama unloads an idle model after 5 minutes by default. Reloading 3.8GB
# from disk costs ~40s here, and the next command pays all of it - that was
# most of a measured 67s "Hello". "-1" pins the model in RAM for the session.
# The cost is the model's size in RAM while idle (~3.8GB of ~15.7GB total).
# Set "30m" or "5m" if you need that memory back for YOLO/mediapipe.
def _keep_alive(name: str, default: str):
    """Ollama wants a NUMBER of seconds (-1 = forever) or a duration STRING
    ("30m"). Env vars are always strings, and passing "-1" gets rejected with
    `time: missing unit in duration` - so numerics have to be coerced to int.
    """
    raw = os.environ.get(name, default).strip()
    try:
        return int(raw)
    except ValueError:
        return raw  # a duration like "30m"/"5m"; Ollama parses it


LLM_KEEP_ALIVE = _keep_alive("ARMGPT_LLM_KEEP_ALIVE", "-1")
# Load the model at startup so the first real command isn't the one paying the
# 40s. Runs in a background thread - the server serves immediately either way.
LLM_WARMUP = _bool("ARMGPT_LLM_WARMUP", True)
# Qwen3 is a hybrid reasoning model. Thinking is off by default: intent parsing
# is narrow extraction, and the robot pipeline blocks on this response.
LLM_THINK = _bool("ARMGPT_LLM_THINK", False)
# Context window handed to Ollama. The intent prompt is ~900 tokens and the
# reply is capped at 256, so 4096 is generous - and it is a *memory* setting
# as much as a capability one: Qwen3.5 advertises a 262k window, and letting
# Ollama size the KV cache for that on a 16GB box is a good way to get a 500
# back from /api/chat when YOLO and mediapipe are also resident.
LLM_NUM_CTX = _int("ARMGPT_LLM_NUM_CTX", 4096)

# --- Camera --------------------------------------------------------------
# A starting guess only - Windows renumbers cameras between reboots, and the
# UI's source picker (whose choice is remembered) is the real control. 0 is
# the built-in/first webcam on almost every machine.
CAMERA_INDEX = _int("ARMGPT_CAMERA_INDEX", 0)
CAMERA_WIDTH = _int("ARMGPT_CAMERA_WIDTH", 1280)
CAMERA_HEIGHT = _int("ARMGPT_CAMERA_HEIGHT", 720)

# --- Robot TCP -----------------------------------------------------------
# "server": ArmGPT listens and the controller/Hercules connects in, commands
#           broadcast to every client. "client": ArmGPT dials out per command
#           to a controller that is itself listening.
ROBOT_MODE = os.environ.get("ARMGPT_ROBOT_MODE", "server")
# In server mode this is the bind interface: 127.0.0.1 accepts only local
# clients (fine for Hercules on the same PC); use 0.0.0.0 to let another
# machine on the LAN connect. In client mode it's the controller's address.
ROBOT_HOST = os.environ.get("ARMGPT_ROBOT_HOST", "127.0.0.1")
ROBOT_PORT = _int("ARMGPT_ROBOT_PORT", 5000)
ROBOT_TIMEOUT_S = _int("ARMGPT_ROBOT_TIMEOUT", 5)  # client mode only
# Dry run: format and log the CSV line but never open a socket. Keep this ON
# until the real controller is on the other end.
ROBOT_DRY_RUN = _bool("ARMGPT_ROBOT_DRY_RUN", True)

# --- Safety --------------------------------------------------------------
# Refuse pick_place while a hand is visible in frame. This is a convenience
# interlock layered on top of the cell's real safety system - MediaPipe misses
# hands, so this must never be the only thing protecting a person.
SAFETY_CHECK = _bool("ARMGPT_SAFETY_CHECK", True)

# --- MongoDB -------------------------------------------------------------
# Optional: without it, history lives in memory for the life of the process.
# 127.0.0.1 rather than "localhost" on purpose - Windows resolves localhost to
# ::1 first, and mongod binds IPv4 only by default, so every connection pays a
# failed IPv6 attempt before falling back.
MONGO_URI = os.environ.get("ARMGPT_MONGO_URI", "mongodb://127.0.0.1:27017")
MONGO_DB = os.environ.get("ARMGPT_MONGO_DB", "armgpt")
# Seconds to wait for mongod at startup before giving up and going in-memory.
MONGO_TIMEOUT_MS = _int("ARMGPT_MONGO_TIMEOUT_MS", 1500)

# --- Flask ---------------------------------------------------------------
HOST = os.environ.get("ARMGPT_HOST", "127.0.0.1")
PORT = _int("ARMGPT_PORT", 5050)
DEBUG = _bool("ARMGPT_DEBUG", False)
