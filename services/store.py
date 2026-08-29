"""Chat history and small settings blobs.

Two backends behind one API:

  * **MongoDB** when `mongod` is reachable - history and settings survive a
    restart.
  * **in-memory** otherwise - the app stays fully usable (you can still hold a
    conversation, and the sidebar still lists this session's chats), you just
    lose it all when the process exits.

Losing history should never stop someone from driving the arm, so a missing or
unreachable Mongo is a downgrade, not an error. `available()` tells the UI
which mode it's in, and `backend()` names it.

The in-memory backend exists because "degrades gracefully" used to mean
"returns an empty list": every reply was written into a session the sidebar
then refused to show, and the LLM lost the back-reference context that makes
"put it on the green one" work. That is a worse experience than it needs to
be for something that costs a dict.
"""
from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone

import config

log = logging.getLogger(__name__)

_client = None
_db = None
_available = False
_reason: str | None = None

# --- in-memory fallback state -------------------------------------------
# Guarded by _mem_lock: Flask serves on many threads and the preview/chat
# paths both touch this.
_mem_lock = threading.RLock()
_mem_sessions: dict[str, dict] = {}
_mem_messages: dict[str, list[dict]] = {}
_mem_settings: dict[str, dict] = {}

# Cap so a long-running session can't grow without bound. Generous: 500 turns
# is far more than anyone scrolls back through, and only the last 6 are ever
# fed to the model.
_MEM_MESSAGE_LIMIT = 500


def init() -> bool:
    """Connect to Mongo if we can. Never raises - the caller has no fallback
    to offer beyond the one this module already implements."""
    global _client, _db, _available, _reason
    try:
        from pymongo import MongoClient
    except ImportError:
        _available = False
        _reason = "pymongo is not installed"
        log.warning("MongoDB disabled: pymongo is not installed - chat history "
                    "will be kept in memory only. Fix with: "
                    "pip install -r requirements.txt "
                    "(and check you are running the venv's Python).")
        return False

    try:
        _client = MongoClient(config.MONGO_URI,
                              serverSelectionTimeoutMS=config.MONGO_TIMEOUT_MS)
        _client.admin.command("ping")
        _db = _client[config.MONGO_DB]
        _db.messages.create_index([("session_id", 1), ("ts", 1)])
        _db.sessions.create_index([("updated_at", -1)])
        _available = True
        _reason = None
        log.info("MongoDB connected: %s/%s", config.MONGO_URI, config.MONGO_DB)
    except Exception as exc:
        _available = False
        # Short on purpose: this goes in a UI tooltip, and PyMongo's
        # ServerSelectionTimeoutError stringifies to a paragraph of topology
        # description. The full text is one log line away.
        _reason = f"cannot reach {config.MONGO_URI} ({type(exc).__name__})"
        _client = _db = None
        log.warning("MongoDB unreachable at %s (%s) - chat history will be "
                    "kept in memory only. Start it with: mongod --dbpath "
                    "<your-data-dir>", config.MONGO_URI,
                    type(exc).__name__)
    return _available


def available() -> bool:
    """True when history is persistent. The UI shows this as a status dot."""
    return _available


def backend() -> str:
    return "mongodb" if _available else "memory"


def reason() -> str | None:
    """Why Mongo isn't in use, for the status tooltip. None when it is."""
    return _reason


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ------------------------------------------------------------------ sessions
def new_session(title: str = "New chat") -> str:
    session_id = uuid.uuid4().hex[:12]
    doc = {"_id": session_id, "title": title,
           "created_at": _now(), "updated_at": _now()}
    if _available:
        _db.sessions.insert_one(doc)
    else:
        with _mem_lock:
            _mem_sessions[session_id] = doc
            _mem_messages[session_id] = []
    return session_id


def ensure_session(session_id: str | None) -> str:
    if not session_id:
        return new_session()
    if _available:
        if _db.sessions.count_documents({"_id": session_id}, limit=1) == 0:
            _db.sessions.insert_one({
                "_id": session_id, "title": "New chat",
                "created_at": _now(), "updated_at": _now(),
            })
    else:
        with _mem_lock:
            if session_id not in _mem_sessions:
                _mem_sessions[session_id] = {
                    "_id": session_id, "title": "New chat",
                    "created_at": _now(), "updated_at": _now()}
                _mem_messages[session_id] = []
    return session_id


def list_sessions(limit: int = 50) -> list[dict]:
    if _available:
        docs = list(_db.sessions.find().sort("updated_at", -1).limit(limit))
    else:
        with _mem_lock:
            docs = sorted(_mem_sessions.values(),
                          key=lambda d: d["updated_at"], reverse=True)[:limit]
    return [{"id": d["_id"], "title": d.get("title", "New chat"),
             "updated_at": d["updated_at"].isoformat()} for d in docs]


def rename_session(session_id: str, title: str) -> None:
    if _available:
        _db.sessions.update_one({"_id": session_id},
                                {"$set": {"title": title[:80]}})
    else:
        with _mem_lock:
            if session_id in _mem_sessions:
                _mem_sessions[session_id]["title"] = title[:80]


def delete_session(session_id: str) -> None:
    if _available:
        _db.messages.delete_many({"session_id": session_id})
        _db.sessions.delete_one({"_id": session_id})
    else:
        with _mem_lock:
            _mem_sessions.pop(session_id, None)
            _mem_messages.pop(session_id, None)


# ------------------------------------------------------------------ messages
def add_message(session_id: str, role: str, content: str,
                meta: dict | None = None) -> dict:
    """Persist one turn. `meta` carries the intent, detections and robot record
    so the transcript can be replayed with its full execution trace."""
    doc = {
        "session_id": session_id, "role": role, "content": content,
        "meta": meta or {}, "ts": _now(),
    }
    if _available:
        _db.messages.insert_one(dict(doc))
        _db.sessions.update_one({"_id": session_id},
                                {"$set": {"updated_at": _now()}})
        # First user line becomes the session title
        if role == "user":
            session = _db.sessions.find_one({"_id": session_id})
            if session and session.get("title") in (None, "New chat"):
                rename_session(session_id, content)
    else:
        with _mem_lock:
            if session_id not in _mem_sessions:
                ensure_session(session_id)
            bucket = _mem_messages.setdefault(session_id, [])
            bucket.append(dict(doc))
            del bucket[:-_MEM_MESSAGE_LIMIT]
            session = _mem_sessions[session_id]
            session["updated_at"] = _now()
            if role == "user" and session.get("title") in (None, "New chat"):
                session["title"] = content[:80]
    doc.pop("_id", None)
    return {"role": role, "content": content, "meta": doc["meta"],
            "ts": doc["ts"].isoformat()}


def get_messages(session_id: str, limit: int = 200) -> list[dict]:
    if _available:
        docs = list(_db.messages.find({"session_id": session_id})
                    .sort("ts", 1).limit(limit))
    else:
        with _mem_lock:
            docs = list(_mem_messages.get(session_id, []))[-limit:]
    return [{"role": d["role"], "content": d["content"],
             "meta": d.get("meta", {}), "ts": d["ts"].isoformat()}
            for d in docs]


# ------------------------------------------------------------------ settings
def save_setting(key: str, value: dict) -> None:
    """Persist a small config blob (e.g. the robot's host/port).

    In-memory this only survives the process, which is the honest behaviour:
    the UI reflects what is in effect now, and the next launch falls back to
    the env vars and config defaults.
    """
    if _available:
        _db.settings.update_one(
            {"_id": key},
            {"$set": {"value": value, "updated_at": _now()}},
            upsert=True,
        )
    else:
        with _mem_lock:
            _mem_settings[key] = dict(value)


def load_setting(key: str) -> dict | None:
    if _available:
        doc = _db.settings.find_one({"_id": key})
        return doc["value"] if doc else None
    with _mem_lock:
        value = _mem_settings.get(key)
        return dict(value) if value else None


def recent_turns(session_id: str, count: int = 6) -> list[dict]:
    """Last N turns as plain {role, content} for the LLM's context window.

    Deliberately small: intent parsing needs just enough history to resolve
    "put it on the blue one" against the previous turn, and a long transcript
    only slows the model down and invites it to drift off-schema.
    """
    if _available:
        docs = list(_db.messages.find({"session_id": session_id})
                    .sort("ts", -1).limit(count))
        docs.reverse()
    else:
        with _mem_lock:
            docs = list(_mem_messages.get(session_id, []))[-count:]
    return [{"role": d["role"], "content": d["content"]} for d in docs]
