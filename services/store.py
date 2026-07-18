"""MongoDB-backed chat history.

Degrades to a no-op if Mongo isn't reachable: losing history should never stop
someone from driving the arm. `available()` tells the UI which mode it's in.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

import config

log = logging.getLogger(__name__)

_client = None
_db = None
_available = False


def init() -> bool:
    global _client, _db, _available
    try:
        from pymongo import MongoClient
        _client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=1500)
        _client.admin.command("ping")
        _db = _client[config.MONGO_DB]
        _db.messages.create_index([("session_id", 1), ("ts", 1)])
        _db.sessions.create_index([("updated_at", -1)])
        _available = True
        log.info("MongoDB connected: %s/%s", config.MONGO_URI, config.MONGO_DB)
    except Exception as exc:
        _available = False
        log.warning("MongoDB unavailable (%s) — history disabled.", exc)
    return _available


def available() -> bool:
    return _available


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ------------------------------------------------------------------ sessions
def new_session(title: str = "New chat") -> str:
    session_id = uuid.uuid4().hex[:12]
    if _available:
        _db.sessions.insert_one({
            "_id": session_id, "title": title,
            "created_at": _now(), "updated_at": _now(),
        })
    return session_id


def ensure_session(session_id: str | None) -> str:
    if not session_id:
        return new_session()
    if _available and _db.sessions.count_documents({"_id": session_id}, limit=1) == 0:
        _db.sessions.insert_one({
            "_id": session_id, "title": "New chat",
            "created_at": _now(), "updated_at": _now(),
        })
    return session_id


def list_sessions(limit: int = 50) -> list[dict]:
    if not _available:
        return []
    out = []
    for doc in _db.sessions.find().sort("updated_at", -1).limit(limit):
        out.append({
            "id": doc["_id"],
            "title": doc.get("title", "New chat"),
            "updated_at": doc["updated_at"].isoformat(),
        })
    return out


def rename_session(session_id: str, title: str) -> None:
    if _available:
        _db.sessions.update_one({"_id": session_id},
                                {"$set": {"title": title[:80]}})


def delete_session(session_id: str) -> None:
    if _available:
        _db.messages.delete_many({"session_id": session_id})
        _db.sessions.delete_one({"_id": session_id})


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
        _db.messages.insert_one(doc)
        _db.sessions.update_one({"_id": session_id},
                                {"$set": {"updated_at": _now()}})
        # First user line becomes the session title
        if role == "user":
            session = _db.sessions.find_one({"_id": session_id})
            if session and session.get("title") in (None, "New chat"):
                rename_session(session_id, content)
    doc.pop("_id", None)
    return {"role": role, "content": content, "meta": doc["meta"],
            "ts": doc["ts"].isoformat()}


def get_messages(session_id: str, limit: int = 200) -> list[dict]:
    if not _available:
        return []
    out = []
    for doc in (_db.messages.find({"session_id": session_id})
                .sort("ts", 1).limit(limit)):
        out.append({
            "role": doc["role"], "content": doc["content"],
            "meta": doc.get("meta", {}), "ts": doc["ts"].isoformat(),
        })
    return out


# ------------------------------------------------------------------ settings
def save_setting(key: str, value: dict) -> None:
    """Persist a small config blob (e.g. the robot's host/port)."""
    if _available:
        _db.settings.update_one(
            {"_id": key},
            {"$set": {"value": value, "updated_at": _now()}},
            upsert=True,
        )


def load_setting(key: str) -> dict | None:
    if not _available:
        return None
    doc = _db.settings.find_one({"_id": key})
    return doc["value"] if doc else None


def recent_turns(session_id: str, count: int = 6) -> list[dict]:
    """Last N turns as plain {role, content} for the LLM's context window.

    Deliberately small: intent parsing needs just enough history to resolve
    "put it on the blue one" against the previous turn, and a long transcript
    only slows the model down and invites it to drift off-schema.
    """
    if not _available:
        return []
    docs = list(_db.messages.find({"session_id": session_id})
                .sort("ts", -1).limit(count))
    return [{"role": d["role"], "content": d["content"]} for d in reversed(docs)]
