"""TCP link to the robot controller. Two modes, switchable at runtime.

    server  (default) - ArmGPT LISTENS. The controller (or Hercules, for
                        testing) connects in as a client, and each command is
                        broadcast to every connected client.
    client            - ArmGPT DIALS OUT to a controller that is itself a
                        listening TCP server, one short connection per command.

Either way this layer sends *pixel* coordinates only - the controller owns
hand-eye calibration and inverse kinematics.

Wire format: one CSV line per command, newline-terminated.

    PICKPLACE,<src_u>,<src_v>,<dst_u>,<dst_v>\\n
    LOCATE,<u>,<v>\\n

mode/host/port/dry-run are *runtime* settings, not import-time constants: the
Robot tab edits them live, and config.py only supplies the startup defaults.
Changes persist to MongoDB when it's available, so a restart keeps them.
"""
from __future__ import annotations

import logging
import socket
import threading
import time
from collections import deque

import config

log = logging.getLogger(__name__)

_send_lock = threading.Lock()  # serialize: the arm executes one thing at a time
_settings_lock = threading.Lock()

_settings = {
    "mode": config.ROBOT_MODE,      # "server" | "client"
    "host": config.ROBOT_HOST,
    "port": config.ROBOT_PORT,
    "timeout": config.ROBOT_TIMEOUT_S,
    "dry_run": config.ROBOT_DRY_RUN,
}

# Recent sends, newest last. Feeds the Robot tab's traffic log so you can see
# exactly what went on the wire without tailing the server console.
_history: deque[dict] = deque(maxlen=50)


class RobotError(RuntimeError):
    pass


# ============================================================ TCP server mode
class _TcpServer:
    """Listening socket that fans each command out to every connected client.

    A per-command connection (as client mode uses) is wrong here: the
    controller connects once and stays connected, and we push to it whenever a
    command produces coordinates. The accept loop runs in a daemon thread; a
    1s accept timeout lets stop() actually stop it. Dead clients are pruned
    lazily on the next broadcast rather than with a separate reaper thread.
    """

    def __init__(self) -> None:
        self._sock: socket.socket | None = None
        self._clients: list[list] = []      # [conn, addr, connected_ts]
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._bind: tuple[str, int] | None = None
        self._error: str | None = None

    def start(self, host: str, port: int) -> bool:
        self.stop()  # idempotent: rebinding on a host/port change
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, int(port)))
            s.listen(8)
            s.settimeout(1.0)
        except OSError as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            log.error("robot server: bind %s:%s failed - %s", host, port,
                      self._error)
            try:
                s.close()
            except Exception:
                pass
            return False

        self._sock = s
        self._bind = (host, int(port))
        self._error = None
        self._running = True
        self._thread = threading.Thread(target=self._accept_loop,
                                        name="robot-server", daemon=True)
        self._thread.start()
        log.info("robot server: listening on %s:%s", host, port)
        return True

    def _accept_loop(self) -> None:
        while self._running:
            try:
                conn, addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._lock:
                self._clients.append([conn, addr, time.time()])
            log.info("robot server: client connected %s:%s", addr[0], addr[1])

    def broadcast(self, data: bytes) -> tuple[int, int]:
        """Send `data` to every client. Returns (delivered, total_at_start)."""
        with self._lock:
            clients = list(self._clients)
        delivered, dead = 0, []
        for entry in clients:
            conn, addr, _ = entry
            try:
                conn.sendall(data)
                delivered += 1
            except OSError:
                dead.append(entry)
        if dead:
            with self._lock:
                for entry in dead:
                    if entry in self._clients:
                        self._clients.remove(entry)
                    try:
                        entry[0].close()
                    except Exception:
                        pass
                    log.info("robot server: client dropped %s:%s",
                             entry[1][0], entry[1][1])
        return delivered, len(clients)

    def stop(self) -> None:
        self._running = False
        with self._lock:
            for conn, _addr, _ in self._clients:
                try:
                    conn.close()
                except Exception:
                    pass
            self._clients.clear()
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        self._bind = None

    def _prune_dead(self) -> None:
        """Drop clients that have closed since the last send.

        We never read from client sockets during normal operation, so a
        disconnect (TCP FIN) goes unnoticed until the next broadcast fails -
        which would leave the UI claiming a client is connected when Hercules
        has already hit Disconnect. select() finds readable sockets; a readable
        socket that peeks empty has been closed by the peer.
        """
        import select
        with self._lock:
            clients = list(self._clients)
        if not clients:
            return
        dead = []
        for entry in clients:
            conn = entry[0]
            try:
                readable, _, _ = select.select([conn], [], [], 0)
                if not readable:
                    continue
                conn.setblocking(False)
                try:
                    if conn.recv(64, socket.MSG_PEEK) == b"":
                        dead.append(entry)      # peer closed
                except BlockingIOError:
                    pass                        # data pending, still alive
                except OSError:
                    dead.append(entry)
                finally:
                    try:
                        conn.setblocking(True)
                    except OSError:
                        pass
            except OSError:
                dead.append(entry)
        if dead:
            with self._lock:
                for entry in dead:
                    if entry in self._clients:
                        self._clients.remove(entry)
                    try:
                        entry[0].close()
                    except Exception:
                        pass
                    log.info("robot server: client gone %s:%s",
                             entry[1][0], entry[1][1])

    def status(self) -> dict:
        self._prune_dead()
        with self._lock:
            addrs = [f"{a[0]}:{a[1]}" for _c, a, _t in self._clients]
        return {
            "listening": self._running and self._sock is not None,
            "bind": self._bind,
            "clients": len(addrs),
            "client_addrs": addrs,
            "error": self._error,
        }


_server = _TcpServer()


def apply_mode() -> None:
    """Bring the server socket in line with the current mode. Called at
    startup and whenever mode/host/port change."""
    cfg = get_settings()
    if cfg["mode"] == "server":
        _server.start(cfg["host"], cfg["port"])
    else:
        _server.stop()


def shutdown() -> None:
    _server.stop()


# ----------------------------------------------------------------- settings
def get_settings() -> dict:
    with _settings_lock:
        return dict(_settings)


def update_settings(host: str | None = None, port: int | None = None,
                    dry_run: bool | None = None, timeout: int | None = None,
                    mode: str | None = None) -> dict:
    """Validate and apply. Raises ValueError on bad input so the API can 400."""
    if mode is not None and mode not in ("server", "client"):
        raise ValueError(f"mode must be 'server' or 'client', got {mode!r}")
    if host is not None:
        host = host.strip()
        if not host:
            raise ValueError("Host cannot be empty.")
        if len(host) > 255 or any(c.isspace() for c in host):
            raise ValueError(f"Not a valid host: {host!r}")
    if port is not None:
        try:
            port = int(port)
        except (TypeError, ValueError):
            raise ValueError(f"Port must be a number, got {port!r}")
        if not 1 <= port <= 65535:
            raise ValueError(f"Port must be 1–65535, got {port}")
    if timeout is not None:
        timeout = int(timeout)
        if not 1 <= timeout <= 60:
            raise ValueError("Timeout must be 1–60 seconds.")

    with _settings_lock:
        # Track the fields that require re-binding the server. A dry-run toggle
        # must NOT drop connected clients, so only these three count.
        before = (_settings["mode"], _settings["host"], _settings["port"])
        if mode is not None:
            _settings["mode"] = mode
        if host is not None:
            _settings["host"] = host
        if port is not None:
            _settings["port"] = port
        if dry_run is not None:
            _settings["dry_run"] = bool(dry_run)
        if timeout is not None:
            _settings["timeout"] = timeout
        after = (_settings["mode"], _settings["host"], _settings["port"])
        current = dict(_settings)

    log.info("robot settings: mode=%s %s:%s dry_run=%s", current["mode"],
             current["host"], current["port"], current["dry_run"])
    _persist(current)
    if before != after:
        apply_mode()
    return current


def _persist(current: dict) -> None:
    from services import store
    try:
        store.save_setting("robot", current)
    except Exception as exc:  # persistence is a nicety, never break a change
        log.debug("could not persist robot settings: %s", exc)


def load_persisted() -> None:
    """Restore saved settings at startup. Env vars still win - an explicit
    ARMGPT_ROBOT_* in the environment should not be silently overridden by
    something typed into the UI three weeks ago. Does NOT start the server;
    call apply_mode() after this."""
    import os

    from services import store
    saved = store.load_setting("robot")
    if not saved:
        return
    with _settings_lock:
        if "ARMGPT_ROBOT_MODE" not in os.environ and saved.get("mode") in ("server", "client"):
            _settings["mode"] = saved["mode"]
        if "ARMGPT_ROBOT_HOST" not in os.environ and saved.get("host"):
            _settings["host"] = saved["host"]
        if "ARMGPT_ROBOT_PORT" not in os.environ and saved.get("port"):
            _settings["port"] = int(saved["port"])
        if "ARMGPT_ROBOT_DRY_RUN" not in os.environ and "dry_run" in saved:
            _settings["dry_run"] = bool(saved["dry_run"])
    log.info("restored robot settings: mode=%s %s:%s dry_run=%s",
             _settings["mode"], _settings["host"], _settings["port"],
             _settings["dry_run"])


# ---------------------------------------------------------------- formatting
def format_pick_place(src: tuple[int, int], dst: tuple[int, int]) -> str:
    return f"PICKPLACE,{src[0]},{src[1]},{dst[0]},{dst[1]}\n"


def format_locate(point: tuple[int, int]) -> str:
    return f"LOCATE,{point[0]},{point[1]}\n"


# --------------------------------------------------------------------- send
def send(line: str) -> dict:
    """Send one preformatted CSV line. Returns a transcript-friendly record.

    Never raises on a delivery failure - the failure is reported in the
    returned dict so the chat can surface it as a message instead of a 500.
    """
    cfg = get_settings()
    record = {
        "line": line.rstrip("\n"),
        "mode": cfg["mode"],
        "host": cfg["host"],
        "port": cfg["port"],
        "dry_run": cfg["dry_run"],
        "sent": False,
        "reply": None,
        "error": None,
        "clients": None,
        "ms": None,
        "ts": time.strftime("%H:%M:%S"),
    }

    if cfg["dry_run"]:
        log.info("[dry-run] would send: %s", record["line"])
        _history.append(record)
        return record

    started = time.perf_counter()
    if cfg["mode"] == "server":
        delivered, total = _server.broadcast(line.encode("ascii"))
        record["clients"] = total
        record["sent"] = delivered > 0
        if total == 0:
            record["error"] = ("No client connected - nothing received the "
                               "command. Connect the controller (or Hercules) "
                               f"to {cfg['host']}:{cfg['port']} first.")
        elif delivered < total:
            record["error"] = (f"Delivered to {delivered} of {total} clients; "
                               f"the rest had dropped.")
    else:
        with _send_lock:
            try:
                with socket.create_connection((cfg["host"], cfg["port"]),
                                              timeout=cfg["timeout"]) as sock:
                    sock.sendall(line.encode("ascii"))
                    record["sent"] = True
                    # The controller may or may not ack; a timeout is not an
                    # error, it just means this one doesn't reply.
                    try:
                        reply = sock.recv(256)
                        if reply:
                            record["reply"] = reply.decode("ascii", "replace").strip()
                    except socket.timeout:
                        pass
            except OSError as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                log.warning("robot send failed: %s", record["error"])

    record["ms"] = round((time.perf_counter() - started) * 1000)
    _history.append(record)
    return record


def history() -> list[dict]:
    return list(_history)


def clear_history() -> None:
    _history.clear()


# ------------------------------------------------------------------- status
def test_connection(host: str | None = None, port: int | None = None,
                    timeout: float = 3.0) -> dict:
    """Health check for the Test button.

    server mode: report whether we're listening and how many clients are on.
    client mode: open a socket to the controller and close it (sends no bytes,
    so it cannot move the arm). Takes optional host/port so a candidate
    address can be tried before it's committed.
    """
    cfg = get_settings()
    if cfg["mode"] == "server":
        st = _server.status()
        ok = st["listening"]
        return {"ok": ok, "mode": "server", "host": cfg["host"],
                "port": cfg["port"], "ms": 0, "clients": st["clients"],
                "client_addrs": st["client_addrs"],
                "error": None if ok else (st["error"] or "Server is not listening.")}

    host = host or cfg["host"]
    port = int(port or cfg["port"])
    started = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
        return {"ok": True, "mode": "client", "host": host, "port": port,
                "ms": round((time.perf_counter() - started) * 1000), "error": None}
    except OSError as exc:
        return {"ok": False, "mode": "client", "host": host, "port": port,
                "ms": round((time.perf_counter() - started) * 1000),
                "error": f"{type(exc).__name__}: {exc}"}


def status() -> dict:
    """Drives the sidebar indicator and the Robot tab banner."""
    cfg = get_settings()
    base = {"mode": cfg["mode"], "dry_run": cfg["dry_run"],
            "host": cfg["host"], "port": cfg["port"]}

    if cfg["mode"] == "server":
        st = _server.status()
        # reachable is the one tri-state the UI keys colours off: None in dry
        # run, True when listening with >=1 client, False otherwise.
        if cfg["dry_run"]:
            reachable = None
        else:
            reachable = bool(st["listening"] and st["clients"] > 0)
        return {**base, "listening": st["listening"], "clients": st["clients"],
                "client_addrs": st["client_addrs"], "error": st["error"],
                "reachable": reachable}

    # client mode
    if cfg["dry_run"]:
        return {**base, "reachable": None, "clients": None}
    try:
        with socket.create_connection((cfg["host"], cfg["port"]), timeout=1.5):
            reachable = True
    except OSError:
        reachable = False
    return {**base, "reachable": reachable, "clients": None}
