#!/usr/bin/env python3
"""
offline_arc_server.py — drop-in offline replacement for the ARC competition
game server for the Heuristic-System agent (offline mode).

the agent's client/client.py talks to a real ARC server over HTTP
(SERVER_URL/game/start, /game/action, /game/last-step, /game/stop). For offline play
we run the game OFFLINE (no scorecard, no network) via arc_agi's OFFLINE env —
this module is a tiny stdlib HTTP server
that holds a persistent arc_agi OFFLINE environment and answers the same four
endpoints with the same JSON payload shape the client expects:

    POST /game/start   {"game_id","seed"}      -> {"frame": <FRAME>, "session": {"session_token": ...}}
    POST /game/action  {"action","data?"}       -> {"frame": <FRAME>}   (header X-Session-Token)
    GET  /game/last-step                         -> {"frame": <FRAME>}
    POST /game/stop                              -> {"ok": true}

where <FRAME> = {
    "frame": [[...64x64...]],   # list of frame layers (last = settled)
    "state": "NOT_FINISHED"|"WIN"|"GAME_OVER",
    "levels_completed": int, "win_levels": int,
    "available_actions": [int,...], "action_input": {"id","data"},
    "step_index": int, "guid": str, "game_id": str,
}

This keeps client/client.py unchanged except SERVER_URL.
The env runs in THIS process (persistent across the many client subprocess
calls the agent spawns), so a single long-lived server backs a whole run.

Faithfulness note: state strings are mapped to the SERVER's vocabulary
(NOT_FINISHED/WIN/GAME_OVER) — the same strings the real ARC server emits
and that client.py/session_inspector already understand — NOT an alternative
RUNNING vocabulary. We do this here (server boundary) so nothing downstream
needs changing.
"""
from __future__ import annotations

import json
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

# Make the vendored ARC SDK importable from within this repo (no machine .pth).
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths  # noqa: F401,E402  (side-effect: puts vendor/arc on sys.path)
from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: F401,E402  (vendored)

VALID_ACTIONS = {"RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4",
                 "ACTION5", "ACTION6", "ACTION7"}


def _state_to_server_str(state) -> str:
    """arcengine GameState -> the SERVER vocabulary client.py expects.

    arcengine: NOT_FINISHED / NOT_PLAYED / WIN / GAME_OVER. client.py and
    session_inspector treat anything not GAME_OVER/WIN as in-progress and rely
    on levels_completed for completion, so a faithful passthrough is correct.
    """
    s = str(getattr(state, "name", state) or "").upper()
    if "GAME_OVER" in s:
        return "GAME_OVER"
    if s == "WIN":
        return "WIN"
    return "NOT_FINISHED"


class OfflineGame:
    """Persistent arc_agi OFFLINE env + the step bookkeeping the server needs."""

    def __init__(self, game_id: str, environments_dir: str, seed: int = 0):
        self.game_id = game_id
        self.seed = seed
        self.arcade = Arcade(arc_api_key="",
                             operation_mode=OperationMode.OFFLINE,
                             environments_dir=environments_dir)
        self._env = self.arcade.make(game_id, seed=seed,
                                     scorecard_id=None, render_mode=None)
        if self._env is None:
            raise RuntimeError(f"failed to make OFFLINE env for {game_id!r}")
        self.session_token = uuid.uuid4().hex
        self.step_index = 0
        self._last_frame = None  # cached server frame payload
        self._lock = threading.Lock()

    # ── payload construction ────────────────────────────────────────────────
    def _frame_payload(self, raw, action_id: str, action_data: dict | None) -> dict:
        frame_layers = getattr(raw, "frame", None) or []
        layers = [(l.tolist() if hasattr(l, "tolist") else l) for l in frame_layers]
        win_levels = int(getattr(raw, "win_levels", 0) or 0)
        avail = list(getattr(raw, "available_actions", []) or [])
        payload = {
            "game_id": self.game_id,
            "state": _state_to_server_str(getattr(raw, "state", None)),
            "levels_completed": int(getattr(raw, "levels_completed", 0) or 0),
            "win_levels": win_levels,
            "available_actions": avail,
            "action_input": {"id": action_id, "data": action_data or {}},
            "guid": getattr(raw, "guid", None) or self.session_token,
            "full_reset": bool(getattr(raw, "full_reset", False)),
            "frame": layers,
            "step_index": self.step_index,
        }
        return payload

    def start(self) -> dict:
        with self._lock:
            raw = self._env.reset()
            if raw is None:
                raise RuntimeError("OFFLINE env.reset() returned None")
            self.step_index = 0
            self._last_frame = self._frame_payload(raw, "RESET", {})
            return self._last_frame

    def action(self, action: str, data: dict | None) -> dict:
        with self._lock:
            if action not in VALID_ACTIONS:
                raise ValueError(f"unknown action {action!r}")
            ga = getattr(GameAction, action)
            step_data = None
            if action == "ACTION6" and data:
                step_data = {k: int(data[k]) for k in ("x", "y") if k in data}
            if action == "RESET":
                raw = self._env.step(ga)
            else:
                raw = self._env.step(ga, data=step_data) if step_data else self._env.step(ga)
            if raw is None:
                raise RuntimeError(f"OFFLINE env.step({action}) returned None")
            # server increments a monotonic step index per accepted action
            self.step_index += 1
            self._last_frame = self._frame_payload(raw, action, step_data or {})
            return self._last_frame

    def last_step(self) -> dict:
        with self._lock:
            if self._last_frame is None:
                raise RuntimeError("no step yet")
            return self._last_frame


def make_handler(game_holder: dict):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # silence
            pass

        def _send(self, code: int, obj: dict):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read(self) -> dict:
            n = int(self.headers.get("Content-Length", 0))
            if not n:
                return {}
            return json.loads(self.rfile.read(n).decode("utf-8"))

        def do_POST(self):
            try:
                if self.path == "/game/start":
                    p = self._read()
                    g = game_holder["make"](p["game_id"], int(p.get("seed", 0)))
                    game_holder["game"] = g
                    return self._send(200, {"frame": g.start(),
                                            "session": {"session_token": g.session_token}})
                g = game_holder.get("game")
                if g is None:
                    return self._send(400, {"error": "no active game"})
                if self.path == "/game/action":
                    p = self._read()
                    frame = g.action(p.get("action", ""), p.get("data"))
                    # real server echoes session on action too (client.py:472
                    # refreshes its token from response["session"]).
                    return self._send(200, {"frame": frame,
                                            "session": {"session_token": g.session_token}})
                if self.path == "/game/stop":
                    return self._send(200, {"ok": True})
                return self._send(404, {"error": "not found"})
            except Exception as e:  # noqa: BLE001
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def do_GET(self):
            try:
                if self.path == "/game/last-step":
                    g = game_holder.get("game")
                    if g is None:
                        return self._send(400, {"error": "no active game"})
                    return self._send(200, {"frame": g.last_step()})
                return self._send(404, {"error": "not found"})
            except Exception as e:  # noqa: BLE001
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})

    return Handler


class OfflineARCServer:
    def __init__(self, environments_dir: str, port: int = 0):
        self.environments_dir = environments_dir
        holder = {"game": None,
                  "make": lambda gid, seed: OfflineGame(gid, environments_dir, seed)}
        self.holder = holder
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(holder))
        self.port = self.httpd.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()
        return self.base

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        if self._thread:
            self._thread.join(timeout=2)


if __name__ == "__main__":
    # standalone smoke: serve ft09 offline and print the start frame summary
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="ft09")
    ap.add_argument("--environments-dir", required=True)
    ap.add_argument("--port", type=int, default=8899)
    args = ap.parse_args()
    srv = OfflineARCServer(args.environments_dir, port=args.port)
    srv.start()
    print(f"offline ARC server on {srv.base}")
    import time
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        srv.stop()
