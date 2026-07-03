#!/usr/bin/env python3
"""
competition_proxy_server.py — competition counterpart of offline_arc_server.py.

In COMPETITION mode the agent's client/client.py must talk to the REAL ARC
server at https://three.arcprize.org. But that session is fragile: the server
keys "which games this session can see" off a GAMESESSION cookie, and AWS ALB
pins the session to a backend node via AWSALBAPP-* cookies. Prior history proved
(a split-session smoke test) that splitting the session across processes / new
Arcade instances yields 400 "game not found" on RESET. So competition REQUIRES a
single shared Arcade whose cookie jar persists for the whole run.

This module keeps client/client.py BYTE-IDENTICAL (it still just POSTs to a
localhost server) by putting a thin local HTTP proxy in front of ONE remote
game. The proxy:
  - is constructed with a SHARED Arcade (competition mode, already holding the
    GAMESESSION/ALB cookies from open_scorecard) + the shared card_id + ONE
    fully-qualified game id (e.g. "ft09-0d8bbf25").
  - on /game/start does shared_arcade.make(full_game_id, scorecard_id=card_id)
    -> a RemoteEnvironmentWrapper, then reset()s it.
  - translates /game/{start,action,last-step,stop} into the remote wrapper's
    reset()/step(), emitting the SAME <FRAME> payload shape offline_arc_server
    emits (fields verified field-for-field: FrameDataRaw has the same attributes
    in OFFLINE and COMPETITION).

One proxy server == one game == one swarm worker. The shared Arcade is created
ONCE by swarm and passed to every proxy; close_scorecard happens once
at the end on that same Arcade.
"""
from __future__ import annotations
# Make the vendored ARC SDK importable from within this repo (no machine .pth).
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths  # noqa: F401  (side-effect: puts vendor/arc on sys.path)

import json
import random
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from arcengine import GameAction  # noqa: F401  (validated available)

# Reuse the offline server's payload vocabulary + valid-action set verbatim so
# the two servers are byte-compatible from client.py's perspective.
from offline_arc_server import VALID_ACTIONS, _state_to_server_str


class RemoteGameAdapter:
    """Wraps a competition RemoteEnvironmentWrapper to expose the same
    start()/action()/last_step()/session_token interface OfflineGame does."""

    def __init__(self, shared_arcade, card_id: str, full_game_id: str, seed: int = 0):
        self.game_id = full_game_id
        self.card_id = card_id
        self.shared_arcade = shared_arcade
        self.seed = seed
        # ★ make() EXACTLY ONCE here, and reuse this wrapper forever (start() only
        # reset()s it). HARD-LEARNED (2026-06-30, reproduced locally): the public
        # ARC server registers a game against the guid minted by the FIRST make();
        # calling make() AGAIN for the same game (a new guid) gets an immediate
        # 400 "game not found". A prior "fix" that re-made on every retry caused
        # ALL 25 games to die at 0 steps. So: make once, never re-make.
        self._env = shared_arcade.make(full_game_id, scorecard_id=card_id)
        if self._env is None:
            raise RuntimeError(f"failed to make COMPETITION env for {full_game_id!r}")
        # ★ Dead-wrapper guard (fix for the "game not found" lost-game bug). make()
        # runs a first reset() internally; if THAT reset hit a ReadTimeout, the
        # server may have already minted a guid while the client kept _guid=None.
        # Such a wrapper is DEAD: every later reset() omits the guid -> the server
        # says the (card,game) already has one -> 400 "game not found" forever, and
        # if we cache it the game dies at 0 steps (games were lost this way).
        # With the reset timeout now 30s the trigger window is nearly gone; here we
        # additionally refuse to cache a _guid=None wrapper — retry the reset a few
        # times with backoff to mint the guid, and if it never comes, raise so the
        # holder is left empty and run_client's retry makes a fresh attempt.
        if getattr(self._env, "_guid", None) is None:
            for _i in range(3):
                time.sleep(2.0 * (_i + 1))   # 2/4/6s: let a slow server settle
                try:
                    if self._env.reset() is not None and getattr(self._env, "_guid", None):
                        break
                except Exception:
                    pass
            if getattr(self._env, "_guid", None) is None:
                raise RuntimeError(
                    f"first reset never minted guid for {full_game_id!r} "
                    f"(dead wrapper not cached)")
        # Local token handed to client.py. The REAL session identity lives in the
        # shared Arcade's cookie jar + the remote wrapper's own _guid; client.py
        # only needs *a* token to echo, it never reaches the remote server itself.
        self.session_token = uuid.uuid4().hex
        self.step_index = 0
        self._last_frame = None
        self._lock = threading.Lock()

    def _frame_payload(self, raw, action_id: str, action_data: dict | None) -> dict:
        # Identical shape to OfflineGame._frame_payload (fields verified same).
        frame_layers = getattr(raw, "frame", None) or []
        layers = [(l.tolist() if hasattr(l, "tolist") else l) for l in frame_layers]
        return {
            "game_id": self.game_id,
            "state": _state_to_server_str(getattr(raw, "state", None)),
            "levels_completed": int(getattr(raw, "levels_completed", 0) or 0),
            "win_levels": int(getattr(raw, "win_levels", 0) or 0),
            "available_actions": list(getattr(raw, "available_actions", []) or []),
            "action_input": {"id": action_id, "data": action_data or {}},
            "guid": getattr(raw, "guid", None) or self.session_token,
            "full_reset": bool(getattr(raw, "full_reset", False)),
            "frame": layers,
            "step_index": self.step_index,
        }

    def start(self) -> dict:
        with self._lock:
            # reset() the SAME wrapper (reuse the guid minted at make() time).
            # reset is idempotent and reusable (verified locally). Short retry
            # only (a couple of seconds) so we stay well inside client.py's hard
            # 30s POST timeout — we do NOT re-make (that would mint a new guid and
            # get 400 "game not found"). A genuine reset failure surfaces fast and
            # the agent-level run_client retry (which re-runs `client.py start`)
            # provides the outer retry budget.
            raw = None
            for attempt in range(3):
                raw = self._env.reset()
                if raw is not None:
                    break
                time.sleep(2.0 * (attempt + 1))   # 2s,4s — total <10s, inside 30s
            if raw is None:
                status = getattr(self._env, "_last_reset_http_status", None)
                body = getattr(self._env, "_last_reset_body", None)
                raise RuntimeError(
                    f"COMPETITION env.reset() returned None "
                    f"(status={status} body={str(body)[:120]})")
            self.step_index = 0
            self._last_frame = self._frame_payload(raw, "RESET", {})
            return self._last_frame

    def _is_gns(self) -> bool:
        se = getattr(self._env, "_last_server_error", None)
        if not se:
            return False
        return "GAME_NOT_STARTED" in str(
            se.get("error_code") or se.get("message") or "")

    def _do_step(self, ga, step_data):
        if step_data:
            return self._env.step(ga, data=step_data)
        return self._env.step(ga)

    def action(self, action: str, data: dict | None) -> dict:
        with self._lock:
            if action not in VALID_ACTIONS:
                raise ValueError(f"unknown action {action!r}")
            ga = getattr(GameAction, action)
            step_data = None
            if action == "ACTION6" and data:
                step_data = {k: int(data[k]) for k in ("x", "y") if k in data}
            raw = self._do_step(ga, step_data)
            # Defense-in-depth (SAFE variant): a step that returns None
            # with a GAME_NOT_STARTED body is the symptom the session-contention
            # report flagged. The primary fix is per-game cookie isolation in the
            # SDK (base._create_remote_wrapper), which removes the shared-jar
            # AWSALBAPP cross-write entirely. As belt-and-suspenders, if a GNS
            # still occurs we retry the SAME action ONCE: this is idempotent and
            # NEVER fabricates a RESET, so a genuine post-GAME_OVER GNS (the real
            # cause of the observed GNS bursts — a dead game the agent keeps
            # poking) still surfaces unchanged, while any transient routing blip
            # self-heals instead of wasting the action.
            if raw is None and action != "RESET" and self._is_gns():
                raw = self._do_step(ga, step_data)
            if raw is None:
                raise RuntimeError(f"COMPETITION env.step({action}) returned None")
            self.step_index += 1
            self._last_frame = self._frame_payload(raw, action, step_data or {})
            return self._last_frame

    def last_step(self) -> dict:
        with self._lock:
            if self._last_frame is None:
                raise RuntimeError("no step yet")
            return self._last_frame


def make_handler(game_holder: dict):
    """Same endpoint contract as offline_arc_server.make_handler, but /game/start
    IGNORES the client-sent game_id and uses the proxy's fixed full_game_id (this
    proxy serves exactly one remote game)."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # silence
            pass

        def _send(self, code: int, obj: dict):
            # Client may have already disconnected (e.g. client.py hit its hard 30s
            # POST timeout while proxy was retrying a reset). Writing to a dead
            # socket raises BrokenPipeError/ConnectionResetError; that exception,
            # when _send is itself called from an except block in do_POST, escapes
            # to the ThreadingHTTPServer thread and (observed 2026-06-30) took down
            # the whole swarm process. Swallow write failures: a dropped client
            # just means that one start/action is lost — the swarm must NOT die.
            try:
                body = json.dumps(obj).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass  # client gone; drop the response, keep the server alive

        def _read(self) -> dict:
            n = int(self.headers.get("Content-Length", 0))
            if not n:
                return {}
            return json.loads(self.rfile.read(n).decode("utf-8"))

        def do_POST(self):
            try:
                if self.path == "/game/start":
                    self._read()  # consume body; game_id is fixed by the proxy
                    # ★ REUSE the adapter across /game/start calls (Bug② fix,
                    # 2026-07-01). agent.py run_client retries `client.py start`
                    # up to 5x on failure; each POST used to `make()` a BRAND-NEW
                    # RemoteGameAdapter (fresh wrapper, _guid=None) = a re-make,
                    # which the server rejects with 400 "game not found" (it only
                    # honours the guid from the FIRST make). So a single transient
                    # first-reset hiccup turned into 5 guaranteed-dead re-makes ->
                    # game exited at 0 steps. Now: make ONCE, and every
                    # subsequent /game/start just reset()s the SAME wrapper (reuse
                    # the live guid) — which always succeeds.
                    g = game_holder.get("game")
                    if g is None:
                        g = game_holder["make"]()
                        game_holder["game"] = g
                    return self._send(200, {"frame": g.start(),
                                            "session": {"session_token": g.session_token}})
                g = game_holder.get("game")
                if g is None:
                    return self._send(400, {"error": "no active game"})
                if self.path == "/game/action":
                    p = self._read()
                    frame = g.action(p.get("action", ""), p.get("data"))
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


class CompetitionProxyServer:
    """One local proxy in front of ONE remote competition game, backed by the
    SHARED Arcade. Same lifecycle API as OfflineARCServer (start()->base, stop())."""

    def __init__(self, shared_arcade, card_id: str, full_game_id: str,
                 port: int = 0, seed: int = 0):
        self.shared_arcade = shared_arcade
        self.card_id = card_id
        self.full_game_id = full_game_id
        holder = {"game": None,
                  "make": lambda: RemoteGameAdapter(shared_arcade, card_id,
                                                    full_game_id, seed)}
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
