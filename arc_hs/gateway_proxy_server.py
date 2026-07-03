#!/usr/bin/env python3
"""
gateway_proxy_server.py — hosted-gateway counterpart of competition_proxy_server.py.

In online mode the agent's client/client.py must talk to the LOCAL arc gateway
(arc_agi server) instead of the public three.arcprize.org. The gateway speaks
the SDK's /api/cmd/* protocol; client.py speaks the harness /game/* protocol.
This proxy is the translation layer, identical in shape to the competition
proxy but pointed at an ONLINE Arcade (gateway:8001) instead of a COMPETITION
Arcade (public site).

Key differences from competition_proxy_server:
  - The shared Arcade is OperationMode.ONLINE with arc_base_url=<gateway> (set up
    by swarm._setup_online), NOT COMPETITION.
  - We do NOT use a scorecard_id: make() lets the gateway use/create its default
    scorecard, which is exactly what gets settled by the hosting framework at the
    end of a rerun (00 scaffold: rerun plays only, never opens/closes a card).
    Scoring is per-action: each env.step() POSTs /api/cmd/ACTION* and the gateway
    updates its scorecard live.

One proxy == one game == one swarm worker. The shared ONLINE Arcade is created
ONCE by swarm and passed to every proxy. No close at the end.
"""
from __future__ import annotations
# Make the vendored ARC SDK importable from within this repo (no machine .pth).
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths  # noqa: F401  (side-effect: puts vendor/arc on sys.path)

import threading
import uuid

from arcengine import GameAction  # noqa: F401  (validated available)

# Reuse the offline server's payload vocabulary + valid-action set, AND the
# competition proxy's generic HTTP handler / lifecycle (both are backend-
# agnostic: the handler only needs an adapter with start()/action()/last_step()/
# session_token, and ThreadingHTTPServer plumbing is identical).
from offline_arc_server import VALID_ACTIONS, _state_to_server_str
from competition_proxy_server import make_handler


class GatewayRemoteAdapter:
    """Wraps an ONLINE RemoteEnvironmentWrapper to expose the same
    start()/action()/last_step()/session_token interface OfflineGame does.
    Same as competition's RemoteGameAdapter but WITHOUT a card_id (gateway uses
    its default scorecard)."""

    def __init__(self, shared_arcade, full_game_id: str, seed: int = 0):
        self.game_id = full_game_id
        self.seed = seed
        # ONLINE make() -> RemoteEnvironmentWrapper, using the SHARED Arcade
        # (arc_base_url=gateway). No scorecard_id: gateway uses its default card.
        self._env = shared_arcade.make(full_game_id)
        if self._env is None:
            raise RuntimeError(f"failed to make ONLINE env for {full_game_id!r}")
        # Local token client.py echoes; real session identity lives in the shared
        # Arcade cookie jar + the remote wrapper's _guid.
        self.session_token = uuid.uuid4().hex
        self.step_index = 0
        self._last_frame = None
        self._lock = threading.Lock()

    def _frame_payload(self, raw, action_id: str, action_data: dict | None) -> dict:
        # Identical shape to OfflineGame / competition RemoteGameAdapter (fields
        # verified same across OFFLINE/ONLINE/COMPETITION).
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
            raw = self._env.reset()
            if raw is None:
                raise RuntimeError("ONLINE env.reset() returned None")
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
            elif step_data:
                raw = self._env.step(ga, data=step_data)
            else:
                raw = self._env.step(ga)
            if raw is None:
                raise RuntimeError(f"ONLINE env.step({action}) returned None")
            self.step_index += 1
            self._last_frame = self._frame_payload(raw, action, step_data or {})
            return self._last_frame

    def last_step(self) -> dict:
        with self._lock:
            if self._last_frame is None:
                raise RuntimeError("no step yet")
            return self._last_frame


class GatewayProxyServer:
    """One local proxy in front of ONE gateway game, backed by the SHARED ONLINE
    Arcade. Same lifecycle API as OfflineARCServer / CompetitionProxyServer."""

    def __init__(self, shared_arcade, full_game_id: str, port: int = 0, seed: int = 0):
        from http.server import ThreadingHTTPServer
        self.shared_arcade = shared_arcade
        self.full_game_id = full_game_id
        holder = {"game": None,
                  "make": lambda: GatewayRemoteAdapter(shared_arcade,
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
