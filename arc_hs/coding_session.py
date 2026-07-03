#!/usr/bin/env python3
"""
coding_session.py — the coding agent that grows a game's Heuristic System.

Given a prompt, this session autonomously reads/writes files and runs shell
commands inside a per-game working directory until it decides the requested
chunk of work is done. It is the one component that talks to the LLM; the rest
of the harness (agent.py + protocols + prompts + workspace + verify/client
tooling) is model-agnostic orchestration around it. The session is driven by an
open OpenAI-compatible chat model — a self-hosted Qwen (Qwen3.6-27B via vLLM) by
default, any tool-calling chat endpoint in general — via two building blocks:

  - QwenVLLMPolicyAdapter : the multimodal chat-completions client (handles image
                            blocks + image budget pruning) in compat/qwen_policy.py.
  - sandbox.run_supervised: the strace execution sandbox in compat/sandbox.py.

Interface used by agent_funs/agent.py:
    .send(prompt) -> runs an autonomous tool loop, returns when the model yields
    .new_session() -> clears conversation history (fresh thread)

Tool set = generic coding only, NO game-specific tools: bash / read_file /
write_file / edit_file / grep_text / list_dir. The agent operates the game by
running `python3 client/client.py move ...` inside bash and inspects frames by
reading client/session/*.txt|*.png. There is deliberately NO game-play API
(no view_game / ActionSession): the agent must grow code that plays, not play by
hand.

Conversation continuity across send() calls: messages accumulate on
self.messages. A per-send budget bounds the autonomous loop.
"""
from __future__ import annotations

import base64
import json
import re
import sys
from pathlib import Path

import os
import subprocess
import time

# Put the in-repo compat dir (sandbox.py / qwen_policy.py / policy.py) on
# sys.path so the imports below resolve to the bundled copies — no external
# directory, no machine-level .pth.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ensure_compat_on_path  # noqa: E402
ensure_compat_on_path()

# strace sandbox is for local dev. In an already-isolated container strace may be
# unavailable, so we allow a no-sandbox fallback (WM_NO_SANDBOX=1, or auto when
# `import sandbox` fails). Local default stays sandboxed.
NO_SANDBOX = os.getenv("WM_NO_SANDBOX", "") == "1"
try:
    import sandbox  # noqa: E402  (bundled strace sandbox)
except Exception:
    sandbox = None
    NO_SANDBOX = True
from qwen_policy import QwenVLLMPolicyAdapter  # noqa: E402

# Per-send autonomous-loop budget (a self-hosted model needs an explicit cap).
# A single protocol prompt (e.g. "continue solving") may legitimately need many
# tool calls: read frames, write engine, run verify, fix, run client move, etc.
DEFAULT_MAX_TOOL_CALLS_PER_SEND = 60
BASH_TIMEOUT = 120
TOOL_RESULT_CHARS = 12000
# Cap single-generation length. The upstream default is 32768, but the model sometimes
# runs thinking away and generates the full 32768 tokens (~20 min on one FP8
# card), hanging a worker — httpx read-timeout never fires because vLLM keeps
# streaming (the read interval never exceeds the timeout). Halve to 16384 to
# bound the worst case (an earlier run showed every game
# crashing on a runaway generate before the Heuristic System was ever written).
# request_timeout is also tightened as a total-time backstop.
GEN_MAX_TOKENS = 16384
GEN_REQUEST_TIMEOUT = 1500.0
# Hard wall-clock timeout for ONE generate() call. httpx read-timeout can't catch
# a runaway that keeps slowly streaming tokens (read interval never exceeds it),
# so we wrap generate in a thread and give up after this many seconds, returning
# None so the caller abandons this generate instead of hanging the whole worker.
# 16384 tok @ ~28 tok/s ≈ 10 min, so 720s (12 min) covers legitimate work.
GEN_HARD_TIMEOUT = 720.0

# Tool schemas (generic coding; no game-specific tools).
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "bash", "description":
        "Run a shell command in the working directory (where hs_*.py, "
        "client/, verify_hs.py, etc. live). This is your main tool: use "
        "it to run python3 verify_hs.py, python3 client/client.py "
        "move ..., python3 run_main_planner.py, and any analysis scripts.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description":
        "Read a file (e.g. a frame .txt, hs_engine.py, a mismatch dump).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "offset": {"type": "integer"},
            "limit": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description":
        "Write/overwrite a file with full content (use for hs_engine.py, "
        "hs_state_io.py, hs.md, planners, etc.).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "edit_file", "description":
        "Replace a unique old_text with new_text in a file (local edit).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "old_text": {"type": "string"},
            "new_text": {"type": "string"}},
            "required": ["path", "old_text", "new_text"]}}},
    {"type": "function", "function": {
        "name": "grep_text", "description": "Regex search files under a path.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}, "path": {"type": "string"},
            "include": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "list_dir", "description": "List a directory.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}}}}},
]

_IMG_EXT = {".png"}


def _safe_json(obj) -> str:
    """json.dumps that never raises on non-serializable values (bytes, sets,
    numpy scalars, etc.). A tool result that contains raw bytes used to crash the
    whole send() with `TypeError: Object of type bytes is not JSON serializable`,
    aborting the game's init (observed: sb26). default= coerces any stray type to
    a string instead of throwing."""
    def _coerce(o):
        if isinstance(o, (bytes, bytearray)):
            try:
                return o.decode("utf-8", "replace")
            except Exception:
                return repr(o)
        return str(o)
    try:
        return json.dumps(obj, default=_coerce)
    except Exception as e:  # noqa: BLE001 — last-resort, must never raise
        return json.dumps({"_unserializable": str(e), "repr": repr(obj)[:2000]})


class CodingSession:
    """An autonomous coding loop that maintains the Heuristic System."""

    def __init__(self, work_dir, log_file, model=None, reasoning_effort="medium",
                 error_handling_manual=False, server_url=None,
                 vllm_base="http://127.0.0.1:8000/v1",
                 max_tool_calls_per_send=DEFAULT_MAX_TOOL_CALLS_PER_SEND,
                 vllm_api_key="EMPTY", deadline=None, multimodal=True):
        self.work_dir = Path(work_dir).resolve()
        self.log_file = Path(log_file)
        # multimodal=True: read_file on a .png returns its base64 so the model
        # sees the rendered frame. multimodal=False (text-only models, e.g.
        # qwen3.7-max): .png reads are refused with a pointer to the .txt ASCII
        # frame (the client writes both, same integer grid -> no info lost), so
        # no image block is ever sent. Lets a non-vision model drive this harness.
        self.multimodal = multimodal
        self.server_url = server_url  # injected into bash env for client.py
        # Per-game wall-clock deadline (epoch seconds), checked INSIDE send() —
        # before each generate AND after each tool call — so a single long send
        # can't blow past the per-game budget (the bug that let ft09 run 351min
        # under a 180min cap, then starve the later games). None = no deadline.
        self.deadline = deadline
        # The sandbox network policy allows ONLY localhost:<runtime_port>. The
        # agent's client.py must reach the offline arc server, so we pass that
        # server's port as the allowed runtime_port. Parse it from server_url.
        self.runtime_port = None
        if server_url:
            try:
                self.runtime_port = int(server_url.rsplit(":", 1)[1].split("/")[0])
            except Exception:
                self.runtime_port = None
        self.max_tool_calls_per_send = max_tool_calls_per_send
        self.messages: list = []  # persistent conversation (thread)
        self.policy = QwenVLLMPolicyAdapter(
            base_url=vllm_base,
            model=model or "Qwen/Qwen3.6-27B",
            api_key=vllm_api_key or "EMPTY",
            max_tokens=GEN_MAX_TOKENS,
            request_timeout=GEN_REQUEST_TIMEOUT)
        self._bash_seq = 0
        # Context compaction (the missing piece that OOM'd the first run). The
        # session_dir is where client.py writes recorded frames; Layer 2 reads it.
        from hs_compaction import HSCompactionEngine
        self.compaction = HSCompactionEngine(
            work_dir=self.work_dir,
            session_dir=self.work_dir / "client" / "session",
            side_query_fn=self._layer3_side_query,
            logger=self._log)
        # HS per-game action hard-cap. session_dir = where client.py writes
        # step_*_metadata.json (one per REAL game action). _hard_stop, once set,
        # makes send() short-circuit and stop_condition end the game. Cap is
        # env-configurable (HS_PER_GAME_ACTION_CAP, default 750, matching the old
        # per-level LEVEL_STEP_CAP value); 0/negative disables the cap.
        self.session_dir = self.work_dir / "client" / "session"
        self._hard_stop = None
        self._per_game_action_cap = int(os.getenv("HS_PER_GAME_ACTION_CAP", "750"))

    def _count_actions_fast(self) -> int:
        """Real cumulative game-action count for THIS game = number of
        step_*_metadata.json across all level_*/attempt_* dirs. Semantics match
        session_inspector.n_steps_total but this only globs (no json parse), so it
        is cheap enough to call after every bash. NOT the count of intermediate
        files (an attempt can have hundreds of those but far fewer real steps)."""
        sd = self.session_dir
        if not sd.is_dir():
            return 0
        n = 0
        for attempt in sd.iterdir():
            if attempt.is_dir() and attempt.name.startswith("level_"):
                n += sum(1 for _ in attempt.glob("step_*_metadata.json"))
        return n

    def _layer3_side_query(self, messages, prompt):
        """One-shot, TOOL-FREE LLM call for the Layer-3 handoff note. Must not
        recurse into compaction or tools — it just summarizes."""
        wire = list(messages) + [{"role": "user", "content": prompt}]
        resp = self.policy.generate(wire, tools=None, max_calls=1)
        return resp.text or ""


    # ── coding-agent interface ────────────────────────────────────
    def new_session(self) -> None:
        self.messages = []

    def _generate_guarded(self, messages, tools, max_calls):
        """policy.generate with a hard wall-clock timeout AND crash isolation.
        Runs in a worker thread; returns None (so the caller abandons this
        generate gracefully) when EITHER:
          - it exceeds GEN_HARD_TIMEOUT (runaway generation that httpx
            read-timeout can't catch — vLLM keeps slowly streaming so the read
            interval never trips), OR
          - the generate call raises/returns a bad response. This notably covers
            the global-deadline teardown race (Bug#2): when the swarm tears vLLM
            down at the global deadline, an in-flight request can make
            policy.generate hit `resp.choices` on a None response and raise
            AttributeError. We must NOT let that crash the whole iteration loop
            (it abandoned a whole wave of games) — the already-POSTed
            actions are already scored; just yield this send cleanly.
        The orphaned thread (on timeout) is left to die with the eventual
        request_timeout — it holds no lock. (qwen_policy.py is read-only,
        so the guard lives here, in our layer.)"""
        import concurrent.futures as _fut
        ex = _fut.ThreadPoolExecutor(max_workers=1)
        fut = ex.submit(self.policy.generate, messages, tools, max_calls)
        try:
            return fut.result(timeout=GEN_HARD_TIMEOUT)
        except _fut.TimeoutError:
            return None
        except Exception as e:  # noqa: BLE001 — isolate any generate failure
            self._log({"kind": "generate_error",
                       "error": f"{type(e).__name__}: {str(e)[:200]}"})
            return None
        finally:
            ex.shutdown(wait=False)

    def send(self, prompt: str, max_calls: int | None = None):
        """Append a user prompt and run the autonomous tool loop until the model
        stops calling tools (yields a final text) or the per-send budget is hit.
        Returns a small summary list of events.

        max_calls: optional per-send override of the tool-call budget. Light
        continuation/stuck/reset nudges pass a small cap so a single such send
        cannot churn tools to the deadline; heavy code-rewrite sends omit it and
        get the full self.max_tool_calls_per_send. Clamped to the instance cap."""
        budget = self.max_tool_calls_per_send
        if max_calls is not None:
            budget = max(1, min(int(max_calls), self.max_tool_calls_per_send))
        # Text-only models (multimodal=False): the main prompt is written for a
        # vision model ("inspect the PNG with your own eyes"). Prepend a one-time
        # override so the model uses ASCII .txt frames and never wastes turns
        # trying to read PNGs (read_file refuses .png in this mode anyway).
        if not self.multimodal and not getattr(self, "_text_only_notice_sent", False):
            self.messages.append({"role": "user", "content": (
                "IMPORTANT — you are a TEXT-ONLY model: you cannot see images. "
                "Ignore every instruction about inspecting PNG frames 'with your "
                "own eyes' or using image/animation visual inspection. For every "
                "frame, read the ASCII dump instead: the client writes a matching "
                "`.txt` next to each `.png` (e.g. step_0001_final.txt) from the "
                "same integer grid — it contains the full game state. Do NOT call "
                "read_file on .png files (it is disabled); always read the .txt. "
                "Do all frame/mismatch analysis numerically on the ASCII grids.")})
            self._text_only_notice_sent = True
        self.messages.append({"role": "user", "content": prompt})
        calls = 0
        last_text = ""
        while calls < budget:
            # HS hard-cap: if a per-game hard stop is set, this send (and every
            # later one this game) must not issue any more actions. Short-circuit
            # so the game winds down to stop_condition cleanly.
            if self._hard_stop is not None:
                break
            # Deadline check BEFORE each generate: stops a long send from blowing
            # past the per-game cutoff (self.deadline). Without this the deadline
            # is only seen between sends, so one send could run hours past it.
            if self.deadline is not None and time.time() >= self.deadline:
                self._log({"kind": "deadline_stop", "where": "pre_generate",
                           "calls": calls})
                break
            # compact BEFORE each generate so we never feed an over-budget
            # context to vLLM (the OOM fix). Layer-1 runs cheaply every turn;
            # full L1+L2+L3 only fires when over the trigger threshold.
            self.messages = self.compaction.maybe_compact(self.messages)
            resp = self._generate_guarded(self.messages, TOOL_SCHEMAS, max_calls=8)
            if resp is None:
                # generation hard-timed-out (runaway). Record + yield this send so
                # the worker doesn't hang; the agent's main loop continues with the
                # next iteration/protocol prompt. (Pre-fix, this hung ~20 min then
                # raised APITimeoutError, killing the whole game at 0 levels.)
                self._log({"kind": "gen_timeout", "note": "generate exceeded "
                           "hard timeout; abandoning this send"})
                break
            last_text = resp.text or last_text
            self.messages.append({
                "role": "assistant", "content": resp.text or "",
                "tool_calls": [{"id": tc.id, "name": tc.name,
                                "arguments": tc.arguments}
                               for tc in (resp.tool_calls or [])]})
            self._log({"kind": "assistant", "text": (resp.text or "")[:2000],
                       "n_tool_calls": len(resp.tool_calls or [])})
            if not resp.tool_calls:
                break  # model yielded — this send's chunk of work is done
            hit_deadline = False
            for tc in resp.tool_calls:
                calls += 1
                result, image_b64 = self._dispatch(tc.name, tc.arguments)
                self.messages.append({
                    "role": "tool", "tool_call_id": tc.id, "name": tc.name,
                    "content": _safe_json(result)[:TOOL_RESULT_CHARS]})
                # deliver any read image as a follow-up user image message
                # (tool messages must stay text-only — qwen_policy constraint)
                if image_b64:
                    self.messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"[image: {tc.arguments.get('path')}]"},
                            {"type": "image", "source": {"type": "base64",
                             "media_type": "image/png", "data": image_b64}}]})
                if calls >= budget:
                    break
                # Deadline check AFTER each tool call: a send with many cheap tool
                # calls shouldn't run minutes past the cutoff either. Worst-case
                # overshoot is now one tool call + one in-flight generate, not the
                # whole remaining send.
                if self.deadline is not None and time.time() >= self.deadline:
                    self._log({"kind": "deadline_stop", "where": "post_tool",
                               "calls": calls})
                    hit_deadline = True
                    break
                # HS per-game action hard-cap (a single bash `for i in
                # {1..200}; do client.py move ...` batch-fires actions, bypassing
                # the iteration-level LEVEL_STEP_CAP). Only bash issues actions, so
                # only check after bash. Reads the REAL cumulative action count (glob
                # of step_*_metadata.json) and, if over cap, sets _hard_stop so this
                # send + all later ones wind down, then stop_condition ends the game.
                if (tc.name == "bash" and self._per_game_action_cap
                        and self._count_actions_fast() >= self._per_game_action_cap):
                    self._log({"kind": "hard_stop", "reason": "per_game_action_cap",
                               "cap": self._per_game_action_cap, "calls": calls})
                    self._hard_stop = "per_game_action_cap"
                    hit_deadline = True   # reuse the break-out mechanism
                    break
            if hit_deadline:
                break
        self._log({"kind": "send_done", "tool_calls_used": calls})
        return [{"summary": last_text[:500], "tool_calls_used": calls}]

    # ── tool dispatch (sandboxed) ──────────────────────────────
    def _dispatch(self, name, args):
        try:
            if name == "bash":
                return self._bash(args.get("command", "")), None
            if name == "read_file":
                return self._read_file(args)
            if name == "write_file":
                return self._write_file(args), None
            if name == "edit_file":
                return self._edit_file(args), None
            if name == "grep_text":
                return self._grep(args), None
            if name == "list_dir":
                return self._list_dir(args), None
            return {"ok": False, "error": f"unknown tool {name}"}, None
        except Exception as e:  # contain everything; never crash the loop
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}, None

    def _bash(self, command: str) -> dict:
        self._bash_seq += 1
        # ── command guard (Bug#1 fix) ────────────────────────────────────────
        # The write_file/edit_file tools already refuse to touch client/session
        # and the protected tooling files (_writable). But bash is a back door:
        # the run 2026-06-16 showed Qwen, when stuck, ran
        #   rm -rf client/session && python3 client/client.py start <name>
        # which DELETED the recorded game truth (3 high-yield games self-wiped to
        # 0 levels locally). Block destructive commands that target the protected
        # paths; allow everything else (incl. the legit `client.py move/start`).
        guard = self._guard_command(command)
        if guard is not None:
            self._log({"kind": "bash_blocked", "seq": self._bash_seq,
                       "command": command[:500], "reason": guard})
            return {"ok": False, "exit_code": 126, "timed_out": False,
                    "stdout": "", "stderr": f"BLOCKED by harness guard: {guard}",
                    "sandbox_mode": "guard"}
        env_overrides = {}
        if self.server_url:
            env_overrides["ARC_SERVER_URL"] = self.server_url
        # client.py resolves its session dir from ARC_CLIENT_ROOT (default = its
        # own dir under work_dir); leave default so session lands in work_dir.
        if NO_SANDBOX or sandbox is None:
            out = self._bash_nosandbox(command, env_overrides)
        else:
            res = sandbox.run_supervised(
                command, str(self.work_dir), str(self.work_dir),
                runtime_port=self.runtime_port,
                timeout=BASH_TIMEOUT, env_overrides=env_overrides,
                run_root=str(self.work_dir))
            out = {"ok": (not res.timed_out and res.exit_code == 0),
                   "exit_code": res.exit_code, "timed_out": res.timed_out,
                   "stdout": res.stdout, "stderr": res.stderr,
                   "sandbox_mode": res.mode}
            if getattr(res, "violation", None):
                out["violation"] = res.violation
        self._log({"kind": "bash", "seq": self._bash_seq,
                   "command": command[:500], "exit_code": out.get("exit_code"),
                   "stdout": (out.get("stdout") or "")[:1500],
                   "stderr": (out.get("stderr") or "")[:800]})
        return out

    def _bash_nosandbox(self, command: str, env_overrides: dict) -> dict:
        """no-strace path: run the command directly via subprocess in
        work_dir. The eval container is already isolated, so no strace needed.
        The command guard (_guard_command) still runs upstream in _bash, so the
        Bug#1 self-wipe protection holds even without the sandbox."""
        env = dict(os.environ)
        env.update(env_overrides)
        try:
            r = subprocess.run(command, shell=True, cwd=str(self.work_dir),
                               capture_output=True, text=True, timeout=BASH_TIMEOUT,
                               env=env)
            return {"ok": r.returncode == 0, "exit_code": r.returncode,
                    "timed_out": False, "stdout": r.stdout, "stderr": r.stderr,
                    "sandbox_mode": "none"}
        except subprocess.TimeoutExpired as e:
            # On timeout, subprocess may hand back stdout/stderr as raw BYTES even
            # under text=True (CPython quirk) -> would later crash json.dumps. Coerce.
            def _s(v):
                if isinstance(v, (bytes, bytearray)):
                    return v.decode("utf-8", "replace")
                return v or ""
            return {"ok": False, "exit_code": -1, "timed_out": True,
                    "stdout": _s(e.stdout), "stderr": _s(e.stderr),
                    "sandbox_mode": "none"}

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else (self.work_dir / p)

    def _read_file(self, args):
        p = self._resolve(args["path"])
        if not p.exists():
            return {"ok": False, "error": f"not found: {p}"}, None
        if p.suffix.lower() in _IMG_EXT:
            if not self.multimodal:
                # text-only model: don't send an image; point to the ASCII frame.
                txt = p.with_suffix(".txt")
                hint = (f"image reads are disabled (text-only model). Read the "
                        f"ASCII frame instead: {txt}" if txt.name else
                        "image reads are disabled (text-only model)")
                return {"ok": False, "kind": "image_disabled", "path": str(p),
                        "hint": hint}, None
            b64 = base64.b64encode(p.read_bytes()).decode("ascii")
            return {"ok": True, "kind": "image", "path": str(p)}, b64
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as e:
            return {"ok": False, "error": f"read error: {e}"}, None
        off = int(args.get("offset", 0)); lim = int(args.get("limit", 2000))
        chunk = lines[off:off + lim]
        return {"ok": True, "path": str(p), "content": "\n".join(chunk),
                "total_lines": len(lines)}, None

    def _write_file(self, args) -> dict:
        p = self._resolve(args["path"])
        if not self._writable(p):
            return {"ok": False, "error": f"forbidden write: {p}"}
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(args["content"], encoding="utf-8")
        return {"ok": True, "path": str(p), "bytes": len(args["content"])}

    def _edit_file(self, args) -> dict:
        p = self._resolve(args["path"])
        if not self._writable(p):
            return {"ok": False, "error": f"forbidden edit: {p}"}
        if not p.exists():
            return {"ok": False, "error": f"not found: {p}"}
        s = p.read_text(encoding="utf-8")
        old = args["old_text"]
        n = s.count(old)
        if n == 0:
            return {"ok": False, "error": "old_text not found"}
        if n > 1:
            return {"ok": False, "error": f"old_text not unique ({n} matches)"}
        p.write_text(s.replace(old, args["new_text"], 1), encoding="utf-8")
        return {"ok": True, "path": str(p)}

    def _grep(self, args) -> dict:
        import re
        root = self._resolve(args.get("path", "."))
        try:
            rx = re.compile(args["pattern"])
        except re.error as e:
            return {"ok": False, "error": f"bad regex: {e}"}
        inc = args.get("include", "*")
        matches = []
        files = [root] if root.is_file() else root.rglob(inc)
        for f in files:
            if not f.is_file() or f.suffix.lower() in _IMG_EXT:
                continue
            try:
                for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if rx.search(line):
                        matches.append({"file": str(f), "line": i, "text": line[:200]})
                        if len(matches) >= 200:
                            break
            except Exception:
                continue
            if len(matches) >= 200:
                break
        return {"ok": True, "matches": matches, "count": len(matches)}

    def _list_dir(self, args) -> dict:
        p = self._resolve(args.get("path", "."))
        if not p.is_dir():
            return {"ok": False, "error": f"not a directory: {p}"}
        entries = [
            {"name": c.name, "is_dir": c.is_dir(),
             "size": (c.stat().st_size if c.is_file() else None)}
            for c in sorted(p.iterdir())]
        return {"ok": True, "path": str(p), "entries": entries}

    def _guard_command(self, command: str) -> str | None:
        """Bug#1 guard: return a reason string to BLOCK a destructive bash
        command that DELETES the protected game truth (client/session) or
        OVERWRITES a protected tooling file, else None to allow.

        Design rule (after a false-positive postmortem): only block when we can
        identify the ACTUAL target of a destructive op. Specifically:
          - a delete/move verb (rm/mv/...) whose arguments name a protected path, OR
          - a file-truncating redirect (`>`/`>>`) whose RESOLVED TARGET TOKEN is a
            protected file or lives under client/session.
        Everything else is allowed. In particular these are NOT blocked (they were
        all false-positives that killed real work):
          - stderr/fd redirects: `2>&1`, `2>/dev/null`, `&>`, `>&`
          - `>` used as a comparison/arrow inside `python3 -c "..."` / heredocs
          - merely MENTIONING a protected filename (e.g. `python3 verify_hs.py`,
            `cat verify_hs.py`, reading frames under client/session)
        The agent runs the game via `python3 client/client.py move/start ...`, reads
        recorded frames under client/session, and runs verify/planner scripts — all
        must pass.
        """
        cmd = command or ""
        low = cmd.lower()
        # bare protected tooling filenames (no directory) the agent must not clobber
        protected_files = {
            "verify_hs.py", "verify_main_planner.py", "session_tools.py",
            "game_status.py", "script_tools.py", "mismatch_artifacts.py",
            "run_main_planner.py", "plan_executor.py", "frame_plot_lib.py",
            "load_initial_full_frame.py", "timeout_tools.py", "client.py",
        }

        def _hits_session(token: str) -> bool:
            t = token.strip().strip("'\"").replace("\\", "/").lower()
            return "client/session" in t

        def _hits_protected_file(token: str) -> bool:
            t = token.strip().strip("'\"").replace("\\", "/").lower()
            base = t.rsplit("/", 1)[-1]
            return base in protected_files

        # ── 1) delete/move verbs aimed at a protected path → BLOCK ──────────────
        # Tokenize on whitespace; if a destructive verb appears AND any later token
        # names client/session or a protected file, block. (Conservative: only the
        # explicit destructive verbs, not arbitrary `>`.)
        destructive_verbs = {"rm", "rmdir", "mv", "shred", "truncate", "unlink"}
        toks = re.split(r"[\s;|&]+", cmd)
        verbs_present = {t.lower() for t in toks} & destructive_verbs
        if verbs_present:
            for t in toks:
                if _hits_session(t):
                    return "destructive command targets client/session (the recorded game truth)"
                if _hits_protected_file(t):
                    return f"destructive command targets a protected file ({t.strip()})"
            # exact self-wipe pattern seen in the original run (rm -rf ... session)
            if re.search(r"\brm\s+-[a-z]*r[a-z]*f?\b.*session", low) or \
               re.search(r"\brm\s+-[a-z]*f?r[a-z]*\b.*session", low):
                return "rm -rf of a session directory is not allowed"

        # ── 2) file-truncating redirect whose TARGET is protected → BLOCK ───────
        # Match real file redirects only: optional single fd digit, then `>`/`>>`,
        # then the target token. This deliberately does NOT match `2>&1`, `>&2`,
        # `&>` fd-dups (target starting with `&`), nor `>` used as comparison
        # inside quotes (those have no bareword filename target right after).
        for m in re.finditer(r"(?<![0-9&>])>>?\s*([^\s;|&<>]+)", cmd):
            target = m.group(1)
            if target.startswith("&"):   # `>&1` etc. — fd dup, not a file
                continue
            if _hits_session(target):
                return "redirect would overwrite something under client/session"
            if _hits_protected_file(target):
                return f"redirect would overwrite a protected file ({target})"
        return None

    def _writable(self, p: Path) -> bool:
        """Agent may write anywhere in work_dir EXCEPT client/session (the
        recorded game truth, written only by client.py) and the read-only
        verify/client tooling files (the harness-owned boundary)."""
        try:
            rel = p.resolve().relative_to(self.work_dir)
        except ValueError:
            return False
        parts = rel.parts
        if parts and parts[0] == "client" and len(parts) > 1 and parts[1] == "session":
            return False
        # protect the verify/IO contract scripts from being silently overwritten
        protected = {"verify_hs.py", "verify_main_planner.py",
                     "session_tools.py", "game_status.py", "script_tools.py",
                     "mismatch_artifacts.py", "run_main_planner.py",
                     "plan_executor.py", "frame_plot_lib.py",
                     "load_initial_full_frame.py", "timeout_tools.py"}
        if len(parts) == 1 and parts[0] in protected:
            return False
        return True

    def _log(self, obj):
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except Exception:
            pass
