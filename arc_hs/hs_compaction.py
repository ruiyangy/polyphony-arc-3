#!/usr/bin/env python3
"""
hs_compaction.py — context auto-compaction for the Heuristic-System agent.

REWRITTEN after repeated single-game experiments exposed two fatal flaws in the previous
per-turn-Layer-1 design:

  1. WRONG TOKEN METER: estimate_tokens used chars/4, but dense ASCII grid frames
     tokenize at ~1.01 chars/token (every grid digit is its own token). So the
     meter under-counted by ~4x on grid content: it reported 170K while the real
     context was 229K -> the 200K trigger never fired -> hard OOM at 229377.
     FIX: use the REAL Qwen tokenizer (measured: local AutoTokenizer 214ms vs
     vLLM /tokenize 245ms on a 131K-char payload, identical counts -> local).

  2. PER-TURN LAYER-1 WAS HARMFUL: clearing old tool outputs EVERY turn (a)
     rewrote the middle of history each turn -> destroyed prefix-cache hits ->
     slow generation; (b) deleted the agent's WORKING MEMORY (which cells it
     clicked + what each did, which hypotheses it ruled out) -> the agent kept
     re-discovering rules it had already established and repeating dead-end
     actions. A run with NO compaction at all reached further than the per-turn
     Layer-1 runs — the per-turn trim actively degraded capability.
     FIX: NO per-turn trimming. Pure append until the real token count crosses
     the trigger, then ONE auto-compact (summarize + rebuild).

New design (adapted to a self-looping agent):
  - Trigger: real tokens > COMPACT_TRIGGER_TOKENS (200K). A HARD safety at 215K
    forces a compact even if the Layer-3 side-query fails, so we never hit 229376.
  - On compact: run a STRONG Layer-3 handoff side-query (7 dimensions), persist
    it to progress_notes.md (survives trouble_protocol2's new_session), then
    rebuild history =
        [compact-notice (user)] +
        [Layer-2 mechanical state pointer (user)] +
        [Layer-3 summary (user)] +
        [last-k REAL conversation turns, newest->oldest up to ~20K tokens].
  - We keep last-k REAL turns (assistant reasoning + recent bash results with
    clicked-coords/diffs), NOT just "recent user messages" — our agent
    self-loops, its "user" messages are protocol templates + frames,
    not human intent. The real cognition lives in assistant turns + bash output.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Optional

# ── thresholds (REAL tokens; ctx=262144, generate max_tokens=32768 -> input
# hard wall = 229376). Trigger at 200K leaves headroom; HARD at 215K is the
# last-resort forced compact so a single big append-burst can't reach 229376.
# Env overrides (WM_COMPACT_TRIGGER / WM_COMPACT_HARD) let smoke tests force an
# early compact without editing defaults.
COMPACT_TRIGGER_TOKENS = int(os.getenv("WM_COMPACT_TRIGGER", "200000"))
COMPACT_HARD_TOKENS = int(os.getenv("WM_COMPACT_HARD", "215000"))
RECENT_TURNS_MAX_TOKENS = 20_000      # cap on recent-turn tokens kept verbatim
# Safe input budget for the Layer-3 side-query itself. The side-query appends
# LAYER3_PROMPT and asks for up to 32768 output, so its input must stay well under
# the 229376 wall. We feed it at most this many tokens of history (newest-first),
# which also keeps the summary focused on recent work. (A known edge case
# is "can't even send the summarization request"; we trim from the oldest.)
LAYER3_INPUT_BUDGET_TOKENS = 150_000
LAYER3_MAX_OUTPUT_TOKENS = 20_000
MAX_CONSECUTIVE_L3_FAILURES = 3

# Identifies a rebuilt-history summary message so a LATER compaction can drop the
# old summary instead of nesting summaries.
SUMMARY_PREFIX = "[[WM_COMPACT_SUMMARY]]"
COMPACT_NOTICE = (
    "You have just undergone a CONTEXT COMPACTION. The earlier exploration "
    "conversation has been replaced by the handoff below. Everything durable is "
    "on disk. Do NOT assume you remember details that were compacted away — "
    "FIRST read the files named below and run `python3 verify_hs.py` to "
    "re-ground, THEN continue. Do not re-discover rules you already established."
)

# ── Strong Layer-3 handoff prompt (7 dimensions, from the 6x-run audit). Tuned
# for the observed HYBRID mode: agent builds a verified Heuristic System AND explores
# by clicking. The biggest losses under the old design were dims 4 & 5 (ruled-out
# directions + already-clicked coords) — explicitly demanded here.
LAYER3_PROMPT = (
    "STOP. Do NOT call any tool or function. Do NOT emit <tool_call>, read_file, "
    "or bash. This is a WRITING task only: based ENTIRELY on the conversation "
    "above (you already have all the information you need in it), write a plain "
    "Markdown handoff. Output ONLY the Markdown text of the handoff, nothing else.\n\n"
    "You are checkpointing an in-progress ARC-AGI-3 Heuristic-System engineering "
    "session for a FRESH instance of yourself that will have NONE of this "
    "conversation — only the files on disk and this handoff. Write a dense, "
    "structured handoff so the next instance resumes WITHOUT re-discovering "
    "anything. Cover ALL of:\n\n"
    "## 1. Disk file state\n"
    "hs_engine.py / hs_state_io.py / hs_planner.py "
    "/ hs.md — what each currently contains, and the LAST "
    "`verify_hs.py` result (which levels PASS, which FAIL and at what "
    "step/why).\n\n"
    "## 2. Confirmed rules (hard-won, do NOT re-derive)\n"
    "Per level: background colour, toggle colour pair(s), target-encoding rule, "
    "HUD/budget formula, and the GAME_OVER action-budget threshold.\n\n"
    "## 3. Current active hypothesis + open questions\n"
    "What mechanic on the current level you are still pinning down; mark each as "
    "untested/partially-confirmed.\n\n"
    "## 4. Ruled-out directions (negative results — CRITICAL)\n"
    "Hypotheses you TRIED and DISPROVED. The next instance must NOT retry these.\n\n"
    "## 5. Key clicks already made + results\n"
    "A compact table of ACTION6 coords you clicked -> observed change "
    "(e.g. '(38,36) -> BR block 9->8'), and known-useless coords. So the next "
    "instance does not re-click the same cells.\n\n"
    "## 6. Real game progress\n"
    "Current level / levels completed / current toggle or action count / how much "
    "budget remains before GAME_OVER.\n\n"
    "## 7. Exact next step\n"
    "Given the verify state: either 'fix engine so level N verify PASSes (the "
    "mismatch is ...)' or 'run planner->executor on the verified model to clear "
    "level N' — be specific.\n\n"
    "Be concise but COMPLETE on dims 2/4/5 (those are what gets lost). Under "
    "2500 words. This is the next instance's only memory of your thinking.\n\n"
    "REMINDER: output ONLY the Markdown handoff prose. No tool calls, no "
    "<tool_call>, no function invocations — you are summarizing, not acting."
)


# ── real tokenizer (loaded once, process-wide) ───────────────────────────────
_TOKENIZER = None
_TOKENIZER_TRIED = False


def _get_tokenizer():
    global _TOKENIZER, _TOKENIZER_TRIED
    if _TOKENIZER_TRIED:
        return _TOKENIZER
    _TOKENIZER_TRIED = True
    # Tokenizer path: env WM_TOKENIZER_DIR (e.g. the model dataset dir) >
    # local default. CRITICAL: if this falls back to chars/4 (tokenizer=None),
    # ASCII grid frames undercount ~4x -> compaction never triggers -> OOM. Any
    # HF dir with the Qwen tokenizer works (the FP8 model dir has it).
    import os
    tok_dir = os.getenv("WM_TOKENIZER_DIR", "").strip() or \
        "Qwen/Qwen3.6-27B"
    try:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(
            tok_dir, trust_remote_code=True)
    except Exception:
        _TOKENIZER = None
    return _TOKENIZER


def _text_of(m: dict) -> str:
    """Flatten a message to the text we send, for token counting."""
    parts = []
    c = m.get("content")
    if isinstance(c, str):
        parts.append(c)
    elif isinstance(c, list):
        for b in c:
            if isinstance(b, dict):
                if b.get("type") == "text":
                    parts.append(b.get("text", ""))
                elif b.get("type") in ("image", "image_url"):
                    parts.append("\x00IMG\x00")  # sentinel, counted separately
                else:
                    parts.append(json.dumps(b))
            else:
                parts.append(str(b))
    if m.get("tool_calls"):
        parts.append(json.dumps(m["tool_calls"]))
    return "\n".join(parts)


def count_tokens(messages: list) -> int:
    """REAL token count via the Qwen tokenizer. Images counted at a flat ~1700.
    Falls back to a conservative chars/3 (NOT chars/4 — grid content is denser)
    only if the tokenizer can't load."""
    tok = _get_tokenizer()
    n_images = 0
    texts = []
    for m in messages:
        t = _text_of(m)
        n_images += t.count("\x00IMG\x00")
        texts.append(t.replace("\x00IMG\x00", ""))
    blob = "\n".join(texts)
    if tok is not None:
        try:
            n_text = len(tok.encode(blob))
        except Exception:
            n_text = len(blob) // 3
    else:
        n_text = len(blob) // 3
    return n_text + n_images * 1700


# ── last-k real turns selection ──────────────────────────────────────────────
def select_recent_turns(messages: list, max_tokens: int = RECENT_TURNS_MAX_TOKENS) -> list:
    """Keep the newest real conversation turns up to a token budget, newest->oldest
    then reversed. Skips any prior compaction summary/notice (avoid nesting).
    Never starts with an orphan tool result (must include its assistant call)."""
    tok = _get_tokenizer()

    def toks(m):
        t = _text_of(m).replace("\x00IMG\x00", "")
        if tok is not None:
            try:
                return len(tok.encode(t))
            except Exception:
                return len(t) // 3
        return len(t) // 3

    selected, total = [], 0
    for m in reversed(messages):
        # drop previous compaction artifacts so summaries don't nest
        c = m.get("content")
        if isinstance(c, str) and (c.startswith(SUMMARY_PREFIX)
                                   or c.startswith(COMPACT_NOTICE[:40])):
            continue
        if m.get("_meta", {}).get("compact"):
            continue
        cost = toks(m)
        if total + cost > max_tokens and selected:
            break
        selected.append(m)
        total += cost
    selected.reverse()
    while selected and selected[0].get("role") == "tool":
        selected.pop(0)
    return selected


# ── Layer 2: mechanical state pointer (no Store; reads disk) ──────────────────
WM_FILES = ["hs.md", "hs_engine.py", "hs_state_io.py",
            "hs_planner.py", "progress_notes.md"]


def layer2_state_pointer(work_dir: Path, session_dir: Path) -> str:
    lines = ["== System-recovered state (mechanical, not an LLM summary — trust "
             "this over memory) =="]
    try:
        import sys
        if str(work_dir) not in sys.path:
            sys.path.insert(0, str(work_dir))
        from session_inspector import inspect_sessions  # type: ignore
        insp = inspect_sessions(session_dir)
        lines.append(f"Game: current_level={insp.current_level_index} "
                     f"steps_total={insp.n_steps_total} "
                     f"steps_current_level={insp.n_steps_current_level} "
                     f"is_game_over={insp.is_game_over} is_solved={insp.is_solved}")
    except Exception as e:
        lines.append(f"(game progress unavailable: {type(e).__name__}: {e})")
    lines.append("")
    lines.append("On-disk files (authoritative; re-read before trusting memory):")
    for fn in WM_FILES:
        p = work_dir / fn
        if p.exists():
            st = p.stat()
            lines.append(f"  - {fn}  ({st.st_size} bytes)")
    logs = sorted(work_dir.glob("level_*_reasoning_log.md"))
    if logs:
        lines.append(f"  - {logs[-1].name}  (latest reasoning log)")
    return "\n".join(lines)


# ── engine ───────────────────────────────────────────────────────────────────
class HSCompactionEngine:
    """Context auto-compaction for one CodingSession. side_query_fn(
    messages, prompt) -> str|None drives the Layer-3 handoff summary (a one-shot
    tool-free LLM call). Same constructor signature as before (drop-in)."""

    def __init__(self, work_dir, session_dir, side_query_fn: Optional[Callable] = None,
                 logger=None):
        self.work_dir = Path(work_dir)
        self.session_dir = Path(session_dir)
        self.side_query_fn = side_query_fn
        self.logger = logger
        self.consecutive_l3_failures = 0
        self.compact_count = 0

    def maybe_compact(self, messages: list) -> list:
        """Pure append until real tokens cross the trigger, then ONE compaction.
        No per-turn trimming (that destroyed prefix cache + working memory)."""
        raw = count_tokens(messages)
        if raw <= COMPACT_TRIGGER_TOKENS:
            return messages  # untouched -> prefix cache stays valid
        return self._compact(messages, raw)

    def _compact(self, messages: list, tokens_before: int) -> list:
        self.compact_count += 1
        force = tokens_before > COMPACT_HARD_TOKENS
        allow_l3 = (self.side_query_fn is not None
                    and self.consecutive_l3_failures < MAX_CONSECUTIVE_L3_FAILURES)
        l2 = layer2_state_pointer(self.work_dir, self.session_dir)
        l3 = self._run_layer3(messages) if allow_l3 else ""

        # persist summary to disk (survives trouble_protocol2 new_session)
        if l3:
            try:
                (self.work_dir / "progress_notes.md").write_text(
                    l3, encoding="utf-8")
            except Exception:
                pass

        recent = select_recent_turns(messages)
        # If forced (over hard) and even recent turns are huge, shrink budget.
        if force:
            recent = select_recent_turns(messages, max_tokens=RECENT_TURNS_MAX_TOKENS // 2)

        new_msgs = [
            {"role": "user", "content": COMPACT_NOTICE, "_meta": {"compact": "notice"}},
            {"role": "user", "content": l2, "_meta": {"compact": "l2"}},
        ]
        if l3:
            new_msgs.append({"role": "user",
                             "content": f"{SUMMARY_PREFIX}\n{l3}",
                             "_meta": {"compact": "l3"}})
        new_msgs.append({"role": "assistant",
                         "content": "Understood — I just compacted. I will read "
                         "progress_notes.md, hs.md and the latest "
                         "reasoning log, run verify_hs.py to re-ground, "
                         "and continue without re-deriving confirmed rules."})
        new_msgs.extend(recent)

        after = count_tokens(new_msgs)
        if self.logger:
            try:
                self.logger({"kind": "compact", "n": self.compact_count,
                             "tokens_before": tokens_before, "tokens_after": after,
                             "forced": force, "layer3": bool(l3),
                             "recent_turns_kept": len(recent)})
            except Exception:
                pass
        return new_msgs

    def _run_layer3(self, messages: list) -> str:
        try:
            # Trim what we feed the side-query so the summarization request itself
            # cannot exceed the context wall (history is ~200K+ at this point;
            # side-query adds the prompt + wants 32768 output). Keep newest.
            slim = select_recent_turns(messages, max_tokens=LAYER3_INPUT_BUDGET_TOKENS)
            out = self.side_query_fn(slim, LAYER3_PROMPT)
            if not out or not out.strip():
                self.consecutive_l3_failures += 1
                return ""
            for tag in ("<think>", "</think>", "<analysis>", "</analysis>"):
                out = out.replace(tag, "")
            out = out.strip()
            # Reject tool-call spam: with tools=None the model can still EMIT
            # <tool_call>/<function= text (it's just returned as text). If the
            # "summary" is actually the model trying to act instead of summarize,
            # it's useless as a handoff — treat as failure, don't persist garbage.
            low = out.lower()
            if (out.count("<tool_call>") + low.count("<function=")) >= 2 or \
               (out.startswith("<tool_call>") or out.startswith("<function=")):
                self.consecutive_l3_failures += 1
                return ""
            # cap by real tokens
            tok = _get_tokenizer()
            if tok is not None:
                ids = tok.encode(out)
                if len(ids) > LAYER3_MAX_OUTPUT_TOKENS:
                    out = tok.decode(ids[:LAYER3_MAX_OUTPUT_TOKENS])
            self.consecutive_l3_failures = 0
            return out
        except Exception:
            self.consecutive_l3_failures += 1
            return ""
