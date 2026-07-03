"""
qwen_policy.py — QwenVLLMPolicyAdapter for the coding-agent loop.

Drives the coding agent with an open Qwen3.6-27B served by vLLM over the
OpenAI-compatible Chat Completions API (POST /v1/chat/completions). This is the
production policy boundary; /v1/responses is deliberately NOT used (vLLM
compatibility risk).

Contract (policy.py):
    generate(messages, tools, max_calls) -> PolicyResponse(text, tool_calls,
                                                            stop_reason, usage)

Design notes:
  - `tool_schemas` are ALREADY Chat Completions shape
    ({type:function, function:{name,description,parameters}}) → passed through.
  - `messages` are already OpenAI-chat shaped (role/content/tool_calls,
    role=tool/tool_call_id). We sanitize them for the wire: drop internal
    `_meta`, coerce assistant.tool_calls[].arguments to JSON strings, and keep
    tool messages text-only.
  - Sampling: Qwen3.6 thinking-mode precise-coding recommendation
    (temperature=0.6, top_p=0.95, top_k=20) — top_k via extra_body (SDK promotes).
  - Server must run with: --reasoning-parser qwen3 --enable-auto-tool-choice
    --tool-call-parser qwen3_coder.
  - Thinking: `--reasoning-parser qwen3` strips <think> from content; we read
    message.reasoning (dual getattr/model_extra) for telemetry only;
    reasoning is not fed back into the loop.
  - timeout / 5xx-connection retry / context-overflow error mapping live here,
    not in the loop (provider concerns live in the adapter).
"""
from __future__ import annotations

import json
import time
from typing import Optional

from policy import PolicyResponse, ToolCall

# Qwen3.6 thinking-mode precise-coding sampling (Qwen official recommendation).
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
DEFAULT_MAX_TOKENS = 32768
TRANSPORT_RETRY_MAX = 6
TRANSPORT_BACKOFF_CAP = 60

# Multimodal image budgets (PNG feedback).
# Hard cap MUST match vLLM `--limit-mm-per-prompt '{"image":30}'`; we never send
# a request exceeding it. Soft cap bounds per-request images well under the hard
# limit; oldest image blocks beyond it are stripped to a text placeholder.
MAX_IMAGES_HARD = 30
MAX_IMAGES_SOFT = 12


def _backoff(attempt: int) -> int:
    return min(5 * (2 ** attempt), TRANSPORT_BACKOFF_CAP)


# ── multimodal helpers (PNG feedback) ───────────────────────────────────────
def _image_data_url(block: dict) -> Optional[str]:
    """Extract a `data:image/...;base64,...` URL from a content block, or None.

    Accepts both shapes:
      OpenAI:   {"type":"image_url","image_url":{"url":"data:image/png;base64,.."}}
      internal: {"type":"image","source":{"type":"base64","media_type":"image/png","data":".."}}
    """
    if not isinstance(block, dict):
        return None
    t = block.get("type")
    if t == "image_url":
        iu = block.get("image_url")
        if isinstance(iu, dict) and isinstance(iu.get("url"), str):
            return iu["url"]
        if isinstance(iu, str):
            return iu
    if t == "image":
        src = block.get("source")
        if isinstance(src, dict) and src.get("type") == "base64":
            media = src.get("media_type", "image/png")
            data = src.get("data", "")
            return f"data:{media};base64,{data}"
    return None


def _to_openai_multimodal(content: list) -> list:
    """Convert a list of internal content blocks to OpenAI multimodal parts.

    text blocks → {"type":"text","text":...}; image blocks → {"type":"image_url",
    "image_url":{"url":"data:..."}}. Unknown blocks are stringified to text.
    """
    parts = []
    for block in content:
        if not isinstance(block, dict):
            parts.append({"type": "text", "text": str(block)})
            continue
        url = _image_data_url(block)
        if url is not None:
            parts.append({"type": "image_url", "image_url": {"url": url}})
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append({"type": "text", "text": block["text"]})
            continue
        # unknown block shape → fold into text so nothing is silently dropped
        parts.append({"type": "text", "text": _stringify_nonimage(block)})
    return parts


def _stringify_nonimage(content) -> str:
    if content is None:
        return ""
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


def _count_images(out: list) -> int:
    n = 0
    for m in out:
        c = m.get("content")
        if isinstance(c, list):
            n += sum(1 for p in c if isinstance(p, dict) and p.get("type") == "image_url")
    return n


def _prune_images(out: list, soft: int, hard: int) -> None:
    """Keep only the newest images within `soft` (hard-capped at `hard`).

    Walks messages newest→oldest, keeps the first `keep` image parts, and
    replaces older image parts with a text placeholder so the model still sees
    that a frame existed (and its frame_ref) without paying the token/limit cost.
    """
    total = _count_images(out)
    keep = min(soft, hard)
    if total <= keep:
        # still enforce the hard cap defensively (soft<=hard so usually a no-op)
        if total <= hard:
            return
        keep = hard
    seen = 0
    for m in reversed(out):
        c = m.get("content")
        if not isinstance(c, list):
            continue
        for i, p in enumerate(c):
            if not (isinstance(p, dict) and p.get("type") == "image_url"):
                continue
            seen += 1
            if seen <= keep:
                continue
            # strip this (older) image → placeholder
            ref = p.get("frame_ref") or (p.get("image_url", {}) or {}).get("frame_ref")
            note = f"[old image omitted: frame_ref={ref}, reason=image_budget]" \
                if ref else "[old image omitted, reason=image_budget]"
            c[i] = {"type": "text", "text": note}


# Injected as the first user turn when the loop has only seeded a system
# prompt (the Qwen chat template requires a user message).
_BOOTSTRAP_USER = (
    "Begin. Read the current game observation, analyze the board, and act "
    "through the client. Use your workspace tools."
)


class QwenVLLMPolicyAdapter:
    """PolicyBase impl backed by vLLM Chat Completions (Qwen3.6-27B)."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000/v1",
                 model: str = "Qwen/Qwen3.6-27B",
                 api_key: str = "EMPTY",
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 temperature: float = TEMPERATURE,
                 request_timeout: float = 600.0):
        from openai import OpenAI  # lazy; only needed when this policy is used
        import httpx
        base = (base_url or "").rstrip("/")
        if base and not base.endswith("/v1"):
            base += "/v1"
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = OpenAI(
            base_url=base, api_key=api_key or "EMPTY",
            timeout=httpx.Timeout(connect=30.0, read=request_timeout,
                                  write=120.0, pool=30.0),
            max_retries=0,
        )
        self._call_seq = 0

    # ── PolicyBase ─────────────────────────────────────────────────────────
    def generate(self, messages, tools, max_calls) -> PolicyResponse:
        payload_messages = self._sanitize_messages(messages)
        # Sampling params are provider-aware via env gates (default = all ON, i.e.
        # unchanged Qwen/vLLM behaviour). Some OpenAI-compatible gateways reject
        # `top_k` (a vLLM extension carried in extra_body), and some reasoning
        # models reject a custom temperature. Set SAMPLING_NO_TOP_K=1 /
        # SAMPLING_NO_TOP_P=1 / SAMPLING_NO_TEMPERATURE=1 at launch to drop the
        # offending fields when pointing at such an endpoint.
        import os as _os
        kwargs = {
            "model": self.model,
            "messages": payload_messages,
            "max_tokens": self.max_tokens,
        }
        if _os.getenv("SAMPLING_NO_TEMPERATURE", "") != "1":
            kwargs["temperature"] = self.temperature
        if _os.getenv("SAMPLING_NO_TOP_P", "") != "1":
            kwargs["top_p"] = TOP_P
        if _os.getenv("SAMPLING_NO_TOP_K", "") != "1":
            kwargs["extra_body"] = {"top_k": TOP_K}
        if tools:
            kwargs["tools"] = tools          # already Chat Completions shape
            kwargs["tool_choice"] = "auto"

        resp = self._call_with_retry(kwargs)
        return self._to_policy_response(resp)

    # ── message sanitation (canonical → wire) ─────────────────────────────
    @staticmethod
    def _sanitize_messages(messages) -> list:
        out = []
        for m in messages:
            role = m.get("role")
            if role == "assistant":
                msg = {"role": "assistant", "content": m.get("content") or ""}
                tcs = m.get("tool_calls") or []
                if tcs:
                    wire_tcs = []
                    for tc in tcs:
                        args = tc.get("arguments", {})
                        if not isinstance(args, str):
                            try:
                                args = json.dumps(args)
                            except Exception:
                                args = "{}"
                        wire_tcs.append({
                            "id": tc.get("id"),
                            "type": "function",
                            "function": {"name": tc.get("name", ""),
                                         "arguments": args},
                        })
                    msg["tool_calls"] = wire_tcs
                    # OpenAI allows content="" with tool_calls; keep as-is.
                out.append(msg)
            elif role == "tool":
                # tool result: text-only content, must carry tool_call_id.
                # (image feedback is delivered as a separate user game_update
                # message, never inside a tool result.)
                content = m.get("content")
                if not isinstance(content, str):
                    content = _stringify_nonimage(content)
                out.append({"role": "tool",
                            "tool_call_id": m.get("tool_call_id", ""),
                            "content": content})
            else:
                # system / user — may carry multimodal content (list of blocks
                # with text + image). Convert to OpenAI multimodal parts instead
                # of json.dumps-ing (which would hide the image from vLLM).
                content = m.get("content")
                if isinstance(content, str):
                    out.append({"role": role, "content": content})
                elif isinstance(content, list):
                    parts = _to_openai_multimodal(content)
                    out.append({"role": role, "content": parts})
                else:
                    out.append({"role": role,
                                "content": _stringify_nonimage(content)})

        # Image budget pruning: keep newest images within MAX_IMAGES_SOFT (and
        # never exceed MAX_IMAGES_HARD), replacing stripped images with a text
        # placeholder. Mutates `out` in place.
        _prune_images(out, soft=MAX_IMAGES_SOFT, hard=MAX_IMAGES_HARD)

        # Qwen's chat template requires at least one user message ("No user
        # query found in messages" 400 otherwise). The loop seeds only a
        # system prompt on turn 1, so inject a minimal bootstrap user turn.
        if not any(mm.get("role") == "user" for mm in out):
            out.append({"role": "user", "content": _BOOTSTRAP_USER})
        return out

    # ── transport ──────────────────────────────────────────────────────────
    def _call_with_retry(self, kwargs):
        self._call_seq += 1
        last_exc = None
        for attempt in range(TRANSPORT_RETRY_MAX):
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as e:  # noqa: BLE001
                last_exc = e
                err = str(e).lower()
                status = (getattr(e, "status_code", None)
                          or getattr(getattr(e, "response", None), "status_code", None))
                # context overflow → not retryable here; surface to loop/compaction.
                if any(s in err for s in ("context length", "maximum context",
                                          "context_length_exceeded",
                                          "reduce the length")):
                    raise
                transient = (
                    (status is not None and 500 <= int(status) < 600)
                    or "apiconnectionerror" in type(e).__name__.lower()
                    or "connection" in err or "timeout" in err
                )
                if transient and attempt < TRANSPORT_RETRY_MAX - 1:
                    time.sleep(_backoff(attempt))
                    continue
                raise
        if last_exc:
            raise last_exc

    # ── response parsing (Chat Completions → PolicyResponse) ───────────────
    @staticmethod
    def _to_policy_response(resp) -> PolicyResponse:
        choice = resp.choices[0]
        msg = choice.message
        finish = getattr(choice, "finish_reason", None) or "stop"
        text = getattr(msg, "content", None) or ""

        tool_calls = []
        for i, tc in enumerate(getattr(msg, "tool_calls", None) or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}
            tc_id = getattr(tc, "id", None) or f"qwen_tc_{i}"
            tool_calls.append(ToolCall(id=tc_id, name=tc.function.name,
                                       arguments=args))

        # finish_reason → stop_reason understood by the loop.
        #   tool_calls present  → "tool_calls" (loop dispatches)
        #   length              → "length" (truncated; loop treats as text turn)
        #   else                → "text" (lets [ACTIONS] fallback parse text)
        if finish == "tool_calls" or tool_calls:
            stop_reason = "tool_calls"
        elif finish == "length":
            stop_reason = "length"
        else:
            stop_reason = "text"

        # reasoning (telemetry only; NOT fed back).
        reasoning = getattr(msg, "reasoning", None)
        if reasoning is None:
            me = getattr(msg, "model_extra", None)
            if isinstance(me, dict):
                reasoning = me.get("reasoning")

        u = getattr(resp, "usage", None)
        usage = {
            "input_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(u, "completion_tokens", 0) or 0),
            "reasoning_chars": len(reasoning) if reasoning else 0,
        } if u is not None else None

        return PolicyResponse(text=text, tool_calls=tool_calls,
                              stop_reason=stop_reason, usage=usage)
