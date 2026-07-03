"""
policy.py — policy boundary for the coding-agent loop.

The policy abstracts the LLM behind a small dataclass contract
(PolicyResponse / ToolCall / PolicyBase). The concrete adapter
(QwenVLLMPolicyAdapter in qwen_policy.py) targets an OpenAI-compatible Chat
Completions endpoint (POST /v1/chat/completions).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class PolicyResponse:
    text: Optional[str]
    tool_calls: list  # list[ToolCall]
    stop_reason: str
    usage: Optional[dict] = None


class PolicyBase(Protocol):
    def generate(self, messages: list, tools: list, max_calls: int) -> PolicyResponse:
        ...
