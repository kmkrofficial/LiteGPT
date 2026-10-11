"""`litegpt-chat-v1` (ChatML-style) renderer: one source of truth for SFT, pretraining instruction data and inference.

    <|bos|><|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>\n

`render_ids` inserts the special ids explicitly and encodes message text as ordinary text. Loss masks and the Jinja twin are M4.
"""
from __future__ import annotations

from litegpt.tokenizer import BOS, IM_END, IM_START, LiteTokenizer

VERSION = "litegpt-chat-v1"
ROLES = ("system", "user", "assistant")
RESERVED_ROLES = ("tool",)


def _check(messages: list[dict]) -> None:
    if not messages:
        raise ValueError("no messages")
    for m in messages:
        role = m.get("role")
        if role in RESERVED_ROLES:
            raise ValueError(f"role {role!r} is reserved and not enabled in {VERSION}")
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}")
        if not isinstance(m.get("content"), str):
            raise ValueError("message content must be a string")


def render_ids(tok: LiteTokenizer, messages: list[dict], add_generation_prompt: bool = False) -> list[int]:
    _check(messages)
    nl = tok.encode_text("\n")
    ids = [BOS]
    for m in messages:
        ids += [IM_START] + tok.encode_text(m["role"] + "\n" + m["content"]) + [IM_END] + nl
    if add_generation_prompt:
        ids += [IM_START] + tok.encode_text("assistant\n")
    return ids


def render_text(messages: list[dict], add_generation_prompt: bool = False) -> str:
    """Human-readable form with literal special-token strings (for docs and debugging; ids come from `render_ids`)."""
    _check(messages)
    out = "<|bos|>" + "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")
