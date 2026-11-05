"""`litegpt-chat-v1` (ChatML-style) renderer: one source of truth for SFT, pretraining instruction data and inference.

    <|bos|><|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>\n

`render_ids` inserts the special ids explicitly and encodes message text as ordinary text. `render` adds the assistant-only loss
mask. `JINJA_PATH` is the same template for Hugging Face tools (tests/test_chat_template.py checks the ids are identical); it cannot
stop special-token strings inside a message from being read as control tokens, so LiteGPT's own code always uses `render_ids`.
"""
from __future__ import annotations

from pathlib import Path

from litegpt.tokenizer import BOS, IM_END, IM_START, LiteTokenizer

VERSION = "litegpt-chat-v1"
JINJA_PATH = Path(__file__).resolve().parent / "assets" / "tokenizer" / "litegpt-chat-v1.jinja"
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


def render(tok: LiteTokenizer, messages: list[dict]) -> tuple[list[int], list[int]]:
    """The ids of `render_ids` and a mask that is 1 on what the model must learn to write: assistant text and its <|im_end|>.
    An assistant message starting with a newline merges with the header, so its ids would differ from the generation prompt at
    inference: that raises ValueError, and SFT preparation counts and drops it."""
    _check(messages)
    nl = tok.encode_text("\n")
    ids, mask = [BOS], [0]
    for m in messages:
        body = tok.encode_text(m["role"] + "\n" + m["content"])
        head = tok.encode_text(m["role"] + "\n")
        learn = int(m["role"] == "assistant")
        if learn and body[:len(head)] != head:
            raise ValueError("assistant text merges with the header")
        ids += [IM_START] + body + [IM_END] + nl
        mask += [0] * (1 + len(head)) + [learn] * (len(body) - len(head) + 1) + [0] * len(nl)
    return ids, mask


def render_text(messages: list[dict], add_generation_prompt: bool = False) -> str:
    """Human-readable form with literal special-token strings (for docs and debugging; ids come from `render_ids`)."""
    _check(messages)
    out = "<|bos|>" + "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")
