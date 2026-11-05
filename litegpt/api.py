"""Offline Python API on the inference engine: load a checkpoint, then generate, chat, stream or extract.

    import litegpt
    m = litegpt.load("weights.pt")
    litegpt.generate(m, "Once upon a time", max_new_tokens=32)
    litegpt.chat(m, "What is a contract?", system="Be brief.")
    litegpt.extract(m, "banking77", "My card has not arrived.")

Options are the fields of `GenerationConfig`. Greedy by default. Nothing here reaches the network. A checkpoint is read with
`weights_only=True`, so a file from someone else cannot run code when it is loaded.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from litegpt import chat_template
from litegpt import t2d as X
from litegpt.config import ModelConfig
from litegpt.inference import Engine, GenerationConfig
from litegpt.model import LiteGPT
from litegpt.tokenizer import BOS, LiteTokenizer

EXTRACT_NEW_TOKENS = 256        # EXP-M4-T2D recipe
CHECKPOINT_GLOBALS = [np._core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.UInt32DType]     # a training checkpoint's numpy RNG state; nothing else is allowed


@dataclass
class LoadedModel:
    model: LiteGPT
    tok: LiteTokenizer
    path: Path
    step: int
    tokens: int
    layout: str | None          # how a fine-tuned checkpoint was trained to read its prompt; None if it is not one
    arm: str | None

    @property
    def engine(self) -> Engine:
        return Engine(self.model, self.tok)


def load(path, device=None, dtype=torch.bfloat16) -> LoadedModel:
    """A LiteGPT checkpoint in eval mode, on the GPU if there is one. The optimizer state is not loaded."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no checkpoint at {path}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    with torch.serialization.safe_globals(CHECKPOINT_GLOBALS):
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    model = LiteGPT(ModelConfig.from_dict({**state["model_config"], "attention_backend": "sdpa"}))
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in state["model"].items()})
    arm = (state.get("t2d") or {}).get("arm")
    layout = None if arm is None else "chat" if arm.endswith("-chat") else "plain"
    return LoadedModel(model.to(device=device, dtype=dtype).eval(), LiteTokenizer.load(), path, state.get("step", 0), state.get("tokens", 0), layout, arm)


def _config(m: LoadedModel, prompt: list[int], options: dict) -> GenerationConfig:
    cfg = GenerationConfig(**options)
    if len(prompt) + cfg.max_new_tokens > m.model.config.block_size:
        raise ValueError(f"prompt of {len(prompt)} tokens + max_new_tokens {cfg.max_new_tokens} exceeds the context of {m.model.config.block_size}")
    return cfg


def _complete(m: LoadedModel, prompt: list[int], options: dict) -> tuple[str, str]:
    r = m.engine.generate_ids([prompt], _config(m, prompt, options))[0]
    return r.text, r.finish_reason


def _chat_ids(m: LoadedModel, messages, system: str | None) -> list[int]:
    if m.layout == "plain":
        raise ValueError(f"{m.arm} was fine-tuned on plain prompts, not chat: use generate or extract")
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if system is not None:
        messages = [{"role": "system", "content": system}, *messages]
    return chat_template.render_ids(m.tok, messages, add_generation_prompt=True)


def generate(m: LoadedModel, prompt: str, **options) -> str:
    """Continue plain text."""
    return _complete(m, [BOS] + m.tok.encode_text(prompt), options)[0]


def chat(m: LoadedModel, messages, system: str | None = None, **options) -> str:
    """The assistant's reply to a user string or a list of {role, content} messages. Only a chat-trained checkpoint is any use here."""
    return _complete(m, _chat_ids(m, messages, system), options)[0]


def chat_stream(m: LoadedModel, messages, system: str | None = None, **options) -> Iterator[str]:
    """`chat`, as text pieces while they are written."""
    prompt = _chat_ids(m, messages, system)
    return m.engine.stream(prompt, _config(m, prompt, options))


def extract(m: LoadedModel, task: str, text: str) -> dict:
    """One document through a registered EXP-M4-T2D task, with the exact prompt and greedy decoding the checkpoint was scored with.
    Works only on a checkpoint fine-tuned for these tasks. Invalid JSON gives `data` None and `valid` False: nothing is repaired."""
    if task not in X.PRIMARY:
        raise ValueError(f"unknown task {task!r}: choose from {', '.join(X.PRIMARY)}")
    if m.layout is None:
        raise ValueError("this checkpoint was not fine-tuned for text-to-data: use the weights of a fine-tune run")
    user = X.user_text(task, {"input": text})
    if m.layout == "chat":
        prompt = chat_template.render_ids(m.tok, [{"role": "user", "content": user}], add_generation_prompt=True)
    else:
        prompt = [BOS] + m.tok.encode_text(user + X.ANSWER_CUE)
    output, finish = _complete(m, prompt, {"max_new_tokens": EXTRACT_NEW_TOKENS})
    data = output.strip() if task == "banking77" else X.first_json(output)
    return {"task": task, "output": output, "data": data, "valid": bool(data) if task == "banking77" else data is not None, "finish_reason": finish}


def info(m: LoadedModel) -> dict:
    cfg = m.model.config
    p = next(m.model.parameters())
    return {"path": str(m.path), "parameters": sum(x.numel() for x in m.model.parameters()), "layers": cfg.n_layer, "embedding": cfg.n_embd, "context": cfg.block_size,
            "vocab_size": cfg.vocab_size, "step": m.step, "tokens": m.tokens, "fine_tune_arm": m.arm, "layout": m.layout, "device": str(p.device), "dtype": str(p.dtype).removeprefix("torch.")}


def dumps(value) -> str:
    return json.dumps(value, indent=1, ensure_ascii=False)
