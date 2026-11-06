"""Generation engine: prefill + incremental decode on the model's KV cache, batched with left padding.

Tokens in, tokens out. Text handling (chat templates, stop strings) is done here only as far as needed for stop strings and
streaming; the Python API (M4) builds on this.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from litegpt.inference.sampling import select_next
from litegpt.inference.stopping import StreamDecoder, find_stop
from litegpt.model import KVCache
from litegpt.tokenizer import EOT, IM_END, PAD


@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.0            # 0 = greedy
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_token_ids: tuple = (EOT, IM_END)
    stop_strings: tuple = ()
    seed: int | None = None


@dataclass
class GenerationResult:
    ids: list[int]
    finish_reason: str                  # "stop" | "length"
    text: str | None = None
    prompt_tokens: int = 0
    usage: dict = field(default_factory=dict)


class Engine:
    def __init__(self, model, tokenizer=None, device=None, dtype=None):
        self.model = model.eval()
        self.tok = tokenizer
        p = next(model.parameters())
        self.device = device or p.device
        self.dtype = dtype or p.dtype

    def _cache(self, batch: int, length: int) -> KVCache:
        return KVCache(self.model.config, batch, max_len=length, device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def generate_ids(self, prompts: list[list[int]], cfg: GenerationConfig | None = None) -> list[GenerationResult]:
        cfg = cfg or GenerationConfig()
        if not prompts or any(len(p) == 0 for p in prompts):
            raise ValueError("every prompt needs at least one token")
        B, L = len(prompts), max(len(p) for p in prompts)
        total = L + cfg.max_new_tokens
        if total > self.model.config.block_size:
            raise ValueError(f"prompt + max_new_tokens = {total} exceeds the context of {self.model.config.block_size}")
        idx = torch.full((B, L), PAD, dtype=torch.long, device=self.device)
        mask = torch.zeros((B, total), dtype=torch.bool, device=self.device)
        for b, p in enumerate(prompts):
            idx[b, L - len(p):] = torch.tensor(p, device=self.device)
            mask[b, L - len(p):L] = True
        gen = torch.Generator(device=self.device).manual_seed(cfg.seed) if cfg.seed is not None else None
        cache = self._cache(B, total)
        padded = bool((~mask[:, :L]).any())
        logits, _ = self.model(idx, kv_cache=cache, start_pos=0, key_padding_mask=mask[:, :L] if padded else None)
        out: list[list[int]] = [[] for _ in range(B)]
        done = [False] * B
        reason = ["length"] * B
        stop_ids = set(cfg.stop_token_ids)
        for step in range(cfg.max_new_tokens):
            nxt = select_next(logits[:, -1, :], out, cfg, gen)
            for b in range(B):
                if done[b]:
                    continue
                t = int(nxt[b])
                if t in stop_ids:
                    done[b], reason[b] = True, "stop"
                    continue
                out[b].append(t)
                if cfg.stop_strings and self.tok is not None:
                    text = self.tok.decode(out[b])
                    if find_stop(text, cfg.stop_strings) is not None:
                        done[b], reason[b] = True, "stop"
            if all(done) or step == cfg.max_new_tokens - 1:
                break
            feed = torch.tensor([[out[b][-1] if (out[b] and not done[b]) else PAD] for b in range(B)], device=self.device)
            pos = L + step
            mask[:, pos] = torch.tensor([not d for d in done], device=self.device)
            logits, _ = self.model(feed, kv_cache=cache, start_pos=pos, key_padding_mask=mask[:, :pos + 1] if padded or any(done) else None)
        results = []
        for b in range(B):
            text = None
            if self.tok is not None:
                text = self.tok.decode(out[b])
                cut = find_stop(text, cfg.stop_strings) if cfg.stop_strings else None
                if cut is not None:
                    text = text[:cut]
            results.append(GenerationResult(out[b], reason[b], text, len(prompts[b]),
                                            {"prompt_tokens": len(prompts[b]), "completion_tokens": len(out[b])}))
        return results

    @torch.no_grad()
    def stream(self, prompt: list[int], cfg: GenerationConfig | None = None):
        """Yield UTF-8-safe text deltas for one prompt (requires a tokenizer)."""
        cfg = cfg or GenerationConfig()
        if self.tok is None:
            raise ValueError("streaming needs a tokenizer")
        dec = StreamDecoder(self.tok.decode, cfg.stop_strings)
        cache = self._cache(1, len(prompt) + cfg.max_new_tokens)
        idx = torch.tensor([prompt], device=self.device)
        logits, _ = self.model(idx, kv_cache=cache, start_pos=0)
        gen = torch.Generator(device=self.device).manual_seed(cfg.seed) if cfg.seed is not None else None
        produced: list[int] = []
        for step in range(cfg.max_new_tokens):
            t = int(select_next(logits[:, -1, :], [produced], cfg, gen)[0])
            if t in cfg.stop_token_ids:
                break
            produced.append(t)
            delta = dec.push(t)
            if delta:
                yield delta
            if dec.stopped:
                return
            logits, _ = self.model(torch.tensor([[t]], device=self.device), kv_cache=cache)
        tail = dec.flush()
        if tail:
            yield tail
