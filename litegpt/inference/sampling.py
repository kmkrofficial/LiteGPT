"""Logits processors and token selection (penalties, temperature, top-k, top-p, min-p, greedy).

All functions take logits of shape (B, V) in float32 and return a new tensor; `generated` is a list of per-row token lists
(the tokens produced so far, used by the penalties).
"""
from __future__ import annotations

import torch


def apply_penalties(logits: torch.Tensor, generated: list[list[int]], repetition: float = 1.0, frequency: float = 0.0,
                    presence: float = 0.0) -> torch.Tensor:
    if repetition == 1.0 and frequency == 0.0 and presence == 0.0:
        return logits
    out = logits.clone()
    for b, toks in enumerate(generated):
        if not toks:
            continue
        ids, counts = torch.unique(torch.tensor(toks, device=logits.device), return_counts=True)
        row = out[b, ids]
        if repetition != 1.0:                                   # CTRL-style: shrink positive, grow negative logits
            row = torch.where(row > 0, row / repetition, row * repetition)
        row = row - frequency * counts.to(row.dtype) - presence
        out[b, ids] = row
    return out


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.size(-1):
        return logits
    kth = torch.topk(logits, k, dim=-1).values[:, -1:]
    return logits.masked_fill(logits < kth, float("-inf"))


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits
    sorted_logits, idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum = probs.cumsum(dim=-1)
    remove = (cum - probs) > p                                   # keep the smallest prefix whose mass reaches p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, idx, sorted_logits)


def apply_min_p(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    if min_p <= 0.0:
        return logits
    probs = torch.softmax(logits, dim=-1)
    thresh = probs.max(dim=-1, keepdim=True).values * min_p
    return logits.masked_fill(probs < thresh, float("-inf"))


def select_next(logits: torch.Tensor, generated: list[list[int]], cfg, generator: torch.Generator | None = None) -> torch.Tensor:
    """Return next token ids (B,) after penalties and, for sampling, temperature / top-k / top-p / min-p."""
    logits = apply_penalties(logits.float(), generated, cfg.repetition_penalty, cfg.frequency_penalty, cfg.presence_penalty)
    if cfg.temperature <= 0.0:
        return logits.argmax(dim=-1)
    logits = logits / cfg.temperature
    logits = apply_top_k(logits, cfg.top_k)
    logits = apply_top_p(logits, cfg.top_p)
    logits = apply_min_p(logits, cfg.min_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)
