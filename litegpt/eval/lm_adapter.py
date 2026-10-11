"""Native LiteGPT model for lm-evaluation-harness (EXP-M1B-SUITE).

Implements lm-eval's `LM` interface on top of `litegpt.model.LiteGPT`, `litegpt.tokenizer.LiteTokenizer` and
`litegpt.inference.Engine`, so M2 proxies and V2 checkpoints are scored by exactly the same tasks as the HF baselines.
Tokenisation follows the lm-eval HF convention: the continuation tokens are the tail of encode(context + continuation) beyond
encode(context); every sequence starts with <|bos|>.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from litegpt.inference import Engine, GenerationConfig
from litegpt.tokenizer import BOS, EOT, IM_END


def _split(tok, context: str, continuation: str) -> tuple[list[int], list[int]]:
    whole = tok.encode_text(context + continuation)
    ctx = tok.encode_text(context)
    n = len(ctx)
    if whole[:n] != ctx:                         # merge across the boundary: fall back to separate encoding
        return ctx, tok.encode_text(continuation)
    return ctx, whole[n:]


@torch.no_grad()
def score_continuations(model, tok, pairs: list[tuple[str, str]], batch_size: int = 16, device=None) -> list[tuple[float, bool]]:
    """Sum of log-probabilities of each continuation given its context, and whether it is the greedy continuation."""
    device = device or next(model.parameters()).device
    maxlen = model.config.block_size
    reqs = []
    for i, (c, x) in enumerate(pairs):
        ctx, cont = _split(tok, c, x)
        seq = ([BOS] + ctx + cont)[-(maxlen + 1):]
        reqs.append((i, seq, len(cont)))
    order = sorted(range(len(reqs)), key=lambda k: -len(reqs[k][1]))
    out: list = [None] * len(reqs)
    for s in range(0, len(order), batch_size):
        chunk = [reqs[k] for k in order[s:s + batch_size]]
        T = max(len(seq) for _, seq, _ in chunk) - 1
        inp = torch.zeros((len(chunk), T), dtype=torch.long, device=device)
        for r, (_, seq, _) in enumerate(chunk):
            inp[r, :len(seq) - 1] = torch.tensor(seq[:-1], device=device)       # right padding: causal, so harmless
        logits, _ = model(inp, logits_all=True)
        logp = F.log_softmax(logits.float(), dim=-1)
        for r, (i, seq, n) in enumerate(chunk):
            L = len(seq) - 1
            tgt = torch.tensor(seq[L - n + 1:L + 1], device=device)
            lp = logp[r, L - n:L, :]
            out[i] = (float(lp.gather(-1, tgt[:, None]).sum()), bool((lp.argmax(-1) == tgt).all()))
    return out


@torch.no_grad()
def rolling_loglikelihood(model, tok, text: str, device=None) -> float:
    """Total log-likelihood of `text` with disjoint context windows (lm-eval `loglikelihood_rolling`)."""
    device = device or next(model.parameters()).device
    ids = [BOS] + tok.encode_text(text)
    W = model.config.block_size
    total = 0.0
    pos = 1
    while pos < len(ids):
        start = max(0, pos - W)
        end = min(len(ids), start + W + 1)
        window = ids[start:end]
        logits, _ = model(torch.tensor([window[:-1]], device=device), logits_all=True)
        logp = F.log_softmax(logits.float(), dim=-1)[0]
        tgt = torch.tensor(window[1:], device=device)
        lp = logp.gather(-1, tgt[:, None])[:, 0]
        first = pos - start - 1
        total += float(lp[first:].sum())
        pos = end
    return total


def make_lm_class():
    """Build the lm-eval LM subclass lazily, so importing this module does not require lm-eval."""
    from lm_eval.api.model import LM

    class LiteGPTLM(LM):
        def __init__(self, model, tok, batch_size: int = 16, max_gen_toks: int = 256):
            super().__init__()
            self.model, self.tok, self.bs, self.max_gen_toks = model.eval(), tok, batch_size, max_gen_toks
            self.engine = Engine(model, tok)

        def loglikelihood(self, requests):
            return score_continuations(self.model, self.tok, [r.args for r in requests], self.bs)

        def loglikelihood_rolling(self, requests):
            return [rolling_loglikelihood(self.model, self.tok, r.args[0]) for r in requests]

        def generate_until(self, requests):
            outs = []
            for r in requests:
                ctx, kw = r.args
                until = kw.get("until") or []
                until = [until] if isinstance(until, str) else list(until)
                n = int(kw.get("max_gen_toks", self.max_gen_toks))
                prompt = ([BOS] + self.tok.encode_text(ctx))[-(self.model.config.block_size - n):]
                cfg = GenerationConfig(max_new_tokens=n, temperature=0.0, stop_token_ids=(EOT, IM_END), stop_strings=tuple(until))
                outs.append(self.engine.generate_ids([prompt], cfg)[0].text)
            return outs

    return LiteGPTLM
