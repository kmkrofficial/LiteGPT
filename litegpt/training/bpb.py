"""Bits per byte for training-time evaluation: token-level validation sets and raw-text `val_text` (any tokenizer)."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

GPT2_EOT = 50256


def gpt2_tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_pretrained("openai-community/gpt2")


def token_byte_lengths(kind: str, vocab_size: int) -> np.ndarray:
    """UTF-8 bytes per token id (0 for special ids and padding ids beyond the real vocabulary)."""
    if kind == "litegpt":
        from litegpt.tokenizer import LiteTokenizer
        nb = LiteTokenizer.load().token_byte_lengths()
    else:
        from litegpt.tokenizer import _byte_decoder
        dec = _byte_decoder()
        tok = gpt2_tokenizer()
        nb = np.zeros(tok.get_vocab_size(), dtype=np.int32)
        for t, i in tok.get_vocab().items():
            nb[i] = 0 if i == GPT2_EOT else sum(1 for ch in t if ch in dec)
    out = np.zeros(vocab_size, dtype=np.int32)
    out[: len(nb)] = nb[:vocab_size]
    return out


class TextEncoder:
    """Encodes raw validation text for a tokenizer kind; returns ids with the document-start token the model was trained with."""

    def __init__(self, kind: str):
        self.kind = kind
        if kind == "litegpt":
            from litegpt.tokenizer import BOS, LiteTokenizer
            self.tok, self.bos = LiteTokenizer.load(), BOS
        else:
            self.tok, self.bos = gpt2_tokenizer(), GPT2_EOT          # GPT-2 documents are separated by <|endoftext|>

    def encode(self, text: str) -> list[int]:
        ids = self.tok.encode_text(text) if self.kind == "litegpt" else self.tok.encode(text, add_special_tokens=False).ids
        return [self.bos] + ids


@torch.no_grad()
def doc_nll(model, ids: list[int], window: int, device) -> float:
    """Sum of NLL (nats) of ids[1:], sliding window of `window` tokens, stride window // 2."""
    stride = max(window // 2, 1)
    total, scored, n = 0.0, 1, len(ids)
    while scored < n:
        end = min(n, scored + stride)
        start = max(0, end - window - 1)
        chunk = torch.tensor([ids[start:end]], device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _ = model(chunk[:, :-1], logits_all=True)
        lp = F.log_softmax(logits.float(), dim=-1)[0]
        tgt = chunk[0, 1:]
        tok_lp = lp.gather(-1, tgt[:, None])[:, 0]
        total -= float(tok_lp[scored - start - 1:].sum())
        scored = end
    return total


@torch.no_grad()
def val_text_bpb(model, encoder: TextEncoder, docs: list[str], window: int, device) -> dict:
    nll = 0.0
    nbytes = 0
    for text in docs:
        nll += doc_nll(model, encoder.encode(text), window, device)
        nbytes += len(text.encode("utf-8"))
    return {"bpb": nll / math.log(2) / max(nbytes, 1), "bytes": nbytes, "docs": len(docs)}


def token_nll(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-token NLL (float32) of a (B, T, V) batch, flattened to (B*T,). Computed one sequence at a time: identical values to
    one batched call, but the float32 copy of the logits is T x V instead of B x T x V (3 GiB at 8 x 2048 x 50k)."""
    return torch.cat([F.cross_entropy(logits[i].float(), y[i], reduction="none") for i in range(logits.size(0))])


@torch.no_grad()
def token_val(model, loader, n_batches: int, byte_lengths: np.ndarray, device) -> dict:
    """Mean loss and BPB over the first `n_batches` batches of a validation index (special tokens excluded from BPB)."""
    loss_sum = nll_bytes = 0.0
    n_tok = n_bytes = 0
    loader.position = 0
    nb_t = torch.as_tensor(byte_lengths, device=device)
    for _ in range(n_batches):
        try:
            x, y = loader.next_batch()
        except StopIteration:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _ = model(x, logits_all=True)
        nll = token_nll(logits, y)
        b = nb_t[y.view(-1)]
        loss_sum += float(nll.sum())
        n_tok += nll.numel()
        nll_bytes += float((nll * (b > 0)).sum())
        n_bytes += int(b.sum())
    return {"loss": loss_sum / max(n_tok, 1), "bpb": nll_bytes / math.log(2) / max(n_bytes, 1), "tokens": n_tok}
