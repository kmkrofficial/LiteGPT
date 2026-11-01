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


def doc_windows(ids: list[int], window: int) -> list[tuple[list[int], list[int]]]:
    """The windows doc_nll scores, as (inputs, targets) with -1 for targets scored by an earlier window."""
    stride = max(window // 2, 1)
    out, scored, n = [], 1, len(ids)
    while scored < n:
        end = min(n, scored + stride)
        start = max(0, end - window - 1)
        chunk = ids[start:end]
        skip = scored - start - 1
        out.append((chunk[:-1], [-1] * skip + chunk[1 + skip:]))
        scored = end
    return out


@torch.no_grad()
def val_text_bpb(model, encoder: TextEncoder, docs: list[str], window: int, device, batch_size: int = 0) -> dict:
    """batch_size 0: one window at a time (doc_nll). Otherwise the same windows of every document, right-padded to `window`
    and scored `batch_size` rows at a time at one fixed shape (compile-friendly); padding is causal-invisible and ignored."""
    nbytes = sum(len(text.encode("utf-8")) for text in docs)
    if not batch_size:
        nll = sum(doc_nll(model, encoder.encode(text), window, device) for text in docs)
        return {"bpb": nll / math.log(2) / max(nbytes, 1), "bytes": nbytes, "docs": len(docs)}
    rows = [w for text in docs for w in doc_windows(encoder.encode(text), window)]
    rows += [([0], [-1])] * (-len(rows) % batch_size)                 # fill the last batch: same shape, nothing scored
    nll = torch.zeros((), dtype=torch.float64, device=device)
    for i in range(0, len(rows), batch_size):
        x = torch.zeros(batch_size, window, dtype=torch.long)
        y = torch.full((batch_size, window), -1, dtype=torch.long)
        for r, (inp, tgt) in enumerate(rows[i:i + batch_size]):
            x[r, :len(inp)] = torch.tensor(inp)
            y[r, :len(tgt)] = torch.tensor(tgt)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _ = model(x, logits_all=True)
        for r in range(batch_size):                                    # float32 logits one row at a time (see token_nll)
            nll += F.cross_entropy(logits[r].float(), y[r], ignore_index=-1, reduction="sum").double()
    return {"bpb": float(nll) / math.log(2) / max(nbytes, 1), "bytes": nbytes, "docs": len(docs)}


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
