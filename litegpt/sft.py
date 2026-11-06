"""Chat SFT data for M4 (EXP-M4-SFT): prepared conversations and whole-conversation packing.

A prepared split is `<split>.bin` (uint16: token id in the low 15 bits, assistant-only loss mask in the top bit; the vocabulary is
32,768) and `<split>_offsets.npy` (int64 start of each conversation, then the end). Conversations are never cut. Rows have a fixed
length so a compiled graph keeps its shape; the rest of a row is padding, which is its own document and carries no loss.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from litegpt import chat_template, data as D
from litegpt.dataprep import content_hash
from litegpt.tokenizer import PAD, LiteTokenizer

MASK_BIT = 1 << 15
SALT = "litegpt-m4-sft-1"
HELDOUT_FRACTION = 0.01
MAX_TOKENS = 2048
RESERVED = ("smol-summarize", "smol-rewrite", "smol-constraints")
ALL_SUBSETS = RESERVED + ("smol-magpie-ultra",)
DROPS = ("bad_roles", "no_assistant", "assistant_starts_with_newline", "over_max_tokens", "decontam", "decay_partition")


def process_file(path, subset: str, ngrams, partition_salt: str, decay_fraction: float, tok: LiteTokenizer | None = None) -> dict:
    """One parquet file: the kept conversations, encoded, and a count for every rule."""
    import pyarrow.parquet as pq
    tok = tok or LiteTokenizer.load()
    counts = {"read": 0, "kept": 0, "tokens": 0, "bytes": 0, "decontam_bytes": 0, **{k: 0 for k in DROPS}}
    decontam: dict = {}
    out = {"train": [], "heldout": []}
    for row in pq.read_table(path, columns=["messages"]).to_pylist():
        counts["read"] += 1
        msgs = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
        key = content_hash(msgs)
        if subset in RESERVED and D.partition_u(partition_salt, "instruction_style", key) < decay_fraction:
            counts["decay_partition"] += 1          # the same split as M1: these conversations were in the decay slice
            continue
        try:
            raw = chat_template.render_text(msgs)
        except ValueError:
            counts["bad_roles"] += 1
            continue
        if not any(m["role"] == "assistant" for m in msgs):
            counts["no_assistant"] += 1
            continue
        size = len(raw.encode("utf-8", "replace"))
        hit = ngrams.match(raw) if ngrams is not None else None
        if hit is not None:
            counts["decontam"] += 1
            counts["decontam_bytes"] += size
            decontam[ngrams.names[hit]] = decontam.get(ngrams.names[hit], 0) + 1
            continue
        try:
            ids, mask = chat_template.render(tok, msgs)
        except ValueError:
            counts["assistant_starts_with_newline"] += 1
            continue
        if len(ids) > MAX_TOKENS:
            counts["over_max_tokens"] += 1
            continue
        counts["kept"] += 1
        counts["tokens"] += len(ids)
        counts["bytes"] += size
        split = "heldout" if D.partition_u(SALT, "heldout", key) < HELDOUT_FRACTION else "train"
        out[split].append((np.asarray(ids, dtype=np.uint16) | (np.asarray(mask, dtype=np.uint16) << 15)).tobytes())
    return {"subset": subset, "file": Path(path).name, "counts": counts, "decontam": decontam, "train": out["train"], "heldout": out["heldout"]}


def write_split(dest: Path, split: str, conversations: list[bytes]) -> dict:
    offsets = np.zeros(len(conversations) + 1, dtype=np.int64)
    with open(dest / f"{split}.bin", "wb") as f:
        for i, c in enumerate(conversations):
            f.write(c)
            offsets[i + 1] = offsets[i] + len(c) // 2
    np.save(dest / f"{split}_offsets.npy", offsets)
    return {"conversations": len(conversations), "tokens": int(offsets[-1])}


class Conversations:
    """A prepared split, memory-mapped."""

    def __init__(self, data_dir, split: str, keep_fraction: float = 1.0, seed: int = 0):
        d = Path(data_dir)
        self.tokens = np.memmap(d / f"{split}.bin", dtype=np.uint16, mode="r")
        self.offsets = np.load(d / f"{split}_offsets.npy")
        n = len(self.offsets) - 1
        self.index = np.arange(n)
        if keep_fraction < 1.0:                     # the sweep's fixed subset: the same conversations for every learning rate
            self.index = np.sort(np.random.default_rng(seed).permutation(n)[:max(1, round(n * keep_fraction))])
        self.lengths = (self.offsets[1:] - self.offsets[:-1])[self.index]

    def __len__(self) -> int:
        return len(self.index)

    def get(self, i: int) -> np.ndarray:
        j = self.index[i]
        return self.tokens[self.offsets[j]:self.offsets[j + 1]]


def pack_rows(lengths, row_len: int, order, window: int = 256) -> list[list[int]]:
    """Fill fixed-length rows with whole conversations, in `order`. Each conversation goes in the first of the last `window` rows it
    fits, so the shuffled order mostly survives. Pure and deterministic."""
    rows: list[list[int]] = []
    free: list[int] = []
    for i in order:
        n = int(lengths[i])
        if n > row_len:
            raise ValueError(f"conversation of {n} tokens does not fit a row of {row_len}")
        for r in range(max(0, len(rows) - window), len(rows)):
            if free[r] >= n:
                rows[r].append(int(i))
                free[r] -= n
                break
        else:
            rows.append([int(i)])
            free.append(row_len - n)
    return rows


def epoch_rows(lengths, row_len: int, seed: int, epoch: int, batch_rows: int) -> list[list[int]]:
    """One epoch's rows. The order depends on the seed and the epoch only, so a resume rebuilds it exactly. Rows are shuffled again
    so short conversations placed late do not cluster, and the last incomplete batch is dropped."""
    rng = random.Random(f"{seed}:{epoch}")
    order = list(range(len(lengths)))
    rng.shuffle(order)
    rows = pack_rows(lengths, row_len, order)
    rng.shuffle(rows)
    return rows[:len(rows) // batch_rows * batch_rows]


@dataclass
class PackedBatch:
    x: torch.Tensor
    y: torch.Tensor
    cu_seqlens: torch.Tensor
    doc: torch.Tensor
    position_ids: torch.Tensor
    real_tokens: int
    target_tokens: int


def make_batch(conv: Conversations, rows: list[list[int]], row_len: int, device) -> PackedBatch:
    """Inputs, targets (-1 = no loss) and document layout. Attention never crosses a conversation (`cu_seqlens`), and the target at a
    position is the next token only when that token is in the same conversation and marked for learning."""
    B, T = len(rows), row_len
    x = np.full((B, T), PAD, dtype=np.int64)
    y = np.full((B, T), -1, dtype=np.int64)
    doc = np.zeros((B, T), dtype=np.int64)
    pos = np.zeros((B, T), dtype=np.int64)
    cu = [0]
    real, d = 0, 0
    for r, row in enumerate(rows):
        at = 0
        for i in row:
            a = np.asarray(conv.get(i))
            n = len(a)
            ids = (a & (MASK_BIT - 1)).astype(np.int64)
            x[r, at:at + n] = ids
            y[r, at:at + n - 1] = np.where(a[1:] >= MASK_BIT, ids[1:], -1)
            doc[r, at:at + n] = d
            pos[r, at:at + n] = np.arange(n)
            at += n
            real += n
            d += 1
            cu.append(r * T + at)
        if at < T:
            doc[r, at:] = d
            pos[r, at:] = np.arange(T - at)
            d += 1
            cu.append((r + 1) * T)
    to = lambda a: torch.from_numpy(a).to(device, non_blocking=True)  # noqa: E731
    return PackedBatch(to(x), to(y), torch.tensor(cu, dtype=torch.int32, device=device), to(doc), to(pos), real, int((y >= 0).sum()))


def read_manifest(data_dir) -> dict:
    return json.loads((Path(data_dir) / "manifest.json").read_text(encoding="utf-8"))
