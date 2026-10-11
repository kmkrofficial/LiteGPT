"""Training-data plumbing (ADR-008): partitions, uint16 shards, mixture index, resumable loader, bits-per-byte.

Tokens live once, in per-source uint16 shard files. The training order is a separate small index of (source, block) pairs built
from the mixture weights, so the stable and decay streams, validation sets and proxy subsets are all views over the same shards.
`MixtureLoader.position` is a single integer: saving it in `data_position` of a checkpoint gives exact resume.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

BLOCK_TOKENS = 2048
_U_SCALE = float(2 ** 64)


# --- partitions ------------------------------------------------------------------------------------------------------
def partition_u(salt: str, source: str, key: str, suffix: str = "") -> float:
    d = hashlib.sha256(f"{salt}:{source}:{key}{suffix}".encode("utf-8")).digest()
    return int.from_bytes(d[:8], "big") / _U_SCALE


def assign_partition(u: float, val_fraction: float, decay_fraction: float, decay_eligible: bool = True) -> str:
    if u < val_fraction:
        return "val"
    if decay_eligible and u < val_fraction + decay_fraction:
        return "decay"
    return "stable"


# --- shards ----------------------------------------------------------------------------------------------------------
class ShardWriter:
    """Appends uint16 token arrays to numbered .bin files of about `shard_tokens` each; records counts and sha256."""

    def __init__(self, directory, prefix: str = "shard", shard_tokens: int = 256_000_000):
        self.dir, self.prefix, self.shard_tokens = Path(directory), prefix, shard_tokens
        self.dir.mkdir(parents=True, exist_ok=True)
        self.files: list[dict] = []
        self._buf: list[np.ndarray] = []
        self._n = 0

    def add(self, ids) -> None:
        a = np.asarray(ids, dtype=np.uint16)
        self._buf.append(a)
        self._n += len(a)
        if self._n >= self.shard_tokens:
            self._flush()

    def _flush(self) -> None:
        if not self._n:
            return
        arr = np.concatenate(self._buf)
        path = self.dir / f"{self.prefix}-{len(self.files):05d}.bin"
        arr.tofile(path)
        self.files.append({"file": path.name, "tokens": int(len(arr)), "sha256": hashlib.sha256(arr.tobytes()).hexdigest()})
        self._buf, self._n = [], 0

    def close(self) -> list[dict]:
        self._flush()
        return self.files


class ShardSet:
    """Read-only view of several .bin files as one token stream."""

    def __init__(self, paths):
        self.paths = [Path(p) for p in paths]
        self.maps = [np.memmap(p, dtype=np.uint16, mode="r") for p in self.paths]
        self.lengths = [len(m) for m in self.maps]
        self.starts = np.concatenate(([0], np.cumsum(self.lengths)))[:-1] if self.maps else np.empty(0, dtype=np.int64)

    def __len__(self) -> int:
        return int(sum(self.lengths))

    def read(self, start: int, n: int) -> np.ndarray:
        if start < 0 or start + n > len(self):
            raise IndexError(f"read [{start}, {start + n}) outside stream of {len(self)} tokens")
        out = np.empty(n, dtype=np.uint16)
        pos, filled = start, 0
        while filled < n:
            i = int(np.searchsorted(self.starts, pos, side="right")) - 1
            off = pos - int(self.starts[i])
            take = min(n - filled, self.lengths[i] - off)
            out[filled:filled + take] = self.maps[i][off:off + take]
            pos += take
            filled += take
        return out


def block_count(n_tokens: int, block_len: int = BLOCK_TOKENS) -> int:
    """Blocks of `block_len` inputs; each block reads block_len + 1 tokens for the target shift."""
    return max((n_tokens - 1) // block_len, 0)


# --- mixture index ---------------------------------------------------------------------------------------------------
def build_index(n_blocks: dict, weights: dict, total_blocks: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Smooth weighted round-robin over sources (exact proportions at every prefix, no randomness) with a seeded
    permutation of each source's blocks. Returns (source_id uint8, block_id uint32, source names in id order)."""
    names = sorted(weights)
    total_w = sum(weights[n] for n in names)
    w = [weights[n] / total_w for n in names]
    k = len(names)
    need = [0] * k
    cur = [0.0] * k
    src = np.empty(total_blocks, dtype=np.uint8)
    rng_k = range(k)
    for i in range(total_blocks):
        for j in rng_k:
            cur[j] += w[j]
        pick = max(rng_k, key=cur.__getitem__)      # ties go to the lowest index: deterministic
        cur[pick] -= 1.0
        src[i] = pick
        need[pick] += 1
    need = dict(zip(names, need))
    for n in names:
        if need[n] > n_blocks.get(n, 0):
            raise ValueError(f"source {n!r} needs {need[n]} blocks but has {n_blocks.get(n, 0)}")
    blk = np.empty(total_blocks, dtype=np.uint32)
    for si, n in enumerate(names):
        perm = np.random.default_rng([seed, si]).permutation(n_blocks[n])[: need[n]]
        blk[src == si] = perm
    return src, blk, names


def index_sha256(src, blk, names) -> str:
    return hashlib.sha256(src.tobytes() + blk.tobytes() + "|".join(names).encode()).hexdigest()


def save_index(path, src, blk, names, meta: dict | None = None) -> str:
    np.savez(path, src=src, blk=blk, names=np.array(names), meta=np.array(json.dumps(meta or {})))
    return index_sha256(src, blk, names)


def load_index(path):
    z = np.load(path, allow_pickle=False)
    return z["src"], z["blk"], [str(n) for n in z["names"]], json.loads(str(z["meta"]))


# --- loader ----------------------------------------------------------------------------------------------------------
class MixtureLoader:
    """Deterministic batches from an index. `position` counts blocks consumed and is the whole resumable state."""

    def __init__(self, src, blk, names, shards: dict, batch_size: int, block_len: int = BLOCK_TOKENS, position: int = 0):
        self.src, self.blk, self.names, self.shards = src, blk, names, shards
        self.batch_size, self.block_len, self.position = batch_size, block_len, position

    def state_dict(self) -> dict:
        return {"position": self.position, "block_len": self.block_len, "batch_size": self.batch_size}

    def load_state_dict(self, state: dict) -> None:
        if state["block_len"] != self.block_len:
            raise ValueError("block_len mismatch on resume")
        self.position = int(state["position"])

    def block(self, i: int) -> np.ndarray:
        s = self.names[int(self.src[i])]
        b = int(self.blk[i])
        return self.shards[s].read(b * self.block_len, self.block_len + 1)

    def next_batch(self):
        import torch
        if self.position + self.batch_size > len(self.src):
            raise StopIteration("index exhausted")
        rows = np.stack([self.block(self.position + k) for k in range(self.batch_size)]).astype(np.int64)
        self.position += self.batch_size
        t = torch.from_numpy(rows)
        return t[:, :-1].contiguous(), t[:, 1:].contiguous()


# --- evaluation helpers ----------------------------------------------------------------------------------------------
def bits_per_byte(nll_sum_nats: float, n_bytes: int) -> float:
    return nll_sum_nats / math.log(2) / max(n_bytes, 1)


def target_bytes(token_ids: np.ndarray, token_byte_lengths: np.ndarray) -> int:
    """UTF-8 bytes covered by target tokens; special / reserved ids count 0."""
    return int(token_byte_lengths[token_ids].sum())
