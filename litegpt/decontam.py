"""13-gram word-overlap decontamination (ADR-008, EXP-M1-DECONTAM).

Text is lowercased and split into word characters; every 13-word window is hashed (crc32 per word, 64-bit polynomial over the
window). A document is contaminated if any of its windows is in the reference set. The same code filters pretraining data and,
later, SFT / reasoning / DPO data.
"""
from __future__ import annotations

import re
import zlib

import numpy as np

NGRAM = 13
_WORD = re.compile(r"\w+")
_MULT = np.uint64(0x9E3779B97F4A7C15)


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def window_hashes(text: str, n: int = NGRAM) -> np.ndarray:
    ws = words(text)
    m = len(ws) - n + 1
    if m <= 0:
        return np.empty(0, dtype=np.uint64)
    codes = np.fromiter((zlib.crc32(w.encode("utf-8")) for w in ws), dtype=np.uint64, count=len(ws))
    h = np.zeros(m, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(n):
            h = h * _MULT + codes[j:j + m]
    return h


class NgramIndex:
    """Sorted hashes with the id of the first benchmark that contributed each hash."""

    def __init__(self, hashes: np.ndarray, bench_ids: np.ndarray, names: list[str]):
        self.hashes, self.bench_ids, self.names = hashes, bench_ids, names

    @classmethod
    def from_texts(cls, named_texts: dict) -> "NgramIndex":
        names = list(named_texts)
        all_h, all_b = [], []
        for bi, name in enumerate(names):
            for t in named_texts[name]:
                h = window_hashes(t)
                if len(h):
                    all_h.append(h)
                    all_b.append(np.full(len(h), bi, dtype=np.int16))
        if not all_h:
            return cls(np.empty(0, np.uint64), np.empty(0, np.int16), names)
        h, b = np.concatenate(all_h), np.concatenate(all_b)
        order = np.argsort(h, kind="stable")
        h, b = h[order], b[order]
        keep = np.concatenate(([True], h[1:] != h[:-1]))
        return cls(h[keep], b[keep], names)

    def __len__(self) -> int:
        return len(self.hashes)

    def match(self, text: str) -> int | None:
        """Index of the first benchmark with a shared 13-gram, or None."""
        if not len(self.hashes):
            return None
        h = window_hashes(text)
        if not len(h):
            return None
        pos = np.searchsorted(self.hashes, h)
        pos[pos == len(self.hashes)] = 0
        hit = self.hashes[pos] == h
        return int(self.bench_ids[pos[hit][0]]) if hit.any() else None

    def save(self, path) -> None:
        np.savez_compressed(path, hashes=self.hashes, bench_ids=self.bench_ids, names=np.array(self.names))

    @classmethod
    def load(cls, path) -> "NgramIndex":
        z = np.load(path, allow_pickle=False)
        return cls(z["hashes"], z["bench_ids"], [str(n) for n in z["names"]])
