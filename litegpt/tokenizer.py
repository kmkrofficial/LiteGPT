"""LiteGPT tokenizer: 32,768-entry byte-level BPE (ADR-008, spec `tokenizer.*`).

Ids 0-4 are the named specials, ids 5-36 are `<|reserved_0|>` ... `<|reserved_31|>`; the rest is the learned vocabulary.
Special-token text appearing inside documents is encoded as ordinary text (`encode_special_tokens=True`), so a web page that
contains "<|endoftext|>" can never inject a control id; special ids are only added explicitly by `encode_document` and the chat
renderer. `train_bpe` is the only training entry point.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

SPECIAL_NAMES = ["<|bos|>", "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>"]
RESERVED_NAMES = [f"<|reserved_{i}|>" for i in range(32)]
ALL_SPECIALS = SPECIAL_NAMES + RESERVED_NAMES
BOS, EOT, IM_START, IM_END, PAD = range(5)
N_SPECIAL = len(ALL_SPECIALS)            # 37
VOCAB_SIZE = 32768
DEFAULT_PATH = Path(__file__).resolve().parent / "assets" / "tokenizer" / "litegpt-32k.json"
# Qwen2-style pre-tokenizer: contractions, letters, single digits, punctuation runs, newlines, whitespace.
QWEN2_REGEX = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


def _build_empty():
    from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN2_REGEX), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    return tok


def train_bpe(text_iterator, vocab_size: int = VOCAB_SIZE, show_progress: bool = False):
    """Train a tokenizer from an iterator of strings. Returns a `tokenizers.Tokenizer` (not yet wrapped)."""
    from tokenizers import pre_tokenizers, trainers
    tok = _build_empty()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=ALL_SPECIALS, show_progress=show_progress,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(text_iterator, trainer=trainer)
    return tok


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _byte_decoder() -> dict:
    """Inverse of the GPT-2 byte-to-unicode map used by the ByteLevel pre-tokenizer."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class LiteTokenizer:
    def __init__(self, tok, path: str | None = None):
        from tokenizers import Tokenizer  # noqa: F401
        self.tok = tok
        self.tok.encode_special_tokens = True
        self.path = path
        self.vocab_size = tok.get_vocab_size()
        for i, name in enumerate(ALL_SPECIALS):
            if tok.token_to_id(name) != i:
                raise ValueError(f"special token {name} must have id {i}, found {tok.token_to_id(name)}")

    @classmethod
    def load(cls, path=None) -> "LiteTokenizer":
        from tokenizers import Tokenizer
        p = Path(path or DEFAULT_PATH)
        return cls(Tokenizer.from_file(str(p)), str(p))

    def save(self, path) -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.tok.save(str(path))
        return file_sha256(path)

    def encode_text(self, text: str) -> list[int]:
        """Ordinary text. Special-token strings inside the text are ordinary characters."""
        return self.tok.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self.tok.encode_batch(texts, add_special_tokens=False)]

    def encode_document(self, text: str) -> list[int]:
        return [BOS] + self.encode_text(text) + [EOT]

    def decode(self, ids) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=False)

    def token_byte_lengths(self) -> np.ndarray:
        """UTF-8 byte length contributed by each id (0 for special/reserved ids), for bits-per-byte accounting."""
        dec = _byte_decoder()
        out = np.zeros(self.vocab_size, dtype=np.int32)
        for token, i in self.tok.get_vocab().items():
            if i >= N_SPECIAL:
                out[i] = sum(1 for ch in token if ch in dec)
        return out
