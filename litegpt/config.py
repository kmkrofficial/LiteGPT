"""Model configuration for LiteGPT V2. Source of truth for approved values: docs/spec/v2.yaml."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

CONFIG_VERSION = 1
ATTENTION_BACKENDS = ("auto", "sdpa", "flash")


@dataclass
class ModelConfig:
    block_size: int = 2048
    vocab_size: int = 32768
    n_layer: int = 28
    n_embd: int = 768
    n_head: int = 12
    n_kv_head: int = 4
    ffn_hidden: int = 2048
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    qk_norm: bool = True
    tie_embeddings: bool = True
    dropout: float = 0.0
    attention_backend: str = "auto"  # auto | sdpa | flash. 'auto' == sdpa until flash is adopted (ADR-006)
    init_std: float = 0.02
    config_version: int = field(default=CONFIG_VERSION)

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        assert self.attention_backend in ATTENTION_BACKENDS, f"attention_backend must be one of {ATTENTION_BACKENDS}"
        assert (self.n_embd // self.n_head) % 2 == 0, "head_dim must be even for RoPE"

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


PRESETS = {
    "litegpt-v2-28l": ModelConfig(),
    "litegpt-v2-24l": ModelConfig(n_layer=24),
    # tiny config for CPU tests / smoke runs; same architecture, ~1M non-embedding params
    "litegpt-tiny": ModelConfig(block_size=128, vocab_size=512, n_layer=2, n_embd=64, n_head=4, n_kv_head=2,
                                ffn_hidden=160),
}


def get_config(name: str, **overrides) -> ModelConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown config {name!r}; choose from {sorted(PRESETS)}")
    return ModelConfig.from_dict({**PRESETS[name].to_dict(), **overrides})
