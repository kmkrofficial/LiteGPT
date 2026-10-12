"""Training run configuration (shared by the M2 proxy ladder and M3 pretraining)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

GPT2_VOCAB_PADDED = 50304          # GPT-2's 50,257 rounded up to a multiple of 64 (as V1)
LITEGPT_VOCAB = 32768


@dataclass
class TrainConfig:
    name: str
    model_kind: str                       # "ablation" (benchmarks/ablation.py ladder step) | "litegpt"
    model: dict                           # ablation: {"step": "s1", ...overrides}; litegpt: {"preset": "...", ...overrides}
    tokenizer: str                        # "gpt2" | "litegpt" (fixes the vocabulary and the bits-per-byte accounting)
    data_run: str                         # data run directory holding shards/ and shards/indexes/
    train_index: str                      # e.g. proxy_gpt2_train, proxy_1b, stable
    val_index: str                        # e.g. proxy_gpt2_val, val_core
    tokens: float                         # training tokens
    micro_batch: int
    accum: int
    lr: float
    val_text_dir: str | None = None       # raw-text validation (tokenizer-independent BPB)
    val_text_domains: list = field(default_factory=lambda: ["web", "synthetic", "code", "math", "chat"])
    warmup_frac: float = 0.01
    decay_frac: float = 0.2
    final_lr_frac: float = 0.1
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    seed: int = 1
    eval_every_tokens: float = 100e6
    eval_blocks: int = 512                # token-level validation blocks per evaluation
    checkpoint_every_minutes: float = 30.0
    keep_last: int = 2
    compile: bool = True
    device: str = "cuda"
    telemetry: bool = True
    abort_temp_c: float = 90.0
    block_len: int = 2048

    @property
    def vocab_size(self) -> int:
        return GPT2_VOCAB_PADDED if self.tokenizer == "gpt2" else LITEGPT_VOCAB

    @property
    def tokens_per_step(self) -> int:
        return self.micro_batch * self.accum * self.block_len

    @property
    def total_steps(self) -> int:
        return int(self.tokens // self.tokens_per_step)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["betas"] = list(self.betas)
        return d


def load_config(path, **overrides) -> TrainConfig:
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    d.update({k: v for k, v in overrides.items() if v is not None})
    if "betas" in d:
        d["betas"] = tuple(d["betas"])
    for k in ("tokens", "eval_every_tokens", "lr"):
        if isinstance(d.get(k), str):
            d[k] = float(d[k])
    return TrainConfig(**d)


def build_model(cfg: TrainConfig):
    """Model for a run: an ablation ladder step or LiteGPT (preset + overrides), with the vocabulary of the tokenizer."""
    if cfg.model_kind == "ablation":
        from benchmarks.ablation import AblationConfig, AblationGPT, LADDER
        m = dict(cfg.model)
        step = m.pop("step")
        return AblationGPT(AblationConfig(vocab_size=cfg.vocab_size, block_size=cfg.block_len, **{**LADDER[step], **m}))
    if cfg.model_kind == "litegpt":
        from litegpt.config import ModelConfig, get_config
        from litegpt.model import LiteGPT
        m = dict(cfg.model)
        preset = m.pop("preset", None)
        base = get_config(preset).to_dict() if preset else {}
        if m.pop("proxy", False):
            from benchmarks.ablation import LITEGPT_PROXY
            base.update(LITEGPT_PROXY)
        return LiteGPT(ModelConfig.from_dict({**base, **m, "vocab_size": cfg.vocab_size, "block_size": cfg.block_len}))
    raise ValueError(f"unknown model_kind {cfg.model_kind!r}")
