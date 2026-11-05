"""Proxy-ladder model with switchable V1 / V2 parts (EXP-M2-LADDER).

Each switch changes exactly one component, so a ladder step isolates one change:

    positional  learned | rope           (V1: learned absolute positions; V2: RoPE)
    norm        layernorm | rmsnorm      (V1: LayerNorm without bias, eps 1e-5; V2: RMSNorm)
    qk_norm     False | True             (V2: per-head RMSNorm on q and k before RoPE)
    mlp         gelu | swiglu            (V1: GELU 4x; V2: SwiGLU)
    n_kv_head   == n_head | < n_head     (V1: multi-head; V2: grouped-query)

Parameter names follow `litegpt.model.LiteGPT` (HF Qwen3 layout), so with every switch on V2 the state dict loads into
LiteGPT unchanged (tested), and with every switch on V1 it reproduces `scripts/legacy_v1/model.py` (tested). Production code
stays in `litegpt/`; this module exists only for the ablation (ADR-007).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from litegpt.model import RMSNorm, apply_rope, build_rope_cache


@dataclass
class AblationConfig:
    n_layer: int = 8
    n_embd: int = 576
    n_head: int = 9
    n_kv_head: int = 9
    ffn_hidden: int = 2304
    vocab_size: int = 50304
    block_size: int = 2048
    positional: str = "learned"
    norm: str = "layernorm"
    qk_norm: bool = False
    mlp: str = "gelu"
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6            # RMSNorm eps (LayerNorm always uses 1e-5, as V1)
    init_std: float = 0.02

    def __post_init__(self):
        assert self.positional in ("learned", "rope") and self.norm in ("layernorm", "rmsnorm") and self.mlp in ("gelu", "swiglu")
        assert self.n_embd % self.n_head == 0 and self.n_head % self.n_kv_head == 0

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def to_dict(self) -> dict:
        return asdict(self)


def _norm(cfg: AblationConfig, dim: int) -> nn.Module:
    return nn.LayerNorm(dim, eps=1e-5, bias=False) if cfg.norm == "layernorm" else RMSNorm(dim, cfg.norm_eps)


class _Attention(nn.Module):
    def __init__(self, cfg: AblationConfig):
        super().__init__()
        self.cfg = cfg
        hd = cfg.head_dim
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * hd, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * hd, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * hd, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * hd, cfg.n_embd, bias=False)
        self.q_norm = RMSNorm(hd, cfg.norm_eps) if cfg.qk_norm else None
        self.k_norm = RMSNorm(hd, cfg.norm_eps) if cfg.qk_norm else None

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        c = self.cfg
        q = self.q_proj(x).view(B, T, c.n_head, c.head_dim)
        k = self.k_proj(x).view(B, T, c.n_kv_head, c.head_dim)
        v = self.v_proj(x).view(B, T, c.n_kv_head, c.head_dim)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if cos is not None:
            q, k = apply_rope(q, cos, sin).to(v.dtype), apply_rope(k, cos, sin).to(v.dtype)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=c.n_kv_head != c.n_head)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, -1))


class _MLP(nn.Module):
    def __init__(self, cfg: AblationConfig):
        super().__init__()
        self.kind = cfg.mlp
        if cfg.mlp == "swiglu":
            self.gate_proj = nn.Linear(cfg.n_embd, cfg.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(cfg.n_embd, cfg.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_hidden, cfg.n_embd, bias=False)

    def forward(self, x):
        if self.kind == "swiglu":
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return self.down_proj(F.gelu(self.up_proj(x)))


class _Block(nn.Module):
    def __init__(self, cfg: AblationConfig):
        super().__init__()
        self.input_layernorm = _norm(cfg, cfg.n_embd)
        self.self_attn = _Attention(cfg)
        self.post_attention_layernorm = _norm(cfg, cfg.n_embd)
        self.mlp = _MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class _Backbone(nn.Module):
    def __init__(self, cfg: AblationConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        if cfg.positional == "learned":
            self.embed_positions = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.layers = nn.ModuleList([_Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = _norm(cfg, cfg.n_embd)


class AblationGPT(nn.Module):
    def __init__(self, cfg: AblationConfig):
        super().__init__()
        self.config = cfg
        self.model = _Backbone(cfg)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight
        if cfg.positional == "rope":
            cos, sin = build_rope_cache(cfg.head_dim, cfg.block_size, cfg.rope_theta)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        for m in self.modules():                                   # same initialisation scheme as LiteGPT for every step
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=cfg.init_std)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=cfg.init_std / math.sqrt(2 * cfg.n_layer))

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.model.embed_tokens.weight.numel()
            if self.config.positional == "learned":
                n -= self.model.embed_positions.weight.numel()
        return n

    def flops_per_token(self, seq_len: int | None = None) -> float:
        c = self.config
        T = seq_len or c.block_size
        return 6 * self.num_params() + 12 * c.n_layer * c.n_head * c.head_dim * T

    def forward(self, idx, targets=None, logits_all: bool = False):
        B, T = idx.shape
        x = self.model.embed_tokens(idx)
        cos = sin = None
        if self.config.positional == "learned":
            x = x + self.model.embed_positions(torch.arange(T, device=idx.device))
        else:
            cos = self.rope_cos[:T][None, None].to(x.dtype)
            sin = self.rope_sin[:T][None, None].to(x.dtype)
        for block in self.model.layers:
            x = block(x, cos, sin)
        x = self.model.norm(x)
        if targets is not None:
            logits = self.lm_head(x)
            return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return self.lm_head(x if logits_all else x[:, [-1], :]), None

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        from litegpt.model import LiteGPT
        return LiteGPT.configure_optimizers(self, weight_decay, learning_rate, betas, device_type)


# ---------------------------------------------------------------- ladder presets (EXP-M2-LADDER) ------------------------
LADDER = {
    "s1": dict(),
    "s2": dict(positional="rope"),
    "s3": dict(positional="rope", norm="rmsnorm", qk_norm=True),
    "s4": dict(positional="rope", norm="rmsnorm", qk_norm=True, mlp="swiglu", ffn_hidden=1536),
    "s5": dict(positional="rope", norm="rmsnorm", qk_norm=True, mlp="swiglu", ffn_hidden=1792, n_kv_head=3),
}
# steps 6-8 use litegpt.model.LiteGPT itself (deep-thin proxy shape); see litegpt/training/config.py
LITEGPT_PROXY = dict(n_layer=18, n_embd=384, n_head=6, n_kv_head=2, ffn_hidden=1216, block_size=2048)


def ladder_config(step: str, vocab_size: int = 50304) -> AblationConfig:
    return AblationConfig(vocab_size=vocab_size, **LADDER[step])


def from_legacy_v1(legacy) -> "AblationGPT":
    """Build the V1-configured ablation model with the weights of a `scripts/legacy_v1/model.py` GPT (bias=False)."""
    lc = legacy.config
    assert not lc.bias, "the ladder's V1 baseline has no biases (V1 trained with bias=False)"
    cfg = AblationConfig(n_layer=lc.n_layer, n_embd=lc.n_embd, n_head=lc.n_head, n_kv_head=lc.n_head, ffn_hidden=4 * lc.n_embd,
                         vocab_size=lc.vocab_size, block_size=lc.block_size)
    m = AblationGPT(cfg)
    sd = legacy.state_dict()
    out = {"model.embed_tokens.weight": sd["transformer.wte.weight"], "model.embed_positions.weight": sd["transformer.wpe.weight"],
           "model.norm.weight": sd["transformer.ln_f.weight"], "lm_head.weight": sd["transformer.wte.weight"]}
    d = lc.n_embd
    for i in range(lc.n_layer):
        p, q = f"transformer.h.{i}.", f"model.layers.{i}."
        qkv = sd[p + "attn.c_attn.weight"]
        out[q + "self_attn.q_proj.weight"], out[q + "self_attn.k_proj.weight"], out[q + "self_attn.v_proj.weight"] = qkv[:d], qkv[d:2 * d], qkv[2 * d:]
        out[q + "self_attn.o_proj.weight"] = sd[p + "attn.c_proj.weight"]
        out[q + "input_layernorm.weight"] = sd[p + "ln_1.weight"]
        out[q + "post_attention_layernorm.weight"] = sd[p + "ln_2.weight"]
        out[q + "mlp.up_proj.weight"] = sd[p + "mlp.c_fc.weight"]
        out[q + "mlp.down_proj.weight"] = sd[p + "mlp.c_proj.weight"]
    m.load_state_dict(out)
    return m
