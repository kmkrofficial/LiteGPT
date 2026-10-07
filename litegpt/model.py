"""
LiteGPT V2 model: a decoder-only transformer with RoPE, RMSNorm + QK-norm, SwiGLU and grouped-query attention.

Everything architectural lives in this file; it knows nothing about tokenizers, sampling or chat templates.
Module / parameter names deliberately mirror HF `Qwen3ForCausalLM` (model.embed_tokens, model.layers.N.self_attn.q_proj,
..., model.norm, lm_head) so that export to Hugging Face is a plain tensor copy (see ADR-005). The implementation itself
is our own. Structure follows nanoGPT/V1 (pre-norm residual blocks, tied embeddings, no biases).

Attention backends (ADR-006): PyTorch SDPA is mandatory and always available. flash-attn is optional; it was adopted for
'auto' on measured evidence (EXP-M0-FLASH: parity tests pass, +12% tok/s). 'auto' uses flash only when flash-attn is installed,
the tensors are on CUDA in bf16/fp16 and no KV cache is used; everything else (CPU, fp32, cached decoding) uses SDPA.
'flash' forces flash (error if unavailable); 'sdpa' forces SDPA.
"""

from __future__ import annotations

import inspect
import math
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


@lru_cache(maxsize=1)
def flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False


def resolve_backend(requested: str) -> str:
    """Config-level resolution: 'flash' | 'auto' (flash if installed, decided per call) | 'sdpa'."""
    if requested == "flash":
        if not flash_attn_available():
            raise RuntimeError("attention_backend='flash' requested but flash-attn is not installed (use 'sdpa' or 'auto')")
        return "flash"
    if requested == "auto" and flash_attn_available():
        return "auto"
    return "sdpa"


def effective_attention_backend(requested: str, device_type: str, dtype: torch.dtype, kv_cache: bool = False) -> str:
    """What actually runs for a given call: 'flash' or 'sdpa'. Used by the model and by run records."""
    r = resolve_backend(requested)
    if r == "flash":
        return "flash"
    if r == "auto" and device_type == "cuda" and dtype in (torch.bfloat16, torch.float16) and not kv_cache:
        return "flash"
    return "sdpa"


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


def build_rope_cache(head_dim: int, max_len: int, theta: float, device=None):
    """cos/sin tables of shape (max_len, head_dim) using the rotate-half layout (matches HF Qwen/Llama)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(max_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    """x: (B, n_heads, T, head_dim); cos/sin broadcastable to (B or 1, 1, T, head_dim)."""
    return x * cos + _rotate_half(x) * sin


class KVCache:
    """Preallocated key/value cache: [n_layer, B, n_kv_head, max_len, head_dim]."""

    def __init__(self, config: ModelConfig, batch_size: int, max_len: int | None = None, device=None, dtype=torch.bfloat16):
        max_len = max_len or config.block_size
        assert max_len <= config.block_size
        shape = (config.n_layer, batch_size, config.n_kv_head, max_len, config.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.max_len = max_len
        self.length = 0

    def update(self, layer: int, k_new, v_new, start_pos: int):
        T = k_new.size(2)
        assert start_pos + T <= self.max_len, f"KV cache overflow: {start_pos + T} > {self.max_len}"
        self.k[layer, :, :, start_pos:start_pos + T] = k_new
        self.v[layer, :, :, start_pos:start_pos + T] = v_new
        return self.k[layer, :, :, :start_pos + T], self.v[layer, :, :, :start_pos + T]

    def reset(self):
        self.length = 0


def doc_ids_from_cu_seqlens(cu_seqlens: torch.Tensor, B: int, T: int):
    """cu_seqlens: (n_docs+1,) cumulative lengths over the flattened B*T tokens (flash-attn varlen convention).
    Documents must not cross row boundaries, i.e. every multiple of T must appear in cu_seqlens.
    Returns (doc_id (B,T), position_ids (B,T) restarting at 0 for every document)."""
    pos = torch.arange(B * T, device=cu_seqlens.device)
    doc = torch.searchsorted(cu_seqlens[1:].contiguous(), pos, right=True)
    position_ids = pos - cu_seqlens[doc]
    return doc.view(B, T), position_ids.view(B, T)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head, self.n_kv_head, self.head_dim = config.n_head, config.n_kv_head, config.head_dim
        d = config.n_embd
        self.q_proj = nn.Linear(d, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, d, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.norm_eps) if config.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, config.norm_eps) if config.qk_norm else None
        self.dropout = config.dropout
        self.backend = resolve_backend(config.attention_backend)

    def forward(self, x, cos, sin, kv_cache=None, start_pos=0, attn_mask=None, cu_seqlens=None, max_seqlen=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        if self.q_norm is not None:  # QK-norm: per-head RMSNorm over head_dim, before RoPE
            q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (B, h, T, hd)
        q, k = apply_rope(q, cos, sin).to(v.dtype), apply_rope(k, cos, sin).to(v.dtype)
        p = self.dropout if self.training else 0.0

        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, k, v, start_pos)

        if kv_cache is None and (self.backend == "flash" or (
                self.backend == "auto" and effective_attention_backend("auto", q.device.type, q.dtype) == "flash")):
            from flash_attn import flash_attn_func, flash_attn_varlen_func
            qf, kf, vf = (t.transpose(1, 2) for t in (q, k, v))  # (B, T, h, hd)
            if cu_seqlens is not None:
                cu_seqlens = cu_seqlens.to(device=q.device, dtype=torch.int32)
                if max_seqlen is None:
                    max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())
                y = flash_attn_varlen_func(qf.reshape(B * T, -1, self.head_dim), kf.reshape(B * T, -1, self.head_dim),
                                           vf.reshape(B * T, -1, self.head_dim), cu_seqlens, cu_seqlens, max_seqlen,
                                           max_seqlen, dropout_p=p, causal=True).view(B, T, -1, self.head_dim)
            else:
                y = flash_attn_func(qf, kf, vf, dropout_p=p, causal=True)
            y = y.reshape(B, T, -1)
        else:
            if attn_mask is not None:            # packed documents (block-diagonal causal) or cache offset prefill
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=p, enable_gqa=True)
            elif T == 1 and kv_cache is not None:  # single-token decode attends to everything cached
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, enable_gqa=True)
            else:
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=True, enable_gqa=True)
            y = y.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(y)


class MLP(nn.Module):
    """SwiGLU: down(silu(gate(x)) * up(x))."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(config.ffn_hidden, config.n_embd, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(config.n_embd, config.norm_eps)
        self.self_attn = Attention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.n_embd, config.norm_eps)
        self.mlp = MLP(config)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x, **attn_kwargs):
        x = x + self.drop(self.self_attn(self.input_layernorm(x), **attn_kwargs))
        x = x + self.drop(self.mlp(self.post_attention_layernorm(x)))
        return x


class _Backbone(nn.Module):
    """Holds embed_tokens / layers / norm so parameter names match HF (model.*)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd, config.norm_eps)


class LiteGPT(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = _Backbone(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        cos, sin = build_rope_cache(config.head_dim, config.block_size, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # scaled init of residual output projections (1/sqrt(2*n_layer)), per GPT-2 / ADR-005
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=config.init_std / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    # ----- introspection -------------------------------------------------------------------------------------------
    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())  # tied weights are counted once
        if non_embedding:
            n -= self.model.embed_tokens.weight.numel()
        return n

    def flops_per_token(self, seq_len: int | None = None, attention: str = "full") -> float:
        """Training FLOPs per token (fwd+bwd). 'full' = PaLM/nanoGPT convention (6N + 12*L*H*hd*T, used for MFU);
        'causal' halves the attention term (what a causal kernel actually computes)."""
        c = self.config
        T = seq_len or c.block_size
        attn = 12 * c.n_layer * c.n_head * c.head_dim * T
        return 6 * self.num_params() + (attn if attention == "full" else attn / 2)

    # ----- forward -------------------------------------------------------------------------------------------------
    def forward(self, idx, targets=None, kv_cache: KVCache | None = None, start_pos: int | None = None,
                cu_seqlens: torch.Tensor | None = None, max_seqlen: int | None = None, logits_all: bool = False):
        """idx: (B, T) token ids. targets: (B, T) with -1 = ignore (assistant-only loss masks use this).
        kv_cache: incremental decoding; start_pos defaults to kv_cache.length. cu_seqlens: packed documents (see
        doc_ids_from_cu_seqlens), RoPE positions restart per document and attention is block-diagonal causal.
        Returns (logits, loss). Without targets only the last position's logits are returned unless logits_all."""
        B, T = idx.shape
        cfg = self.config
        if start_pos is None:
            start_pos = kv_cache.length if kv_cache is not None else 0
        assert start_pos + T <= cfg.block_size, f"sequence end {start_pos + T} exceeds block_size {cfg.block_size}"
        assert not (kv_cache is not None and cu_seqlens is not None), "packed sequences are for training, not cached decoding"

        attn_mask = None
        if cu_seqlens is not None:
            doc, position_ids = doc_ids_from_cu_seqlens(cu_seqlens, B, T)
            cos, sin = self.rope_cos[position_ids].unsqueeze(1), self.rope_sin[position_ids].unsqueeze(1)
            if resolve_backend(cfg.attention_backend) != "flash":  # 'auto'/'sdpa' may need the mask (SDPA fallback)
                causal = torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()
                attn_mask = ((doc[:, :, None] == doc[:, None, :]) & causal).unsqueeze(1)
        else:
            cos = self.rope_cos[start_pos:start_pos + T][None, None]
            sin = self.rope_sin[start_pos:start_pos + T][None, None]
            if kv_cache is not None and T > 1 and start_pos > 0:  # chunked prefill / continuation with cached prefix
                i = torch.arange(T, device=idx.device)[:, None] + start_pos
                j = torch.arange(start_pos + T, device=idx.device)[None, :]
                attn_mask = (j <= i)[None, None]

        x = self.model.embed_tokens(idx)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        for block in self.model.layers:
            x = block(x, cos=cos, sin=sin, kv_cache=kv_cache, start_pos=start_pos, attn_mask=attn_mask,
                      cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        x = self.model.norm(x)
        if kv_cache is not None:
            kv_cache.length = start_pos + T

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x if logits_all else x[:, [-1], :])
            loss = None
        return logits, loss

    # ----- optimizer (same grouping as V1: decay 2D weights, no decay for norms) ----------------------------------
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        params = {n: p for n, p in self.named_parameters() if p.requires_grad}
        decay = [p for p in params.values() if p.dim() >= 2]
        nodecay = [p for p in params.values() if p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay}, {"params": nodecay, "weight_decay": 0.0}]
        use_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters and device_type == "cuda"
        extra = dict(fused=True) if use_fused else {}
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **extra)
