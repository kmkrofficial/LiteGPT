import math

import pytest
import torch
import torch.nn.functional as F

from litegpt.config import get_config
from litegpt.model import KVCache, LiteGPT, apply_rope, build_rope_cache, doc_ids_from_cu_seqlens, flash_attn_available


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    m = LiteGPT(get_config("litegpt-tiny")).eval()
    # randomise norm weights so QK-norm / RMSNorm paths are non-trivial in parity checks
    for n, p in m.named_parameters():
        if "norm" in n:
            torch.nn.init.uniform_(p, 0.5, 1.5)
    return m


def test_tied_embeddings_and_param_sharing(tiny):
    assert tiny.lm_head.weight is tiny.model.embed_tokens.weight


def test_init_statistics():
    cfg = get_config("litegpt-v2-24l", n_layer=24)
    torch.manual_seed(0)
    with torch.device("cpu"):
        m = LiteGPT(get_config("litegpt-tiny", n_layer=8, n_embd=256, n_head=8, n_kv_head=4, ffn_hidden=704))
    c = m.config
    q = m.model.layers[0].self_attn.q_proj.weight
    assert q.std().item() == pytest.approx(0.02, rel=0.1)
    o = m.model.layers[0].self_attn.o_proj.weight
    assert o.std().item() == pytest.approx(0.02 / math.sqrt(2 * c.n_layer), rel=0.1)
    d = m.model.layers[3].mlp.down_proj.weight
    assert d.std().item() == pytest.approx(0.02 / math.sqrt(2 * c.n_layer), rel=0.1)
    assert torch.all(m.model.norm.weight == 1)
    assert cfg.n_layer == 24


def test_kv_cache_incremental_equals_full_forward(tiny):
    cfg = tiny.config
    idx = torch.randint(0, cfg.vocab_size, (2, 40))
    with torch.no_grad():
        full, _ = tiny(idx, logits_all=True)
        cache = KVCache(cfg, batch_size=2, max_len=64, dtype=torch.float32)
        pre, _ = tiny(idx[:, :24], kv_cache=cache, logits_all=True)
        assert cache.length == 24
        torch.testing.assert_close(pre, full[:, :24], atol=1e-5, rtol=1e-4)
        outs = []
        for t in range(24, 40):  # one token at a time
            lo, _ = tiny(idx[:, t:t + 1], kv_cache=cache)
            outs.append(lo)
        torch.testing.assert_close(torch.cat(outs, 1), full[:, 24:], atol=1e-5, rtol=1e-4)
        assert cache.length == 40


def test_chunked_prefill_with_cached_prefix_equals_full(tiny):
    cfg = tiny.config
    idx = torch.randint(0, cfg.vocab_size, (1, 50))
    with torch.no_grad():
        full, _ = tiny(idx, logits_all=True)
        cache = KVCache(cfg, 1, 64, dtype=torch.float32)
        a, _ = tiny(idx[:, :20], kv_cache=cache, logits_all=True)
        b, _ = tiny(idx[:, 20:50], kv_cache=cache, logits_all=True)  # T>1 with start_pos>0 -> offset causal mask
    torch.testing.assert_close(torch.cat([a, b], 1), full, atol=1e-5, rtol=1e-4)


def test_rope_is_relative_and_norm_preserving():
    hd, n = 16, 64
    cos, sin = build_rope_cache(hd, n, 10000.0)
    torch.manual_seed(1)
    q, k = torch.randn(1, 1, 1, hd), torch.randn(1, 1, 1, hd)

    def score(i, j):
        qi = apply_rope(q, cos[i][None, None, None], sin[i][None, None, None])
        kj = apply_rope(k, cos[j][None, None, None], sin[j][None, None, None])
        return (qi * kj).sum().item()

    assert score(10, 3) == pytest.approx(score(30, 23), abs=1e-4)   # depends only on i-j
    assert score(10, 3) != pytest.approx(score(10, 4), abs=1e-3)
    r = apply_rope(q, cos[7][None, None, None], sin[7][None, None, None])
    assert r.norm().item() == pytest.approx(q.norm().item(), rel=1e-5)
    c0, s0 = cos[0], sin[0]
    assert torch.all(c0 == 1) and torch.all(s0 == 0)  # position 0 is the identity rotation


def test_gqa_matches_explicit_mha_with_repeated_kv(tiny):
    """Model attention (SDPA + enable_gqa) == hand-written softmax attention with KV heads repeated."""
    cfg = tiny.config
    attn = tiny.model.layers[0].self_attn
    B, T = 2, 17
    x = torch.randn(B, T, cfg.n_embd)
    cos, sin = (t[None, None, :T] for t in build_rope_cache(cfg.head_dim, cfg.block_size, cfg.rope_theta))
    with torch.no_grad():
        got = attn(x, cos[0], sin[0])
        q = attn.q_norm(attn.q_proj(x).view(B, T, cfg.n_head, cfg.head_dim)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(x).view(B, T, cfg.n_kv_head, cfg.head_dim)).transpose(1, 2)
        v = attn.v_proj(x).view(B, T, cfg.n_kv_head, cfg.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos[0], sin[0]), apply_rope(k, cos[0], sin[0])
        rep = cfg.n_head // cfg.n_kv_head
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(cfg.head_dim)
        s = s.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf"))
        ref = attn.o_proj((s.softmax(-1) @ v).transpose(1, 2).reshape(B, T, -1))
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-4)


def test_cu_seqlens_packing_equals_separate_sequences(tiny):
    cfg = tiny.config
    a, b, c = (torch.randint(0, cfg.vocab_size, (1, n)) for n in (13, 22, 9))
    packed = torch.cat([a, b, c], 1)  # one row, T=44
    cu = torch.tensor([0, 13, 35, 44], dtype=torch.int32)
    with torch.no_grad():
        lp, _ = tiny(packed, cu_seqlens=cu, logits_all=True)
        for seg, (s, e) in zip((a, b, c), ((0, 13), (13, 35), (35, 44))):
            ls, _ = tiny(seg, logits_all=True)
            torch.testing.assert_close(lp[:, s:e], ls, atol=1e-5, rtol=1e-4)


def test_doc_ids_and_position_reset():
    cu = torch.tensor([0, 3, 8, 12], dtype=torch.int32)
    doc, pos = doc_ids_from_cu_seqlens(cu, B=2, T=6)
    assert doc.tolist() == [[0, 0, 0, 1, 1, 1], [1, 1, 2, 2, 2, 2]]
    assert pos.tolist() == [[0, 1, 2, 0, 1, 2], [3, 4, 0, 1, 2, 3]]


def test_loss_ignore_index_masks_positions(tiny):
    cfg = tiny.config
    idx = torch.randint(0, cfg.vocab_size, (3, 20))
    tgt = torch.randint(0, cfg.vocab_size, (3, 20))
    masked = tgt.clone()
    masked[:, :8] = -1
    with torch.no_grad():
        logits, loss_masked = tiny(idx, masked)
        ref = F.cross_entropy(logits[:, 8:].reshape(-1, cfg.vocab_size), tgt[:, 8:].reshape(-1))
        changed = masked.clone()
        changed[:, 8:] = tgt[:, 8:]  # same unmasked targets ...
        _, loss_same = tiny(idx, changed)
        other = masked.clone()
        other[:, 8:] = (tgt[:, 8:] + 1) % cfg.vocab_size  # ... vs different unmasked targets
        _, loss_other = tiny(idx, other)
    assert loss_masked.item() == pytest.approx(ref.item(), rel=1e-5)
    assert loss_same.item() == pytest.approx(loss_masked.item(), rel=1e-5)
    assert loss_other.item() != loss_masked.item()  # unmasked targets do matter; masked ones do not


def test_backward_gives_finite_grads_everywhere():
    torch.manual_seed(0)
    m = LiteGPT(get_config("litegpt-tiny"))
    x = torch.randint(0, 512, (2, 32))
    _, loss = m(x, torch.roll(x, -1, 1))
    loss.backward()
    for n, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n
        assert p.grad.abs().sum() > 0, f"no gradient reaches {n}"


def test_context_overflow_is_rejected(tiny):
    with pytest.raises(AssertionError):
        tiny(torch.zeros(1, tiny.config.block_size + 1, dtype=torch.long))


def test_flash_backend_unavailable_raises_clear_error_or_matches_sdpa():
    cfg = get_config("litegpt-tiny", attention_backend="flash")
    if not flash_attn_available():
        with pytest.raises(RuntimeError, match="flash-attn"):
            LiteGPT(cfg)


@pytest.mark.skipif(not (flash_attn_available() and torch.cuda.is_available()), reason="flash-attn/CUDA not available")
def test_flash_matches_sdpa_on_gpu():
    torch.manual_seed(0)
    base = get_config("litegpt-tiny", n_embd=128, n_head=4, n_kv_head=2, ffn_hidden=320)
    a = LiteGPT(base).cuda().to(torch.bfloat16).eval()
    b = LiteGPT(get_config("litegpt-tiny", n_embd=128, n_head=4, n_kv_head=2, ffn_hidden=320,
                           attention_backend="flash")).cuda().to(torch.bfloat16).eval()
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, 512, (2, 64), device="cuda")
    with torch.no_grad():
        la, _ = a(x, logits_all=True)
        lb, _ = b(x, logits_all=True)
    torch.testing.assert_close(la, lb, atol=5e-2, rtol=5e-2)


def test_auto_backend_falls_back_to_sdpa_on_cpu_fp32_and_cached_decoding():
    from litegpt.model import effective_attention_backend
    assert effective_attention_backend("auto", "cpu", torch.float32) == "sdpa"
    assert effective_attention_backend("auto", "cuda", torch.float32) == "sdpa"
    assert effective_attention_backend("auto", "cuda", torch.bfloat16, kv_cache=True) == "sdpa"
    assert effective_attention_backend("sdpa", "cuda", torch.bfloat16) == "sdpa"
    m = LiteGPT(get_config("litegpt-tiny", attention_backend="auto")).eval()  # must run on CPU fp32 either way
    with torch.no_grad():
        logits, _ = m(torch.randint(0, 512, (1, 8)))
    assert torch.isfinite(logits).all()


@pytest.mark.skipif(not (flash_attn_available() and torch.cuda.is_available()), reason="flash-attn/CUDA not available")
def test_auto_uses_flash_on_cuda_bf16_and_matches_sdpa():
    from litegpt.model import effective_attention_backend
    assert effective_attention_backend("auto", "cuda", torch.bfloat16) == "flash"
    torch.manual_seed(0)
    kw = dict(n_embd=128, n_head=4, n_kv_head=2, ffn_hidden=320)
    a = LiteGPT(get_config("litegpt-tiny", attention_backend="sdpa", **kw)).cuda().to(torch.bfloat16).eval()
    b = LiteGPT(get_config("litegpt-tiny", attention_backend="auto", **kw)).cuda().to(torch.bfloat16).eval()
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, 512, (2, 32), device="cuda")
    cu = torch.tensor([0, 20, 32, 64], dtype=torch.int32, device="cuda")
    with torch.no_grad():
        for kwargs in ({}, {"cu_seqlens": cu}):  # plain causal and packed documents (max_seqlen derived automatically)
            la, _ = a(x, logits_all=True, **kwargs)
            lb, _ = b(x, logits_all=True, **kwargs)
            torch.testing.assert_close(la, lb, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not (flash_attn_available() and torch.cuda.is_available()), reason="flash-attn/CUDA not available")
def test_flash_varlen_packing_matches_sdpa_block_mask_on_gpu():
    """cu_seqlens: flash varlen kernel vs SDPA block-diagonal causal mask, incl. a document boundary at a row boundary."""
    torch.manual_seed(0)
    kw = dict(n_embd=128, n_head=4, n_kv_head=2, ffn_hidden=320)
    a = LiteGPT(get_config("litegpt-tiny", **kw)).cuda().to(torch.bfloat16).eval()
    b = LiteGPT(get_config("litegpt-tiny", attention_backend="flash", **kw)).cuda().to(torch.bfloat16).eval()
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, 512, (2, 32), device="cuda")
    cu = torch.tensor([0, 20, 32, 64], dtype=torch.int32, device="cuda")  # row0: docs 20+12, row1: one doc of 32
    with torch.no_grad():
        la, _ = a(x, cu_seqlens=cu, max_seqlen=32, logits_all=True)
        lb, _ = b(x, cu_seqlens=cu, max_seqlen=32, logits_all=True)
    torch.testing.assert_close(la, lb, atol=5e-2, rtol=5e-2)


def _triton_available() -> bool:
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not (torch.cuda.is_available() and _triton_available()), reason="needs CUDA and Triton (WSL2/Linux)")
def test_compiled_matches_eager_on_gpu():
    """torch.compile must not change the model's outputs beyond bf16 tolerance (forward logits and loss)."""
    torch.manual_seed(0)
    kw = dict(n_embd=128, n_head=4, n_kv_head=2, ffn_hidden=320)
    m = LiteGPT(get_config("litegpt-tiny", attention_backend="sdpa", **kw)).cuda().eval()
    c = torch.compile(m)
    x = torch.randint(0, 512, (2, 64), device="cuda")
    y = torch.roll(x, -1, 1)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        le, loss_e = m(x, y)
        lc, loss_c = c(x, y)
    torch.testing.assert_close(le.float(), lc.float(), atol=5e-2, rtol=5e-2)
    assert abs(loss_e.item() - loss_c.item()) < 1e-2


@pytest.mark.skipif(not (torch.cuda.is_available() and _triton_available() and flash_attn_available()),
                    reason="needs CUDA, Triton and flash-attn (WSL2/Linux)")
def test_compiled_flash_matches_eager_sdpa_on_gpu():
    """flash-attn under torch.compile must match the eager SDPA reference (plain causal and packed documents)."""
    torch.manual_seed(0)
    kw = dict(n_embd=128, n_head=4, n_kv_head=2, ffn_hidden=320)
    ref = LiteGPT(get_config("litegpt-tiny", attention_backend="sdpa", **kw)).cuda().eval()
    fl = LiteGPT(get_config("litegpt-tiny", attention_backend="flash", **kw)).cuda().eval()
    fl.load_state_dict(ref.state_dict())
    cfl = torch.compile(fl)
    x = torch.randint(0, 512, (2, 64), device="cuda")
    cu = torch.tensor([0, 40, 64, 128], dtype=torch.int32, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for kwargs in ({}, {"cu_seqlens": cu, "max_seqlen": 64}):
            la, _ = ref(x, logits_all=True, **kwargs)
            lb, _ = cfl(x, logits_all=True, **kwargs)
            torch.testing.assert_close(la.float(), lb.float(), atol=5e-2, rtol=5e-2)
