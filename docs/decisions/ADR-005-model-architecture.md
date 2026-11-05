# ADR-005: V2 model architecture

Status: Accepted (evidence pending M2 proxy ladder)
Date: 2026-10-06
Spec fields: `model`, `tokenizer`, `chat_template`

## Context

V1 is GPT-2 Small:
- 12 layers × 768, multi-head attention, learned absolute positions, 1024 context.
- LayerNorm, GELU MLP, GPT-2 BPE with a 50,304-token vocabulary.
- 124.5M parameters, of which 31% are embeddings.

V2 must improve capability per parameter and per unit of compute, and stay trainable on 16 GB.

## Decision

| Component | V1 | V2 | Reason |
|---|---|---|---|
| Positions | learned `wpe` table | RoPE θ=10k (100k for 4096 extension) | relative-position bias, extensible context, no parameters |
| Norm | LayerNorm | RMSNorm (pre-norm) | same quality, cheaper |
| Attention stability | none | QK-norm (per-head RMSNorm before RoPE) | bounded logits, higher usable LR, fewer loss spikes |
| FFN | 4d GELU | SwiGLU, hidden 2048 | lower loss at matched parameters and FLOPs |
| Attention | MHA 12 heads | GQA 12 query / 4 KV | parameters moved into depth, 3× smaller KV cache |
| Shape | 12 × 768 | **28 × 768** | depth beats width under 1B parameters |
| Tokenizer | GPT-2 50k | custom 32k byte-level BPE, single digits, chat + 32 reserved tokens | arithmetic, chat structure, smaller embedding and logits |
| Context | 1024 | 2048 (4096 extension in decay) | multi-turn chat, worked solutions |
| Init | 0.02, scaled `c_proj` | 0.02, scaled `o_proj` and `w2` (1/√(2L)) | stable start at 28 layers |
| Kept from V1 | — | decoder-only pre-norm, tied embeddings, no biases, AdamW, bf16, SDPA | proven, simple |

- **Size:** 201.4M parameters total (176.2M non-embedding, 25.2M tied embedding).
- **Training cost:** about 1.47 GFLOP per token.
- **HF compatibility:** the tensor layout is identical to HF `Qwen3ForCausalLM`, so export is a weight rename verified by a logit-parity test. The implementation is our own.

## Alternatives rejected
- **MoE:** costs VRAM and adds complexity.
- **Multi-token prediction:** hurts below about 1B parameters.
- **MLA, linear attention, SSM hybrids:** need custom kernels and give little benefit at 2k context.
- **Sliding-window attention:** pointless at 2k.
- **Logit soft-capping:** breaks the flash path.
- **Vocabulary above 100k:** would be about 40% of the parameters.
- **MQA:** too much quality loss.
- **Simply widening GPT-2:** rejected by the project brief.

## Consequences
- `from_pretrained('gpt2')` loading and the V1 ONNX export don't apply to V2.
- The V1 code stays in place for baselines until M5, then moves to `legacy/v1/`.

## Evidence
- Pending M2: the proxy architecture ladder (V1-arch → +RoPE → +RMSNorm/QK-norm → +SwiGLU → +GQA → +deep-thin → +tokenizer → +mixture), with registry IDs and the report section generated from the records.
