# Distribution and export

**Status: nothing is released, and export is not built.** Blocked by the licence gate ([ADR-004](decisions/ADR-004-licensing.md), proposal in [ADR-011](decisions/ADR-011-model-licence.md)).

## Where it will be published

| What | Where |
|---|---|
| Code | `github.com/kmkrofficial/LiteGPT`, where V1 already is |
| Base weights | `huggingface.co/kmkrworks/LiteGPT-V2-Base` |
| Instruct weights | `huggingface.co/kmkrworks/LiteGPT-Llama-V2-Instruct` |

V2's commits go on top of V1's, so V1's history stays. V1's models stay at `koganrath/LiteGPT-Base` and `koganrath/LiteGPT-Instruct` on Hugging Face. The V2 model repos above don't exist yet, and no V2 code is public yet.

## Planned

| Item | State |
|---|---|
| `LiteGPT-V2-Base` and `LiteGPT-Llama-V2-Instruct` weights as safetensors, with `litegpt.json` metadata | Not built |
| `litegpt/export/to_hf.py` with a logit-parity test | Not built. Parameter names mirror `Qwen3ForCausalLM`, but whether that class reproduces V2's QK-norm is untested. If logits don't match, the release loads through the `litegpt` package instead, and says so |
| Model cards | Drafts: [Base](release/model_card_base.md), [Instruct](release/model_card_instruct.md) |
| Notice files | Drafts: [Base](release/NOTICE.txt), [Instruct](release/NOTICE_instruct.txt) |
| Frozen report `runs/report-v2.0.0/` | Not built |

## Not distributed

The text-to-data fine-tunes: see [ADR-011](decisions/ADR-011-model-licence.md). Training data is not redistributed either: it is rebuilt from pinned public revisions.

## Later

Quantization (INT8, then INT4) is v2.1. GGUF is a stretch goal.
