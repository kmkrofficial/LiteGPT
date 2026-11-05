# Distribution and export

**Status: nothing is released, and export is not built.** Blocked by the licence gate ([ADR-004](decisions/ADR-004-licensing.md), proposal in [ADR-011](decisions/ADR-011-model-licence.md)).

## Planned

| Item | State |
|---|---|
| `LiteGPT-V2-Base` and `LiteGPT-V2-Instruct` weights as safetensors, with `litegpt.json` metadata | Not built |
| `litegpt/export/to_hf.py` with a logit-parity test | Not built. Parameter names mirror `Qwen3ForCausalLM`, but whether that class reproduces V2's QK-norm is untested. If logits don't match, the release loads through the `litegpt` package instead, and says so |
| Model cards | Drafts: [Base](release/model_card_base.md), [Instruct](release/model_card_instruct.md) |
| Notice files | Drafts: [Base](release/NOTICE.txt), [Instruct](release/NOTICE_instruct.txt) |
| Frozen report `runs/report-v2.0.0/` | Not built |

## Not distributed

The text-to-data fine-tunes: see [ADR-011](decisions/ADR-011-model-licence.md). Training data is not redistributed either: it is rebuilt from pinned public revisions.

## Later

Quantization (INT8, then INT4) is v2.1. GGUF is a stretch goal.
