# Distribution and export

**Status: v2.0.0 is released.** The licence gate passed ([ADR-004](decisions/ADR-004-licensing.md), [ADR-011](decisions/ADR-011-model-licence.md)).

## Where it is

| What | Where |
|---|---|
| Code | [kmkrofficial/LiteGPT](https://github.com/kmkrofficial/LiteGPT), on top of V1's history |
| Base weights | [kmkrworks/LiteGPT-V2-Base](https://huggingface.co/kmkrworks/LiteGPT-V2-Base) |
| Instruct weights | [kmkrworks/LiteGPT-Llama-V2-Instruct](https://huggingface.co/kmkrworks/LiteGPT-Llama-V2-Instruct) |

V1's models stay at `koganrath/LiteGPT-Base` and `koganrath/LiteGPT-Instruct` on Hugging Face.

## What a model folder holds

Weights as safetensors at bfloat16, config, tokenizer, chat template, `litegpt.json` metadata, model card, notice and licence files. In float32, logits and greedy output are identical through Transformers and through LiteGPT (EXP-M5-EXPORT).

```bash
uv run --group hf python -m litegpt.export.to_hf --checkpoint milestone-final.pt --kind base --out LiteGPT-V2-Base
```

The frozen report for this release is [`runs/report-v2.0.0/REPORT.md`](../runs/report-v2.0.0/REPORT.md).

## Not distributed

The text-to-data fine-tunes: see [ADR-011](decisions/ADR-011-model-licence.md). Training data is not redistributed either: it is rebuilt from pinned public revisions.

## Later

Quantization (INT8, then INT4) is v2.1. GGUF is a stretch goal.
