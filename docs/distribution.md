# Distribution and export

**Status: nothing is released.** The licence gate has passed ([ADR-004](decisions/ADR-004-licensing.md), [ADR-011](decisions/ADR-011-model-licence.md)). Publishing still needs the owner's final yes.

## Where it will be published

| What | Where |
|---|---|
| Code | `github.com/kmkrofficial/LiteGPT`, where V1 already is |
| Base weights | `huggingface.co/kmkrworks/LiteGPT-V2-Base` |
| Instruct weights | `huggingface.co/kmkrworks/LiteGPT-Llama-V2-Instruct` |

V2's commits go on top of V1's, so V1's history stays. V1's models stay at `koganrath/LiteGPT-Base` and `koganrath/LiteGPT-Instruct` on Hugging Face. The V2 model repos above don't exist yet, and no V2 code is public yet.

## State

| Item | State |
|---|---|
| `litegpt/export/to_hf.py` | Built. Writes safetensors at bfloat16, config, tokenizer, chat template, `litegpt.json`, model card, notice and licence files |
| Both models exported | Done locally, 387 MB each. To be rebuilt once more when the cards are final |
| Parity with Transformers (EXP-M5-EXPORT) | Logits and greedy output identical in float32. **One rule failed as written:** tokenizer ids matched on 195 of 200 strings, all 5 holding literal control-token text. Awaiting the owner's decision |
| Model cards | Drafts: [Base](release/model_card_base.md), [Instruct](release/model_card_instruct.md) |
| Notice files | Drafts: [Base](release/NOTICE.txt), [Instruct](release/NOTICE_instruct.txt) |
| Frozen report `runs/report-v2.0.0/` | Not built |

```bash
uv run --group hf python -m litegpt.export.to_hf --checkpoint milestone-final.pt --kind base --out LiteGPT-V2-Base
```

## Not distributed

The text-to-data fine-tunes: see [ADR-011](decisions/ADR-011-model-licence.md). Training data is not redistributed either: it is rebuilt from pinned public revisions.

## Later

Quantization (INT8, then INT4) is v2.1. GGUF is a stretch goal.
